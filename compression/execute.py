"""Local, resumable CPU treatment preparation followed by optional GPU evaluation."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import torch
from evaluation.adapters import LocalAdapter
from evaluation.config import EvaluationConfig
from experiments.store import read_json, write_json
from experiments.provenance import canonical_hash, file_hash
from compression.multimodal import base_identity, protected_identity, export_treatment


def execute(config, plan, output, *, resume=False, authorize_gpu=False):
    allowed = {'stages', 'max_model_bytes', 'calibration_data', 'recovery_batches', 'evaluate'}
    if not isinstance(plan, dict) or set(plan) - allowed:
        raise ValueError('Unknown execution plan fields')
    stages = plan.get('stages')
    if not isinstance(stages, list) or not 1 <= len(stages) <= 4 or any(s not in {'dense', 'prune', 'recover', 'awq'} for s in stages) or len(set(stages)) != len(stages):
        raise ValueError('Provide 1..4 unique dense/prune/recover/awq stages')
    if stages != sorted(stages, key=['dense','prune','recover','awq'].index):
        raise ValueError('Stage order must be dense, prune, recover, awq')
    budget = plan.get('max_model_bytes', 64*1024*1024)
    if type(budget) is not int or budget < 1:
        raise ValueError('Explicit model byte budget required')
    root = Path(output)
    inputs = {key: file_hash(plan[key], 8*1024*1024) for key in ('calibration_data','recovery_batches') if key in plan}
    if any(value['status'] != 'HASHED' for value in inputs.values()):
        raise ValueError('Preparation data must be local and within the 8 MiB input budget')
    identity = canonical_hash({'config':config.to_dict(), 'plan':plan, 'inputs':inputs})
    if root.exists():
        if not resume or read_json(root/'plan.json')['identity'] != identity:
            raise ValueError('Existing output requires --resume with the identical plan/config')
    else:
        root.mkdir(parents=True)
        write_json(root/'plan.json', {'identity':identity, 'plan':plan, 'config':config.to_dict(), 'inputs':inputs})
    # CPU preparation is intentional. The target CUDA dtype/device only applies
    # to subsequent evaluation, preventing accidental full-model GPU allocation.
    cpu = replace(config, device='cpu', dtype='float32', treatment='baseline', treatment_artifact_path=None)
    from evaluation.inspection import compatibility
    inspected = compatibility(cpu)
    disk_weights = sum((inspected.get(key) or {}).get('artifact_bytes') or 0 for key in ('model','vision'))
    if inspected['errors'] or disk_weights > budget:
        raise ValueError('Model files fail compatibility or preparation byte budget')
    import psutil
    # Conservative admission for FP32 restoration, training/export copies and
    # temporary buffers. This is an estimate, not an OS-enforced memory quota.
    if psutil.virtual_memory().available < disk_weights * 8:
        raise ValueError('Insufficient available host RAM for conservative preparation estimate')
    adapter=LocalAdapter(); adapter.load(cpu)
    source=base_identity(cpu); protected=protected_identity(adapter.model)
    widths=[b.mlp.fc1.out_features for b in adapter.model.model.layers]
    if sum(t.numel()*t.element_size() for t in adapter.model.state_dict().values()) > budget:
        raise ValueError('Loaded model exceeds preparation byte budget')
    recovery=None
    for index, stage in enumerate(stages):
        target=root/f'{index:02d}-{stage}'
        receipt=root/f'{index:02d}-complete.json'
        if receipt.exists():
            from compression.multimodal import apply_treatment
            done=read_json(receipt)
            if read_json(target/'treatment.json')['sha256'] != done['artifact_sha256']:
                raise ValueError('Checkpoint receipt identity mismatch')
            restored_config=replace(cpu,treatment=done['treatment'],treatment_artifact_path=str(target),max_treatment_bytes=budget)
            # Reload from base for every checkpoint; never apply deltas twice.
            adapter=LocalAdapter(); adapter.load(restored_config)
            recovery=done.get('recovery')
            continue
        if target.exists():
            raise ValueError(f'Incomplete checkpoint at {target}; preserve it and use a new output root')
        start=time.perf_counter()
        if stage=='prune':
            from compression.pruning import prune, PruningConfig
            prune(adapter.model,PruningConfig(method='mlp_neurons',sparsity=.25))
        elif stage=='recover':
            from compression.recovery import recover_lora, RecoveryConfig
            batches=read_json(plan['recovery_batches'])
            batches=[{k:torch.tensor(v,dtype=torch.long) for k,v in b.items()} for b in batches]
            trained,recovery=recover_lora(adapter.model,batches,RecoveryConfig(max_parameter_elements=budget//4))
            adapter.model=trained.merge_and_unload(safe_merge=True).eval()
        elif stage=='awq':
            from VLM.quantization.awq_calibration import load_calibration_text, build_calibration_samples
            from VLM.quantization.awq_config import AWQConfig
            from VLM.quantization.awq_quantize import quantize_bunny_awq
            q=AWQConfig(calibration_samples=2,sequence_length=32,max_tokens_per_linear=64,scale_grid=3,clip_grid=4)
            samples=build_calibration_samples(adapter.tokenizer,load_calibration_text(plan['calibration_data']),2,32,q.seed)
            quantize_bunny_awq(adapter.model,samples,q)
        document=export_treatment(adapter,target,source_identity=source,original_widths=widths,
                                  protected_sha256=protected,recovery=recovery)
        write_json(receipt, {'treatment':document['treatment'],'artifact_sha256':document['sha256'],
            'recovery':recovery,'seconds':time.perf_counter()-start,'preparation_device':'cpu', 'peak_vram_bytes':None,
            'host_rss_bytes_after_stage': psutil.Process().memory_info().rss,
            'host_available_bytes_after_stage': psutil.virtual_memory().available})
    if plan.get('evaluate'):
        from evaluation.runner import evaluate
        last=root/f'{len(stages)-1:02d}-{stages[-1]}'
        document=read_json(last/'treatment.json')
        final=replace(config,treatment=document['treatment'],treatment_artifact_path=str(last),max_treatment_bytes=budget)
        if not (root/'evaluation-complete.json').exists():
            run=evaluate(final,root/'evaluations',authorize_gpu=authorize_gpu)
            write_json(root/'evaluation-complete.json',{'run':str(run.path)})
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--authorize-gpu-execution',action='store_true')
    args=parser.parse_args()
    plan=read_json(args.plan)
    for key in ('recovery_batches','calibration_data'):
        if key in plan: plan[key]=str((Path(args.plan).resolve().parent/plan[key]).resolve())
    print(execute(EvaluationConfig.load(args.config),plan,args.output,resume=args.resume,authorize_gpu=args.authorize_gpu_execution))


if __name__=='__main__': main()
