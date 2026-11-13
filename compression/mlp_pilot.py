"""Opt-in, CPU-only single-MLP feasibility pilot; never loads a complete VLM."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch
from torch import nn

from compression.pruning import prune, PruningConfig
from evaluation.checkpoint_layout import TensorCheckpoint
from experiments.store import read_json, write_json
from experiments.provenance import canonical_hash, file_hash
from scripts.reproduce_bunny_baseline import BUNNY

MAX_COMPONENT_BYTES = 256 * 1024**2
REQUIRED_FREE_RAM = 3 * 1024**3
REQUIRED_FREE_DISK = 512 * 1024**2
FIELDS = ('fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias')


def inspect_component(root, manifest_path, layer=0):
    root = Path(root).resolve()
    document = read_json(manifest_path)
    if not isinstance(document,dict) or (document.get('identifier'), document.get('revision')) != BUNNY or document.get('status') != 'VERIFIED_LOCAL_CONTENT':
        raise ValueError('Pilot requires the pinned Bunny local content manifest')
    config = read_json(root/'config.json')
    if not isinstance(config,dict) or config.get('model_type') != 'bunny-phi' or type(config.get('num_hidden_layers')) is not int or type(layer) is not int or not 0 <= layer < config['num_hidden_layers']:
        raise ValueError('Pilot requires an existing Bunny Phi layer')
    hidden, intermediate = config.get('hidden_size'), config.get('intermediate_size')
    if any(type(v) is not int or not 1 <= v <= 65536 for v in (hidden, intermediate)) or intermediate < 4:
        raise ValueError('Invalid MLP dimensions')
    shapes = [(intermediate, hidden), (intermediate,), (hidden, intermediate), (hidden,)]
    keys = [f'model.layers.{layer}.mlp.{field}' for field in FIELDS]
    with TensorCheckpoint(root) as checkpoint:
        for key, shape in zip(keys, shapes):
            if key not in checkpoint.tensors or checkpoint.shape(key) != shape:
                raise ValueError('Source MLP shape/coverage mismatch')
            if checkpoint.tensors[key].get_slice(key).get_dtype() not in {'F16','BF16','F32'}:
                raise ValueError('Unsupported source MLP dtype')
    source_bytes = 4*(2*hidden*intermediate+intermediate+hidden)
    if source_bytes > MAX_COMPONENT_BYTES:
        raise ValueError('Single-MLP FP32 component exceeds 256 MiB cap')
    kept = intermediate-int(intermediate*.25)
    target_bytes = 4*(2*hidden*kept+kept+hidden)
    if (root/'model.safetensors').is_file():
        files = ['config.json', 'model.safetensors']
    else:
        mapping = read_json(root/'model.safetensors.index.json')['weight_map']
        files = ['config.json', 'model.safetensors.index.json', *sorted({mapping[key] for key in keys})]
    for name in files:
        digest = document.get('files', {}).get(name, {})
        if digest.get('status') != 'HASHED' or not digest.get('sha256'):
            raise ValueError('Selected source file lacks manifest hash')
    return {'status':'PLAN_ONLY_NOT_EXECUTED','scope':'SINGLE_PRETRAINED_MLP_OPERATOR_ONLY',
        'layer':layer,'hidden_size':hidden,'original_intermediate':intermediate,'kept_intermediate':kept,
        'source_fp32_bytes':source_bytes,'target_fp32_bytes':target_bytes,
        'required_free_ram_bytes':REQUIRED_FREE_RAM,'required_free_disk_bytes':REQUIRED_FREE_DISK,
        'estimated_component_tensor_peak_bytes':source_bytes+max(16*hidden*intermediate,3*target_bytes),
        'estimate_status':'ANALYTICAL_NOT_MEASURED_PEAK','keys':keys,'source_files':files,
        'manifest_sha256':canonical_hash(document),'activation':config.get('hidden_act','gelu_new')}


def _load_pair(root, plan):
    from transformers.activations import ACT2FN
    if plan['activation'] not in ACT2FN:
        raise ValueError('Unsupported source activation')
    with TensorCheckpoint(root) as checkpoint:
        values = [checkpoint.tensors[key].get_tensor(key).float().clone() for key in plan['keys']]
    model = nn.Module(); model.model = nn.Module()
    block = nn.Module(); block.mlp = nn.Module()
    for name, weight, bias in [('fc1',values[0],values[1]),('fc2',values[2],values[3])]:
        linear = nn.Linear(weight.shape[1],weight.shape[0],device='meta')
        linear.weight = nn.Parameter(weight,requires_grad=False)
        linear.bias = nn.Parameter(bias,requires_grad=False)
        setattr(block.mlp,name,linear)
    model.model.layers = nn.ModuleList([block])
    model.config = SimpleNamespace(intermediate_size=plan['original_intermediate'])
    model.eval()
    return model, ACT2FN[plan['activation']]


def observation():
    high_water = None
    try:
        import resource, sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        high_water = value if sys.platform == 'darwin' else value*1024
    except ImportError:
        pass
    return {'rss_bytes':psutil.Process().memory_info().rss,
            'process_lifetime_peak_rss_bytes':high_water}


def pilot(root, manifest_path, *, layer=0, output=None, execute=False):
    if type(execute) is not bool:
        raise ValueError('Execution must be explicitly boolean')
    plan = inspect_component(root, manifest_path, layer)
    if not execute:
        return plan
    root = Path(root).resolve()
    if output is None:
        raise ValueError('Explicit new output directory required')
    destination = Path(output).resolve()
    if destination.is_relative_to(root) or destination.exists() or not destination.parent.is_dir():
        raise ValueError('Output must be new, have an existing parent, and be outside assets')
    if psutil.virtual_memory().available < REQUIRED_FREE_RAM:
        raise ValueError('Single-MLP pilot requires 3 GiB available host RAM')
    import shutil
    if shutil.disk_usage(destination.parent).free < REQUIRED_FREE_DISK:
        raise ValueError('Single-MLP pilot requires 512 MiB free output disk')
    source = read_json(manifest_path)
    if canonical_hash(source) != plan['manifest_sha256']:
        raise ValueError('Source manifest changed after planning')
    verified = {}
    for name in plan['source_files']:
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Unsafe source file')
        expected = source['files'][name]
        actual = file_hash(path, None)
        if actual != expected:
            raise ValueError('Selected source file differs from pinned local manifest')
        verified[name] = actual
    checkpoints = {'before_load':observation()}
    model, activation = _load_pair(root,plan)
    a, b = model.model.layers[0].mlp.fc1, model.model.layers[0].mlp.fc2
    checkpoints['after_load'] = observation()
    inputs = torch.randn(1,16,plan['hidden_size'],generator=torch.Generator(device='cpu').manual_seed(42),device='cpu',dtype=torch.float32)
    with torch.inference_mode():
        dense = b(activation(a(inputs)))
        report = prune(model,PruningConfig(method='mlp_neurons',sparsity=.25))
        kept = report['layers'][0]['kept_neurons']
        mask = torch.zeros(plan['original_intermediate'],device='cpu',dtype=torch.float32); mask[kept] = 1
        reference = b(activation(a(inputs))*mask)
        mlp = model.model.layers[0].mlp
        result = mlp.fc2(activation(mlp.fc1(inputs)))
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite pruned operator output')
        torch.testing.assert_close(result,reference,rtol=1e-4,atol=1e-4)
    checkpoints['after_prune_and_probe'] = observation()
    destination.mkdir(exist_ok=False)
    from safetensors.torch import save_file, load_file
    artifact = destination/'mlp.safetensors'
    save_file(dict(mlp.state_dict()),str(artifact))
    if artifact.stat().st_size > MAX_COMPONENT_BYTES:
        raise ValueError('Micro artifact exceeds byte cap; retained for inspection')
    restored = load_file(str(artifact))
    if any(not torch.equal(value,restored[key]) for key,value in mlp.state_dict().items()):
        raise ValueError('Micro artifact reload differs')
    checkpoints['after_save_reload'] = observation()
    summary = {**plan,'status':'CPU_OPERATOR_PILOT_COMPLETED',
        'format':'phi-mlp-micro-pilot-v1','source_files_verified':verified,'pruning':report,
        'artifact':file_hash(artifact,None),'resource_observations':checkpoints,
        'synthetic_probe_shape':list(inputs.shape),'synthetic_probe_seed':42,
        'masked_reference_max_abs_error':float((result-reference).abs().max()),
        'dense_vs_pruned_probe_rmse':float((result-dense).square().mean().sqrt()),
        'model_quality':'NOT_MEASURED','full_model_execution':'NOT_PERFORMED','gpu_execution':'NOT_PERFORMED',
        'artifact_scope':'ONE_MLP_PAIR_NOT_A_DEPLOYABLE_BUNNY_CHECKPOINT'}
    write_json(destination/'pilot.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--layer',type=int,default=0)
    parser.add_argument('--output')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    print(json.dumps(pilot(args.model_dir,args.manifest,layer=args.layer,output=args.output,execute=args.execute),indent=2))


if __name__=='__main__':main()
