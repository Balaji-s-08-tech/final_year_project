"""Bounded CPU comparison suite. Full-model/GPU experiments are deliberately excluded."""
from dataclasses import dataclass, asdict
import copy
import hashlib
import time
import csv
import io
from pathlib import Path
import torch

from compression.pruning import PruningConfig, prune, materialize_masks
from evaluation.data import normalize
from evaluation.inspection import architecture
from experiments.provenance import canonical_hash, file_hash
from experiments.schema import benchmark, measurement
from experiments.store import ExperimentStore, write_json, publish, read_json
from runtime_policy import admit_model, admit_device

TREATMENTS=('dense','unstructured','structured_mlp','awq_int4','awq_int8','unstructured_awq','structured_awq')


@dataclass(frozen=True)
class PipelineConfig:
    treatments: tuple[str,...] = TREATMENTS
    sparsity: float = .25
    group_size: int = 8
    seed: int = 42
    max_parameter_elements: int = 1_000_000
    max_model_bytes: int = 64*1024*1024
    scale_grid: int = 3
    clip_grid: int = 4

    def validate(self):
        if not self.treatments or len(set(self.treatments))!=len(self.treatments) or set(self.treatments)-set(TREATMENTS):
            raise ValueError('Unique supported treatments required')
        PruningConfig(sparsity=self.sparsity).validate()
        for name in ('group_size','seed','max_parameter_elements','max_model_bytes','scale_grid','clip_grid'):
            value=getattr(self,name)
            if type(value) is not int or value < (0 if name=='seed' else 1): raise ValueError(f'Invalid {name}')
        return self


class TextPredictor:
    task='text'
    def __init__(self,tokenizer,max_new_tokens=2,max_input_tokens=128):
        if type(max_new_tokens) is not int or max_new_tokens<1: raise ValueError('Invalid generation limit')
        self.tokenizer,self.max_new_tokens,self.max_input_tokens=tokenizer,max_new_tokens,max_input_tokens

    def __call__(self,model,record):
        from transformers import GenerationConfig
        encoded=self.tokenizer(record['question'],return_tensors='pt',truncation=False)
        ids=encoded['input_ids']
        if ids.shape[1]>self.max_input_tokens: raise ValueError('Input exceeds configured bound')
        settings=GenerationConfig.from_dict(model.generation_config.to_dict())
        settings.update(max_new_tokens=self.max_new_tokens,do_sample=False,num_beams=1,use_cache=True,pad_token_id=self.tokenizer.pad_token_id)
        self.effective_config=settings.to_dict()
        with torch.inference_mode():
            output=model.generate(input_ids=ids,attention_mask=encoded.get('attention_mask',torch.ones_like(ids)),
                                  generation_config=settings)
        tokens=output[0,ids.shape[1]:].tolist()
        eos=self.tokenizer.eos_token_id
        if eos in tokens: tokens=tokens[:tokens.index(eos)+1]
        return {'text':self.tokenizer.decode(tokens,skip_special_tokens=True),'token_ids':tokens}


def _identity(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_suite(base_model,tokenizer,calibration_samples,records,*,calibration_ids,protocol,output_root,config=PipelineConfig(),predictor=None):
    """Caller supplies small local assets; no loader, GPU flag, remote client or fallback."""
    config.validate(); admit_model(base_model)
    size=sum(t.numel()*t.element_size() for t in list(base_model.parameters())+list(base_model.buffers()))
    if sum(p.numel() for p in base_model.parameters())>config.max_parameter_elements or size>config.max_model_bytes:
        raise ValueError('CPU fixture budget exceeded; full-model execution is not supported by this suite')
    if any((t.is_floating_point() or t.is_complex()) and not torch.isfinite(t).all() for t in base_model.state_dict().values()):
        raise ValueError('Nonfinite source state')
    for field in ('dataset_identifier','dataset_revision','split','task','prompt_template','model_identifier','model_revision'):
        if not isinstance(protocol.get(field),str) or not protocol[field].strip(): raise ValueError(f'Missing protocol {field}')
    if protocol['task'] not in {'text','vqa'}: raise ValueError('Unsupported task')
    from string import Formatter
    if [field for _,field,_,_ in Formatter().parse(protocol['prompt_template']) if field is not None]!=['question']:
        raise ValueError('Protocol must format one question')
    ids=[r['id'] for r in records]
    if not ids or len(ids)>128 or len(set(ids))!=len(ids) or len(set(calibration_ids))!=len(calibration_ids) or set(ids)&set(calibration_ids):
        raise ValueError('Unique disjoint evaluation/calibration IDs required')
    if len(calibration_ids)!=len(calibration_samples) or not 1<=len(calibration_samples)<=32: raise ValueError('Calibration provenance/count mismatch')
    for sample in calibration_samples:
        admit_device(str(sample.device))
        if sample.ndim!=2 or sample.shape[0]!=1 or not 1<=sample.shape[1]<=128: raise ValueError('Bounded single-sequence calibration tensors required')
    if not all(isinstance(r.get('question'),str) and r['question'].strip() and isinstance(r.get('answers'),list) and r['answers'] and all(isinstance(a,str) and normalize(a) for a in r['answers']) for r in records):
        raise ValueError('Invalid held-out records')
    if protocol['task']=='text' and any(r.get('image') for r in records): raise ValueError('Do not discard visual inputs in text evaluation')
    if protocol['task']=='vqa' and any(not r.get('image') or not Path(r['image']).is_file() for r in records):
        raise ValueError('Visual evaluation requires local images')
    if predictor is None:
        if protocol['task']!='text': raise ValueError('VQA requires an explicit complete multimodal predictor')
        predictor=TextPredictor(tokenizer)
    if getattr(predictor,'task',None)!=protocol['task']: raise ValueError('Predictor modality differs from protocol')
    heldout_tokens=[tokenizer(protocol['prompt_template'].format(question=r['question']),return_tensors='pt')['input_ids'].tolist() for r in records] if protocol['task']=='text' else []
    if any(t.tolist() in heldout_tokens for t in calibration_samples):
        raise ValueError('Calibration token content overlaps held-out evaluation')
    data=[]
    for record in records:
        item=copy.deepcopy(record)
        if item.get('image'): item['image_sha256']=file_hash(item['image'],None)['sha256']
        data.append(item)
    fixed={'protocol':protocol,'held_out_fingerprint':canonical_hash(data), 'base_state_sha256':_identity(base_model),
           'calibration_ids':list(calibration_ids),'calibration_token_sha256':canonical_hash([t.tolist() for t in calibration_samples]),
           'evidence':'SYNTHETIC_OR_SMALL_CPU_ONLY','seed':config.seed}
    suite=ExperimentStore(output_root).create('compression_suite',asdict(config)|fixed)
    suite.transition('RUNNING'); results=[]
    try:
        for treatment in config.treatments:
            torch.manual_seed(config.seed)
            run=ExperimentStore(suite.path/'runs').create(treatment,asdict(config)|fixed|{'treatment':treatment},
                model={'identifier':protocol['model_identifier'],'revision':protocol['model_revision'],'checkpoint_sha256':fixed['base_state_sha256']},
                dataset={'identifier':protocol['dataset_identifier'],'revision':protocol['dataset_revision'],
                         'fingerprint':fixed['held_out_fingerprint'],'evaluation':protocol})
            run.transition('RUNNING'); started=time.perf_counter()
            try:
                run.log(f'Starting {treatment}; CPU fixture evidence only')
                model=copy.deepcopy(base_model).eval(); pruning=None
                from evaluation.integrity import guard_logits
                guard_logits(model)
                if treatment in {'unstructured','unstructured_awq','structured_mlp','structured_awq'}:
                    method='mlp_neurons' if treatment.startswith('structured') else 'magnitude_mask'
                    pruning=prune(model,PruningConfig(method=method,sparsity=config.sparsity))
                quant=None
                if 'awq' in treatment:
                    materialize_masks(model)
                    from VLM.quantization.awq_config import AWQConfig
                    from VLM.quantization.awq_quantize import quantize_bunny_awq
                    bits=8 if treatment=='awq_int8' else 4
                    q=AWQConfig(bits=bits,group_size=config.group_size,seed=config.seed,scale_grid=config.scale_grid,
                                clip_grid=config.clip_grid,calibration_samples=len(calibration_samples))
                    quant=quantize_bunny_awq(model,calibration_samples,q)
                    quant={'method':'AWQ','bits':bits,'calibration':fixed['calibration_token_sha256'],'summary':quant}
                outputs=[]
                for record in records:
                    prepared=dict(record); prepared['question']=protocol['prompt_template'].format(question=record['question'])
                    prediction=predictor(model,prepared)
                    if not prediction.get('token_ids'): raise ValueError('No actual generated tokens')
                    outputs.append({'id':record['id'],'prediction':prediction['text'],'token_ids':prediction['token_ids'],
                                    'correct':normalize(prediction['text']) in {normalize(a) for a in record['answers']}})
                metadata=fixed|{'treatment':treatment,'pruning':pruning,'quantization':quant,'architecture':architecture(model),
                    'effective_generation':getattr(predictor,'effective_config',None),
                    'tokenizer_vocab_sha256':canonical_hash(tokenizer.get_vocab()) if hasattr(tokenizer,'get_vocab') else None}
                if metadata['effective_generation'] is None:
                    raise ValueError('Predictor must report its effective inference configuration')
                effective=metadata['effective_generation']
                if 'generation' in effective:
                    effective={k:v for k,v in effective.items() if k!='model'}
                protocol_identity=canonical_hash(fixed|{'effective_inference':effective,'tokenizer_vocab_sha256':metadata['tokenizer_vocab_sha256']})
                artifact=run.path/'model.pt'
                publish(artifact,lambda stream:torch.save({'format':'vlm-compression-v1','model':model,'tokenizer':tokenizer,'metadata':metadata},stream))
                digest=file_hash(artifact,None)
                write_json(run.path/'artifact_manifest.json',digest|{'filename':artifact.name,'format':'vlm-compression-v1'})
                write_json(run.path/'predictions.json',outputs)
                write_json(run.path/'compression.json',metadata)
                metrics=benchmark({('baseline_accuracy' if treatment=='dense' else 'compressed_accuracy'):
                    measurement('baseline_accuracy' if treatment=='dense' else 'compressed_accuracy',sum(r['correct'] for r in outputs)/len(outputs),status='MEASURED',method='normalized_exact_match_v1',metadata={'evidence':fixed['evidence']}),
                    ('original_model_size' if treatment=='dense' else 'compressed_model_size'):
                    measurement('original_model_size' if treatment=='dense' else 'compressed_model_size',artifact.stat().st_size,status='MEASURED',method='identical_pipeline_container_filesystem_stat'),
                    'execution_duration':measurement('execution_duration',time.perf_counter()-started,status='MEASURED',method='perf_counter_compression_and_fixture_eval')},
                    quantization={'method':'AWQ','bits':quant['bits'],'calibration':{'token_sha256':fixed['calibration_token_sha256']}} if quant else None)
                run.log(f'Saved artifact {artifact.name}; quality is fixture-only')
                run.transition('COMPLETED',metrics=metrics,artifacts=[str(artifact)])
                results.append({'treatment':treatment,'path':str(run.path),'protocol_sha256':protocol_identity,'metrics':metrics})
                del model
            except BaseException as exc:
                import traceback
                run.log(traceback.format_exc(),'stderr.log')
                run.transition('FAILED',error=f'{type(exc).__name__}: {exc}'); raise
        if len({r['protocol_sha256'] for r in results})!=1: raise ValueError('Incomparable protocol identities')
        write_json(suite.path/'comparison.json',{'evidence':fixed['evidence'],'results':results,
            'warning':'Quality values concern supplied CPU fixtures only; latency/throughput/VRAM and research gains are not measured'})
        stream=io.StringIO(); writer=csv.writer(stream)
        writer.writerow(['treatment','metric','value','unit','status','evidence'])
        for result in results:
            for name,metric in result['metrics']['metrics'].items():
                writer.writerow([result['treatment'],name,metric['value'],metric['unit'],metric['status'],fixed['evidence']])
        publish(suite.path/'comparison.csv',lambda output:output.write(stream.getvalue().encode()))
        suite.transition('COMPLETED',artifacts=[r['path'] for r in results])
        return suite
    except BaseException as exc:
        suite.transition('FAILED',error=f'{type(exc).__name__}: {exc}'); raise


def load_artifact(run_path,*,trusted=False,max_load_bytes=64*1024*1024):
    from VLM.quantization.awq_model_loader import _torch_load, _enable_project_imports
    root=Path(run_path); manifest=read_json(root/'artifact_manifest.json'); path=root/'model.pt'
    if not trusted or path.stat().st_size>max_load_bytes: raise ValueError('Trusted bounded CPU artifact required')
    if file_hash(path,None)['sha256']!=manifest['sha256']: raise ValueError('Artifact content mismatch')
    _enable_project_imports(); payload=_torch_load(path)
    if payload.get('format')!='vlm-compression-v1': raise ValueError('Unknown pipeline artifact')
    admit_model(payload['model'])
    from evaluation.integrity import validate_packed
    validate_packed(payload['model'])
    if architecture(payload['model'])!=payload['metadata']['architecture']: raise ValueError('Reconstructed architecture differs')
    return payload
