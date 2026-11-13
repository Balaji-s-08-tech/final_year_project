"""Source-bound, incremental physical Phi MLP pruning. No acquisition or GPU work."""
import argparse
import hashlib
from pathlib import Path
import shutil

import psutil
import torch
from torch import nn

from evaluation.checkpoint_layout import TensorCheckpoint
from experiments.manifests import verify_model_manifest
from experiments.provenance import canonical_hash
from experiments.store import read_json, write_json

FORMAT = 'bunny-mlp-recipe-v1'
FIELDS = ('fc1.weight','fc1.bias','fc2.weight','fc2.bias')
RAM_RESERVE = 3*1024**3
DISK_RESERVE = 64*1024**2
PAIR_CAP = 256*1024**2


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def storage_bytes(root):
    from experiments.assets import inventories, AssetLimits
    limits=AssetLimits(max_file_bytes=DISK_RESERVE,max_total_bytes=DISK_RESERVE,
                       max_total_hash_bytes=DISK_RESERVE)
    files=inventories([Path(root)],hash_large=False,limits=limits)[0]
    total=sum(value['size_bytes'] for value in files.values())
    if total>64*1024**2:raise ValueError('Recipe directory exceeds bounded metadata profile')
    return total


def source_plan(root, manifest, layers=None):
    root = Path(root).resolve()
    config = read_json(root/'config.json')
    if not isinstance(config,dict) or config.get('model_type') != 'bunny-phi':
        raise ValueError('Recipe requires Bunny Phi configuration')
    h,i,n = (config.get(k) for k in ('hidden_size','intermediate_size','num_hidden_layers'))
    if any(type(v) is not int or v<1 for v in (h,i,n)) or n>128 or i<4:
        raise ValueError('Unsupported recipe architecture dimensions')
    if 4*(2*h*i+i+h)>PAIR_CAP:
        raise ValueError('MLP pair exceeds bounded FP32 ranking profile')
    selected = list(range(n)) if layers is None else list(layers)
    if not selected or any(type(v) is not int or not 0<=v<n for v in selected) or selected != sorted(set(selected)):
        raise ValueError('Layer selection must be unique, sorted and within the source')
    if len(selected)*(i-int(i*.25))*16+65536 > 8*1024**2:
        raise ValueError('Recipe exceeds metadata budget')
    if not isinstance(manifest,dict) or manifest.get('kind')!='model' or manifest.get('status')!='VERIFIED_LOCAL_CONTENT':
        raise ValueError('Complete source content manifest required')
    with TensorCheckpoint(root) as checkpoint:
        layouts=[]
        for index in selected:
            keys=[f'model.layers.{index}.mlp.{field}' for field in FIELDS]
            expected=[(i,h),(i,),(h,i),(h,)]
            dtypes=[]
            for key,shape in zip(keys,expected):
                if key not in checkpoint.tensors or checkpoint.shape(key)!=shape:
                    raise ValueError('Source MLP dimension/coverage mismatch')
                dtype=checkpoint.tensors[key].get_slice(key).get_dtype()
                if dtype not in {'F16','BF16','F32'}:raise ValueError('Unsupported MLP source dtype')
                dtypes.append(dtype)
            if len(set(dtypes))!=1:raise ValueError('Mixed MLP source dtypes unsupported')
            layouts.append({'layer':index,'keys':keys,'dtype':dtypes[0]})
    return {'format':FORMAT,'source_manifest':manifest,'config_sha256':canonical_hash(config),
        'hidden_size':h,'intermediate_size':i,'num_hidden_layers':n,
        'sparsity':.25,'ranking':'mean_abs_fc1_rows_plus_fc2_columns_float64_stable_v1',
        'selected_layers':selected,'layouts':layouts,'complete_model':selected==list(range(n)),
        'estimated_working_bytes':4*(2*h*i+i+h)+16*h*i,
        'required_available_ram_bytes':RAM_RESERVE,'required_free_disk_bytes':DISK_RESERVE,
        'scope':'MLP_ONLY_NO_RECOVERY_NO_AWQ','estimate_status':'ANALYTICAL_NOT_MEASURED'}


def admit(plan, output_parent=None):
    if psutil.virtual_memory().available < plan['required_available_ram_bytes']:
        raise ValueError('Insufficient RAM for bounded recipe operation')
    if output_parent is not None and shutil.disk_usage(output_parent).free < DISK_RESERVE:
        raise ValueError('Insufficient disk for recipe metadata and receipts')


def reconstruction_ram(config):
    from evaluation.inspection import local_checkpoint
    sizes=[local_checkpoint(path)['artifact_bytes'] for path in (config.model_path,config.vision_path)]
    if any(size is None for size in sizes):raise ValueError('Recipe requires known base/vision weight bytes')
    return max(config.required_free_ram_bytes or 0,2*sum(sizes)+RAM_RESERVE)


def layer_result(checkpoint, layout, hidden, intermediate):
    # Only one original MLP pair is materialized. No whole-model state_dict clone.
    values=[checkpoint.tensors[key].get_tensor(key).clone() for key in layout['keys']]
    if any(not torch.isfinite(value).all() for value in values):raise ValueError('Nonfinite source MLP')
    a,ab,b,bb=values
    scores=a.double().abs().mean(1)+b.double().abs().mean(0)
    keep=torch.argsort(scores,stable=True)[int(intermediate*.25):].sort().values
    result=[a.index_select(0,keep),ab.index_select(0,keep),b.index_select(1,keep),bb.clone()]
    record={'layer':layout['layer'],'kept_neurons':keep.tolist(),'original_width':intermediate,
        'remaining_width':len(keep),'source_dtype':layout['dtype'],
        'tensor_hashes':{field:tensor_hash(value) for field,value in zip(FIELDS,result)}}
    return record,result


def build(root, manifest_path, output, *, layers=None, execute=False, resume=False):
    manifest=read_json(manifest_path)
    plan=source_plan(root,manifest,layers)
    if not execute:return plan
    if execute is not True or type(resume) is not bool:raise ValueError('Explicit boolean execution/resume required')
    root=Path(root).resolve();destination=Path(output).resolve()
    if destination.is_relative_to(root) or not destination.parent.is_dir():raise ValueError('Output must be outside assets with existing parent')
    admit(plan,destination.parent)
    verify_model_manifest(root,manifest,hash_large=True)
    if destination.exists():
        if not resume or read_json(destination/'plan.json')!=plan:raise ValueError('Existing output requires identical resumable plan')
    else:
        destination.mkdir(exist_ok=False);write_json(destination/'plan.json',plan)
    records=[]
    for layout in plan['layouts']:
        admit(plan,destination.parent)
        with TensorCheckpoint(root) as checkpoint:
            record,values=layer_result(checkpoint,layout,plan['hidden_size'],plan['intermediate_size'])
        del checkpoint,values
        receipt=destination/'layers'/f"{layout['layer']:04d}.json"
        if receipt.exists():
            if read_json(receipt)!=record:raise ValueError('Corrupt/incompatible completed layer receipt')
        else:write_json(receipt,record)
        records.append({'layer':layout['layer'],'receipt_sha256':canonical_hash(record)})
    recipe={'plan':plan,'layers':records,'status':'COMPLETE' if plan['complete_model'] else 'PILOT_ONLY'}
    if not (destination/'compilation_resources.json').exists():
        from compression.mlp_pilot import observation
        write_json(destination/'compilation_resources.json',{'scope':'CPU_RECIPE_COMPILATION_ONLY',
            'process_observation':observation(),'processed_layers':plan['selected_layers'],
            'warning':'High-water RSS covers this entire process, not a per-layer allocation delta'})
    wrapper={'format':FORMAT,'recipe_sha256':canonical_hash(recipe),'treatment':'structured_pruning',
             'complete_model':plan['complete_model']}
    for name,document in [('recipe.json',recipe),('treatment.json',wrapper)]:
        if (destination/name).exists():
            if read_json(destination/name)!=document:raise ValueError('Corrupt final recipe; preserve output and use new destination')
        else:write_json(destination/name,document)
    return wrapper


def load(root, source_root, *, require_complete=True, verify_source=True):
    root=Path(root)
    storage_bytes(root)
    wrapper=read_json(root/'treatment.json');recipe=read_json(root/'recipe.json')
    if not isinstance(wrapper,dict) or wrapper.get('format')!=FORMAT or canonical_hash(recipe)!=wrapper.get('recipe_sha256'):
        raise ValueError('Missing/corrupt recipe integrity')
    if not isinstance(recipe,dict) or set(recipe)!={'plan','layers','status'} or not isinstance(recipe.get('plan'),dict):raise ValueError('Invalid recipe object')
    plan=recipe['plan']
    actual=source_plan(source_root,plan.get('source_manifest'),plan.get('selected_layers'))
    if plan!=actual:raise ValueError('Recipe/source configuration or layout mismatch')
    complete=plan['complete_model']
    if wrapper!={'format':FORMAT,'recipe_sha256':canonical_hash(recipe),'treatment':'structured_pruning','complete_model':complete}:
        raise ValueError('Recipe treatment declaration mismatch')
    if require_complete and (not complete or recipe.get('status')!='COMPLETE'):
        raise ValueError('Partial recipe cannot reconstruct a full model')
    if recipe.get('status')!=('COMPLETE' if complete else 'PILOT_ONLY') or not isinstance(recipe.get('layers'),list) or len(recipe['layers'])!=len(plan['layouts']):
        raise ValueError('Incomplete recipe layer coverage')
    expanded=[]
    for reference,layout in zip(recipe['layers'],plan['layouts']):
        if not isinstance(reference,dict) or set(reference)!={'layer','receipt_sha256'} or type(reference['layer']) is not int or reference['layer']!=layout['layer']:
            raise ValueError('Invalid layer receipt reference')
        record=read_json(root/'layers'/f"{layout['layer']:04d}.json")
        if canonical_hash(record)!=reference['receipt_sha256']:raise ValueError('Layer receipt integrity mismatch')
        if not isinstance(record,dict) or set(record)!={'layer','kept_neurons','original_width','remaining_width','source_dtype','tensor_hashes'}:
            raise ValueError('Invalid recipe layer declaration')
        keep=record.get('kept_neurons') if isinstance(record,dict) else None
        if (not isinstance(keep,list) or any(type(k)is not int or not 0<=k<plan['intermediate_size'] for k in keep)
            or keep!=sorted(set(keep)) or len(keep)!=plan['intermediate_size']-int(plan['intermediate_size']*.25)
            or type(record.get('layer')) is not int or record.get('layer')!=layout['layer']):raise ValueError('Invalid recipe neuron selection')
        if record['original_width']!=plan['intermediate_size'] or record['remaining_width']!=len(keep) or record['source_dtype']!=layout['dtype']:
            raise ValueError('Recipe dimension/dtype declaration mismatch')
        if not isinstance(record['tensor_hashes'],dict) or set(record['tensor_hashes'])!=set(FIELDS):raise ValueError('Incomplete recipe tensor hashes')
        from checkpoint_io import validate_digest
        for digest in record['tensor_hashes'].values():validate_digest(digest)
        expanded.append(record)
    if verify_source:
        admit(plan)
        verify_model_manifest(source_root,plan['source_manifest'],hash_large=True)
    return {**recipe,'layers':expanded},wrapper


def apply(adapter, artifact):
    recipe,wrapper=load(artifact,adapter.config.model_path)
    if adapter.config.treatment!='structured_pruning':raise ValueError('Recipe treatment/config mismatch')
    plan=recipe['plan']
    if len(adapter.model.model.layers)!=plan['num_hidden_layers']:raise ValueError('Runtime layer count mismatch')
    if any(t.device.type!='cpu' for t in adapter.model.parameters()):raise ValueError('Reconstruct on CPU before device transfer')
    for expected,layout in zip(recipe['layers'],plan['layouts']):
        admit(plan)
        with TensorCheckpoint(adapter.config.model_path) as checkpoint:
            record,values=layer_result(checkpoint,layout,plan['hidden_size'],plan['intermediate_size'])
        del checkpoint
        if record!=expected:raise ValueError('Recipe ranking or reconstructed values differ from source')
        block=adapter.model.model.layers[layout['layer']]
        if block.mlp.fc1.in_features!=plan['hidden_size'] or block.mlp.fc1.out_features!=plan['intermediate_size'] or block.mlp.fc2.in_features!=plan['intermediate_size'] or block.mlp.fc2.out_features!=plan['hidden_size']:
            raise ValueError('Runtime MLP dimensions disagree with original checkpoint')
        modules=[]
        for old,weight,bias in [(block.mlp.fc1,values[0],values[1]),(block.mlp.fc2,values[2],values[3])]:
            new=nn.Linear(weight.shape[1],weight.shape[0],device='meta',dtype=weight.dtype)
            new.weight=nn.Parameter(weight,requires_grad=old.weight.requires_grad)
            new.bias=nn.Parameter(bias,requires_grad=old.bias.requires_grad)
            new.train(old.training);modules.append(new)
        block.mlp.fc1,block.mlp.fc2=modules
        del values,modules,new,old,weight,bias
    widths=[b.mlp.fc1.out_features for b in adapter.model.model.layers]
    adapter.model.config.pruned_intermediate_sizes=widths
    if len(set(widths))==1:adapter.model.config.intermediate_size=widths[0]
    from evaluation.integrity import validate_state
    validate_state(adapter.model)
    return {**wrapper,'physical_pruning':True,'masked':False,'packed_bits':[],
            'artifact_storage_bytes':storage_bytes(artifact),
            'reconstructed_parameter_bytes_before_runtime_cast':sum(p.numel()*p.element_size() for p in adapter.model.parameters()),
            'quality':'NOT_ESTABLISHED'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--pilot-layers',type=int,nargs='+')
    args=parser.parse_args()
    result=build(args.model_dir,args.manifest,args.output,layers=args.pilot_layers,execute=args.execute,resume=args.resume)
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
