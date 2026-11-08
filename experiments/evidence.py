"""Comparison admission on persisted records. Hash evidence is not producer authentication."""
import json
import re
from pathlib import Path
from experiments.provenance import canonical_hash, file_hash
from experiments.schema import validate_benchmark


def document(path):
    from experiments.store import read_json
    value=read_json(path)
    if not isinstance(value,dict):raise ValueError('Expected JSON object')
    return value


def _hash(value,length=64):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is not None


def assess(path, max_asset_hash_bytes=8*1024*1024):
    """Core omissions invalidate verification; optional environment omissions are conditional."""
    from experiments.store import Run
    path=Path(path); errors=[]; conditional=[]; excluded=[]
    docs={}
    for key,name in [('identity','experiment.json'),('config','config.json'),('environment','environment.json'),('assets','assets.json'),('effective','effective_inference.json')]:
        try: docs[key]=document(path/name)
        except (OSError,ValueError,TypeError) as exc:
            docs[key]={}; errors.append(f'{name}: {exc}')
    try:
        state=Run(path).latest()
        if not isinstance(state,dict): raise ValueError('Invalid lifecycle state')
    except (OSError,ValueError,TypeError) as exc:state={};excluded.append(str(exc))
    if state.get('status')!='COMPLETED':excluded.append(f"Run is {state.get('status','INVALID')}: {state.get('error')}")
    else:
        try: validate_benchmark(state.get('metrics'))
        except (ValueError,TypeError,KeyError) as exc:excluded.append(f'Invalid terminal metrics: {exc}')
    identity=docs['identity']; config=docs['config']
    if identity.get('run_id')!=state.get('run_id') or not isinstance(identity.get('run_id'),str):excluded.append('Run identity/state mismatch')
    if identity.get('config_sha256')!=canonical_hash(config):errors.append('Config hash mismatch')
    for section,fields in [('model',('identifier','revision','architecture','config_sha256','checkpoint_sha256')),('dataset',('identifier','revision','fingerprint'))]:
        values=identity.get(section)
        if not isinstance(values,dict):errors.append(f'Invalid {section}');continue
        for field in fields:
            value=values.get(field)
            if not isinstance(value,str) or not value.strip() or (field in ('config_sha256','checkpoint_sha256','fingerprint') and not _hash(value)):
                errors.append(f'Missing/invalid {section}.{field}')
    dataset=identity.get('dataset'); evaluation=dataset.get('evaluation') if isinstance(dataset,dict) else None
    if not isinstance(evaluation,dict):errors.append('Missing evaluation protocol');evaluation={}
    for key in ('split','scorer','prompt_template'):
        if not isinstance(evaluation.get(key),str) or not evaluation[key].strip():errors.append(f'Missing/invalid evaluation.{key}')
    for key in ('max_input_tokens','max_new_tokens'):
        if type(evaluation.get(key)) is not int or evaluation[key]<1:errors.append(f'Invalid evaluation.{key}')
    assets=docs['assets']; regions={k:v for k,v in assets.items() if k!='sha256'}
    if not _hash(assets.get('sha256')) or assets.get('sha256')!=canonical_hash(regions):errors.append('Invalid assets content identity')
    text=config.get('adapter')=='portable_awq_text' or evaluation.get('task')=='text'
    for region in ('model','encoder'):
        data=assets.get(region)
        if text and region=='encoder' and data=={'status':'UNAVAILABLE'}:continue
        if not isinstance(data,dict) or data.get('status') not in ('CONTENT_HASHED','VERIFIED_LOCAL_CONTENT'):
            errors.append(f'Unverified asset region: {region}');continue
        files=data.get('files')
        if not isinstance(files,dict) or not files:errors.append(f'Empty asset inventory: {region}');continue
        for name,item in files.items():
            if not isinstance(name,str) or not name.strip() or not isinstance(item,dict) or item.get('status')!='HASHED' or not _hash(item.get('sha256')) or type(item.get('size_bytes')) is not int or item['size_bytes']<0:
                errors.append(f'Invalid asset file: {region}.{name}')
                continue
            try:
                root=Path(data['path']).resolve()
                target=root if root.is_file() and name==root.name else (root/name).resolve()
                if root.is_dir() and not target.is_relative_to(root):raise ValueError('Asset path escapes root')
                actual=file_hash(target,max_asset_hash_bytes)
                if actual.get('status')!='HASHED' or actual.get('sha256')!=item['sha256'] or actual.get('size_bytes')!=item['size_bytes']:
                    raise ValueError('Asset missing, changed or exceeds explicit hash budget')
            except (OSError,ValueError,KeyError,TypeError) as exc:errors.append(f'Unverifiable asset {region}.{name}: {exc}')
    model=identity.get('model')
    model_assets=assets.get('model')
    if isinstance(model,dict) and isinstance(model_assets,dict) and isinstance(model_assets.get('files'),dict):
        hashes=[v.get('sha256') for v in model_assets['files'].values() if isinstance(v,dict)]
        if model.get('checkpoint_sha256') not in hashes+[canonical_hash(model_assets['files'])]:errors.append('Checkpoint hash is not bound to asset inventory')
        try:
            root=Path(model_assets['path'])
            if root.is_dir() and 'config.json' in model_assets['files']:
                if max_asset_hash_bytes is not None and (root/'config.json').stat().st_size>max_asset_hash_bytes:raise ValueError('Model configuration exceeds hash budget')
                config_digest=canonical_hash(document(root/'config.json'))
            elif root.is_file():
                if max_asset_hash_bytes is not None and root.stat().st_size>max_asset_hash_bytes:raise ValueError('Portable model exceeds hash budget')
                from VLM.quantization.portable_integrity import header
                config_digest=header(root,max_hash_bytes=max_asset_hash_bytes)['model_config_sha256']
            else:raise ValueError('Missing model configuration evidence')
            if model.get('config_sha256')!=config_digest:raise ValueError('Model configuration hash contradicts assets')
        except (OSError,ValueError,KeyError,TypeError) as exc:errors.append(f'Unverifiable model configuration: {exc}')

    effective=docs['effective']; generation=effective.get('generation')
    if not isinstance(generation,dict):generation={};errors.append('Missing effective generation')
    for key in ('max_new_tokens','num_beams'):
        if type(generation.get(key)) is not int or generation[key]<1:errors.append(f'Invalid generation.{key}')
    for key in ('do_sample','use_cache'):
        if type(generation.get(key)) is not bool:errors.append(f'Invalid generation.{key}')
    for key in ('pad_token_id','eos_token_id'):
        if key not in generation or (generation[key] is not None and not (type(generation[key]) is int and generation[key]>=0) and not (key=='eos_token_id' and isinstance(generation[key],list) and generation[key] and all(type(v)is int and v>=0 for v in generation[key]))):errors.append(f'Invalid generation.{key}')
    if generation.get('max_new_tokens')!=evaluation.get('max_new_tokens'):errors.append('Requested/effective generation limit conflict')
    tokenizer=effective.get('tokenizer')
    if not isinstance(tokenizer,dict) or not all(isinstance(tokenizer.get(k),str) and tokenizer[k].strip() for k in ('class','name')):errors.append('Missing effective tokenizer identity')
    if not text and (not isinstance(effective.get('processor'),dict) or not effective['processor']):errors.append('Missing image processor configuration')
    source=identity.get('source')
    if not isinstance(source,dict) or not _hash(source.get('commit'),40) or source.get('dirty') is not False:errors.append('Missing clean source identity')
    for key in ('python','pytorch'):
        if not isinstance(docs['environment'].get(key),str) or not docs['environment'][key].strip():errors.append(f'Missing software environment: {key}')
    dependencies=docs['environment'].get('dependencies')
    if not isinstance(dependencies,dict) or not dependencies or not all(isinstance(k,str) and k.strip() and isinstance(v,str) and v.strip() for k,v in dependencies.items()):errors.append('Invalid dependency versions')
    for key in ('os','cpu'):
        if not docs['environment'].get(key):conditional.append(f'Missing environment.{key}')
    return {'errors':errors,'conditional':conditional,'exclusions':excluded,'state':state,'documents':docs}
