"""Declarative portable admission before trusted pickle construction, then tensor reconciliation."""
import torch
from experiments.provenance import canonical_hash


def positive(value,name):
    if type(value) is not int or value<1:raise ValueError(f'Invalid positive integer dimension: {name}')
    return value


def validate_architecture(document):
    if not isinstance(document,dict) or type(document.get('schema_version')) is not int or document['schema_version']!=1:raise ValueError('Missing portable architecture header')
    config=document.get('config')
    if not isinstance(config,dict) or config.get('model_type') not in ('phi','bunny-phi','tiny-phi2'):raise ValueError('Unsupported portable architecture')
    blocks=document.get('blocks')
    if not isinstance(blocks,list) or not blocks:raise ValueError('Missing decoder dimensions')
    fixture=config['model_type']=='tiny-phi2'
    if not fixture:
        hidden=positive(config.get('hidden_size'),'hidden_size');heads=positive(config.get('num_attention_heads'),'num_attention_heads')
        kv=positive(config.get('num_key_value_heads',heads),'num_key_value_heads')
        if hidden%heads or kv!=heads:raise ValueError('Unsupported Phi head/KV dependency')
        positive(config.get('intermediate_size'),'intermediate_size')
        if positive(config.get('num_hidden_layers'),'num_hidden_layers')!=len(blocks):raise ValueError('Layer count mismatch')
    for block in blocks:
        if not isinstance(block,dict):raise ValueError('Invalid decoder descriptor')
        h=positive(block.get('hidden'),'residual width');i=positive(block.get('intermediate'),'MLP width')
        norm=block.get('qk_normalization')
        if not isinstance(norm,dict) or type(norm.get('enabled')) is not bool:raise ValueError('Missing/invalid qk_layernorm normalization declaration')
        requested=config.get('qk_layernorm',False)
        if type(requested)is not bool or norm['enabled']!=requested:raise ValueError('qk_layernorm configuration conflict')
        expected_shape=[h] if requested else None
        if norm.get('query_shape')!=expected_shape or norm.get('key_shape')!=expected_shape:raise ValueError('Unsupported query/key normalization dimensions')
        if not fixture:
            if h!=hidden or block.get('heads')!=heads or type(block.get('heads')) is not int or block.get('head_dim')!=hidden//heads or type(block.get('head_dim')) is not int:raise ValueError('Attention head dimension mismatch')
            rotary=positive(block.get('rotary_dim'),'rotary_dim')
            if rotary>hidden//heads or rotary%2:raise ValueError('Invalid rotary dimension')
        expected={'self_attn.q_proj':[h,h],'self_attn.k_proj':[h,h],'self_attn.v_proj':[h,h],'self_attn.dense':[h,h],'mlp.fc1':[h,i],'mlp.fc2':[i,h]}
        if block.get('linears')!=expected:raise ValueError('Attention/MLP projection dimensions contradict residual/config')
        for dims in block['linears'].values():
            if not isinstance(dims,list) or any(type(v)is not int or v<1 for v in dims):raise ValueError('Invalid projection dimensions')
    widths=[b['intermediate'] for b in blocks]
    if config.get('pruned_intermediate_sizes') is not None and config['pruned_intermediate_sizes']!=widths:raise ValueError('Pruning metadata mismatch')
    if not fixture and len(set(widths))==1 and config.get('intermediate_size')!=widths[0]:raise ValueError('MLP configuration mismatch')
    return document


def describe(model):
    config=model.config.to_dict();blocks=[];packed={}
    from VLM.quantization.awq_quantize import PackedAWQLinear
    for block in model.model.layers:
        linears={name:[m.in_features,m.out_features] for name in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.dense','mlp.fc1','mlp.fc2') for m in [block.get_submodule(name)]}
        enabled=getattr(block.self_attn,'qk_layernorm',False)
        if type(enabled)is not bool or enabled!=config.get('qk_layernorm',False):raise ValueError('qk_layernorm normalization configuration conflict')
        shapes=[]
        for name in ('q_layernorm','k_layernorm'):
            norm=getattr(block.self_attn,name,None)
            if enabled:
                if not isinstance(norm,torch.nn.LayerNorm):raise ValueError('Unsupported query/key normalization type')
                shape=list(norm.normalized_shape)
                if shape!=[block.mlp.fc1.in_features] or (norm.weight is not None and list(norm.weight.shape)!=shape) or (norm.bias is not None and list(norm.bias.shape)!=shape):raise ValueError('Unsupported query/key normalization dimensions')
                shapes.append(shape)
            else:shapes.append(None)
        descriptor={'qk_normalization':{'enabled':enabled,'query_shape':shapes[0],'key_shape':shapes[1]},'hidden':block.mlp.fc1.in_features,'intermediate':block.mlp.fc1.out_features,'linears':linears}
        if config.get('model_type')!='tiny-phi2':
            attention=block.self_attn
            for name,expected in [('hidden_size',config['hidden_size']),('num_key_value_heads',config.get('num_key_value_heads',config['num_attention_heads'])),('num_key_value_groups',1)]:
                if type(getattr(attention,name,None)) is not int or getattr(attention,name)!=expected:raise ValueError(f'Attention dependency mismatch: {name}')
            descriptor.update(heads=block.self_attn.num_heads,head_dim=block.self_attn.head_dim,rotary_dim=block.self_attn.rotary_emb.dim)
        blocks.append(descriptor)
    for name,m in model.named_modules():
        if isinstance(m,PackedAWQLinear):packed[name]={'bits':m.bits,'group_size':m.group_size,'in_features':m.in_features,'out_features':m.out_features}
    architecture=validate_architecture({'schema_version':1,'config':config,'blocks':blocks})
    return architecture,packed


def validate_precision(precision,quantization,packed):
    if precision not in ('AWQ INT4','AWQ INT8') or not isinstance(quantization,dict) or quantization.get('method')!='AWQ' or type(quantization.get('bits')) is not int:
        raise ValueError('Unsupported canonical portable precision')
    bits=quantization['bits']
    if bits not in (4,8) or precision!=f'AWQ INT{bits}' or not isinstance(packed,dict) or not packed:raise ValueError('Contradictory precision metadata')
    for name,data in packed.items():
        if not isinstance(name,str) or not isinstance(data,dict) or type(data.get('bits'))is not int or data['bits']!=bits:raise ValueError('Packed/declaration precision conflict')
        for key in ('in_features','out_features','group_size'):positive(data.get(key),key)
        if data['group_size']>data['in_features']:raise ValueError('Invalid effective group size')
    return bits


def header(path, *, max_hash_bytes=64*1024*1024):
    from experiments.artifacts import inspect_artifact
    from experiments.evidence import document
    from pathlib import Path
    sidecar=Path(str(path)+'.metadata.json')
    try:
        if sidecar.stat().st_size>1024*1024:raise ValueError('Portable sidecar exceeds 1 MiB')
        info=document(sidecar)
    except OSError as exc:raise ValueError('Portable checkpoint requires a verified declarative sidecar; legacy unchecked pickle is unsupported') from exc
    validate_architecture(info.get('architecture'))
    validate_precision(info.get('precision'),info.get('quantization'),info.get('packed_layers'))
    for name,data in info['packed_layers'].items():
        parts=name.split('.')
        if len(parts)!=5 or parts[:2]!=['model','layers'] or not parts[2].isdigit():raise ValueError('Invalid packed layer path')
        index=int(parts[2]); suffix='.'.join(parts[3:])
        blocks=info['architecture']['blocks']
        if index>=len(blocks) or suffix not in blocks[index]['linears'] or [data['in_features'],data['out_features']]!=blocks[index]['linears'][suffix]:raise ValueError('Packed dimensions contradict architecture')
    calibration=info['quantization'].get('calibration')
    if not isinstance(calibration,dict) or type(calibration.get('bits'))is not int or calibration['bits']!=info['quantization']['bits']:raise ValueError('Contradictory declared calibration precision')
    requested=calibration.get('group_size')
    if type(requested)is not int or (requested<1 and requested!=-1):raise ValueError('Invalid requested group size in header')
    if info.get('model_config_sha256')!=canonical_hash(info['architecture']['config']):raise ValueError('Header config hash mismatch')
    # Reject invalid declarations before reading the large payload.
    from checkpoint_io import admit_size
    if max_hash_bytes is not None:admit_size(path,max_hash_bytes)
    outer=inspect_artifact(path)
    if outer.get('format')!='portable_awq':raise ValueError('Unsupported portable format')
    if document(sidecar)!=info:raise ValueError('Portable header changed during admission')
    return info


def reconcile(payload,info):
    if not isinstance(payload,dict) or payload.get('format')!='bunny-awq-portable-v1':raise ValueError('Unsupported portable payload')
    metadata=payload.get('metadata')
    if not isinstance(metadata,dict) or not isinstance(metadata.get('quantization'),dict):raise ValueError('Missing payload quantization metadata')
    bits=info['quantization']['bits']
    if canonical_hash(metadata['quantization'])!=canonical_hash(info['quantization'].get('calibration')):raise ValueError('Inner/outer precision or calibration configuration conflict')
    if 'precision' in metadata and metadata['precision']!=info['precision']:raise ValueError('Inner/outer canonical precision conflict')
    if 'method' in metadata['quantization'] and metadata['quantization']['method']!='AWQ':raise ValueError('Unsupported inner quantization method')
    requested=metadata['quantization'].get('group_size')
    if type(requested)is not int or (requested<1 and requested!=-1):raise ValueError('Invalid requested quantization group size')
    if type(metadata['quantization'].get('bits'))is not int or metadata['quantization']['bits']!=bits:raise ValueError('Inner/outer precision conflict')
    if canonical_hash(metadata.get('model_config'))!=canonical_hash(info['architecture']['config']):raise ValueError('Inner/outer model configuration conflict')
    from evaluation.integrity import validate_packed
    validate_packed(payload['model'])
    arch,packed=describe(payload['model'])
    if canonical_hash(arch)!=canonical_hash(info['architecture']) or packed!=info['packed_layers']:raise ValueError('Loaded tensors contradict architecture/precision header')
    if not payload.get('tokenizer'):raise ValueError('Missing tokenizer')
