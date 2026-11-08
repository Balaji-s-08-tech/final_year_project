"""Shared finite-output and checkpoint structural contracts; no resource acquisition."""
import torch


def validate_state(model):
    for name,tensor in model.state_dict().items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite checkpoint state: {name}')
    import math
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.LayerNorm):
            if type(module.eps)not in (float,int) or not math.isfinite(module.eps) or module.eps<=0:raise ValueError(f'Invalid LayerNorm normalization epsilon: {name}')
            if not module.normalized_shape or any(type(v)is not int or v<1 for v in module.normalized_shape):raise ValueError(f'Invalid LayerNorm normalization shape: {name}')
            if any(p is not None and tuple(p.shape)!=tuple(module.normalized_shape) for p in (module.weight,module.bias)):raise ValueError(f'LayerNorm normalization tensor mismatch: {name}')
        if isinstance(module,torch.nn.Linear):
            if module.weight.shape != (module.out_features,module.in_features) or (module.bias is not None and module.bias.shape != (module.out_features,)):
                raise ValueError(f'Checkpoint linear shape mismatch: {name}')
    for block in getattr(getattr(model,'model',None),'layers',[]):
        if hasattr(block,'mlp') and hasattr(block.mlp,'fc1'):
            a,b=block.mlp.fc1,block.mlp.fc2
            if a.out_features!=b.in_features or a.in_features!=b.out_features:
                raise ValueError('Checkpoint MLP dependency mismatch')


def finite_logits_hook(module,args,output):
    logits=getattr(output,'logits',None)
    if logits is not None and (not torch.is_tensor(logits) or not torch.isfinite(logits).all()):
        raise ValueError('Nonfinite logits: refusing generation/scoring')


def guard_logits(model):
    if not getattr(model,'_finite_logits_guard',False):
        model.register_forward_hook(finite_logits_hook)
        model._finite_logits_guard=True


def validate_packed(model):
    from VLM.quantization.awq_quantize import PackedAWQLinear, _dequantize, _unpack_int4
    validate_state(model)
    for name,m in model.named_modules():
        if not isinstance(m,PackedAWQLinear): continue
        if type(m.bits) is not int or m.bits not in (4,8) or min(m.in_features,m.out_features,m.group_size)<1:
            raise ValueError(f'Invalid packed dimensions: {name}')
        columns=(m.in_features+1)//2 if m.bits==4 else m.in_features
        groups=(m.in_features+m.group_size-1)//m.group_size
        if m.packed_weight.shape!=(m.out_features,columns) or m.scales.shape!=(m.out_features,groups) or m.zero_points.shape!=m.scales.shape:
            raise ValueError(f'Packed shape mismatch: {name}')
        if m.bias is not None and m.bias.shape!=(m.out_features,):
            raise ValueError(f'Packed bias shape mismatch: {name}')
        if m.packed_weight.dtype!=torch.uint8 or m.zero_points.dtype!=torch.uint8 or m.zero_points.max()>2**m.bits-1 or not (m.scales>0).all():
            raise ValueError(f'Invalid packed storage/scales: {name}')
        codes=_unpack_int4(m.packed_weight,m.in_features) if m.bits==4 else m.packed_weight
        _dequantize(codes,m.scales.float(),m.zero_points,m.group_size)
    widths=[b.mlp.fc1.out_features for b in getattr(getattr(model,'model',None),'layers',[]) if hasattr(b,'mlp')]
    declared=getattr(getattr(model,'config',None),'pruned_intermediate_sizes',None)
    if declared is not None and widths!=declared:
        raise ValueError('Pruning metadata disagrees with actual MLP dimensions')

    if getattr(getattr(model,'model',None),'layers',None):
        from VLM.quantization.portable_integrity import describe
        describe(model)
