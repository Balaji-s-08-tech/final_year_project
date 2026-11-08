"""Quantization diagnostics and bounded trusted CPU artifact verification."""
from pathlib import Path
import torch
from runtime_policy import admit_device, admit_model


def reconstruction_error(reference, reconstructed):
    admit_device(str(reference.device)); admit_device(str(reconstructed.device))
    if reference.shape != reconstructed.shape or not reference.numel() or not torch.isfinite(reference).all() or not torch.isfinite(reconstructed).all():
        raise ValueError('Finite nonempty tensors with identical shapes are required')
    error=(reference.double()-reconstructed.double()).abs()
    return {'status':'MEASURED_CPU_TENSOR', 'unit':'weight_units', 'max_absolute_error':error.max().item(),
            'mean_absolute_error':error.mean().item(), 'rmse':error.square().mean().sqrt().item(),
            'relative_l2':(error.norm()/reference.double().norm()).item() if reference.double().norm()>0 else None,
            'quality_claim':'NONE; tensor error is not model task accuracy'}


def verify_checkpoint(path, *, trusted=False, max_load_bytes=64*1024*1024):
    path=Path(path)
    if not trusted or path.stat().st_size > max_load_bytes:
        raise ValueError('Trusted local artifact and explicit sufficient CPU load budget required')
    from experiments.artifacts import inspect_artifact
    from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint
    from VLM.quantization.awq_quantize import PackedAWQLinear, _effective_group_size
    outer=inspect_artifact(path)
    if outer['format'] != 'portable_awq': raise ValueError('Verified portable metadata is required')
    _,model,inner=load_quantized_bunny_checkpoint(str(path))
    admit_model(model)
    if any((t.is_floating_point() or t.is_complex()) and not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint state')
    bits=outer['quantization']['bits']
    config=inner.get('quantization',{})
    if config.get('bits') != bits or config.get('group_size') in (None,0):
        raise ValueError('Inner/outer quantization metadata mismatch')
    count=0
    for module in model.modules():
        if not isinstance(module,PackedAWQLinear): continue
        count+=1
        expected=_effective_group_size(module.in_features,config['group_size'])
        columns=(module.in_features+1)//2 if bits==4 else module.in_features
        groups=(module.in_features+expected-1)//expected
        if module.bits != bits or module.group_size != expected or module.packed_weight.shape != (module.out_features,columns):
            raise ValueError('Packed weight shape/precision mismatch')
        if module.scales.shape != (module.out_features,groups) or module.zero_points.shape != module.scales.shape:
            raise ValueError('Invalid group parameter shapes')
        if not torch.isfinite(module.scales).all() or not (module.scales>0).all():
            raise ValueError('Invalid scales')
        if module.packed_weight.dtype != torch.uint8 or module.zero_points.dtype != torch.uint8 or module.zero_points.max() > (1<<bits)-1:
            raise ValueError('Invalid packed storage/zero points')
    if not count: raise ValueError('No actual packed layers')
    return {'status':'VERIFIED_CPU_ARTIFACT', 'bits':bits, 'packed_linears':count, 'sha256':outer['sha256'],
            'warning':'Trusted pickle verification is not authenticity, recovery lineage or task-quality proof'}
