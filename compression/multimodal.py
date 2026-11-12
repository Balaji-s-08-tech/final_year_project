"""Declarative Bunny language-block treatments; base multimodal assets stay local."""
from pathlib import Path
import hashlib
import torch
from torch import nn

from compression.pruning import PHI_LINEARS, MaskedLinear, set_module
from evaluation.integrity import validate_packed
from experiments.provenance import canonical_hash, file_hash
from experiments.store import read_json, write_json


def base_identity(config):
    from evaluation.provenance import asset_manifest
    from dataclasses import replace
    assets = asset_manifest(replace(config, hash_large_assets=True, treatment_artifact_path=None))
    return canonical_hash({'identifier': config.model_identifier, 'revision': config.model_revision,
        'assets': {k: v.get('files') for k, v in assets.items() if k in ('model', 'encoder') and isinstance(v, dict)}})


def protected_identity(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith('model.layers.'):
            continue
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def describe_treatment(model, original_widths, recovery=None):
    from VLM.quantization.awq_quantize import PackedAWQLinear
    widths = [b.mlp.fc1.out_features for b in model.model.layers]
    if len(widths) != len(original_widths) or any(w > old for w, old in zip(widths, original_widths)):
        raise ValueError('Invalid physical pruning widths')
    physical = any(w < old for w, old in zip(widths, original_widths))
    masked = any(isinstance(m, MaskedLinear) and (~m.weight_mask).any() for m in model.modules())
    bits = {m.bits for m in model.modules() if isinstance(m, PackedAWQLinear)}
    if len(bits) > 1:
        raise ValueError('Mixed packed precision is unsupported')
    if recovery is not None and (not isinstance(recovery, dict) or not recovery.get('losses')):
        raise ValueError('Recovery requires a training ledger')
    if bits:
        treatment = 'pruning_awq' if physical or masked else f'awq_int{next(iter(bits))}'
    elif physical:
        treatment = 'structured_pruning'
    elif masked:
        treatment = 'unstructured_pruning'
    else:
        treatment = 'baseline'
    return {'treatment': treatment, 'physical_pruning': physical, 'masked': bool(masked),
            'packed_bits': sorted(bits), 'recovery': recovery, 'widths': widths}


def export_treatment(adapter, destination, *, source_identity, original_widths, protected_sha256, recovery=None):
    """Save only language blocks; never pickle model/tokenizer/processor objects."""
    from safetensors.torch import save_file
    from VLM.quantization.awq_quantize import PackedAWQLinear
    validate_packed(adapter.model)
    if protected_identity(adapter.model) != protected_sha256:
        raise ValueError('Treatment changed protected embedding/vision/projector/head state')
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    state = {k: v.detach().cpu().contiguous().clone() for k, v in adapter.model.state_dict().items() if k.startswith('model.layers.')}
    linears = {}
    for i, block in enumerate(adapter.model.model.layers):
        for suffix in PHI_LINEARS:
            module = block.get_submodule(suffix)
            linears[f'model.layers.{i}.{suffix}'] = {'in': module.in_features, 'out': module.out_features,
                'kind': 'packed' if isinstance(module, PackedAWQLinear) else 'masked' if isinstance(module, MaskedLinear) else 'dense',
                'bits': getattr(module, 'bits', None), 'group_size': getattr(module, 'group_size', None)}
    path = root / 'blocks.safetensors'
    save_file(state, str(path))
    document = {'format': 'bunny-treatment-v1', 'base_sha256': source_identity,
        'protected_sha256': protected_sha256, 'sha256': file_hash(path, None)['sha256'],
        'linears': linears, 'original_widths': original_widths,
        **describe_treatment(adapter.model, original_widths, recovery)}
    write_json(root / 'treatment.json', document)
    return document


def apply_treatment(adapter, root, *, max_bytes):
    """Reconstruct against verified base assets, strict-load, then classify tensors."""
    from checkpoint_io import load_tensors, verify_identity
    from VLM.quantization.awq_quantize import PackedAWQLinear
    root = Path(root)
    document = read_json(root / 'treatment.json')
    if not isinstance(document, dict) or document.get('format') != 'bunny-treatment-v1' or document.get('base_sha256') != base_identity(adapter.config):
        raise ValueError('Treatment base identity mismatch')
    if document.get('protected_sha256') != protected_identity(adapter.model):
        raise ValueError('Treatment protected multimodal identity mismatch')
    path = root / 'blocks.safetensors'
    verify_identity(path, document['sha256'], max_load_bytes=max_bytes)
    state = load_tensors(path, max_load_bytes=max_bytes)
    original_widths = [b.mlp.fc1.out_features for b in adapter.model.model.layers]
    if document['original_widths'] != original_widths:
        raise ValueError('Original MLP width mismatch')
    expected_paths = {f'model.layers.{i}.{s}' for i in range(len(original_widths)) for s in PHI_LINEARS}
    if not isinstance(document.get('linears'), dict) or set(document['linears']) != expected_paths or any(not k.startswith('model.layers.') for k in state):
        raise ValueError('Treatment contains unsupported modules')
    replacements = []
    for name, spec in document['linears'].items():
        if not isinstance(spec, dict) or set(spec) != {'in','out','kind','bits','group_size'}:
            raise ValueError('Invalid treatment linear declaration')
        current = adapter.model.get_submodule(name)
        a, b = spec['in'], spec['out']
        if type(a) is not int or type(b) is not int or min(a, b) < 1:
            raise ValueError('Invalid treatment dimensions')
        if '.self_attn.' in name and (a, b) != (current.in_features, current.out_features):
            raise ValueError('Physical attention pruning is unsupported')
        if name.endswith('mlp.fc1') and (a != current.in_features or b > current.out_features) or name.endswith('mlp.fc2') and (b != current.out_features or a > current.in_features):
            raise ValueError('Invalid MLP residual dimensions')
        bias = state.get(name + '.bias')
        if spec['kind'] == 'packed':
            if type(spec['bits']) is not int or spec['bits'] not in (4, 8) or type(spec['group_size']) is not int or not 1 <= spec['group_size'] <= a:
                raise ValueError('Invalid packed precision/group declaration')
            module = PackedAWQLinear.__new__(PackedAWQLinear)
            nn.Module.__init__(module)
            module.in_features, module.out_features = a, b
            module.bits, module.group_size = spec['bits'], spec['group_size']
            for key in ('packed_weight', 'scales', 'zero_points'):
                module.register_buffer(key, state[name + '.' + key])
            module.register_buffer('bias', bias)
        else:
            weight = state[name + '.weight']
            if tuple(weight.shape) != (b, a):
                raise ValueError('Treatment weight dimensions disagree')
            module = nn.Linear(a, b, bias=bias is not None, dtype=weight.dtype)
            if spec['kind'] == 'masked':
                mask = state[name + '.weight_mask']
                if mask.dtype != torch.bool or mask.shape != weight.shape:
                    raise ValueError('Invalid treatment mask')
                module = MaskedLinear(module, mask)
            elif spec['kind'] != 'dense':
                raise ValueError('Unsupported treatment module')
        replacements.append((name, module))
    for name, module in replacements:
        set_module(adapter.model, name, module)
    expected = {k for k in adapter.model.state_dict() if k.startswith('model.layers.')}
    if set(state) != expected:
        raise ValueError('Incomplete treatment tensor coverage')
    result = adapter.model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(k.startswith('model.layers.') for k in result.missing_keys):
        raise ValueError('Treatment load coverage failed')
    adapter.model.config.pruned_intermediate_sizes = [b.mlp.fc1.out_features for b in adapter.model.model.layers]
    if len(set(adapter.model.config.pruned_intermediate_sizes)) == 1:
        adapter.model.config.intermediate_size = adapter.model.config.pruned_intermediate_sizes[0]
    validate_packed(adapter.model)
    actual = describe_treatment(adapter.model, original_widths, document.get('recovery'))
    if any(actual[k] != document[k] for k in actual) or actual['treatment'] != adapter.config.treatment:
        raise ValueError('Treatment label contradicts loaded tensors')
    return document
