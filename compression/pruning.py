"""Phi language masks and physical MLP slicing; no vision/head pruning claims."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from runtime_policy import admit_model

PHI_LINEARS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.dense', 'mlp.fc1', 'mlp.fc2')


@dataclass(frozen=True)
class PruningConfig:
    method: str = 'magnitude_mask'
    sparsity: float = .2
    distribution: str = 'layer'
    targets: tuple[str, ...] = ('mlp.fc1', 'mlp.fc2')

    def validate(self):
        if self.method not in {'magnitude_mask', 'mlp_neurons'} or self.distribution not in {'layer', 'global'}:
            raise ValueError('Unsupported pruning method/distribution')
        if type(self.sparsity) not in (float, int) or not math.isfinite(self.sparsity) or not 0 <= self.sparsity < 1:
            raise ValueError('Sparsity must be finite in [0,1)')
        if not self.targets or len(set(self.targets)) != len(self.targets) or set(self.targets) - set(PHI_LINEARS):
            raise ValueError('Only explicit Phi block linear targets are supported')
        if self.method == 'mlp_neurons' and (set(self.targets) != {'mlp.fc1','mlp.fc2'} or self.distribution != 'layer'):
            raise ValueError('Physical MLP reduction requires the fc1/fc2 pair and layer distribution')
        return self


def targets(model, config):
    config.validate()
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None or not len(layers):
        raise ValueError('Expected nonempty Phi model.model.layers')
    selected = {}
    for index, block in enumerate(layers):
        for suffix in config.targets:
            try: module = block.get_submodule(suffix)
            except AttributeError as exc: raise ValueError(f'Missing Phi target {suffix}') from exc
            if type(module) is not nn.Linear:
                raise ValueError('Prune only dense nn.Linear targets; materialize masks before further compression')
            if not torch.isfinite(module.weight).all():
                raise ValueError('Nonfinite pruning weights')
            if module.bias is not None and not torch.isfinite(module.bias).all():
                raise ValueError('Nonfinite pruning bias')
            selected[f'model.layers.{index}.{suffix}'] = module
        if config.method == 'mlp_neurons':
            a, b = block.mlp.fc1, block.mlp.fc2
            if a.out_features != b.in_features or a.in_features != b.out_features:
                raise ValueError('Incompatible Phi MLP dependencies')
    return selected


def set_module(model, path, module):
    parent, _, leaf = path.rpartition('.')
    setattr(model.get_submodule(parent), leaf, module)


class MaskedLinear(nn.Linear):
    def __init__(self, source, mask):
        super().__init__(source.in_features, source.out_features, bias=source.bias is not None,
                         device=source.weight.device, dtype=source.weight.dtype)
        self.weight = nn.Parameter(source.weight.detach().clone() * mask, requires_grad=source.weight.requires_grad)
        if source.bias is not None:
            self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=source.bias.requires_grad)
        self.register_buffer('weight_mask', mask.to(device=self.weight.device, dtype=torch.bool))
        self.train(source.training)

    def forward(self, inputs):
        return nn.functional.linear(inputs, self.weight * self.weight_mask, self.bias)


def materialize_masks(model):
    admit_model(model)
    for path, module in list(model.named_modules()):
        if isinstance(module, MaskedLinear):
            dense = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None,
                              dtype=module.weight.dtype, device=module.weight.device)
            dense.weight = nn.Parameter((module.weight * module.weight_mask).detach().clone(), requires_grad=module.weight.requires_grad)
            if module.bias is not None:
                dense.bias = nn.Parameter(module.bias.detach().clone(), requires_grad=module.bias.requires_grad)
            dense.train(module.training)
            set_module(model, path, dense)
    return model


def _slice(source, rows, columns):
    # The random initialization is immediately replaced, so allocate only metadata.
    result = nn.Linear(len(columns), len(rows), bias=source.bias is not None, dtype=source.weight.dtype, device='meta')
    full_rows = len(rows) == source.out_features and torch.equal(rows, torch.arange(source.out_features, device=rows.device))
    full_columns = len(columns) == source.in_features and torch.equal(columns, torch.arange(source.in_features, device=columns.device))
    if full_columns:
        weight = source.weight.detach().index_select(0, rows)
    elif full_rows:
        weight = source.weight.detach().index_select(1, columns)
    else:
        weight = source.weight.detach().index_select(0, rows).index_select(1, columns)
    result.weight = nn.Parameter(weight, requires_grad=source.weight.requires_grad)
    if source.bias is not None:
        result.bias = nn.Parameter(source.bias.detach().index_select(0, rows), requires_grad=source.bias.requires_grad)
    result.train(source.training)
    return result


def prune(model, config=PruningConfig()):
    admit_model(model)  # This framework is CPU-only; no acknowledgement can enable GPU here.
    selected = targets(model, config)  # prevalidate before changing modules
    before = sum(p.numel() for p in model.parameters())
    details = []
    if config.method == 'magnitude_mask':
        masks = {name: torch.ones_like(module.weight, dtype=torch.bool) for name, module in selected.items()}
        if config.distribution == 'global':
            scores = torch.cat([m.weight.detach().to(torch.float64).abs().flatten() for m in selected.values()])
            order = torch.argsort(scores, stable=True)[:int(scores.numel() * config.sparsity)]
            combined = torch.ones_like(scores, dtype=torch.bool); combined[order] = False
            offset = 0
            for name, mask in masks.items():
                masks[name] = combined[offset:offset+mask.numel()].reshape(mask.shape); offset += mask.numel()
        else:
            for name, module in selected.items():
                order = torch.argsort(module.weight.detach().abs().flatten(), stable=True)[:int(module.weight.numel()*config.sparsity)]
                masks[name].view(-1)[order] = False
        for name, source in selected.items():
            mask = masks[name]
            set_module(model, name, MaskedLinear(source, mask))
            details.append({'path':name, 'masked':int((~mask).sum()), 'elements':mask.numel(),
                            'actual_zero_weights':int(((source.weight.detach()*mask)==0).sum()),
                            'masked_fraction':float((~mask).float().mean()),
                            'actual_zero_fraction':float(((source.weight.detach()*mask)==0).float().mean())})
    else:
        # Prevalidation is complete. Keeping this map would retain every original
        # MLP after replacement, accumulating a whole extra model's MLP storage.
        del selected
        for index, block in enumerate(model.model.layers):
            a, b = block.mlp.fc1, block.mlp.fc2
            remove = int(a.out_features * config.sparsity)
            score = a.weight.detach().to(torch.float64).abs().mean(1) + b.weight.detach().to(torch.float64).abs().mean(0)
            keep = torch.argsort(score, stable=True)[remove:].sort().values
            block.mlp.fc1 = _slice(a, keep, torch.arange(a.in_features, device=a.weight.device))
            block.mlp.fc2 = _slice(b, torch.arange(b.out_features, device=b.weight.device), keep)
            details.append({'layer':index, 'original_intermediate':a.out_features, 'remaining_intermediate':len(keep), 'kept_neurons':keep.tolist()})
    widths=[block.mlp.fc1.out_features for block in model.model.layers]
    if hasattr(model.config,'intermediate_size') and len(set(widths))==1:
        model.config.intermediate_size=widths[0]
    model.config.pruned_intermediate_sizes=widths
    return {'schema_version':1, 'configuration':asdict(config), 'parameter_elements_before':before,
            'parameter_elements_after':sum(p.numel() for p in model.parameters()), 'layers':details,
            'scope':'Phi language blocks only', 'checkpoint_size':'NOT_MEASURED', 'speedup':'NOT_MEASURED'}


def reconstruct_state(model, state):
    """Recreate masks/shapes on a compatible fresh architecture, then strict-load state."""
    admit_model(model)
    for name,tensor in state.items():
        if not torch.is_tensor(tensor) or ((tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all()):
            raise ValueError(f'Nonfinite or invalid state tensor: {name}')
    replacements = []
    for index, block in enumerate(model.model.layers):
        for suffix in PHI_LINEARS:
            path = f'model.layers.{index}.{suffix}'
            current = model.get_submodule(path)
            key = path+'.weight'
            if key not in state: raise ValueError(f'Missing state tensor {key}')
            if not torch.is_tensor(state[key]) or state[key].ndim != 2 or min(state[key].shape) < 1 or not torch.isfinite(state[key]).all():
                raise ValueError('Invalid state weight tensor')
            rows, columns = state[key].shape
            if suffix == 'mlp.fc1' and columns != current.in_features or suffix == 'mlp.fc2' and rows != current.out_features:
                raise ValueError('State changes the MLP residual interface')
            if suffix.startswith('self_attn') and (rows,columns) != tuple(current.weight.shape):
                raise ValueError('Physical attention reconstruction is unsupported')
            new = nn.Linear(columns, rows, bias=current.bias is not None, dtype=current.weight.dtype)
            mask = state.get(path+'.weight_mask')
            if mask is not None:
                if mask.dtype != torch.bool or mask.shape != state[key].shape: raise ValueError('Invalid mask state')
                new = MaskedLinear(new, mask)
            new.train(current.training)
            new.weight.requires_grad_(current.weight.requires_grad)
            if new.bias is not None: new.bias.requires_grad_(current.bias.requires_grad)
            replacements.append((path,new))
        if state[f'model.layers.{index}.mlp.fc1.weight'].shape[0] != state[f'model.layers.{index}.mlp.fc2.weight'].shape[1]:
            raise ValueError('State breaks fc1/fc2 dependencies')
    import copy
    candidate = copy.deepcopy(model)
    for path,new in replacements: set_module(candidate,path,new)
    candidate.load_state_dict(state, strict=True)
    widths=[b.mlp.fc1.out_features for b in candidate.model.layers]
    candidate.config.pruned_intermediate_sizes=widths
    if hasattr(candidate.config,'intermediate_size') and len(set(widths))==1: candidate.config.intermediate_size=widths[0]
    from evaluation.integrity import validate_state
    validate_state(candidate)
    return candidate
