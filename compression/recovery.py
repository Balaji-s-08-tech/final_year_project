"""Bounded CPU LoRA recovery on caller-owned local causal-language fixtures.

This validates recovery mechanics; it does not establish research recovery quality.
"""
from dataclasses import asdict, dataclass
import copy
import math

import torch

from runtime_policy import admit_model
from compression.pruning import PHI_LINEARS


@dataclass(frozen=True)
class RecoveryConfig:
    rank: int = 2
    alpha: int = 4
    learning_rate: float = .001
    steps: int = 2
    seed: int = 42
    targets: tuple[str, ...] = ('fc1', 'fc2')
    max_parameter_elements: int = 1_000_000

    def validate(self):
        for key in ('rank', 'alpha', 'steps', 'max_parameter_elements'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'Positive integer {key} required')
        if self.steps > 1000 or type(self.seed) is not int or self.seed < 0:
            raise ValueError('Recovery exceeds step/seed bounds')
        if type(self.learning_rate) not in (int, float) or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('Finite learning rate in (0,1] required')
        allowed = {name.rsplit('.', 1)[1] for name in PHI_LINEARS}
        if not self.targets or len(set(self.targets)) != len(self.targets) or set(self.targets) - allowed:
            raise ValueError('Unique Phi linear recovery targets required')
        return self


def prepare_recovery_batches(source, batches):
    """Admit tensor shape/content before hashing, cloning a model or training."""
    if not isinstance(batches, (tuple, list)) or not 1 <= len(batches) <= 32:
        raise ValueError('Provide 1..32 local recovery batches')
    vocab = source.get_input_embeddings().num_embeddings
    owned = []
    for batch in batches:
        if not isinstance(batch, dict) or set(batch) != {'input_ids', 'attention_mask', 'labels'}:
            raise ValueError('Explicit input_ids, attention_mask and labels required')
        ids, mask, labels = (batch[key] for key in ('input_ids', 'attention_mask', 'labels'))
        if any(not torch.is_tensor(t) or t.device.type != 'cpu' or t.ndim != 2 for t in (ids, mask, labels)):
            raise ValueError('CPU two-dimensional recovery tensors required')
        if ids.shape != mask.shape or ids.shape != labels.shape or not 1 <= ids.shape[0] <= 8 or not 2 <= ids.shape[1] <= 128:
            raise ValueError('Recovery batch shape exceeds bounds')
        if ids.dtype != torch.long or labels.dtype != torch.long or not ((ids >= 0) & (ids < vocab)).all():
            raise ValueError('Invalid recovery token IDs')
        if not ((mask == 0) | (mask == 1)).all() or not (((labels == -100) | ((labels >= 0) & (labels < vocab))).all()):
            raise ValueError('Invalid recovery mask/labels')
        if not labels[mask == 0].eq(-100).all() or not ((labels[:, 1:] != -100) & (mask[:, 1:] == 1) & (mask[:, :-1] == 1)).any(dim=1).all():
            raise ValueError('Padding must be ignored and each row needs shifted targets')
        owned.append({key: value.detach().clone() for key, value in batch.items()})
    return owned


def recover_lora(source, batches, config=RecoveryConfig()):
    """Return a trained PEFT adapter and loss ledger; preserve the source model.

    Batches contain explicit local input_ids/attention_mask/labels. Labels must
    exclude padding and have at least one shifted supervised target per row.
    Inputs are bounded before copying a model or creating trainable parameters.
    """
    config.validate()
    admit_model(source)
    if sum(p.numel() for p in source.parameters()) > config.max_parameter_elements:
        raise ValueError('CPU recovery fixture parameter budget exceeded')
    if any(p.dtype != torch.float32 for p in source.parameters()):
        raise ValueError('CPU recovery requires float32 parameters')
    owned = prepare_recovery_batches(source, batches)
    from peft import LoraConfig, get_peft_model
    # Fully qualified targets prevent matching vision/projector layers by suffix.
    paths = []
    for index, block in enumerate(source.model.layers):
        for suffix in PHI_LINEARS:
            if suffix.rsplit('.', 1)[1] in config.targets:
                module = block.get_submodule(suffix)
                if type(module) is not torch.nn.Linear:
                    raise ValueError('Recovery requires dense Phi linears; materialize masks first')
                if config.rank > min(module.in_features, module.out_features):
                    raise ValueError('Recovery rank exceeds target dimensions')
                paths.append(f'model.layers.{index}.{suffix}')
    if not paths:
        raise ValueError('No Phi recovery targets')
    adapter_elements = sum(config.rank * (source.get_submodule(path).in_features + source.get_submodule(path).out_features) for path in paths)
    if adapter_elements > config.max_parameter_elements:
        raise ValueError('CPU adapter parameter budget exceeded')
    from evaluation.integrity import validate_state
    validate_state(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = get_peft_model(copy.deepcopy(source), LoraConfig(
            r=config.rank, lora_alpha=config.alpha, target_modules=paths,
            lora_dropout=0., bias='none'))
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
        losses = []
        for step in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            output = model(**owned[step % len(owned)], use_cache=False)
            loss = output.loss
            if loss is None or loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError('Nonfinite recovery loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite recovery gradients')
            optimizer.step()
            losses.append(float(loss.detach()))
        validate_state(model)
        model.eval()
    return model, {'configuration': asdict(config), 'losses': losses,
                   'evidence': 'CPU_FIXTURE_RECOVERY_ONLY', 'research_quality': 'NOT_MEASURED'}
