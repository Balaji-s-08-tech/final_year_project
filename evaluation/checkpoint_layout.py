"""Strict safetensors layout and redundant-vision validation, without fixed counts."""
from contextlib import ExitStack
from pathlib import Path
import logging

import torch
from safetensors import safe_open
from experiments.store import read_json

LOGGER = logging.getLogger(__name__)
EMBEDDED_VISION_PREFIX = 'model.vision_tower.vision_tower.'


class TensorCheckpoint:
    """Read the same single-file or indexed safetensors layout as Transformers."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.stack = ExitStack()
        self.tensors = {}

    def __enter__(self):
        try:
            index = self.root / 'model.safetensors.index.json'
            mapping = None
            if (self.root / 'model.safetensors').is_file():
                names = ['model.safetensors']
            elif index.is_file():
                document = read_json(index)
                mapping = document.get('weight_map') if isinstance(document, dict) else None
                if not isinstance(mapping, dict) or not mapping or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
                    raise ValueError('Invalid safetensors checkpoint weight map')
                names = sorted(set(mapping.values()))
            else:
                names = ['model.safetensors']
            if len(names) > 64:
                raise ValueError('Checkpoint shard count exceeds supported metadata budget')
            actual_mapping = {}
            for name in names:
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root) or not path.is_file() or path.suffix != '.safetensors':
                    raise ValueError('Missing or unsafe safetensors checkpoint shard')
                handle = self.stack.enter_context(safe_open(str(path), framework='pt', device='cpu'))
                for key in handle.keys():
                    if key in self.tensors:
                        raise ValueError(f'Duplicate checkpoint tensor: {key}')
                    self.tensors[key] = handle
                    actual_mapping[key] = name
                    if len(self.tensors) > 100000:
                        raise ValueError('Checkpoint tensor count exceeds metadata budget')
            if mapping is not None and mapping != actual_mapping:
                raise ValueError('Checkpoint index differs from actual shard tensor metadata')
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        self.stack.close()

    def shape(self, key):
        return tuple(self.tensors[key].get_slice(key).get_shape())

    def validate_shapes(self, expected, *, allowed_extra=()):
        actual = set(self.tensors)
        missing = set(expected) - actual
        unexpected = actual - set(expected) - set(allowed_extra)
        if missing or unexpected:
            raise ValueError(f'Checkpoint coverage failed: missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}')
        for key, tensor in expected.items():
            if self.shape(key) != tuple(tensor.shape):
                raise ValueError(f'Checkpoint shape mismatch: {key}')
            if self.tensors[key].get_slice(key).get_dtype() not in {'F16', 'BF16', 'F32', 'F64'}:
                raise ValueError(f'Checkpoint floating tensor dtype mismatch: {key}')


def validate_loading_info(coverage, expected_unused, label):
    if any(coverage.get(key) for key in ('missing_keys', 'mismatched_keys', 'error_msgs')) or set(coverage.get('unexpected_keys', [])) != set(expected_unused):
        raise ValueError(f'Checkpoint {label} coverage failed: {coverage}')


def validate_siglip_checkpoint(root, model, coverage):
    """Allow only the complete configured full SigLIP model's unused tensors."""
    config = read_json(Path(root) / 'config.json')
    expected = model.state_dict()
    if config.get('model_type') == 'siglip':
        from accelerate import init_empty_weights
        from transformers import SiglipConfig, SiglipModel
        with init_empty_weights(include_buffers=True):
            full = SiglipModel(SiglipConfig.from_dict(config))
        expected = full.state_dict()
    elif config.get('model_type') != 'siglip_vision_model':
        raise ValueError('Unsupported SigLIP checkpoint configuration')
    unused = set(expected) - set(model.state_dict())
    if any(not (key.startswith('text_model.') or key in {'logit_scale', 'logit_bias'}) for key in unused):
        raise ValueError('Unrecognized configured SigLIP unused tensor')
    with TensorCheckpoint(root) as checkpoint:
        checkpoint.validate_shapes(expected)
    validate_loading_info(coverage, unused, 'encoder')
    report = {'status': 'VERIFIED_CONFIGURED_LAYOUT', 'loaded_vision_tensors': len(model.state_dict()),
              'verified_unused_tensors': len(unused), 'unused_categories': ['text_model', 'contrastive_head'] if unused else []}
    if unused:
        LOGGER.warning('SigLIP loading warnings reconciled: %d configured text/head tensors are intentionally unused; all %d vision tensors covered', len(unused), len(model.state_dict()))
    return report


def validate_bunny_checkpoint(root, model, coverage, *, chunk_bytes=8*1024*1024):
    """Reconcile lazy tower extras with the actually loaded standalone vision model.

    Embedded FP16 copies may be skipped only with exact key/shape coverage and
    equality after FP16 conversion. Compare bounded row slices on CPU.
    """
    root = Path(root)
    if not (root/'model.safetensors').is_file() and not (root/'model.safetensors.index.json').is_file():
        # Preserve the old explicitly trusted .bin path with strict HF coverage.
        # The new redundant-key exception is available only for safetensors.
        if not (root/'pytorch_model.bin').is_file() and not (root/'pytorch_model.bin.index.json').is_file():
            raise ValueError('Checkpoint weights unavailable')
        validate_loading_info(coverage, set(), 'Bunny')
        return {'status': 'STRICT_TRANSFORMERS_COVERAGE', 'redundant_vision_tensors': 0,
                'redundancy_comparison': 'NO_SAFETENSORS_EXCEPTION_ALLOWED'}
    tower = model.get_vision_tower().vision_tower
    vision = tower.state_dict()
    expected = {key: value for key, value in model.state_dict().items() if not key.startswith(EMBEDDED_VISION_PREFIX)}
    with TensorCheckpoint(root) as checkpoint:
        embedded = {key for key in checkpoint.tensors if key.startswith(EMBEDDED_VISION_PREFIX)}
        checkpoint.validate_shapes(expected, allowed_extra=embedded)
        if embedded:
            required = {EMBEDDED_VISION_PREFIX + key for key in vision}
            if embedded != required:
                raise ValueError('Embedded Bunny vision checkpoint coverage differs from standalone SigLIP')
            for key, target in vision.items():
                source_key = EMBEDDED_VISION_PREFIX + key
                if checkpoint.shape(source_key) != tuple(target.shape):
                    raise ValueError(f'Embedded vision shape mismatch: {key}')
                source = checkpoint.tensors[source_key].get_slice(source_key)
                if source.get_dtype() not in {'F16', 'BF16', 'F32', 'F64'}:
                    raise ValueError(f'Embedded vision dtype mismatch: {key}')
                rows = target.shape[0] if target.ndim else 1
                row_elements = max(1, target.numel() // max(1, rows))
                stride = max(1, chunk_bytes // (row_elements * 8))
                for start in range(0, rows, stride):
                    a = checkpoint.tensors[source_key].get_tensor(source_key) if target.ndim == 0 else source[start:start+stride]
                    b = target if target.ndim == 0 else target[start:start+stride]
                    a, b = a.to(dtype=torch.float16, device='cpu'), b.to(dtype=torch.float16, device='cpu')
                    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.equal(a, b):
                        raise ValueError(f'Embedded Bunny vision weights differ from standalone SigLIP: {key}')
    validate_loading_info(coverage, embedded, 'Bunny')
    report = {'status': 'VERIFIED_CONFIGURED_LAYOUT', 'redundant_vision_tensors': len(embedded),
              'redundancy_comparison': 'EXACT_AFTER_FP16_CONVERSION' if embedded else 'NOT_PRESENT'}
    if embedded:
        LOGGER.warning('Bunny loading warnings reconciled: %d redundant vision tensors verified against standalone SigLIP; language/projector coverage is exact', len(embedded))
    return report
