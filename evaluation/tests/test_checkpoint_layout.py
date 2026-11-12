"""Real Transformers loading paths on small local random checkpoint fixtures."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from evaluation.adapters import LocalAdapter
from evaluation.checkpoint_layout import EMBEDDED_VISION_PREFIX, TensorCheckpoint, validate_loading_info
from evaluation.tests.test_baseline import tiny_bunny


def embed_vision(config, *, full_siglip=False):
    if full_siglip:
        from transformers import SiglipConfig, SiglipModel, SiglipTextConfig, SiglipVisionConfig
        vision = SiglipVisionConfig.from_pretrained(config.vision_path)
        text = SiglipTextConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                               num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16)
        torch.manual_seed(23)
        SiglipModel(SiglipConfig.from_text_vision_configs(text, vision)).save_pretrained(config.vision_path)
    vision = load_file(str(Path(config.vision_path) / 'model.safetensors'))
    path = Path(config.model_path) / 'model.safetensors'
    state = load_file(str(path))
    state.update({EMBEDDED_VISION_PREFIX + key: value.half() for key, value in vision.items() if key.startswith('vision_model.')})
    save_file(state, str(path), metadata={'format': 'pt'})


@pytest.mark.parametrize('full_siglip', [False, True])
def test_embedded_vision_and_full_siglip_are_verified_without_fixed_counts(tiny_bunny, full_siglip, caplog):
    embed_vision(tiny_bunny, full_siglip=full_siglip)
    adapter = LocalAdapter(); adapter.load(tiny_bunny)
    vision = adapter.model.get_vision_tower().vision_tower
    report = adapter.loading_report
    assert report['bunny']['redundant_vision_tensors'] == len(vision.state_dict())
    assert report['bunny']['redundancy_comparison'] == 'EXACT_AFTER_FP16_CONVERSION'
    assert (report['siglip']['verified_unused_tensors'] > 0) == full_siglip
    assert 'loading warnings reconciled' in caplog.text


@pytest.mark.parametrize('damage', ['unknown_language', 'unknown_embedded', 'missing_embedded', 'embedded_shape', 'wrong_vision', 'missing_projector'])
def test_bunny_bad_coverage_or_mismatched_vision_fails(tiny_bunny, damage):
    embed_vision(tiny_bunny)
    path = Path(tiny_bunny.model_path) / 'model.safetensors'
    state = load_file(str(path))
    key = next(k for k in state if k.startswith(EMBEDDED_VISION_PREFIX))
    if damage == 'unknown_language': state['model.surprise.weight'] = torch.ones(1)
    elif damage == 'unknown_embedded': state[EMBEDDED_VISION_PREFIX+'surprise'] = torch.ones(1)
    elif damage == 'missing_embedded': del state[key]
    elif damage == 'embedded_shape': state[key] = torch.ones(1)
    elif damage == 'wrong_vision': state[key] = state[key] + .5
    else: del state['model.mm_projector.0.weight']
    save_file(state, str(path), metadata={'format':'pt'})
    with pytest.raises(ValueError, match='[Cc]heckpoint|[Vv]ision'):
        LocalAdapter().load(tiny_bunny)


@pytest.mark.parametrize('damage', ['extra_text', 'missing_text', 'text_shape', 'extra_other', 'missing_vision', 'vision_shape'])
def test_full_siglip_must_match_configured_metadata(tiny_bunny, damage):
    embed_vision(tiny_bunny, full_siglip=True)
    path = Path(tiny_bunny.vision_path) / 'model.safetensors'
    state = load_file(str(path))
    key = next(k for k in state if k.startswith('text_model.'))
    if damage == 'extra_text': state['text_model.surprise.weight'] = torch.ones(1)
    elif damage == 'missing_text': del state[key]
    elif damage == 'text_shape': state[key] = torch.ones(1)
    elif damage == 'extra_other': state['other.weight'] = torch.ones(1)
    elif damage == 'missing_vision': del state['vision_model.embeddings.patch_embedding.weight']
    else: state['vision_model.embeddings.patch_embedding.weight'] = torch.ones(1)
    save_file(state, str(path), metadata={'format':'pt'})
    with pytest.raises((ValueError, RuntimeError), match='[Cc]heckpoint|size mismatch'):
        LocalAdapter().load(tiny_bunny)


@pytest.mark.parametrize('field', ['missing_keys', 'mismatched_keys', 'error_msgs'])
def test_loading_diagnostics_always_fail(field):
    with pytest.raises(ValueError):
        validate_loading_info({field:['bad'], 'unexpected_keys':['expected']}, {'expected'}, 'fixture')


@pytest.mark.parametrize('damage', ['none','wrong_map','escape','duplicate'])
def test_shard_index_is_checked_against_actual_headers(tiny_bunny, damage):
    embed_vision(tiny_bunny)
    root = Path(tiny_bunny.model_path)
    state = load_file(str(root/'model.safetensors'))
    keys = list(state)
    first, second = {k:state[k] for k in keys[::2]}, {k:state[k] for k in keys[1::2]}
    if damage == 'duplicate': second[keys[0]] = state[keys[0]]
    for name, tensors in [('first.safetensors',first),('second.safetensors',second)]:
        save_file(tensors,str(root/name),metadata={'format':'pt'})
    (root/'model.safetensors').unlink()
    mapping = {k:'first.safetensors' if k in first else 'second.safetensors' for k in keys}
    if damage == 'wrong_map': mapping[keys[0]]='second.safetensors'
    if damage == 'escape': mapping[keys[0]]='../escape.safetensors'
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
    if damage == 'none':
        with TensorCheckpoint(root) as checkpoint: assert set(checkpoint.tensors) == set(state)
    else:
        with pytest.raises(ValueError):
            with TensorCheckpoint(root): pass


def test_single_file_precedes_stale_index_like_transformers(tiny_bunny):
    root = Path(tiny_bunny.model_path)
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'unused':'absent.safetensors'}}))
    with TensorCheckpoint(root) as checkpoint:
        assert 'lm_head.weight' in checkpoint.tensors


def test_explicit_trusted_nonredundant_bin_support_preserved(tiny_bunny):
    from dataclasses import replace
    root = Path(tiny_bunny.model_path)
    state = load_file(str(root/'model.safetensors'))
    torch.save(state, root/'pytorch_model.bin')
    (root/'model.safetensors').unlink()
    adapter = LocalAdapter(); adapter.load(replace(tiny_bunny, trusted_checkpoint=True))
    assert adapter.loading_report['bunny']['redundancy_comparison'] == 'NO_SAFETENSORS_EXCEPTION_ALLOWED'


@pytest.mark.parametrize('s2', [False, True])
def test_both_siglip_tower_loaders_share_strict_full_checkpoint_validation(tiny_bunny, s2):
    from types import SimpleNamespace
    from VLM.quantization.awq_model_loader import _enable_project_imports
    _enable_project_imports()
    from bunny.model.multimodal_encoder.siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
    embed_vision(tiny_bunny, full_siglip=True)
    tower_class = SiglipVisionTowerS2 if s2 else SiglipVisionTower
    tower = tower_class(tiny_bunny.vision_path, SimpleNamespace(s2_scales='4,8'), delay_load=True)
    tower.load_model()
    assert tower.is_loaded and tower.layout_verification['verified_unused_tensors'] > 0
