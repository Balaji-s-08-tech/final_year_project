"""Real 27-block key layout with tiny widths; no pretrained/GPU result claimed."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import SiglipVisionConfig, SiglipVisionModel

from evaluation.tests.test_baseline import tiny_bunny
from evaluation.tests.test_checkpoint_layout import embed_vision
from evaluation.adapters import LocalAdapter
from evaluation.checkpoint_layout import EMBEDDED_VISION_PREFIX, validate_bunny_checkpoint


@pytest.fixture
def partial_vision(tiny_bunny):
    config = SiglipVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=27,
        num_attention_heads=2, image_size=4, patch_size=2)
    torch.manual_seed(71)
    SiglipVisionModel(config).save_pretrained(tiny_bunny.vision_path)
    embed_vision(tiny_bunny)
    path = Path(tiny_bunny.model_path)/'model.safetensors'
    state = load_file(str(path))
    omitted = [k for k in state if k.startswith(EMBEDDED_VISION_PREFIX+'vision_model.encoder.layers.26.')
               or k.startswith(EMBEDDED_VISION_PREFIX+'vision_model.head.')]
    assert len(omitted) == 27
    for key in omitted: del state[key]
    save_file(state, str(path), metadata={'format':'pt'})
    return tiny_bunny


def test_actual_421_448_layout_loads_and_compares_every_embedded_tensor(partial_vision):
    adapter=LocalAdapter(); adapter.load(partial_vision)
    report=adapter.loading_report['bunny']
    assert len(adapter.model.get_vision_tower().vision_tower.state_dict()) == 448
    assert report['redundant_vision_tensors'] == report['required_vision_tensors'] == 421
    assert report['standalone_only_tensors'] == 27
    assert report['feature_hidden_state_index'] == 26


def test_downstream_weights_do_not_affect_selected_features(partial_vision):
    adapter=LocalAdapter(); adapter.load(partial_vision)
    wrapper=adapter.model.get_vision_tower()
    model=wrapper.vision_tower.eval()
    pixels=torch.randn(1,3,4,4)
    with torch.no_grad():
        before=model(pixels,output_hidden_states=True)
        for name,p in model.named_parameters():
            if name.startswith(('vision_model.encoder.layers.26.', 'vision_model.head.', 'vision_model.post_layernorm.')):
                p.add_(torch.randn_like(p)*.5)
        after=model(pixels,output_hidden_states=True)
    assert len(before.hidden_states) == 28
    assert torch.equal(wrapper.feature_select(before), before.hidden_states[26])
    assert torch.equal(wrapper.feature_select(before), wrapper.feature_select(after))
    assert not torch.equal(before.last_hidden_state,after.last_hidden_state)
    assert not torch.equal(before.pooler_output,after.pooler_output)


@pytest.mark.parametrize('damage',['missing_required','missing_postnorm','changed_required','shape','dtype','unknown','language','projector','optional_partial'])
def test_partial_layout_rejects_invalid_embedded_state(partial_vision,damage):
    path=Path(partial_vision.model_path)/'model.safetensors'
    state=load_file(str(path))
    required=EMBEDDED_VISION_PREFIX+'vision_model.encoder.layers.25.layer_norm1.weight'
    if damage=='missing_required':del state[required]
    elif damage=='missing_postnorm':del state[EMBEDDED_VISION_PREFIX+'vision_model.post_layernorm.weight']
    elif damage=='changed_required':state[required]=state[required]+.5
    elif damage=='shape':state[required]=torch.ones(1)
    elif damage=='dtype':state[required]=torch.ones_like(state[required],dtype=torch.int32)
    elif damage=='unknown':state[EMBEDDED_VISION_PREFIX+'vision_model.unknown.weight']=torch.ones(1)
    elif damage=='language':state['model.extra.weight']=torch.ones(1)
    elif damage=='projector':state['model.mm_projector.extra.weight']=torch.ones(1)
    else:state[EMBEDDED_VISION_PREFIX+'vision_model.head.probe']=torch.ones(1,1,8)
    save_file(state,str(path),metadata={'format':'pt'})
    with pytest.raises(ValueError):LocalAdapter().load(partial_vision)


@pytest.mark.parametrize('selection',[-1,28,True,'pooler'])
def test_required_subgraph_tracks_actual_selection(partial_vision,selection):
    adapter=LocalAdapter();adapter.load(partial_vision)
    wrapper=adapter.model.get_vision_tower();wrapper.select_layer=selection
    state=load_file(str(Path(partial_vision.model_path)/'model.safetensors'))
    coverage={'unexpected_keys':[k for k in state if k.startswith(EMBEDDED_VISION_PREFIX)]}
    with pytest.raises(ValueError):validate_bunny_checkpoint(partial_vision.model_path,adapter.model,coverage)


def test_incompatible_depth_rejected(partial_vision):
    adapter=LocalAdapter();adapter.load(partial_vision)
    adapter.model.get_vision_tower().vision_tower.config.num_hidden_layers=26
    with pytest.raises(ValueError,match='depth'):
        validate_bunny_checkpoint(partial_vision.model_path,adapter.model,{})


def test_supplied_optional_head_is_still_value_checked(partial_vision):
    path=Path(partial_vision.model_path)/'model.safetensors'
    state=load_file(str(path))
    standalone=load_file(str(Path(partial_vision.vision_path)/'model.safetensors'))
    state.update({EMBEDDED_VISION_PREFIX+k:v.half() for k,v in standalone.items() if k.startswith('vision_model.head.')})
    state[EMBEDDED_VISION_PREFIX+'vision_model.head.probe'] += .5
    save_file(state,str(path),metadata={'format':'pt'})
    with pytest.raises(ValueError,match='weights differ'):LocalAdapter().load(partial_vision)


def test_unknown_feature_selector_rejected(partial_vision):
    adapter=LocalAdapter();adapter.load(partial_vision)
    adapter.model.get_vision_tower().feature_select=lambda output:output.pooler_output
    with pytest.raises(ValueError,match='feature selection'):
        validate_bunny_checkpoint(partial_vision.model_path,adapter.model,{})
