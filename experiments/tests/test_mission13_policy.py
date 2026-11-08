import pytest
import torch
from experiments.tests.test_mission11 import pair,rewrite
from experiments.store import read_json
from experiments.compare import compare


def test_invalid_effective_temperature_not_verified(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch)
    for r in runs:
        p=r.path/'effective_inference.json';d=read_json(p);d['generation']['do_sample']=True;d['generation']['temperature']=-1;rewrite(p,d)
    assert not compare(*[r.path for r in runs])['equivalent_protocol_verified']


def test_invalid_normalization_epsilon_rejected():
    from compression.synthetic import fixture
    from evaluation.integrity import validate_packed
    model=fixture()[0];model.model.layers[0].input_layernorm.eps=-1
    with pytest.raises(ValueError,match='normalization|LayerNorm'):validate_packed(model)


@pytest.mark.parametrize('field,value',[('temperature',0),('temperature',True),('temperature','warm'),('top_p',1.1),('typical_p',0),('num_return_sequences',2),('top_k',-1),('unknown_setting',1)])
def test_generation_semantic_edges_rejected(field,value):
    from evaluation.inference_policy import validate_generation
    config={'max_new_tokens':2,'do_sample':False,'num_beams':1,'use_cache':True,'pad_token_id':0,'eos_token_id':2,field:value}
    with pytest.raises(ValueError):validate_generation(config)


def test_qk_epsilon_metadata_matches_config(tmp_path):
    from compression.synthetic import fixture
    from VLM.quantization.portable_integrity import describe
    model=fixture()[0];model.config.qk_layernorm=True
    a=model.model.layers[0].self_attn;a.qk_layernorm=True;a.q_layernorm=torch.nn.LayerNorm(16,eps=.2);a.k_layernorm=torch.nn.LayerNorm(16)
    with pytest.raises(ValueError,match='epsilon'):describe(model)


@pytest.mark.parametrize('field,value',[('pad_token_id',True),('eos_token_id',[-1]),('max_new_tokens',-1)])
def test_generation_token_controls_are_typed(field,value):
    from evaluation.inference_policy import validate_generation
    settings={'max_new_tokens':2,'do_sample':False,'num_beams':1,'use_cache':True,'pad_token_id':0,'eos_token_id':2,field:value}
    with pytest.raises(ValueError):validate_generation(settings)


def test_invalid_metric_mapping_is_excluded_not_crashed(tmp_path,monkeypatch):
    from experiments.compare import compare_collection
    runs=pair(tmp_path,monkeypatch)
    p=sorted(runs[1].path.glob('state_*.json'))[-1];state=read_json(p);state['metrics']={'metrics':[]};rewrite(p,state)
    report=compare_collection(runs[0].path,[runs[1].path])
    assert report['included_count']==0 and report['excluded_count']==1
