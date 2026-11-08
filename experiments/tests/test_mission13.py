"""Mission 13 reproductions and adjacent public-boundary regressions."""
from pathlib import Path
import json
import pytest
import torch
from experiments.tests.test_mission11 import pair,exported,rewrite
from experiments.compare import compare,compare_collection
from experiments.store import read_json
from experiments.schema import benchmark
from experiments.provenance import file_hash
from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint


def test_overflow_effective_evidence_never_verifies(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch)
    for r in runs:
        p=r.path/'effective_inference.json';d=read_json(p);d['generation']['temperature']='EXP';p.write_text(json.dumps(d).replace('"EXP"','1e999'))
    assert not compare(*[r.path for r in runs])['equivalent_protocol_verified']


def test_bad_numeric_candidate_does_not_abort_collection(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch);(runs[1].path/'config.json').write_text('{"seed":1e999}')
    report=compare_collection(runs[0].path,[runs[1].path])
    assert report['excluded_count']==1


def test_unmeasured_completed_not_eligible(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch)
    for r in runs:
        p=sorted(r.path.glob('state_*.json'))[-1];s=read_json(p);s['metrics']=benchmark();rewrite(p,s)
    report=compare_collection(runs[0].path,[runs[1].path])
    assert report['included_count']==0 and report['excluded_count']==1


def test_untrusted_loader_stops_before_deserialization(tmp_path,monkeypatch):
    p=exported(tmp_path)
    import VLM.quantization.awq_model_loader as loader
    called=[]
    monkeypatch.setattr(loader,'_torch_load',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='trust'):loader.load_quantized_bunny_checkpoint(str(p))
    assert not called


def test_qk_norm_conflict_rejected(tmp_path):
    p=exported(tmp_path);payload=torch.load(p,weights_only=False)
    a=payload['model'].model.layers[0].self_attn;a.qk_layernorm=True;a.q_layernorm=torch.nn.LayerNorm(7);a.k_layernorm=torch.nn.LayerNorm(7)
    torch.save(payload,p);side=Path(str(p)+'.metadata.json');d=read_json(side);h=file_hash(p,None);d.update(sha256=h['sha256'],size_bytes=h['size_bytes']);rewrite(side,d)
    with pytest.raises(ValueError,match='normalization|qk_layernorm'):load_quantized_bunny_checkpoint(str(p), trusted=True)


@pytest.mark.parametrize('text',['1e999','-1e999','NaN','Infinity','-Infinity','1e-999',str(2**63),str(-(2**63)-1),'{"nested":[{"value":1e999}]}'])
def test_numeric_ingress_rejects_out_of_range(text):
    from experiments.jsonio import loads
    with pytest.raises(ValueError):loads(text)


def test_numeric_boundaries_and_nested_metric_metadata():
    from experiments.jsonio import loads
    from experiments.schema import measurement
    assert loads(str(2**63-1))==2**63-1
    assert loads(str(-(2**63)))==-(2**63)
    assert loads('1e-12')==1e-12
    with pytest.raises(ValueError):measurement('inference_latency',1,status='MEASURED',method='fixture',metadata={'nested':[float('inf')]})


def test_valid_peer_survives_malformed_numeric_candidate(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch);bad=tmp_path/'bad';bad.mkdir();(bad/'config.json').write_text('{"x":1e999}')
    report=compare_collection(runs[0].path,[bad,runs[1].path])
    assert report['included_count']==1 and report['excluded_count']==1


def test_partial_measurements_are_explicit(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch);q=compare(*[r.path for r in runs])
    assert q['common_measured_metrics']==['inference_latency']
    assert all(m['status']=='PARTIAL' for m in q['measurement_coverage'])


def test_valid_qk_normalization_forward_and_reload(tmp_path):
    from compression.synthetic import fixture
    from VLM.quantization.awq_quantize import PackedAWQLinear,_save_output
    from VLM.quantization.awq_config import AWQConfig
    model,tok,*_=fixture();model.config.qk_layernorm=True
    a=model.model.layers[0].self_attn;a.qk_layernorm=True;a.q_layernorm=torch.nn.LayerNorm(16);a.k_layernorm=torch.nn.LayerNorm(16)
    model.model.layers[0].mlp.fc1=PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1,4,8)
    with torch.inference_mode():expected=model(torch.tensor([[1,4]])).logits
    p=_save_output(tmp_path,model,tok,AWQConfig(bits=4,group_size=8),{}, {})
    _,loaded,_=load_quantized_bunny_checkpoint(str(p), trusted=True)
    torch.testing.assert_close(loaded(torch.tensor([[1,4]])).logits,expected)


@pytest.mark.parametrize('name',['portable','recovery','recovery_state','raw_pickle','pipeline','verifier'])
@pytest.mark.parametrize('consent',[False,1,'yes'])
def test_public_pickle_boundaries_reject_without_true(tmp_path,monkeypatch,name,consent):
    import checkpoint_io
    import VLM.quantization.awq_model_loader as loader
    called=[]
    monkeypatch.setattr(torch,'load',lambda *a,**k:called.append(True))
    p=tmp_path/'artifact.pt';p.write_bytes(b'audit sentinel; never unpickle')
    if name=='portable': operation=lambda:loader.load_quantized_bunny_checkpoint(str(p),trusted=consent)
    elif name=='recovery': operation=lambda:loader.load_bunny_recovered_model(base_model=None,pruned_checkpoint=str(p),recovery_checkpoint=None,trusted=consent)
    elif name=='recovery_state': operation=lambda:loader._load_recovery_state(torch.nn.Linear(1,1),p,trusted=consent)
    elif name=='raw_pickle':operation=lambda:checkpoint_io.load_pickle(p,trusted=consent)
    elif name=='pipeline':
        from compression.pipeline import load_artifact
        # Trust must be checked before even reading the artifact manifest.
        operation=lambda:load_artifact(tmp_path,trusted=consent)
    else:
        from compression.quantization import verify_checkpoint
        operation=lambda:verify_checkpoint(p,trusted=consent)
    with pytest.raises(ValueError):operation()
    assert not called


def test_bad_header_rejected_before_weight_hash(tmp_path,monkeypatch):
    p=exported(tmp_path);side=Path(str(p)+'.metadata.json');data=read_json(side);data['architecture']['config']['num_attention_heads']=0;rewrite(side,data)
    import experiments.artifacts as artifacts
    monkeypatch.setattr(artifacts,'file_hash',lambda *a,**k:pytest.fail('Hashed payload before rejecting invalid dimensions'))
    with pytest.raises(ValueError,match='dimension'):load_quantized_bunny_checkpoint(str(p),trusted=True)


def test_tensor_pickle_also_requires_trust(tmp_path,monkeypatch):
    from checkpoint_io import load_tensors
    called=[];monkeypatch.setattr(torch,'load',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='trust'):load_tensors(tmp_path/'unknown.bin')
    assert not called


def test_safetensors_needs_no_pickle_consent(tmp_path):
    from checkpoint_io import load_tensors
    from safetensors.torch import save_file
    p=tmp_path/'state.safetensors';save_file({'weight':torch.ones(2)},p)
    assert torch.equal(load_tensors(p)['weight'],torch.ones(2))
