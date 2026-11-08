from pathlib import Path
import pytest
import torch

def test_backend_rejects_stale_checkpoint_consent(tmp_path,monkeypatch):
    from backend import jobs
    from experiments.provenance import file_hash
    p=tmp_path/'weights.pt';p.write_bytes(b'new owned sentinel, not a pickle')
    current=file_hash(p,None)['sha256']
    monkeypatch.setattr(jobs,'get_checkpoint',lambda _: {'name':p.name,'path':str(p),'format':'portable_awq','checkpoint_sha256':current})
    manager=jobs.JobManager(executor=lambda *a:{'mock_only':True},experiment_root=tmp_path/'runs')
    with pytest.raises(ValueError,match='hash|changed|identity'):
        manager.submit({'operation':'inference','checkpoint_id':'same-path','prompt':'fixture','max_new_tokens':1,'trusted_checkpoint':True,'checkpoint_sha256':'0'*64})


@pytest.mark.parametrize('consent,digest',[(False,None),(True,None),(True,'malformed')])
def test_execution_consent_requires_identity(tmp_path,monkeypatch,consent,digest):
    from backend import jobs
    p=tmp_path/'owned.pt';p.write_bytes(b'owned sentinel, never deserialized')
    monkeypatch.setattr(jobs,'get_checkpoint',lambda _:{'name':p.name,'path':str(p),'format':'portable_awq'})
    manager=jobs.JobManager(executor=lambda *a:pytest.fail('Invalid request reached execution'),experiment_root=tmp_path/'runs')
    with pytest.raises(ValueError):manager.submit({'operation':'inference','checkpoint_id':'fixture','trusted_checkpoint':consent,'checkpoint_sha256':digest})


def test_checkpoint_change_while_queued(tmp_path,monkeypatch):
    from backend import jobs
    from experiments.provenance import file_hash
    import threading
    p=tmp_path/'owned.pt';p.write_bytes(b'owned version one, never deserialized');digest=file_hash(p,None)['sha256']
    checkpoint={'name':p.name,'path':str(p),'format':'portable_awq'}
    monkeypatch.setattr(jobs,'get_checkpoint',lambda _:checkpoint);monkeypatch.setattr(jobs,'resolve_approved_path',lambda v:Path(v))
    manager=jobs.JobManager(experiment_root=tmp_path/'runs');monkeypatch.setattr(manager._pool,'submit',lambda *a:None)
    request={'operation':'inference','checkpoint_id':'fixture','trusted_checkpoint':True,'checkpoint_sha256':digest}
    record=manager.submit(request);p.write_bytes(b'owned version two, never deserialized')
    monkeypatch.setattr(manager,'_infer',lambda *a:pytest.fail('Changed file reached inference'))
    with pytest.raises(ValueError,match='hash changed'):manager._execute(record.id,request,threading.Event(),None,None)


def test_hash_rejected_before_deserialization(tmp_path,monkeypatch):
    from checkpoint_io import load_pickle
    p=tmp_path/'not-pickle.pt';p.write_bytes(b'owned sentinel, never a pickle')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Wrong digest reached deserialization'))
    with pytest.raises(ValueError,match='hash changed'):load_pickle(p,trusted=True,expected_sha256='0'*64)


def test_verified_portable_owned_roundtrip(tmp_path):
    from experiments.tests.test_mission11 import exported
    from experiments.provenance import file_hash
    from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint
    p=exported(tmp_path);digest=file_hash(p,None)['sha256']
    tok,model,metadata=load_quantized_bunny_checkpoint(str(p),trusted=True,expected_sha256=digest)
    with torch.inference_mode():assert torch.isfinite(model(input_ids=tok('calibration',return_tensors='pt')['input_ids']).logits).all()


def test_combined_artifact_replacement_rejected(tmp_path,monkeypatch):
    from compression.synthetic import fixture
    from compression.pipeline import load_artifact
    from evaluation.inspection import architecture
    from experiments.store import write_json
    from experiments.provenance import file_hash
    import VLM.quantization.awq_model_loader as loader
    model,tok,*_=fixture();path=tmp_path/'model.pt'
    payload={'format':'vlm-compression-v1','model':model,'tokenizer':tok,'metadata':{'architecture':architecture(model)}}
    torch.save(payload,path);write_json(tmp_path/'artifact_manifest.json',file_hash(path,None))
    def replace_with_other_owned_fixture():
        # Both payloads are known, locally generated, explicitly trusted synthetic code.
        with torch.no_grad():model.model.embed_tokens.weight.fill_(1)
        torch.save(payload,path)
    monkeypatch.setattr(loader,'_enable_project_imports',replace_with_other_owned_fixture)
    with pytest.raises(ValueError,match='hash|content'):load_artifact(tmp_path,trusted=True)
