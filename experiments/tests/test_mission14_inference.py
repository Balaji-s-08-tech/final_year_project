from pathlib import Path
import pytest

def test_backend_prompt_truncation_rejected(monkeypatch):
    from compression.synthetic import fixture
    from backend.jobs import JobManager
    from types import SimpleNamespace
    import VLM.quantization.awq_model_loader as loader
    import threading
    _,tok,*_=fixture()
    class RecordingModel:
        config=SimpleNamespace(max_position_embeddings=4096)
        def generate(self,**kw):return kw['input_ids']
    monkeypatch.setattr(loader,'load_quantized_bunny_checkpoint',lambda *a,**k:(tok,RecordingModel(),{}))
    # 2051 tokens in 2051 characters: punctuation is separately tokenized.
    prompt='a,'*1025+'a'
    assert len(tok(prompt)['input_ids'])>2048
    with pytest.raises(ValueError,match='prompt|token|truncat'):
        JobManager._infer('owned',Path('not-accessed'),{'prompt':prompt,'max_new_tokens':1,'trusted_checkpoint':True},threading.Event())


@pytest.mark.parametrize('value',[True,1.0,'1'])
def test_backend_generation_not_silently_coerced(value):
    from backend.schemas import JobRequest
    with pytest.raises(ValueError):JobRequest(operation='inference',checkpoint_id='fixture',prompt='fixture',max_new_tokens=value)


def test_actual_owned_portable_inference_records_effective_input(tmp_path):
    from backend.jobs import JobManager
    from experiments.tests.test_mission11 import exported
    from experiments.provenance import file_hash
    import threading
    p=exported(tmp_path)
    result=JobManager._infer('fixture',p,{'prompt':'calibration alpha','max_new_tokens':1,'trusted_checkpoint':True,'checkpoint_sha256':file_hash(p,None)['sha256']},threading.Event())
    assert result['original_input_tokens']==result['effective_input_tokens']==2
    assert result['truncated'] is False
    assert result['effective_generation']['do_sample'] is False
    assert result['effective_generation']['max_new_tokens']==1


@pytest.mark.parametrize('value',[True,0,129,float('nan')])
def test_direct_inference_invalid_token_policy(tmp_path,value,monkeypatch):
    from backend.jobs import JobManager
    from compression.synthetic import fixture
    import VLM.quantization.awq_model_loader as loader
    import threading
    model,tok,*_=fixture()
    monkeypatch.setattr(loader,'load_quantized_bunny_checkpoint',lambda *a,**k:(tok,model,{}))
    with pytest.raises(ValueError):JobManager._infer('fixture',tmp_path/'not-accessed',{'prompt':'calibration','max_new_tokens':value},threading.Event())
