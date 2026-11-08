from pathlib import Path
import pytest
import torch
from experiments.tests.test_mission11 import pair, rewrite
from experiments.store import read_json
from experiments.compare import compare

def test_safetensors_budget_precedes_allocation(tmp_path,monkeypatch):
    from safetensors.torch import save_file
    import checkpoint_io
    p=tmp_path/'weights.safetensors';save_file({'w':torch.ones(2)},p)
    def denied(*args,**kwargs):raise ValueError('CPU tensor byte budget exceeded')
    monkeypatch.setattr(checkpoint_io,'admit_size',denied)
    with pytest.raises(ValueError,match='budget'):checkpoint_io.load_tensors(p)



def test_safe_tensor_roundtrip_without_pickle_consent(tmp_path):
    from safetensors.torch import save_file
    from checkpoint_io import load_tensors
    path=tmp_path/'owned.safetensors';value=torch.arange(8,dtype=torch.float32);save_file({'w':value},path)
    assert torch.equal(load_tensors(path)['w'],value)
    with pytest.raises(ValueError,match='budget'):load_tensors(path,max_load_bytes=1)
