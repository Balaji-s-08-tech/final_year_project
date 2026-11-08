from pathlib import Path
import pytest
import torch
from experiments.tests.test_mission11 import pair, rewrite
from experiments.store import read_json
from experiments.compare import compare

def test_phi_residual_embedding_interface_rejected(tmp_path):
    from compression.synthetic import fixture
    from VLM.quantization.awq_quantize import PackedAWQLinear,_save_output
    from VLM.quantization.awq_config import AWQConfig
    model,tok,*_=fixture();model.model.embed_tokens=torch.nn.Embedding(16,17)
    model.model.layers[0].mlp.fc1=PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1,4,8)
    with pytest.raises(ValueError,match='embedding|residual'):_save_output(tmp_path,model,tok,AWQConfig(bits=4,group_size=8),{}, {})



@pytest.mark.parametrize('damage',['head','embedding_tensor','input_norm','final_norm','norm_epsilon'])
def test_residual_interfaces_and_norms(damage):
    from compression.synthetic import fixture
    from VLM.quantization.portable_integrity import describe
    model,*_=fixture()
    if damage=='head':model.lm_head=torch.nn.Linear(15,16)
    elif damage=='embedding_tensor':model.model.embed_tokens.weight=torch.nn.Parameter(torch.ones(16,15))
    elif damage=='input_norm':model.model.layers[0].input_layernorm=torch.nn.LayerNorm(15)
    elif damage=='final_norm':model.model.final_layernorm=torch.nn.LayerNorm(15)
    else:model.model.final_layernorm.eps=0.01
    with pytest.raises(ValueError,match='dimensions|normalization'):describe(model)


def test_residual_header_rejected_before_pickle(tmp_path,monkeypatch):
    from experiments.tests.test_mission11 import exported
    from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint
    from VLM.quantization import awq_model_loader as loader
    path=exported(tmp_path);side=Path(str(path)+'.metadata.json');doc=read_json(side)
    doc['architecture']['interfaces']['embedding'][1]+=1;rewrite(side,doc)
    monkeypatch.setattr(loader,'_torch_load',lambda *a,**k:pytest.fail('Malformed header reached pickle'))
    with pytest.raises(ValueError,match='embedding|residual'):load_quantized_bunny_checkpoint(str(path),trusted=True)
