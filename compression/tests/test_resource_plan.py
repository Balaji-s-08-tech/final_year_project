import pytest
import torch
from torch import nn

from compression.resource_plan import estimate
from compression.synthetic import fixture
from compression.pruning import prune, PruningConfig
from VLM.quantization.awq_quantize import PackedAWQLinear


def test_storage_estimates_match_actual_phi_and_packed_tensors():
    model=fixture()[0]
    count=sum(p.numel() for p in model.parameters())
    plan=estimate(16,32,1,count,group_size=8)
    assert plan['dense_fp32_model_bytes']==sum(t.numel()*t.element_size() for t in model.state_dict().values())
    prune(model,PruningConfig(method='mlp_neurons',sparsity=.25))
    assert plan['pruned_fp32_model_bytes']==sum(t.numel()*t.element_size() for t in model.state_dict().values())
    block=model.model.layers[0]
    for name,module in list(block.named_modules()):
        if isinstance(module,nn.Linear):
            parent,_,leaf=name.rpartition('.')
            setattr(block.get_submodule(parent),leaf,PackedAWQLinear.from_linear(module,4,8))
    actual=sum(t.numel()*t.element_size() for t in block.state_dict().values())
    assert plan['awq_pruned_block_int4_payload_bytes']==actual


@pytest.mark.parametrize('changes',[{'hidden':True},{'parameters':1},{'sparsity':float('nan')},{'group_size':0}])
def test_invalid_dimensions_rejected(changes):
    args=dict(hidden=16,intermediate=32,layers=1,parameters=10000)
    args.update(changes)
    with pytest.raises(ValueError):estimate(**args)
