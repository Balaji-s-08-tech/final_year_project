from types import SimpleNamespace
import pytest
import torch
from compression.quantization import reconstruction_error, verify_checkpoint
from VLM.quantization.awq_calibration import _bounded_append, build_calibration_samples, collect_linear_inputs
from VLM.quantization.awq_config import AWQConfig
from VLM.quantization.awq_quantize import quantize_bunny_awq, _save_output
from VLM.quantization.tests.test_tiny_pipeline import TinyPhiModel, TinyTokenizer


def test_small_activations_are_bounded_and_own_storage():
    source=torch.full((100,8),1e-12)
    saved=[]; assert _bounded_append(saved,source,2)==0
    assert saved[0].shape==(2,8) and saved[0].dtype==torch.float32
    source.zero_(); assert (saved[0]>0).all()
    assert saved[0].untyped_storage().nbytes()==2*8*4


def test_calibration_failure_restores_hooks_cache_and_modes():
    model=TinyPhiModel().train(); model.model.layers[0].mlp.eval()
    before=[m.training for m in model.modules()]
    linear=model.model.layers[0].mlp.fc1
    def fail(*args,**kwargs): raise RuntimeError('fixture failure')
    model.forward=fail
    with pytest.raises(RuntimeError): collect_linear_inputs(model,{'linear':linear},[torch.tensor([[1]])],torch.device('cpu'),4)
    assert not linear._forward_hooks and model.config.use_cache is True
    assert [m.training for m in model.modules()]==before


@pytest.mark.parametrize('bits',[4,8])
def test_export_cpu_verification_and_error_metrics(tmp_path,bits):
    model=TinyPhiModel(); tokenizer=TinyTokenizer()
    cfg=AWQConfig(bits=bits,group_size=3,scale_grid=2,clip_grid=2,max_tokens_per_linear=4)
    summary=quantize_bunny_awq(model,[torch.tensor([[1,2,3]])],cfg)
    path=_save_output(tmp_path,model,tokenizer,cfg,{'provenance':'synthetic'},summary)
    assert verify_checkpoint(path,trusted=True)['bits']==bits
    with pytest.raises(ValueError): verify_checkpoint(path)
    with pytest.raises(ValueError): verify_checkpoint(path,trusted=True,max_load_bytes=1)
    result=reconstruction_error(torch.tensor([0.,1.]),torch.tensor([0.,.9]))
    assert result['mean_absolute_error']==pytest.approx(.05)
    assert result['max_absolute_error']==pytest.approx(.1)
    assert reconstruction_error(torch.zeros(2),torch.zeros(2))['relative_l2'] is None


def test_calibration_count_and_config_types_fail_explicitly():
    with pytest.raises(ValueError): build_calibration_samples(TinyTokenizer(),['one'],2,8,42)
    with pytest.raises(ValueError): AWQConfig(bits=True).validate()
    with pytest.raises(ValueError): _bounded_append([],torch.tensor([[float('nan')]]),1)


@pytest.mark.parametrize('bias',[1e-12,1e10])
def test_packed_bias_conversion_preserves_float32_range(bias):
    from VLM.quantization.awq_quantize import PackedAWQLinear
    source=torch.nn.Linear(2,1); source.weight.data.zero_(); source.bias.data.fill_(bias)
    packed=PackedAWQLinear.from_linear(source,4,2)
    torch.testing.assert_close(packed(torch.ones(1,2)),source(torch.ones(1,2)),rtol=1e-6,atol=0)
