import io
import pytest
import torch
from compression.pruning import PruningConfig, MaskedLinear, prune, materialize_masks, reconstruct_state
from VLM.quantization.tests.test_tiny_pipeline import TinyPhiModel
from evaluation.tests.test_baseline import tiny_bunny


@pytest.mark.parametrize('distribution',['layer','global'])
def test_masks_shapes_counts_gradients_and_roundtrip(distribution):
    torch.manual_seed(7)
    model = TinyPhiModel(); original = sum(p.numel() for p in model.parameters())
    report = prune(model, PruningConfig(sparsity=.5,distribution=distribution))
    assert report['parameter_elements_after'] == original
    for module in model.modules():
        if isinstance(module,MaskedLinear):
            assert module.weight[~module.weight_mask].eq(0).all()
            module(torch.ones(1,module.in_features)).sum().backward()
            assert module.weight.grad[~module.weight_mask].eq(0).all()
    inputs=torch.tensor([[1,2,3]])
    expected=model(inputs).logits
    restored = reconstruct_state(TinyPhiModel(), model.state_dict())
    torch.testing.assert_close(restored(inputs).logits, expected)
    materialize_masks(restored)
    assert not any(isinstance(m,MaskedLinear) for m in restored.modules())
    torch.testing.assert_close(restored(inputs).logits, expected)


def test_physical_mlp_reduction_and_serialization():
    model=TinyPhiModel()
    report=prune(model,PruningConfig(method='mlp_neurons',sparsity=.5))
    assert model.model.layers[0].mlp.fc1.out_features == model.model.layers[0].mlp.fc2.in_features == 8
    assert report['parameter_elements_after'] < report['parameter_elements_before']
    restored=reconstruct_state(TinyPhiModel(),model.state_dict())
    inputs=torch.tensor([[1,2]])
    torch.testing.assert_close(restored(inputs).logits,model(inputs).logits)
    buffer=io.BytesIO(); torch.save(model,buffer); buffer.seek(0)
    loaded=torch.load(buffer,weights_only=False)
    torch.testing.assert_close(loaded(inputs).logits,model(inputs).logits)


@pytest.mark.parametrize('ratio',[-.1,1.,float('nan'),True])
def test_invalid_sparsity(ratio):
    with pytest.raises(ValueError): prune(TinyPhiModel(),PruningConfig(sparsity=ratio))


def test_invalid_target_and_dependency_rejected_without_mutation():
    model=TinyPhiModel()
    with pytest.raises(ValueError): prune(model,PruningConfig(targets=('mm_projector',)))
    with pytest.raises(ValueError): prune(model,PruningConfig(method='mlp_neurons',targets=('self_attn.q_proj',)))
    assert type(model.model.layers[0].mlp.fc1) is torch.nn.Linear


def test_ties_are_deterministic_and_attention_untouched():
    a,b=TinyPhiModel(),TinyPhiModel()
    b.load_state_dict(a.state_dict())
    before=a.model.layers[0].self_attn.q_proj.weight.clone()
    for m in (a,b): prune(m,PruningConfig(sparsity=.25))
    assert torch.equal(a.model.layers[0].mlp.fc1.weight_mask,b.model.layers[0].mlp.fc1.weight_mask)
    assert torch.equal(before,a.model.layers[0].self_attn.q_proj.weight)


@pytest.mark.parametrize('method',['magnitude_mask','mlp_neurons'])
def test_real_tiny_multimodal_interface_stays_finite(tiny_bunny,method):
    from evaluation.adapters import LocalAdapter
    from evaluation.data import load_examples
    adapter=LocalAdapter(); adapter.load(tiny_bunny)
    projector={k:v.clone() for k,v in adapter.model.model.mm_projector.state_dict().items()}
    prune(adapter.model,PruningConfig(method=method,sparsity=.25))
    record=load_examples(tiny_bunny)[0][0]
    prepared=adapter.prepare(record)
    with torch.inference_mode():
        output=adapter.model(**{k:v.clone() for k,v in prepared.inputs.items()},use_cache=False)
        assert torch.isfinite(output.logits).all()
        assert adapter.decode(prepared,adapter.generate(prepared)).token_ids
    assert all(torch.equal(v,projector[k]) for k,v in adapter.model.model.mm_projector.state_dict().items())
