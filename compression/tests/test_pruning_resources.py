import weakref
import torch
from torch import nn

from compression import pruning
from VLM.quantization.tests.test_tiny_pipeline import TinyPhiModel, TinyPhiBlock


def test_finished_mlp_modules_are_released_before_next_layer(monkeypatch):
    model = TinyPhiModel()
    model.model.layers.append(TinyPhiBlock())
    first = model.model.layers[0].mlp
    old = [weakref.ref(first.fc1), weakref.ref(first.fc2)]
    second_fc1 = model.model.layers[1].mlp.fc1
    original_slice = pruning._slice
    checked = []

    def inspect_slice(source, rows, columns):
        if source is second_fc1:
            assert all(ref() is None for ref in old), 'Finished dense MLPs remain referenced'
            checked.append(True)
        return original_slice(source, rows, columns)

    monkeypatch.setattr(pruning, '_slice', inspect_slice)
    pruning.prune(model, pruning.PruningConfig(method='mlp_neurons', sparsity=.25))
    assert checked


def test_slicing_preserves_values_flags_and_owns_storage():
    for rows, columns in [(torch.tensor([0, 2]), torch.arange(4)),
                          (torch.arange(3), torch.tensor([1, 3])),
                          (torch.tensor([0, 2]), torch.tensor([1, 3]))]:
        source = nn.Linear(4, 3).double().eval()
        source.weight.requires_grad_(False)
        result = pruning._slice(source, rows, columns)
        assert torch.equal(result.weight, source.weight[rows][:, columns])
        assert torch.equal(result.bias, source.bias[rows])
        assert not result.training and not result.weight.requires_grad and result.bias.requires_grad
        before = source.weight.clone()
        with torch.no_grad(): result.weight.add_(1)
        assert torch.equal(source.weight, before)


def test_slicing_does_not_initialize_discarded_random_weights():
    source = nn.Linear(4, 3)
    rng = torch.random.get_rng_state().clone()
    pruning._slice(source, torch.tensor([0, 2]), torch.arange(4))
    assert torch.equal(torch.random.get_rng_state(), rng)
