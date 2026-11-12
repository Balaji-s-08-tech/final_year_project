import torch

from compression.pruning import PruningConfig, prune
from VLM.quantization.tests.test_tiny_pipeline import TinyPhiModel


def test_global_ranking_preserves_float64_differences():
    model = TinyPhiModel().double()
    a, b = model.model.layers[0].mlp.fc1, model.model.layers[0].mlp.fc2
    with torch.no_grad():
        a.weight.fill_(1 + 1e-10)
        b.weight.fill_(1)
    prune(model, PruningConfig(sparsity=.5, distribution='global'))
    assert model.model.layers[0].mlp.fc1.weight_mask.all()
    assert not model.model.layers[0].mlp.fc2.weight_mask.any()


def test_physical_ranking_does_not_overflow_half_precision():
    model = TinyPhiModel().half()
    a, b = model.model.layers[0].mlp.fc1, model.model.layers[0].mlp.fc2
    with torch.no_grad():
        a.weight.fill_(60000)
        b.weight.fill_(60000)
        a.weight[0].fill_(40000)
        b.weight[:, 0].fill_(40000)
    report = prune(model, PruningConfig(method='mlp_neurons', sparsity=1/16))
    assert report['layers'][0]['kept_neurons'] == list(range(1, 16))
