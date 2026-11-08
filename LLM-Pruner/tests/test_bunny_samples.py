"""CPU regression fixtures; no dataset/model downloads or pruning run."""
import sys
import random
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LLMPruner.datasets.example_samples import prepare_bunny_samples, bunny_pruning_loss


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.inputs = []
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images):
        self.inputs.append({"ids": input_ids.clone(), "mask": attention_mask.clone(), "labels": labels.clone(), "images": images})
        ids = input_ids.clone().clamp_min(0)
        return None, None, attention_mask, None, self.embedding(ids), labels


def records():
    return [{"input_ids": torch.tensor([-200, 1, i + 2]), "labels": torch.tensor([-100, 1, i + 2]),
             "image": torch.zeros(3, 2 + i, 2 + i)} for i in range(3)]


def test_unbatched_returns_all_consumer_pairs_without_mutation():
    model, data = Model(), records()
    original = [r['input_ids'].clone() for r in data]
    random.seed(17)
    output = prepare_bunny_samples(model, SimpleNamespace(pad_token_id=1), data, 3, 3)
    assert len(output) == len(model.inputs) == 3
    for embeds, labels in output:
        assert embeds.shape == (1, 3, 4) and labels.shape == (1, 3)
        assert not embeds.requires_grad
        # Returned embeddings remain usable for subsequent importance backprop.
        linear = torch.nn.Linear(4, 1)
        linear(embeds).sum().backward()
        assert linear.weight.grad is not None
    assert all(torch.equal(r['input_ids'], old) for r, old in zip(data, original))


def test_batched_variable_image_shapes_and_eos_padding():
    model, data = Model(), records()
    data[1]['input_ids'] = torch.tensor([-200, 1])
    data[1]['labels'] = torch.tensor([-100, 1])
    output = prepare_bunny_samples(model, SimpleNamespace(pad_token_id=1), data, 3, 3, batch=True, device="cpu")
    embeds, labels, mask = output
    assert embeds.shape == (3, 3, 4) and mask.sum().item() == 8
    images = model.inputs[0]['images']
    assert isinstance(images, list) and all(i.ndim == 4 and i.device.type == 'cpu' for i in images)
    ids = model.inputs[0]['ids']
    assert mask[ids == 1].sum().item() == 3  # real EOS stays attended; padding does not
    assert labels[~mask].eq(-100).all()


def test_identical_image_shapes_stack_and_seeded_subset():
    data = records()
    for r in data:
        r['image'] = torch.zeros(3, 2, 2)
    random.seed(42)
    a = Model()
    prepare_bunny_samples(a, SimpleNamespace(pad_token_id=0), data, 2, 3, batch=True)
    random.seed(42)
    b = Model()
    prepare_bunny_samples(b, SimpleNamespace(pad_token_id=0), data, 2, 3, batch=True)
    assert torch.equal(a.inputs[0]['ids'], b.inputs[0]['ids'])
    assert a.inputs[0]['images'].shape == (2, 3, 2, 2)


@pytest.mark.parametrize('count,length', [(0, 3), (4, 3), (1, 0), (True, 3)])
def test_invalid_requests_fail_before_sampling(count, length):
    with pytest.raises(ValueError):
        prepare_bunny_samples(Model(), SimpleNamespace(pad_token_id=0), records(), count, length)


def test_insufficient_eligible_samples_terminates_instead_of_looping():
    data = records()
    data[0]['image'] = None
    data[1]['input_ids'] = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError, match='eligible'):
        prepare_bunny_samples(Model(), SimpleNamespace(pad_token_id=0), data, 3, 3)


def test_text_bound_does_not_slice_batch_dimension():
    model = Model()
    with pytest.raises(ValueError, match="eligible"):
        prepare_bunny_samples(model, SimpleNamespace(pad_token_id=0), records(), 3, 1, batch=True)
    embeds, labels, mask = prepare_bunny_samples(model, SimpleNamespace(pad_token_id=0), records(), 3, 2, batch=True)
    assert embeds.shape == (3, 2, 4) and labels.shape == mask.shape == (3, 2)


def test_batched_loss_contract_propagates_labels_and_mask_once():
    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)
        def forward(self, inputs_embeds, labels, attention_mask):
            assert labels.shape == attention_mask.shape == inputs_embeds.shape[:2]
            return SimpleNamespace(loss=self.linear(inputs_embeds)[attention_mask].square().mean())
    samples = prepare_bunny_samples(Model(), SimpleNamespace(pad_token_id=0), records(), 3, 3, batch=True)
    model = LanguageModel()
    loss = bunny_pruning_loss(model, samples, "cpu")
    assert torch.isfinite(loss)
    loss.backward()
    assert model.linear.weight.grad is not None
    with pytest.raises(ValueError):
        bunny_pruning_loss(model, samples[:2])
