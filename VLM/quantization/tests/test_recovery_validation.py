"""Full-state safety checks do not establish recovery training provenance."""
import pytest
import torch
from VLM.quantization.awq_model_loader import _load_recovery_state, load_bunny_recovered_model


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape", "nonfinite", "nontensor"])
def test_invalid_state_rejected_before_any_parameter_changes(tmp_path, mutation):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {k: v.clone() + 1 for k, v in before.items()}
    key = list(state)[-1]
    if mutation == "missing":
        del state[key]
    elif mutation == "unexpected":
        state["unexpected"] = torch.ones(1)
    elif mutation == "shape":
        state[key] = torch.ones(4)
    elif mutation == "nonfinite":
        state[key] = torch.full_like(state[key], float("nan"))
    else:
        state[key] = "not a tensor"
    path = tmp_path / "recovery.pt"
    torch.save(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        _load_recovery_state(model, path, trusted=True)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_complete_state_loads_without_inventing_training_evidence(tmp_path):
    model = torch.nn.Linear(2, 3)
    state = {k: torch.full_like(v, .125) for k, v in model.state_dict().items()}
    path = tmp_path / "complete.pt"
    torch.save(state, path)
    _load_recovery_state(model, path, trusted=True)
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())


def test_missing_explicit_recovery_does_not_fall_back_to_pruned_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="Explicit recovery"):
        load_bunny_recovered_model(base_model="unused", pruned_checkpoint=str(tmp_path / "prune.pt"),
                                   recovery_checkpoint=str(tmp_path / "missing.pt"), trusted=True)


def test_raw_serialized_model_is_marked_unverified(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model": torch.nn.Linear(2, 3), "tokenizer": "fixture-tokenizer"}, path)
    _, _, source = load_bunny_recovered_model(base_model=None, pruned_checkpoint=str(path), recovery_checkpoint=None, trusted=True)
    assert source["recovery_provenance"] == "UNVERIFIED"


def test_finite_state_cannot_silently_overflow_a_half_model(tmp_path):
    model = torch.nn.Linear(2, 3).half()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "overflow.pt"
    torch.save({k: torch.full(v.shape, 1e10) for k, v in before.items()}, path)
    with pytest.raises(ValueError, match="overflows"):
        _load_recovery_state(model, path, trusted=True)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
