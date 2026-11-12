import copy
import pytest
import torch
from peft import PeftModel

from compression.synthetic import fixture
from compression.pruning import prune, PruningConfig
from compression.recovery import recover_lora, RecoveryConfig


def setup():
    model, _, samples, _, _ = fixture()
    prune(model, PruningConfig(method='mlp_neurons', sparsity=.25))
    ids = samples[0]
    return model, [{'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': ids.clone()}]


def test_pruned_lora_train_save_reload_and_merge(tmp_path):
    source, batches = setup()
    before = copy.deepcopy(source.state_dict())
    trained, report = recover_lora(source, batches, RecoveryConfig(steps=4))
    assert len(report['losses']) == 4
    assert all(torch.equal(before[k], v) for k, v in source.state_dict().items())
    assert all('lora_' in name for name, p in trained.named_parameters() if p.requires_grad)
    assert any(p.abs().sum() > 0 for name, p in trained.named_parameters() if 'lora_B' in name)
    trained.save_pretrained(tmp_path, safe_serialization=True)
    restored = PeftModel.from_pretrained(copy.deepcopy(source), tmp_path).eval()
    inputs = {k: v for k, v in batches[0].items() if k != 'labels'}
    with torch.no_grad():
        expected = trained(**inputs).logits
        torch.testing.assert_close(restored(**inputs).logits, expected)
        merged = restored.merge_and_unload(safe_merge=True)
        torch.testing.assert_close(merged(**inputs).logits, expected)


@pytest.mark.parametrize('kind', ['padding', 'no_targets', 'tokens'])
def test_recovery_rejects_invalid_supervision(kind):
    model, batches = setup()
    if kind == 'padding':
        batches[0]['attention_mask'][0, -1] = 0
    elif kind == 'no_targets':
        batches[0]['labels'].fill_(-100)
    else:
        batches[0]['input_ids'][0, 0] = 9999
    with pytest.raises(ValueError):
        recover_lora(model, batches)


def test_recovery_is_reproducible_without_changing_rng():
    model, batches = setup()
    state = torch.random.get_rng_state().clone()
    a, ra = recover_lora(model, batches)
    assert torch.equal(state, torch.random.get_rng_state())
    b, rb = recover_lora(model, batches)
    assert ra == rb
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_recovery_integrates_with_prune_awq_artifact(tmp_path):
    from compression.pipeline import run_suite, PipelineConfig, load_artifact
    model, tokenizer, samples, records, protocol = fixture()
    ids = tokenizer('answer alpha', return_tensors='pt')['input_ids']
    batches = [{'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': ids.clone()}]
    suite = run_suite(model, tokenizer, samples, records, calibration_ids=['cal1', 'cal2'],
        protocol=protocol, output_root=tmp_path,
        config=PipelineConfig(treatments=('structured_lora', 'structured_lora_awq')),
        recovery_batches=batches, recovery_ids=['train1'])
    assert suite.latest()['status'] == 'COMPLETED'
    for run in suite.latest()['artifacts']:
        payload = load_artifact(run, trusted=True)
        assert len(payload['metadata']['recovery']['losses']) == 2
        assert payload['metadata']['recovery_inputs']['ids'] == ['train1']
        with torch.no_grad():
            assert torch.isfinite(payload['model'](input_ids=ids).logits).all()


def test_recovery_suite_rejects_overlap_before_output(tmp_path):
    from compression.pipeline import run_suite, PipelineConfig
    model, tokenizer, samples, records, protocol = fixture()
    ids = tokenizer(records[0]['question'], return_tensors='pt')['input_ids']
    batches = [{'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': ids.clone()}]
    with pytest.raises(ValueError, match='overlaps held-out'):
        run_suite(model, tokenizer, samples, records, calibration_ids=['cal1', 'cal2'],
            protocol=protocol, output_root=tmp_path / 'absent',
            config=PipelineConfig(treatments=('structured_lora',)), recovery_batches=batches, recovery_ids=['train1'])
    assert not (tmp_path / 'absent').exists()


def test_recovery_rank_admitted_before_adapter_allocation():
    model, batches = setup()
    with pytest.raises(ValueError, match='rank exceeds'):
        recover_lora(model, batches, RecoveryConfig(rank=10**9))


def test_recovery_suite_admits_batch_before_hashing(tmp_path):
    from compression.pipeline import run_suite, PipelineConfig
    model, tokenizer, samples, records, protocol = fixture()
    with pytest.raises(ValueError, match='tensors required'):
        run_suite(model, tokenizer, samples, records, calibration_ids=['cal1', 'cal2'],
            protocol=protocol, output_root=tmp_path / 'absent', config=PipelineConfig(treatments=('structured_lora',)),
            recovery_batches=[{'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [1, 2]}], recovery_ids=['train1'])
    assert not (tmp_path / 'absent').exists()
