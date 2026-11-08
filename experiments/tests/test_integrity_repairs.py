import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.schema import benchmark, validate_benchmark, measurement
from runtime_policy import admit_device


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_shared_device_gate_never_queries_cuda(device, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("queried CUDA before admission"))
    with pytest.raises(PermissionError): admit_device(device)
    assert admit_device("auto") == "cpu"
    assert admit_device(device, True) == device


def test_awq_load_and_backend_gates_precede_model_access(tmp_path):
    from VLM.quantization.awq_quantize import _device
    from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint
    from backend.jobs import JobManager
    for operation in [lambda: _device("cuda"), lambda: load_quantized_bunny_checkpoint("missing", "cuda"),
                      lambda: JobManager(experiment_root=tmp_path).submit({"operation": "quantize", "device": "cuda"})]:
        with pytest.raises(PermissionError): operation()


@pytest.mark.parametrize("field", ["unit", "metadata"])
def test_serialized_null_fields_fail(field):
    doc = benchmark(); doc["metrics"]["inference_latency"][field] = None
    with pytest.raises(ValueError): validate_benchmark(doc)
    validate_benchmark(benchmark())


def test_nested_measured_zero_and_history_are_detected(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import main
    from experiments.store import ExperimentStore
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "IMPORTED_RESULTS_ROOT", tmp_path / "checkpoints" / "imported_results")
    doc = benchmark({"baseline_accuracy": measurement("baseline_accuracy", 0, status="MEASURED", method="fixture")})
    client = TestClient(main.app)
    result = client.post('/api/results/import', json={"manifest": {"results": {"sample": {"precision": "UNKNOWN", "benchmark": doc}}}})
    assert result.status_code == 201 and result.json()["measured"] is True
    run = ExperimentStore(tmp_path / "checkpoints" / "experiments").create("fixture", {})
    run.transition("RUNNING"); run.transition("COMPLETED", metrics=doc)
    report = client.get('/api/benchmarks').json()
    assert report['status'] == 'measured'
    assert any(item['origin'] == 'local_experiment' for item in report['items'])


def test_supervision_after_truncation_and_nonfinite_loss():
    sys.path.insert(0, str(Path('LLM-Pruner').resolve()))
    from LLMPruner.datasets.example_samples import prepare_bunny_samples, bunny_pruning_loss
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('sampler_fixture', 'LLM-Pruner/tests/test_bunny_samples.py')
    module = module_from_spec(spec); spec.loader.exec_module(module)
    data = module.records()
    for record in data: record['labels'][:] = -100
    with pytest.raises(ValueError, match="eligible"):
        prepare_bunny_samples(module.Model(), SimpleNamespace(pad_token_id=0), data, 1, 3)
    data[0]['labels'][-1] = 2
    assert len(prepare_bunny_samples(module.Model(), SimpleNamespace(pad_token_id=0), data, 1, 3)) == 1
    class Bad(torch.nn.Module):
        def forward(self, **kwargs): return SimpleNamespace(loss=torch.tensor(float('nan')))
    with pytest.raises(ValueError, match="Nonfinite"):
        bunny_pruning_loss(Bad(), (torch.ones(1,3,4), torch.tensor([[-100,1,2]]), torch.ones(1,3)))


def test_asset_changes_are_visible(tmp_path):
    from evaluation.provenance import asset_manifest
    (tmp_path/'tokenizer.json').write_text('{}')
    config = SimpleNamespace(model_path=str(tmp_path), vision_path=None)
    first = asset_manifest(config)
    (tmp_path/'tokenizer.json').write_text('{"changed":true}')
    assert asset_manifest(config)['sha256'] != first['sha256']
