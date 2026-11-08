import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from experiments.artifacts import inspect_artifact
from experiments.compare import compare, csv_report
from experiments.provenance import canonical_hash, file_hash
from experiments.schema import UNITS, benchmark, measurement
from experiments.store import ExperimentStore, publish, read_json, write_json
from VLM.quantization.benchmark import benchmark_manifest


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "runs", Path(__file__).resolve().parents[2])


def test_concurrent_unique_creation_and_duplicate_protection(store):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(lambda _: store.create("awq", {"seed": 42}), range(16)))
    ids = [run.latest()["run_id"] for run in runs]
    assert len(set(ids)) == 16
    assert len(store.runs()) == 16
    with pytest.raises(FileExistsError):
        store.create("different_name", {}, run_id=ids[0])
    for run in runs:
        assert read_json(run.path / "config.json") == {"seed": 42}
        assert read_json(run.path / "experiment.json")["model"]["revision"] is None


def test_immutable_atomic_publication(tmp_path):
    target = tmp_path / "record.json"
    def failing(stream):
        stream.write(b'{"incomplete":')
        raise RuntimeError("disk failure")
    with pytest.raises(RuntimeError):
        publish(target, failing)
    assert not target.exists()
    assert not list(tmp_path.glob(".pending-*"))
    write_json(target, {"original": True})
    with pytest.raises(FileExistsError):
        write_json(target, {"original": False})
    assert read_json(target) == {"original": True}
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", {"x": float("nan")})
    assert not (tmp_path / "nan.json").exists()


@pytest.mark.parametrize("terminal", ["COMPLETED", "FAILED", "CANCELLED"])
def test_lifecycle_finalization_and_failure_preservation(store, terminal):
    run = store.create("test", {})
    initial = (run.path / "experiment.json").read_bytes()
    run.transition("QUEUED")
    with pytest.raises(ValueError):
        run.transition("COMPLETED")
    run.transition("RUNNING")
    run.log("started")
    run.log("output", "stdout.log")
    run.log("error output", "stderr.log")
    run.transition(terminal, error="intentional failure" if terminal == "FAILED" else None)
    assert (run.path / "experiment.json").read_bytes() == initial
    assert len(list(run.path.glob("state_*.json"))) == 4
    assert read_json(run.path / "metrics.json")["metrics"]["peak_gpu_memory"]["value"] is None
    if terminal != "COMPLETED":
        assert read_json(run.path / "failure_report.json")["status"] == terminal
    for operation in [lambda: run.transition("RUNNING"), lambda: run.log("late")]:
        with pytest.raises(ValueError):
            operation()
    assert (run.path / "stdout.log").read_text() == "output\n"


def test_concurrent_state_transition_has_single_winner(store):
    run = store.create("test", {})
    run.transition("RUNNING")
    def finish(_):
        try:
            run.transition("COMPLETED")
            return True
        except ValueError:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(finish, range(4))) == 1


@pytest.mark.parametrize("kwargs", [
    {"value": 0}, {"value": 1, "status": "MEASURED"},
    {"value": float("nan"), "status": "MEASURED", "method": "test"},
    {"value": True, "status": "MEASURED", "method": "test"},
    {"value": -1, "status": "MEASURED", "method": "test"},
    {"unit": "milliseconds"}, {"status": "UNKNOWN"},
])
def test_metric_validation(kwargs):
    with pytest.raises(ValueError):
        measurement("inference_latency", **kwargs)


def test_missing_and_real_zero_measurements():
    metrics = benchmark()["metrics"]
    assert all(m["value"] is None and m["status"] == "UNMEASURED" for m in metrics.values())
    assert set(metrics) == set(UNITS)
    assert measurement("compressed_accuracy", 0, status="MEASURED", method="held_out_exact_match")["value"] == 0
    for name, value in [("compressed_accuracy", 1.1), ("gpu_utilization", 101), ("compressed_model_size", 1.5)]:
        with pytest.raises(ValueError):
            measurement(name, value, status="MEASURED", method="test")


def test_provenance_and_bounded_hashing(store, tmp_path):
    run = store.create("test", {"seed": 7}, source_paths=["experiments/schema.py"])
    identity = read_json(run.path / "experiment.json")
    environment = read_json(run.path / "environment.json")
    assert len(identity["source"]["commit"]) == 40
    assert isinstance(identity["source"]["dirty"], bool)
    assert identity["source"]["files"]["experiments/schema.py"]["status"] == "HASHED"
    assert identity["config_sha256"] == canonical_hash({"seed": 7})
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert environment["python"] and environment["dependencies"]["torch"]
    assert environment["ram"]["available_bytes"] > 0
    artifact = tmp_path / "large"
    artifact.write_bytes(b"12345")
    assert file_hash(artifact, 2)["status"] == "SKIPPED_SIZE_LIMIT"
    assert file_hash(artifact, None)["status"] == "HASHED"


def bound_artifact(path, precision, bits=None):
    path.write_bytes(b"small test artifact")
    digest = file_hash(path, None)
    metadata = {"schema_version": 1, "filename": path.name, "precision": precision, "format": "portable_awq" if bits else "dense",
                "sha256": digest["sha256"], "size_bytes": digest["size_bytes"],
                "quantization": {"method": "AWQ", "bits": bits} if bits else None}
    write_json(path.with_name(path.name + ".metadata.json"), metadata)
    return metadata


@pytest.mark.parametrize("precision,bits", [("AWQ INT4", 4), ("AWQ INT8", 8), ("FP16", None), ("FP32", None), ("BF16", None)])
def test_verified_precision_ignores_argument_label(tmp_path, precision, bits):
    path = tmp_path / "misleading_int4.pt"
    bound_artifact(path, precision, bits)
    report = benchmark_manifest(None, None, None, str(path))
    assert report["pruned_awq_int4"]["precision"] == precision
    assert inspect_artifact(path)["verification"] == "SHA256_VERIFIED"


def test_missing_contradictory_and_stale_metadata(tmp_path):
    unknown = tmp_path / "int8.pt"
    unknown.write_bytes(b"unknown")
    assert inspect_artifact(unknown)["precision"] == "UNKNOWN"
    metadata = bound_artifact(tmp_path / "bad.pt", "AWQ INT4", 8)
    with pytest.raises(ValueError, match="Contradictory"):
        inspect_artifact(tmp_path / "bad.pt")
    path = tmp_path / "stale.pt"
    bound_artifact(path, "FP16")
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="content"):
        inspect_artifact(path)


def test_comparison_missing_protocol_differences_and_csv(store):
    model = {"identifier": "tiny", "architecture": "phi", "revision": "r1", "config_sha256": "test_hash"}
    dataset = {"identifier": "test", "revision": "r1", "fingerprint": "hash", "evaluation": {"split": "test"}}
    baseline = store.create("baseline", {"bits": 8}, model=model, dataset=dataset)
    candidate = store.create("candidate", {"bits": 4}, model=model, dataset=dataset)
    for run, value in [(baseline, 100), (candidate, 60)]:
        run.transition("RUNNING")
        run.transition("COMPLETED", metrics=benchmark({"compressed_model_size": measurement("compressed_model_size", value, status="MEASURED", method="stat")}))
    report = compare(baseline.path, candidate.path)
    assert report["baseline_run"] != report["candidate_run"]
    assert report["deltas"]["compressed_model_size"]["difference"] == -40
    assert report["deltas"]["inference_latency"]["difference"] is None
    assert report["configuration_differences"]["bits"] == {"baseline": 8, "candidate": 4}
    assert report["warnings"] and not report["equivalent_protocol_verified"]
    assert "compressed_model_size,-40,bytes" in csv_report(report)
    assert not compare(baseline.path, baseline.path)["warnings"]


def test_benchmark_cli_output_collision_and_durable_failure(tmp_path):
    artifact = tmp_path / "dense.pt"
    bound_artifact(artifact, "FP16")
    output = tmp_path / "manifest.json"
    command = [sys.executable, "-m", "VLM.quantization.benchmark", "--artifact", str(artifact), "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    original = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original
    bad = subprocess.run([*command[:-1], str(tmp_path / "failed.json"), "--original_fp16", str(tmp_path / "missing")], capture_output=True)
    assert bad.returncode != 0
    states = [run.latest() for run in ExperimentStore(tmp_path / "experiments").runs()]
    assert {s["status"] for s in states} == {"FAILED", "COMPLETED"}


def test_backend_restart_and_request_bits_regression(tmp_path, monkeypatch):
    from backend import jobs
    artifact = tmp_path / "model.pt"
    bound_artifact(artifact, "AWQ INT8", 8)
    monkeypatch.setattr(jobs, "get_checkpoint", lambda _: {"path": str(artifact), "name": artifact.name, "format": "portable_awq"})
    monkeypatch.setattr(jobs, "resolve_approved_path", lambda p: Path(p))
    manager = jobs.JobManager(experiment_root=tmp_path / "runs")
    record = manager.submit({"operation": "benchmark", "checkpoint_id": "fixture", "bits": 4})
    deadline = time.monotonic() + 10
    while manager.get(record.id).status in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(.01)
    result = manager.get(record.id)
    assert result.status == "completed", result.error
    manifest = next(p for p in result.result["output_files"] if Path(p).name == "benchmark_manifest.json")
    assert read_json(manifest)["results"]["artifact"]["precision"] == "AWQ INT8"
    restarted = jobs.JobManager(experiment_root=tmp_path / "runs")
    assert restarted.get(record.id).status == "completed"
    assert restarted.get(record.id).logs == result.logs
    assert (manager._runs[record.id].path / "stdout.log").stat().st_size > 0


@pytest.mark.parametrize("bits", [4, 8])
def test_awq_export_collision_and_metadata(tmp_path, bits):
    from VLM.quantization.awq_config import AWQConfig
    from VLM.quantization.awq_quantize import _save_output, PackedAWQLinear
    from VLM.quantization.tests.test_tiny_pipeline import TinyPhiModel, TinyTokenizer
    model = TinyPhiModel()
    model.model.layers[0].mlp.fc1 = PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1, bits, 3)
    path = _save_output(tmp_path, model, TinyTokenizer(), AWQConfig(bits=bits), {}, {})
    before = path.read_bytes()
    assert inspect_artifact(path)["precision"] == f"AWQ INT{bits}"
    with pytest.raises(FileExistsError):
        _save_output(tmp_path, model, TinyTokenizer(), AWQConfig(bits=bits), {}, {})
    assert path.read_bytes() == before


def test_multiprocess_run_creation_and_id_reservation(tmp_path):
    script = "from experiments.store import ExperimentStore; import sys; ExperimentStore(sys.argv[1]).create('parallel', {}, run_id='a'*32)"
    processes = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path / "runs")], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    results = [p.communicate(timeout=20) for p in processes]
    assert sorted(p.returncode for p in processes) == [0, 1, 1]
    assert len(ExperimentStore(tmp_path / "runs").runs()) == 1


def test_backend_failure_and_interrupted_owner_history(store, tmp_path, monkeypatch):
    from backend import jobs
    from backend.schemas import JobRecord
    monkeypatch.setattr(jobs, "get_checkpoint", lambda _: {"name": "fixture", "path": "unavailable", "format": "bin"})
    def fail(*args):
        raise RuntimeError("controlled executor failure")
    manager = jobs.JobManager(executor=fail, experiment_root=tmp_path / "failed")
    record = manager.submit({"operation": "benchmark", "checkpoint_id": "fixture"})
    deadline = time.monotonic() + 5
    while manager.get(record.id).status in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(.01)
    assert manager.get(record.id).status == "failed"
    assert "controlled executor failure" in (manager._runs[record.id].path / "stderr.log").read_text()
    assert jobs.JobManager(experiment_root=tmp_path / "failed").get(record.id).error == "controlled executor failure"
    run = store.create("dashboard_test", {})
    job = JobRecord(id=run.latest()["run_id"], operation="benchmark", status="running", created_at=jobs._now(), request={})
    run.transition("RUNNING", job=job.model_dump(mode="json"))
    def owner_gone(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(jobs.os, "kill", owner_gone)
    recovered = jobs.JobManager(experiment_root=store.root)
    assert recovered.get(job.id).status == "failed"
    assert run.latest()["status"] == "FAILED"


def test_cli_awq_failure_is_retained_without_model_download(tmp_path):
    command = [sys.executable, "-m", "VLM.quantization.awq_quantize", "--trust-local-checkpoint", "--model_path", str(tmp_path / "missing.pt"),
               "--calibration_data", str(tmp_path / "missing.txt"), "--output_dir", str(tmp_path / "output"), "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, timeout=20)
    assert result.returncode != 0
    runs = ExperimentStore(tmp_path / "output" / "experiments").runs()
    assert len(runs) == 1 and runs[0].latest()["status"] == "FAILED"
    assert "FileNotFoundError" in (runs[0].path / "stderr.log").read_text()
    assert "Explicit recovery checkpoint is unavailable" in runs[0].latest()["error"]


def test_documented_schemas_and_roundtrip(store):
    root = Path(__file__).resolve().parents[2]
    identity_schema = read_json(root / "docs/schemas/experiment-v1.json")
    metric_schema = read_json(root / "docs/schemas/benchmark-v1.json")
    run = store.create("schema", {"seed": 1})
    assert set(identity_schema["required"]) <= set(read_json(run.path / "experiment.json"))
    document = benchmark()
    assert set(metric_schema["properties"]["metrics"]["required"]) == set(document["metrics"])
    assert json.loads(json.dumps(document)) == document


def test_comparison_protocol_and_metric_method_mismatches(store):
    baseline = store.create("baseline", {}, dataset={"evaluation": {"split": "test"}})
    candidate = store.create("candidate", {}, dataset={"evaluation": {"split": "train"}})
    for run, method in [(baseline, "warmup_10"), (candidate, "no_warmup")]:
        run.transition("RUNNING")
        run.transition("COMPLETED", metrics=benchmark({"inference_latency": measurement("inference_latency", 1, status="MEASURED", method=method)}))
    report = compare(baseline.path, candidate.path)
    assert "dataset.evaluation differs or is unavailable" in report["warnings"]
    assert report["deltas"]["inference_latency"]["difference"] is None


def test_versioned_import_and_existing_history_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import main
    monkeypatch.setattr(main, "IMPORTED_RESULTS_ROOT", tmp_path / "imports")
    client = TestClient(main.app,base_url="http://localhost:8000",client=("127.0.0.1",50000))
    assert client.get("/api/experiments").status_code == 200
    manifest = {"results": {"candidate": {"precision": "UNKNOWN", "benchmark": benchmark()}}}
    response = client.post("/api/results/import", json={"manifest": manifest})
    assert response.status_code == 201
    assert response.json()["measured"] is False
    manifest["results"]["candidate"]["benchmark"]["metrics"]["inference_latency"]["unit"] = "ms"
    assert client.post("/api/results/import", json={"manifest": manifest}).status_code == 422


def test_invalid_quantization_schema_and_metadata():
    for quantization in [{"method": "AWQ", "bits": 3}, {"method": None, "bits": 4}, {"method": "unsupported"}]:
        with pytest.raises(ValueError):
            benchmark(quantization=quantization)
    with pytest.raises(ValueError):
        measurement("unknown_metric")
    with pytest.raises(ValueError):
        measurement("inference_latency", metadata="invalid")


def test_awq_cli_success_has_unique_outputs_and_complete_record(tmp_path):
    import torch
    from VLM.quantization.tests.test_tiny_pipeline import TinyPhiModel, TinyTokenizer
    checkpoint = tmp_path / "input.pt"
    torch.save({"model": TinyPhiModel(), "tokenizer": TinyTokenizer()}, checkpoint)
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("small CPU fixture\n")
    command = [sys.executable, "-m", "VLM.quantization.awq_quantize", "--trust-local-checkpoint", "--model_path", str(checkpoint),
               "--calibration_data", str(calibration), "--output_dir", str(tmp_path / "out"),
               "--device", "cpu", "--calibration_samples", "1", "--sequence_length", "4"]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    runs = ExperimentStore(tmp_path / "out" / "experiments").runs()
    assert len(runs) == 2
    for run in runs:
        assert run.latest()["status"] == "COMPLETED"
        assert run.latest()["metrics"]["metrics"]["compressed_model_size"]["value"] > 0
        identity = read_json(run.path / "experiment.json")
        assert identity["model"]["checkpoint_sha256"]
        assert identity["dataset"]["fingerprint"]
        assert "Saved portable AWQ" in (run.path / "stderr.log").read_text()
        artifact = next((run.path / "artifacts").glob("*.pt"))
        result = subprocess.run([sys.executable, "-c",
            "import sys, torch; from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint; "
            "_, model, metadata = load_quantized_bunny_checkpoint(sys.argv[1], trusted=True); "
            "assert torch.isfinite(model(input_ids=torch.tensor([[1,2]])).logits).all()",
            str(artifact)], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr


def test_baseline_candidate_sizes_and_ratio(tmp_path):
    from VLM.quantization.benchmark import summarize
    original, candidate = tmp_path / "original", tmp_path / "candidate"
    original.write_bytes(b"12345678")
    candidate.write_bytes(b"1234")
    report = benchmark_manifest(str(original), None, None, str(candidate))
    summary = summarize(report)["metrics"]
    assert summary["original_model_size"]["value"] == 8
    assert summary["compressed_model_size"]["value"] == 4
    assert summary["compression_ratio"]["value"] == 2
    candidate.write_bytes(b"")
    assert summarize(benchmark_manifest(str(original), None, None, str(candidate)))["metrics"]["compression_ratio"]["value"] is None


def test_export_rejects_dense_or_contradictory_model_precision(tmp_path):
    from VLM.quantization.awq_config import AWQConfig
    from VLM.quantization.awq_quantize import _save_output, PackedAWQLinear
    from VLM.quantization.tests.test_tiny_pipeline import TinyPhiModel, TinyTokenizer
    model = TinyPhiModel()
    with pytest.raises(ValueError, match="precision"):
        _save_output(tmp_path, model, TinyTokenizer(), AWQConfig(bits=4), {}, {})
    model.model.layers[0].mlp.fc1 = PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1, 8, 3)
    with pytest.raises(ValueError, match="precision"):
        _save_output(tmp_path, model, TinyTokenizer(), AWQConfig(bits=4), {}, {})
    assert not list(tmp_path.iterdir())


def test_history_excludes_incomplete_hidden_initialization(store):
    hidden = store.root / "test" / ".creating-incomplete"
    hidden.mkdir(parents=True)
    write_json(hidden / "experiment.json", {"incomplete": True})
    assert store.runs() == []


def test_documented_comparison_cli(store, tmp_path):
    runs = [store.create("comparison", {"seed": i}) for i in [1, 2]]
    for run in runs:
        run.transition("RUNNING")
        run.transition("COMPLETED")
    output, csv = tmp_path / "comparison.json", tmp_path / "comparison.csv"
    command = [sys.executable, "-m", "experiments", "compare", *[str(r.path) for r in runs], "--output", str(output), "--csv", str(csv)]
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert read_json(output)["candidate_run"] == runs[1].latest()["run_id"]
    assert "MISSING_OR_INCOMPARABLE" in csv.read_text()
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_creation_and_history_cli_preserve_provenance_configuration(tmp_path):
    config = tmp_path / "config.json"
    write_json(config, {"seed": 42, "model": {"identifier": "tiny", "revision": "r1"}, "dataset": {"identifier": "fixture", "revision": "d1"}})
    root = tmp_path / "runs"
    command = [sys.executable, "-m", "experiments", "create", "--root", str(root), "--name", "fixture", "--config", str(config)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    identity = read_json(Path(result.stdout.strip()) / "experiment.json")
    assert identity["model"]["revision"] == "r1"
    assert identity["dataset"]["revision"] == "d1"
    assert identity["seed"] == 42
    result = subprocess.run([sys.executable, "-m", "experiments", "history", "--root", str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]["status"] == "CREATED"


def test_optional_gpu_metadata_and_unavailable_git_are_explicit(tmp_path, monkeypatch):
    from experiments import provenance
    monkeypatch.setattr(provenance, "_command", lambda command, cwd=None:
        "CPU test fixture GPU, 580.178.04, 23000" if command[0] == "nvidia-smi" else None)
    source, environment = provenance.capture(tmp_path)
    assert source["commit"] is None and source["dirty"] is None
    assert environment["gpu_devices"] == [{"model": "CPU test fixture GPU", "driver_version": "580.178.04", "available_memory_bytes": 23000 * 1024**2}]
