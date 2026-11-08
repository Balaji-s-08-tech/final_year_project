import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.jobs import JobManager
from backend.main import app
from backend.checkpoints import resolve_approved_path


client = TestClient(app,base_url="http://localhost:8000",client=("127.0.0.1",50000))


def test_health_system_models_and_empty_metrics_are_real():
    health = client.get("/api/health")
    system = client.get("/api/system/info")
    models = client.get("/api/models")
    benchmarks = client.get("/api/benchmarks")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert system.status_code == 200
    data = system.json()
    assert data["torch_version"]
    assert "memory_total_bytes" in data["cuda"]
    assert data["system_ram"]["total_bytes"] > 0
    assert models.status_code == 200
    assert models.json()[0]["backbone"] == "Phi-2"
    assert benchmarks.json()["status"] in {"measured", "not_measured"}


def test_path_and_request_validation():
    assert client.get("/api/checkpoints").status_code == 200
    assert client.get("/api/calibration-data").status_code == 200
    invalid_bits = client.post("/api/jobs", json={"operation": "quantize", "checkpoint_id": "unknown", "bits": 3})
    missing_checkpoint = client.post("/api/jobs", json={"operation": "quantize", "checkpoint_id": "unknown", "calibration_id": "unknown"})
    missing_inference_prompt = client.post("/api/inference", json={"operation": "inference", "checkpoint_id": "unknown"})
    assert invalid_bits.status_code == 422
    assert missing_checkpoint.status_code == 404
    assert missing_inference_prompt.status_code == 422


def test_artifact_paths_must_stay_under_registered_roots():
    from backend.config import PROJECT_ROOT

    try:
        resolve_approved_path(str(PROJECT_ROOT / "README.md"))
    except ValueError as exc:
        assert "outside the approved" in str(exc)
    else:
        raise AssertionError("A project source file must not be accepted as a checkpoint.")


def test_unsupported_operations_are_explicit():
    response = client.post("/api/jobs", json={"operation": "recovery", "checkpoint_id": "x"})
    assert response.status_code == 422
    assert client.get("/api/pipeline/capabilities").json()["operations"]["image_inference"]["status"] == "unavailable"


def test_job_lifecycle_and_structured_logs_with_light_executor(monkeypatch):
    from backend import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "get_checkpoint", lambda _key: {"name": "unit.bin", "format": "bin", "path": "unused"})
    started = threading.Event()

    def executor(job_id, request, cancel_event, log, set_process):
        started.set()
        log(job_id, "INFO", "lightweight executor started")
        while not cancel_event.wait(0.01):
            break
        return {"verified": True}

    manager = JobManager(executor=executor)
    record = manager.submit({"operation": "benchmark", "checkpoint_id": "fixture"})
    deadline = time.monotonic() + 2
    while manager.get(record.id).status not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(0.01)
    result = manager.get(record.id)
    assert started.is_set()
    assert result.status == "completed"
    assert result.result == {"verified": True}
    assert any(item.message == "lightweight executor started" for item in result.logs)


def test_cancel_running_job_with_light_executor(monkeypatch):
    from backend import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "get_checkpoint", lambda _key: {"name": "unit.bin", "format": "bin", "path": "unused"})
    started = threading.Event()

    def executor(job_id, request, cancel_event, log, set_process):
        started.set()
        cancel_event.wait(2)
        raise RuntimeError("Cancelled")

    manager = JobManager(executor=executor)
    record = manager.submit({"operation": "benchmark", "checkpoint_id": "fixture"})
    assert started.wait(1)
    assert manager.cancel(record.id).status == "running"
    deadline = time.monotonic() + 2
    while manager.get(record.id).status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.get(record.id).status == "cancelled"


def test_http_job_submission_with_mock_executor(monkeypatch):
    from backend import jobs as jobs_module
    from backend import main as main_module

    monkeypatch.setattr(jobs_module, "get_checkpoint", lambda _key: {"name": "unit.bin", "format": "bin", "path": "unused"})

    def executor(job_id, request, cancel_event, log, set_process):
        log(job_id, "INFO", "mock executor reached through HTTP")
        return {"test_executor": True}

    manager = JobManager(executor=executor)
    monkeypatch.setattr(main_module, "jobs", manager)
    submitted = client.post("/api/jobs", json={"operation": "benchmark", "checkpoint_id": "fixture"})
    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    deadline = time.monotonic() + 2
    while manager.get(job_id).status not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(0.01)
    detail = client.get(f"/api/jobs/{job_id}")
    logs = client.get(f"/api/jobs/{job_id}/logs")
    assert detail.status_code == 200
    assert detail.json()["status"] == "completed"
    assert detail.json()["result"]["test_executor"] is True
    assert any(entry["message"] == "mock executor reached through HTTP" for entry in logs.json())


def test_results_import_preserves_metadata_and_unmeasured_values(monkeypatch, tmp_path: Path):
    from backend import main as main_module

    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "IMPORTED_RESULTS_ROOT", tmp_path / "checkpoints" / "imported_results")
    manifest = {
        "metadata": {"run_id": "controlled-fixture", "model": "tiny-test-only"},
        "results": {
            "pruned_awq_int4": {
                "precision": "AWQ INT4",
                "artifact_size_bytes": 1234,
                "task_performance": "NOT MEASURED",
                "generation_latency": "NOT MEASURED",
            }
        },
    }
    imported = client.post("/api/results/import", json={"manifest": manifest, "source_label": "fixture.json"})
    assert imported.status_code == 201
    assert imported.json()["measured"] is True
    report = main_module.benchmark_results()
    item = next(entry for entry in report["items"] if entry["origin"] == "imported")
    assert report["status"] == "measured"
    assert item["source_label"] == "fixture.json"
    assert item["metadata"] == manifest["metadata"]
    assert item["results"] == manifest["results"]
    assert item["results"]["pruned_awq_int4"]["task_performance"] == "NOT MEASURED"


def test_results_import_rejects_invalid_artifact_size():
    response = client.post("/api/results/import", json={"manifest": {"candidate": {"precision": "INT4", "artifact_size_bytes": -1}}})
    assert response.status_code == 422
