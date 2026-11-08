import logging
import math
import os
import platform
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.checkpoints import get_checkpoint, list_calibration_files, list_checkpoints
from backend.config import FRONTEND_ORIGIN, OUTPUT_ROOT, PROJECT_ROOT
from backend.jobs import JobManager
from backend.schemas import ImportResultsRequest, JobRequest, JobSubmission
from experiments.store import ExperimentStore, write_json
from experiments.schema import validate_benchmark

logging.basicConfig(level=os.environ.get("VLM_LOG_LEVEL", "INFO"))
app = FastAPI(title="VLM Compression Research API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[FRONTEND_ORIGIN], allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
from backend.admission import RequestAdmission
app.add_middleware(RequestAdmission)
jobs = JobManager()
IMPORTED_RESULTS_ROOT = OUTPUT_ROOT / "imported_results"


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "vlm-compression-api", "version": app.version}


@app.get("/api/system/info")
@app.get("/api/system")
def system_info():
    ram = psutil.virtual_memory()
    cuda = {"available": False, "name": None, "memory_total_bytes": None, "memory_allocated_bytes": None, "memory_reserved_bytes": None, "memory_free_bytes": None, "version": None}
    try:
        import torch
        cuda["version"] = torch.version.cuda
        from experiments.provenance import capture
        _, observed = capture(PROJECT_ROOT)
        cuda["available"] = bool(observed.get("gpu_devices") and torch.version.cuda)
        if observed.get("gpu_devices"):
            device = observed["gpu_devices"][0]
            cuda.update(name=device["model"], memory_free_bytes=device["available_memory_bytes"])
    except Exception as exc:
        cuda["error"] = str(exc)
    return {"python_version": platform.python_version(), "python_executable": sys.executable, "platform": platform.platform(), "torch_version": _torch_version(), "cuda": cuda, "system_ram": {"total_bytes": ram.total, "available_bytes": ram.available, "used_bytes": ram.used, "percent_used": ram.percent}}


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


@app.get("/api/models")
def models():
    checkpoints = list_checkpoints()
    return [{"id": "bunny-v1_0-3b-phi2", "name": "Bunny-v1.0-3B", "family": "Bunny LVLM", "backbone": "Phi-2", "architecture_importable": _bunny_importable(), "registered_checkpoints": checkpoints, "capabilities": {"pruning": {"available": False, "status": "not exposed by dashboard job runner; historical scripts are quarantined; use the CPU-validated compression.pruning library"}, "recovery": {"available": False, "status": "requires explicit trained checkpoint; no cloud worker configured"}, "awq_quantization": {"available": True, "precisions": [4, 8], "engine": "portable PyTorch dequantization fallback"}, "text_inference": {"available": any(item["format"] == "portable_awq" for item in checkpoints)}, "image_inference": {"available": False, "reason": "Registered portable AWQ checkpoints contain the Phi-2 language backbone, not the full Bunny vision encoder/projector."}}}]


def _bunny_importable():
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "VLM"))
        from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
        return BunnyPhiForCausalLM.__name__ == "BunnyPhiForCausalLM"
    except Exception:
        logging.exception("Bunny architecture import check failed")
        return False


@app.get("/api/checkpoints")
def checkpoints():
    return list_checkpoints()


@app.get("/api/calibration-data")
def calibration_data():
    return list_calibration_files()


@app.get("/api/pipeline/status")
def pipeline_status():
    all_jobs = jobs.list()
    active = next((job for job in all_jobs if job.status in {"queued", "running"}), None)
    return {"status": "idle" if active is None else active.status, "current_job": None if active is None else active.model_dump(mode="json"), "jobs": len(all_jobs)}


@app.get("/api/jobs")
def list_jobs():
    return [job.model_dump(mode="json") for job in jobs.list()]


@app.post("/api/jobs", response_model=JobSubmission, status_code=202)
def create_job(request: JobRequest):
    try:
        record = jobs.submit(request.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JobSubmission(job_id=record.id, status=record.status, message="Job accepted and queued. Poll its job endpoint for actual progress.")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return jobs.get(job_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    try:
        return [log.model_dump(mode="json") for log in jobs.get(job_id).logs]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return jobs.cancel(job_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/inference")
def create_inference(request: JobRequest):
    if request.operation != "inference":
        raise HTTPException(status_code=422, detail="operation must be inference")
    return create_job(request)


@app.get("/api/benchmarks")
def benchmark_results():
    from experiments.store import read_json
    manifests = []
    invalid=[]
    for path in (PROJECT_ROOT / "checkpoints").rglob("benchmark_manifest.json"):
        try:
            payload = read_json(path)
            if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
                results = payload["results"]
                metadata = payload.get("metadata", {})
                origin = payload.get("origin", "generated")
                imported_at = payload.get("imported_at")
                source_label = payload.get("source_label")
            else:
                results = payload
                metadata = {}
                origin = "generated"
                imported_at = None
                source_label = None
            if not isinstance(results, dict):
                continue
            manifests.append({"path": str(path), "origin": origin, "imported_at": imported_at, "source_label": source_label, "metadata": metadata, "results": results})
        except (OSError, ValueError):
            continue
    from experiments.store import read_json
    for run in ExperimentStore(PROJECT_ROOT / "checkpoints" / "experiments", PROJECT_ROOT).runs():
        try:
            state = run.latest()
            if state["status"] == "COMPLETED" and "metrics" in state:
                manifests.append({"path": str(run.path), "origin": "local_experiment", "imported_at": None,
                    "source_label": state["run_id"], "metadata": read_json(run.path / "experiment.json"),
                    "results": {"experiment": {"precision": "UNKNOWN", "benchmark": state["metrics"]}}})
        except (OSError,ValueError,TypeError,KeyError) as exc:
            invalid.append({'path':str(run.path),'error':str(exc)})
    valid=[]; measured=False
    for item in manifests:
        try:
            measured = _has_numeric_measurement(item['results']) or measured
            valid.append(item)
        except (ValueError,TypeError,KeyError) as exc:
            invalid.append({'path':item['path'],'error':str(exc)})
    manifests=valid
    return {"status": "measured" if measured else "not_measured", "items": manifests, "invalid_records": invalid}


def _has_numeric_measurement(results: dict) -> bool:
    from experiments.results import has_measurement
    return has_measurement(results)


@app.post("/api/results/import", status_code=201)
def import_results(request: ImportResultsRequest):
    from experiments.jsonio import validate
    try: validate(request.manifest)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    manifest = request.manifest
    if isinstance(manifest.get("results"), dict):
        results = manifest["results"]
        metadata = manifest.get("metadata", {})
    else:
        results = manifest
        metadata = {}
    if not isinstance(results, dict) or not results:
        raise HTTPException(status_code=422, detail="Manifest must contain a non-empty results mapping.")
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=422, detail="Manifest metadata must be an object.")
    for name, record in results.items():
        if not isinstance(name, str) or not isinstance(record, dict) or not isinstance(record.get("precision"), str):
            raise HTTPException(status_code=422, detail="Each result must be an object with a precision string.")
        size = record.get("artifact_size_bytes")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise HTTPException(status_code=422, detail=f"{name}.artifact_size_bytes must be a non-negative integer or null.")
        for field, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise HTTPException(status_code=422, detail=f"{name}.{field} must be finite.")
        if "benchmark" in record:
            try:
                validate_benchmark(record["benchmark"])
            except (ValueError, TypeError, KeyError) as exc:
                raise HTTPException(status_code=422, detail="Invalid versioned benchmark schema") from exc
    imported_at = datetime.now(timezone.utc).isoformat()
    payload = {"schema_version": 1, "origin": "imported", "imported_at": imported_at, "source_label": request.source_label, "metadata": metadata, "results": results}
    try:
        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Manifest must contain finite JSON values only.") from exc
    run = ExperimentStore(IMPORTED_RESULTS_ROOT, PROJECT_ROOT).create("import", {"source_label": request.source_label, "manifest": manifest})
    import_id = run.latest()["run_id"]
    target_dir = run.path
    run.transition("RUNNING")
    target_path = target_dir / "benchmark_manifest.json"
    write_json(target_path, payload)
    run.log("Imported external results; supplied measurements/provenance are not independently verified")
    run.transition("COMPLETED", artifacts=[str(target_path)])
    return {"import_id": import_id, "path": str(target_path), "origin": "imported", "imported_at": imported_at, "measured": _has_numeric_measurement(results)}


@app.get("/api/experiments")
def experiment_history():
    from experiments.store import read_json
    rows=[]
    for run in ExperimentStore(OUTPUT_ROOT/'experiments',PROJECT_ROOT).runs():
        try:
            identity=read_json(run.path/'experiment.json');state=run.latest()
            if not isinstance(identity,dict) or not isinstance(state,dict):raise ValueError('Invalid experiment/history object')
            rows.append({'path':str(run.path),'experiment':identity,'state':state})
        except (OSError,ValueError,TypeError,KeyError) as exc:
            rows.append({'path':str(run.path),'experiment':None,'state':{'status':'INVALID','error':str(exc)}})
    return rows


@app.get("/api/pipeline/capabilities")
def pipeline_capabilities():
    return {"operations": {"pruning": {"status": "not_exposed", "command": None, "reason": "Historical scripts are quarantined; compression.pruning is a CPU-validated library, not an API worker."}, "recovery": {"status": "unavailable", "reason": "No recovery worker is configured; full recovery training is not started by this API."}, "quantize": {"status": "available", "bits": [4, 8]}, "benchmark": {"status": "available", "metrics": ["artifact size"], "other_metrics": "NOT MEASURED"}, "text_inference": {"status": "available_for_portable_awq_checkpoint", "device": "CPU"}, "image_inference": {"status": "unavailable", "reason": "The portable quantized checkpoint is language-only."}}}
