import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend.checkpoints import get_calibration_file, get_checkpoint, resolve_approved_path
from backend.config import OUTPUT_ROOT, PROJECT_ROOT
from backend.schemas import JobLog, JobRecord
from experiments.store import ExperimentStore, read_json, now
from experiments.schema import benchmark, measurement

logger = logging.getLogger("vlm_backend.jobs")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    def __init__(self, executor: Optional[Callable] = None, experiment_root: Optional[Path] = None):
        self._executor = executor or self._execute
        self._jobs: dict[str, JobRecord] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-job")
        self.store = ExperimentStore(experiment_root or OUTPUT_ROOT / "experiments", PROJECT_ROOT)
        self._runs = {}
        for run in self.store.runs():
            state = run.latest()
            if "job" not in state:
                continue
            record = JobRecord.model_validate(state["job"])
            self._runs[record.id] = run
            self._jobs[record.id] = record
            self._cancel_events[record.id] = threading.Event()
            if state["status"] in {"QUEUED", "RUNNING"}:
                pid = read_json(run.path / "experiment.json")["owner_pid"]
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    record.status = "failed"
                    record.error = "Backend owner exited; interrupted jobs are not resumed automatically"
                    record.finished_at = _now()
                    run.transition("FAILED", job=record.model_dump(mode="json"), error=record.error)
                except PermissionError:
                    pass  # An inaccessible owner is not evidence that it exited.

    def list(self) -> list[JobRecord]:
        with self._lock:
            return sorted((job.model_copy(deep=True) for job in self._jobs.values()), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Job not found.")
            return job.model_copy(deep=True)

    def submit(self, payload: dict) -> JobRecord:
        from runtime_policy import admit_device
        admit_device(payload.get("device", "cpu"))
        operation = payload["operation"]
        if operation == "inference" and payload.get("image"):
            raise ValueError("The registered AWQ checkpoints are language-only; image inference is not supported.")
        checkpoint = get_checkpoint(payload["checkpoint_id"])
        if operation == "quantize":
            if not payload.get("calibration_id"):
                raise ValueError("AWQ quantization requires a registered local calibration text file.")
            calibration = get_calibration_file(payload["calibration_id"])
            resolve_approved_path(calibration["path"])
            if payload.get("device") == "cuda" and not self._cuda_has_headroom():
                raise RuntimeError("CUDA quantization is blocked: at least 5 GiB free VRAM is required by the current resource policy.")
        if operation == "inference" and checkpoint["format"] != "portable_awq":
            raise ValueError("Text inference is enabled only for registered portable AWQ checkpoints.")
        record = JobRecord(
            id=uuid.uuid4().hex,
            operation=operation,
            status="queued",
            created_at=_now(),
            request=payload,
            logs=[],
        )
        record.logs.append(JobLog(timestamp=_now(), level="INFO", message=f"Accepted {operation} job for {checkpoint['name']}. It has not started yet."))
        with self._lock:
            active = [job for job in self._jobs.values() if job.status in {"queued", "running"}]
            if active:
                raise RuntimeError("A GPU/ML job is queued or running. Wait for it to finish before submitting another.")
            effective = {**payload, "seed": 42 if operation == "quantize" else None}
            if operation == "quantize":
                effective.update(calibration_samples=4, sequence_length=32)
            run = self.store.create("dashboard_" + operation, effective, run_id=record.id,
                model={"checkpoint": checkpoint.get("path"), "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
                       "precision": checkpoint.get("precision"), "quantization": checkpoint.get("quantization")},
                dataset={"identifier": calibration["path"], "calibration": effective} if operation == "quantize" else None,
                source_paths=["backend/jobs.py", "VLM/quantization/awq_quantize.py", "VLM/quantization/benchmark.py"])
            run.transition("QUEUED", job=record.model_dump(mode="json"))
            self._runs[record.id] = run
            self._jobs[record.id] = record
            self._cancel_events[record.id] = threading.Event()
        self._pool.submit(self._run, record.id)
        return record.model_copy(deep=True)

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Job not found.")
            if job.status not in {"queued", "running"}:
                raise ValueError(f"Job cannot be cancelled in state {job.status}.")
            event = self._cancel_events[job_id]
            event.set()
            process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                process.terminate()
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = _now()
            job.logs.append(JobLog(timestamp=_now(), level="WARNING", message="Cancellation requested."))
            run = self._runs[job_id]
            if job.status == "cancelled":
                run.transition("CANCELLED", job=job.model_dump(mode="json"))
            else:
                run.snapshot_job(job.model_dump(mode="json"))
            return job.model_copy(deep=True)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            cancel_event = self._cancel_events[job_id]
            if cancel_event.is_set():
                return
            job.status = "running"
            job.started_at = _now()
            job.logs.append(JobLog(timestamp=_now(), level="INFO", message="Worker started."))
            self._runs[job_id].transition("RUNNING", job=job.model_dump(mode="json"))
        started = time.perf_counter()
        try:
            result = self._executor(job_id, job.request, cancel_event, self._append_log, self._set_process)
            with self._lock:
                job = self._jobs[job_id]
                if cancel_event.is_set():
                    job.status = "cancelled"
                    job.logs.append(JobLog(timestamp=_now(), level="WARNING", message="Job cancelled."))
                else:
                    job.status = "completed"
                    job.progress = 100
                    job.result = result or {}
                    job.logs.append(JobLog(timestamp=_now(), level="SUCCESS", message="Job completed."))
                job.finished_at = _now()
                metrics = self._result_metrics(result or {}, time.perf_counter() - started)
                self._runs[job_id].transition(job.status.upper(), job=job.model_dump(mode="json"), metrics=metrics,
                    artifacts=(result or {}).get("output_files", []))
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "cancelled" if cancel_event.is_set() else "failed"
                job.error = None if cancel_event.is_set() else str(exc)
                job.logs.append(JobLog(timestamp=_now(), level="ERROR", message="Job cancelled." if cancel_event.is_set() else str(exc)))
                job.finished_at = _now()
                run = self._runs[job_id]
                run.log(traceback.format_exc(), "stderr.log")
                run.transition(job.status.upper(), job=job.model_dump(mode="json"), error=job.error,
                    metrics=self._result_metrics({}, time.perf_counter() - started))
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def _append_log(self, job_id: str, level: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.logs.append(JobLog(timestamp=_now(), level=level, message=message[:4000]))
                run = self._runs[job_id]
                run.log(f"{now()} {level} {message}")
                run.snapshot_job(job.model_dump(mode="json"))

    @staticmethod
    def _result_metrics(result, duration):
        metrics = benchmark()
        for filename in result.get("output_files", []):
            if Path(filename).name == "benchmark_manifest.json":
                report = read_json(filename)
                records = report.get("results", report)
                from VLM.quantization.benchmark import summarize
                metrics = summarize(records)
            elif Path(filename).suffix == ".pt" and Path(filename + ".metadata.json").is_file():
                from experiments.artifacts import inspect_artifact
                metadata = inspect_artifact(filename)
                metrics = benchmark({"compressed_model_size": measurement("compressed_model_size", Path(filename).stat().st_size, status="MEASURED", method="filesystem_stat")}, quantization=metadata["quantization"])
        metrics["metrics"]["execution_duration"] = measurement("execution_duration", duration, status="MEASURED", method="perf_counter")
        if "latency_seconds" in result:
            metrics["metrics"]["inference_latency"] = measurement("inference_latency", result["latency_seconds"], status="MEASURED", method="CPU_generate_perf_counter_no_warmup")
        return metrics

    def _set_process(self, job_id: str, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes[job_id] = process

    @staticmethod
    def _cuda_has_headroom() -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes >= 5 * 1024**3
        except Exception:
            return False

    def _execute(self, job_id: str, request: dict, cancel_event: threading.Event, log, set_process) -> dict:
        checkpoint = get_checkpoint(request["checkpoint_id"])
        path = resolve_approved_path(checkpoint["path"])
        if request["operation"] == "inference":
            return self._infer(job_id, path, request, cancel_event)
        run = self._runs[job_id]
        output_dir = run.path / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=False)
        if request["operation"] == "quantize":
            calibration = get_calibration_file(request["calibration_id"])
            calibration_path = resolve_approved_path(calibration["path"])
            command = [sys.executable, "-m", "VLM.quantization.awq_quantize", "--model_path", str(path), "--calibration_data", str(calibration_path), "--bits", str(request["bits"]), "--group_size", str(request["group_size"]), "--calibration_samples", "4", "--sequence_length", "32", "--device", request["device"], "--output_dir", str(output_dir)]
        elif request["operation"] == "benchmark":
            output_file = output_dir / "benchmark_manifest.json"
            command = [sys.executable, "-m", "VLM.quantization.benchmark", "--artifact", str(path), "--output", str(output_file)]
        else:
            raise ValueError("Unsupported job operation.")
        command += ["--experiment-dir", str(run.path)]
        # Keep raw stdout/stderr distinct and durable, while forwarding stdout
        # to structured API logs. stderr stays available even if the worker dies.
        stderr_stream = (run.path / "stderr.log").open("a")
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=stderr_stream, text=True, encoding="utf-8", errors="replace", bufsize=1)
        stderr_stream.close()
        set_process(job_id, process)
        for line in iter(process.stdout.readline, ""):
            run.log(line.rstrip(), "stdout.log")
            if cancel_event.is_set() and process.poll() is None:
                process.terminate()
            if line.strip():
                log(job_id, "INFO", line.rstrip())
        return_code = process.wait()
        with (run.path / "stderr.log").open("rb") as stream:
            os.fsync(stream.fileno())
        if cancel_event.is_set():
            raise RuntimeError("Cancelled")
        if return_code != 0:
            raise RuntimeError(f"Pipeline command failed with exit code {return_code}.")
        output_paths = [str(item) for item in output_dir.rglob("*") if item.is_file()]
        return {"output_dir": str(output_dir), "output_files": output_paths, "command": command[1:]}

    @staticmethod
    def _infer(job_id: str, path: Path, request: dict, cancel_event: threading.Event) -> dict:
        if cancel_event.is_set():
            raise RuntimeError("Cancelled")
        from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint
        import torch
        tokenizer, model, metadata = load_quantized_bunny_checkpoint(str(path), device="cpu")
        encoded = tokenizer(request["prompt"], return_tensors="pt", truncation=True, max_length=2048)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=request["max_new_tokens"], do_sample=False)
        elapsed = time.perf_counter() - started
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"text": text, "latency_seconds": elapsed, "max_new_tokens": request["max_new_tokens"], "modality": "text-only", "metadata": metadata}
