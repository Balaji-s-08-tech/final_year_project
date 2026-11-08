"""Bounded local provenance collection; never reads environment/secret values."""
import hashlib
import csv
import io
import importlib.metadata
import importlib
import json
import os
import platform
import subprocess
from pathlib import Path


def canonical_hash(value):
    from experiments.jsonio import validate
    validate(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path, max_bytes=8 * 1024 * 1024):
    path = Path(path)
    if not path.is_file():
        return {"sha256": None, "status": "UNAVAILABLE"}
    if max_bytes is not None and path.stat().st_size > max_bytes:
        return {"sha256": None, "status": "SKIPPED_SIZE_LIMIT", "size_bytes": path.stat().st_size}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "status": "HASHED", "size_bytes": path.stat().st_size}


def _command(command, cwd=None):
    try:
        return subprocess.check_output(command, cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture(root, source_paths=(), max_hash_bytes=8 * 1024 * 1024):
    root = Path(root)
    status = _command(["git", "status", "--porcelain"], root)
    source = {"commit": _command(["git", "rev-parse", "HEAD"], root),
              "branch": _command(["git", "branch", "--show-current"], root),
              "dirty": None if status is None else bool(status),
              "files": {str(p): file_hash(root / p, max_hash_bytes) for p in source_paths}}
    dependencies = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    import psutil
    ram = psutil.virtual_memory()
    # Query only; do not create a CUDA context, load a model, or run GPU kernels.
    gpu = _command(["nvidia-smi", "--query-gpu=name,driver_version,memory.free", "--format=csv,noheader,nounits"])
    devices = []
    for row in csv.reader(io.StringIO(gpu or "")):
        if len(row) == 3:
            try:
                free_bytes = int(float(row[2].strip()) * 1024**2)
            except ValueError:
                free_bytes = None
            devices.append({"model": row[0].strip(), "driver_version": row[1].strip(), "available_memory_bytes": free_bytes})
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        torch = None
    environment = {"python": platform.python_version(), "os": platform.platform(),
                   "pytorch": dependencies.get("torch"), "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
                   "gpu_query": gpu, "gpu_query_unit": "MiB for memory.free", "gpu_status": "AVAILABLE" if gpu else "UNAVAILABLE",
                   "gpu_devices": devices or None,
                   "cpu": {"model": platform.processor() or None, "architecture": platform.machine(), "count": os.cpu_count()},
                   "ram": {"total_bytes": ram.total, "available_bytes": ram.available}, "dependencies": dependencies}
    return source, environment
