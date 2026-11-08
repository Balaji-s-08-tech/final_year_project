"""Exclusive atomic publication and append-only lifecycle revisions on local POSIX disks."""
import contextlib
import fcntl
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from experiments.provenance import canonical_hash, capture
from experiments.schema import benchmark, validate_benchmark

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
TRANSITIONS = {"CREATED": {"QUEUED", "RUNNING", "FAILED", "CANCELLED"}, "QUEUED": {"RUNNING", "FAILED", "CANCELLED"}, "RUNNING": TERMINAL}


def now():
    return datetime.now(timezone.utc).isoformat()


def publish(path, writer):
    """Publish a complete file without replacing any existing directory entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-replace; competing publication fails.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def write_json(path, value):
    from experiments.jsonio import validate
    validate(value)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    from experiments.jsonio import DEFAULT_LIMITS
    if len(payload)>DEFAULT_LIMITS.max_bytes:raise ValueError("Serialized JSON exceeds byte budget")
    publish(path, lambda stream: stream.write(payload))


def read_json(path, *, limits=None):
    from experiments.jsonio import read,DEFAULT_LIMITS
    return read(path,limits=DEFAULT_LIMITS if limits is None else limits)


class Run:
    def __init__(self, path):
        self.path = Path(path)

    @contextlib.contextmanager
    def lock(self):
        with (self.path / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def latest(self):
        from experiments.jsonio import files
        revisions = sorted(files(self.path, "state_*.json", recursive=False))
        if not revisions:
            raise ValueError("Incomplete experiment initialization")
        return read_json(revisions[-1])

    def _append(self, state):
        write_json(self.path / f"state_{state['revision']:06d}.json", state)

    def transition(self, status, *, job=None, error=None, metrics=None, artifacts=None):
        if metrics is not None:
            validate_benchmark(metrics)
        with self.lock():
            previous = self.latest()
            if status not in TRANSITIONS.get(previous["status"], set()):
                raise ValueError(f"Invalid transition {previous['status']} -> {status}")
            state = {**previous, "status": status, "revision": previous["revision"] + 1, "updated_at": now()}
            if job is not None:
                state["job"] = job
            if status in TERMINAL:
                state.update(metrics=metrics or benchmark(), artifacts=artifacts or [], error=error)
                # The final snapshot is authoritative. Derived files are published first;
                # a crash before the snapshot is explicit and can be recovered as FAILED.
                for name, value in [("metrics.json", state["metrics"]), ("artifacts.json", state["artifacts"]),
                                    ("failure_report.json", {"error": error, "status": status, "at": now()})]:
                    if name == "failure_report.json" and status == "COMPLETED":
                        continue
                    if not (self.path / name).exists():
                        write_json(self.path / name, value)
            self._append(state)
            return state

    def snapshot_job(self, job):
        with self.lock():
            state = self.latest()
            if state["status"] in TERMINAL:
                return
            self._append({**state, "revision": state["revision"] + 1, "updated_at": now(), "job": job})

    def log(self, message, filename="execution.log"):
        if filename not in {"execution.log", "stdout.log", "stderr.log"}:
            raise ValueError("Invalid log name")
        with self.lock():
            if self.latest()["status"] in TERMINAL:
                raise ValueError("Finalized experiment logs are immutable")
            with (self.path / filename).open("a") as stream:
                stream.write(message + "\n")
                stream.flush()
                os.fsync(stream.fileno())


class ExperimentStore:
    def __init__(self, root, project_root=None):
        self.root = Path(root)
        self.project_root = Path(project_root or Path.cwd())

    def create(self, name, config, *, model=None, dataset=None, source_paths=(), run_id=None, max_hash_bytes=8 * 1024 * 1024):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Experiment name must be a safe path component")
        run_id = run_id or uuid.uuid4().hex
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Run ID must be a UUID hex string")
        # IDs are globally unique within this store, irrespective of experiment name.
        self.root.mkdir(parents=True, exist_ok=True)
        claim = self.root / ".ids" / run_id
        publish(claim, lambda stream: stream.write(name.encode()))
        destination = self.root / name / run_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".creating-", dir=destination.parent))
        source, environment = capture(self.project_root, source_paths, max_hash_bytes)
        identity = {"schema_version": 1, "run_id": run_id, "name": name, "created_at": now(),
                    "config_sha256": canonical_hash(config), "seed": config.get("seed"), "source": source,
                    "model": {**dict.fromkeys(("identifier", "architecture", "revision", "config_sha256", "checkpoint", "checkpoint_sha256", "precision", "quantization")), **(model or {})},
                    "dataset": {**dict.fromkeys(("identifier", "revision", "fingerprint", "calibration", "evaluation")), **(dataset or {})},
                    "owner_pid": os.getpid()}
        # Fingerprint explicitly supplied small local inputs only. Remote IDs,
        # directories, and files exceeding the policy retain explicit status.
        from experiments.provenance import file_hash
        for section, key, target in [("model", "checkpoint", "checkpoint_sha256"), ("dataset", "identifier", "fingerprint")]:
            value = identity[section].get(key)
            if value and identity[section].get(target) is None:
                digest = file_hash(value, max_hash_bytes)
                identity[section][target] = digest["sha256"]
                identity[section]["hash_status"] = digest["status"]
        for filename, value in [("experiment.json", identity), ("config.json", config), ("environment.json", environment),
                                ("state_000000.json", {"run_id": run_id, "status": "CREATED", "revision": 0, "updated_at": now()})]:
            write_json(staging / filename, value)
        for filename in ("execution.log", "stdout.log", "stderr.log"):
            publish(staging / filename, lambda stream: None)
        os.rename(staging, destination)
        parent = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return Run(destination)

    def runs(self, *, max_records=1000, max_entries=10000):
        from experiments.jsonio import files
        if type(max_records) is not int or max_records < 1:
            raise ValueError("Positive history record budget required")
        paths = []
        for path in files(self.root, "experiment.json", max_entries=max_entries):
            paths.append(path)
            if len(paths) > max_records:
                raise ValueError("History exceeds record budget; select a smaller store or explicit reviewed limit")
        return [Run(path.parent) for path in sorted(paths)]
