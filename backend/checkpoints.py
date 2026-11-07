import hashlib
import json
from pathlib import Path
from typing import Optional

from backend.config import approved_roots
from experiments.artifacts import inspect_artifact


def resolve_approved_path(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not any(path.is_relative_to(root) for root in approved_roots()):
        raise ValueError("Path is outside the approved checkpoint and calibration directories.")
    if not path.is_file():
        raise ValueError("The selected artifact must be a file.")
    return path


def _artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _identifier(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def list_checkpoints() -> list[dict]:
    supported = {".bin", ".pt", ".pth", ".safetensors"}
    records = []
    for root in approved_roots():
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in supported:
                continue
            lowered = path.name.lower()
            try:
                metadata = inspect_artifact(path)
            except (OSError, ValueError):
                metadata = {"precision": "UNKNOWN", "format": None}
            awq = metadata["format"] == "portable_awq"
            precision = metadata["precision"]
            fmt = "portable_awq" if awq else path.suffix.lower().lstrip(".")
            records.append({
                "id": _identifier(path),
                "name": path.name,
                "path": str(path),
                "format": fmt,
                "precision": precision,
                "checkpoint_sha256": metadata.get("sha256"),
                "quantization": metadata.get("quantization"),
                "size_bytes": path.stat().st_size,
                "parameters": None,
                "lineage": "AWQ output" if awq else "Serialized model checkpoint; inspect metadata before interpreting as recovered",
                "multimodal": False,
                "available": True,
            })
    return sorted(records, key=lambda record: (record["name"], record["path"]))


def get_checkpoint(checkpoint_id: str) -> dict:
    matches = [item for item in list_checkpoints() if item["id"] == checkpoint_id]
    if not matches:
        raise KeyError("Checkpoint is not registered or is outside approved directories.")
    return matches[0]


def list_calibration_files() -> list[dict]:
    records = []
    for root in approved_roots():
        for path in root.rglob("*.txt"):
            if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024:
                records.append({"id": _identifier(path), "name": path.name, "path": str(path), "size_bytes": path.stat().st_size})
    return sorted(records, key=lambda record: record["path"])


def get_calibration_file(calibration_id: str) -> dict:
    matches = [item for item in list_calibration_files() if item["id"] == calibration_id]
    if not matches:
        raise KeyError("Calibration file is not registered or is outside approved directories.")
    return matches[0]
