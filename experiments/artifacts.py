"""Validate artifact-bound JSON metadata without unpickling model objects."""
from pathlib import Path
import json
import re
from experiments.provenance import file_hash


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate artifact metadata key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite artifact metadata value: {value}")


def inspect_artifact(path):
    path = Path(path)
    sidecar = path.with_name(path.name + ".metadata.json")
    unknown = {"precision": "UNKNOWN", "format": None, "quantization": None, "verification": "UNAVAILABLE"}
    if not sidecar.is_file():
        return unknown
    if sidecar.stat().st_size > 1024 * 1024:
        raise ValueError("Artifact metadata exceeds the 1 MiB header limit")
    metadata = json.loads(sidecar.read_text(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(metadata, dict):
        raise ValueError("Artifact metadata must be an object")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1 or metadata.get("filename") != path.name:
        raise ValueError("Invalid artifact metadata identity/version")
    if type(metadata.get("size_bytes")) is not int or metadata["size_bytes"] < 0:
        raise ValueError("Artifact byte size must be a nonnegative integer")
    if not isinstance(metadata.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", metadata["sha256"]):
        raise ValueError("Invalid artifact SHA256")
    if (metadata.get("format") is not None and not isinstance(metadata["format"], str)) or metadata.get("format") not in {"dense", "portable_awq", None}:
        raise ValueError("Unsupported artifact metadata format")
    precision = metadata.get("precision")
    quantization = metadata.get("quantization")
    if not isinstance(precision, str) or precision not in {"FP16", "FP32", "BF16", "AWQ INT4", "AWQ INT8", "UNKNOWN"}:
        raise ValueError("Invalid artifact precision")
    if precision.startswith("AWQ"):
        if not isinstance(quantization, dict) or quantization.get("method") != "AWQ" or type(quantization.get("bits")) is not int or quantization["bits"] != int(precision[-1]):
            raise ValueError("Contradictory AWQ metadata")
        if metadata.get("format") != "portable_awq":
            raise ValueError("AWQ precision requires portable_awq format")
        if quantization.get("calibration") is not None and not isinstance(quantization["calibration"], dict):
            raise ValueError("Calibration metadata must be an object or null")
    elif quantization is not None:
        raise ValueError("Dense/unknown precision cannot declare quantization")
    if metadata.get("format") == "portable_awq" and not precision.startswith("AWQ"):
        raise ValueError("Portable AWQ format requires AWQ precision")
    actual = file_hash(path, max_bytes=None)
    if actual["sha256"] != metadata.get("sha256") or actual.get("size_bytes") != metadata.get("size_bytes"):
        raise ValueError("Artifact metadata does not match file content")
    return {"precision": precision, "format": metadata.get("format"), "quantization": quantization, "verification": "SHA256_VERIFIED", "sha256": actual["sha256"]}
