"""Validate artifact-bound JSON metadata without unpickling model objects."""
from pathlib import Path
from experiments.provenance import file_hash
from experiments.store import read_json


def inspect_artifact(path):
    path = Path(path)
    sidecar = path.with_name(path.name + ".metadata.json")
    unknown = {"precision": "UNKNOWN", "format": None, "quantization": None, "verification": "UNAVAILABLE"}
    if not sidecar.is_file():
        return unknown
    metadata = read_json(sidecar)
    if metadata.get("schema_version") != 1 or metadata.get("filename") != path.name:
        raise ValueError("Invalid artifact metadata identity/version")
    precision = metadata.get("precision")
    quantization = metadata.get("quantization")
    if precision not in {"FP16", "FP32", "BF16", "AWQ INT4", "AWQ INT8", "UNKNOWN"}:
        raise ValueError("Invalid artifact precision")
    if precision.startswith("AWQ"):
        if not isinstance(quantization, dict) or quantization.get("method") != "AWQ" or type(quantization.get("bits")) is not int or quantization["bits"] != int(precision[-1]):
            raise ValueError("Contradictory AWQ metadata")
    elif quantization is not None:
        raise ValueError("Dense/unknown precision cannot declare quantization")
    if metadata.get("format") == "portable_awq" and not precision.startswith("AWQ"):
        raise ValueError("Portable AWQ format requires AWQ precision")
    actual = file_hash(path, max_bytes=None)
    if actual["sha256"] != metadata.get("sha256") or actual.get("size_bytes") != metadata.get("size_bytes"):
        raise ValueError("Artifact metadata does not match file content")
    return {"precision": precision, "format": metadata.get("format"), "quantization": quantization, "verification": "SHA256_VERIFIED", "sha256": actual["sha256"]}
