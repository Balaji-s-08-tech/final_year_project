"""Benchmark schema v1. Units are explicit and never inferred from values."""
import math

VERSION = 1
UNITS = {
    "original_model_size": "bytes", "compressed_model_size": "bytes",
    "original_parameter_count": "parameters", "remaining_parameter_count": "parameters",
    "compression_ratio": "ratio", "quantization_error": "weight_units",
    "inference_latency": "seconds", "median_latency": "seconds", "p95_latency": "seconds",
    "throughput": "samples/second", "tokens_per_second": "tokens/second",
    "peak_gpu_memory": "bytes", "gpu_utilization": "percent", "execution_duration": "seconds",
    "baseline_accuracy": "fraction", "compressed_accuracy": "fraction",
    "accuracy_retention": "ratio", "perplexity": "dimensionless",
}
STATUSES = {"MEASURED", "UNMEASURED", "UNAVAILABLE", "FAILED"}


def measurement(name, value=None, *, status="UNMEASURED", method=None, unit=None, metadata=None):
    if name not in UNITS:
        raise ValueError(f"Unknown metric: {name}")
    unit = UNITS[name] if unit is None else unit
    if name not in UNITS or unit != UNITS[name]:
        raise ValueError(f"Invalid metric/unit: {name}/{unit}")
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError("Invalid measurement status")
    if status == "MEASURED":
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("Measured values must be finite nonnegative numbers")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("Measured values require a method")
        if name in {"baseline_accuracy", "compressed_accuracy"} and value > 1:
            raise ValueError("Accuracy is a fraction in [0, 1]")
        if name == "gpu_utilization" and value > 100:
            raise ValueError("GPU utilization is a percentage in [0, 100]")
        if unit in {"bytes", "parameters"} and not isinstance(value, int):
            raise ValueError("Counts and bytes must be integers")
    elif value is not None:
        raise ValueError("Unmeasured values must be null")
    if method is not None and not isinstance(method, str):
        raise ValueError("Measurement method must be a string or null")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Measurement metadata must be an object")
    return {"value": value, "unit": unit, "method": method, "status": status, "metadata": metadata or {}}


def benchmark(metrics=None, *, quantization=None):
    supplied = metrics or {}
    if set(supplied) - set(UNITS):
        raise ValueError("Unknown metric names")
    result = {name: measurement(name, **supplied[name]) if name in supplied else measurement(name) for name in UNITS}
    quantization = quantization or {"method": None, "bits": None, "calibration": None}
    if not isinstance(quantization, dict) or not (quantization.get("method") is None or isinstance(quantization.get("method"), str)) or quantization.get("method") not in {None, "AWQ"}:
        raise ValueError("Unsupported quantization method")
    if quantization.get("calibration") is not None and not isinstance(quantization["calibration"], dict):
        raise ValueError("Calibration configuration must be an object or null")
    if quantization.get("method") == "AWQ":
        if type(quantization.get("bits")) is not int or quantization["bits"] not in {4, 8}:
            raise ValueError("AWQ supports only INT4 and INT8")
    elif quantization.get("bits") is not None:
        raise ValueError("Bit width requires a quantization method")
    return {"schema_version": VERSION, "metrics": result, "quantization": quantization}


def validate_benchmark(document):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != VERSION:
        raise ValueError("Invalid benchmark schema/version")
    if not isinstance(document.get("metrics"), dict) or set(document["metrics"]) != set(UNITS):
        raise ValueError("Invalid benchmark metric fields")
    required = {"value", "unit", "method", "status", "metadata"}
    if any(not isinstance(metric, dict) or set(metric) != required for metric in document["metrics"].values()):
        raise ValueError("Incomplete or unknown measurement fields")
    for name, metric in document["metrics"].items():
        if metric["unit"] != UNITS[name] or not isinstance(metric["metadata"], dict):
            raise ValueError("Serialized units and metadata must be explicit and typed")
    quantization = document.get("quantization")
    if not isinstance(quantization, dict) or not {"method", "bits"}.issubset(quantization):
        raise ValueError("Invalid benchmark schema/version")
    benchmark(document["metrics"], quantization=document.get("quantization"))
    return document
