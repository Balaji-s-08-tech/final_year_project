"""Read legacy scalars and versioned measurements without manufacturing zeros."""
import math
from experiments.schema import validate_benchmark


def has_measurement(results):
    for record in results.values():
        if not isinstance(record, dict):
            continue
        if isinstance(record.get("benchmark"), dict):
            validate_benchmark(record["benchmark"])
            if any(m["status"] == "MEASURED" for m in record["benchmark"]["metrics"].values()):
                return True
        for key in {"artifact_size_bytes", "peak_gpu_memory", "generation_latency", "tokens_per_second", "task_performance", "accuracy"}:
            value = record.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                return True
    return False
