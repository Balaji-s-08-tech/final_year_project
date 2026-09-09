"""Transparent benchmark manifest for pruning/quantization comparisons.

It never manufactures task accuracy.  Size is measured from supplied artifacts;
latency and performance remain explicitly pending unless a separate evaluator is
connected to the relevant Bunny benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def artifact_size_bytes(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    item = Path(path)
    if not item.exists():
        raise FileNotFoundError(item)
    return sum(file.stat().st_size for file in item.rglob("*") if file.is_file()) if item.is_dir() else item.stat().st_size


def benchmark_manifest(original_fp16: Optional[str], pruned_fp16: Optional[str], pruned_int8: Optional[str], pruned_awq_int4: Optional[str]) -> Dict[str, Dict[str, object]]:
    entries = (("original_fp16", original_fp16, "FP16"), ("pruned_fp16", pruned_fp16, "FP16"), ("pruned_int8", pruned_int8, "INT8"), ("pruned_awq_int4", pruned_awq_int4, "AWQ INT4"))
    return {name: {"precision": precision, "artifact_size_bytes": artifact_size_bytes(path), "peak_gpu_memory": "NOT IMPLEMENTED / PENDING", "generation_latency": "NOT IMPLEMENTED / PENDING", "tokens_per_second": "NOT IMPLEMENTED / PENDING", "task_performance": "NOT IMPLEMENTED / PENDING"} for name, path, precision in entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-fabricated pruning/quantization benchmark manifest.")
    parser.add_argument("--original_fp16")
    parser.add_argument("--pruned_fp16")
    parser.add_argument("--pruned_int8")
    parser.add_argument("--pruned_awq_int4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = benchmark_manifest(args.original_fp16, args.pruned_fp16, args.pruned_int8, args.pruned_awq_int4)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
