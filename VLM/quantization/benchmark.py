"""Transparent benchmark manifest for pruning/quantization comparisons.

It never manufactures task accuracy.  Size is measured from supplied artifacts;
latency and performance remain explicitly pending unless a separate evaluator is
connected to the relevant Bunny benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import time
import copy
from pathlib import Path
from typing import Dict, Optional
from experiments.artifacts import inspect_artifact
from experiments.schema import benchmark, measurement
from experiments.store import ExperimentStore, Run, publish, write_json


def artifact_size_bytes(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    item = Path(path)
    if not item.exists():
        raise FileNotFoundError(item)
    return sum(file.stat().st_size for file in item.rglob("*") if file.is_file()) if item.is_dir() else item.stat().st_size


def benchmark_manifest(original_fp16: Optional[str], pruned_fp16: Optional[str], pruned_int8: Optional[str], pruned_awq_int4: Optional[str]) -> Dict[str, Dict[str, object]]:
    entries = (("original_fp16", original_fp16), ("pruned_fp16", pruned_fp16), ("pruned_int8", pruned_int8), ("pruned_awq_int4", pruned_awq_int4))
    results = {}
    for name, path in entries:
        metadata = inspect_artifact(path) if path and Path(path).is_file() else {"precision": "UNKNOWN", "quantization": None}
        size = artifact_size_bytes(path)
        size_metric = "original_model_size" if name == "original_fp16" else "compressed_model_size"
        metrics = benchmark({size_metric: measurement(size_metric, size, status="MEASURED" if size is not None else "UNMEASURED", method="filesystem_stat" if size is not None else None)}, quantization=metadata.get("quantization"))
        results[name] = {"precision": metadata["precision"], "artifact_size_bytes": size, "peak_gpu_memory": "NOT MEASURED", "generation_latency": "NOT MEASURED", "tokens_per_second": "NOT MEASURED", "task_performance": "NOT MEASURED", "benchmark": metrics}
    return results


def summarize(report):
    """Use the last supplied role as candidate; retain baseline size if supplied."""
    supplied = [r for r in report.values() if r.get("artifact_size_bytes") is not None]
    result = copy.deepcopy(supplied[-1]["benchmark"]) if supplied else benchmark()
    original = report.get("original_fp16", {}).get("artifact_size_bytes")
    if original is not None:
        result["metrics"]["original_model_size"] = measurement("original_model_size", original, status="MEASURED", method="filesystem_stat")
        compressed = result["metrics"]["compressed_model_size"]["value"]
        if compressed:
            result["metrics"]["compression_ratio"] = measurement("compression_ratio", original / compressed, status="MEASURED", method="original_artifact_bytes/compressed_artifact_bytes")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-fabricated pruning/quantization benchmark manifest.")
    parser.add_argument("--original_fp16")
    parser.add_argument("--pruned_fp16")
    parser.add_argument("--pruned_int8")
    parser.add_argument("--pruned_awq_int4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", help="Artifact to benchmark; precision comes from its metadata")
    parser.add_argument("--experiment-dir", help="Existing RUNNING parent experiment")
    args = parser.parse_args()
    output = Path(args.output)
    # Reserve the requested output name for the entire command; never replace it.
    # The lock is retained on failure as evidence and prevents ambiguous retries.
    if output.exists():
        raise FileExistsError(output)
    publish(output.with_name(output.name + ".reservation"), lambda stream: None)
    run = Run(args.experiment_dir) if args.experiment_dir else ExperimentStore(output.parent / "experiments").create(
        "benchmark", vars(args), model={"checkpoint": args.artifact or args.pruned_awq_int4 or args.pruned_int8 or args.pruned_fp16 or args.original_fp16},
        source_paths=["VLM/quantization/benchmark.py"])
    if not args.experiment_dir:
        run.transition("RUNNING")
    elif run.latest()["status"] != "RUNNING":
        raise ValueError("Parent experiment must be RUNNING")
    started = time.perf_counter()
    try:
        from experiments.output import record_output
        with record_output(run) if not args.experiment_dir else nullcontext():
            report = benchmark_manifest(args.original_fp16, args.pruned_fp16, args.pruned_int8, args.artifact or args.pruned_awq_int4)
            if args.artifact:
                report["artifact"] = report.pop("pruned_awq_int4")
            payload = {"schema_version": 1, "metadata": {"run_id": run.latest()["run_id"]}, "results": report}
            write_json(output, payload)
            run.log(f"Benchmark manifest: {output}")
            print(json.dumps(payload, indent=2))
            if not args.experiment_dir:
                primary = summarize(report)
                primary["metrics"]["execution_duration"] = measurement("execution_duration", time.perf_counter() - started, status="MEASURED", method="perf_counter")
                run.transition("COMPLETED", metrics=primary, artifacts=[str(output)])
    except BaseException as exc:
        if not args.experiment_dir:
            import traceback
            run.log(traceback.format_exc(), "stderr.log")
            run.transition("CANCELLED" if isinstance(exc, KeyboardInterrupt) else "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
