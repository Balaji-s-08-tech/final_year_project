"""Measured evaluation with explicit timing boundaries and immutable outputs."""
import math
import random
import statistics
import time
from pathlib import Path
import torch

from evaluation.adapters import Adapter, LocalAdapter
from evaluation.data import load_examples, normalize
from evaluation.preflight import preflight
from experiments.provenance import canonical_hash
from experiments.schema import benchmark, measurement
from experiments.store import ExperimentStore, write_json
from experiments.output import record_output


def nearest_rank(values, percentile):
    if not values:
        raise ValueError("No measured samples")
    return sorted(values)[max(0, math.ceil(percentile * len(values)) - 1)]


def evaluate(config, output_root, *, adapter: Adapter | None = None, authorize_gpu=False):
    from runtime_policy import admit_device
    admit_device(config.device, authorize_gpu)
    config.validate()
    if config.device.startswith("cuda") and not authorize_gpu:
        raise PermissionError("GPU execution requires separate explicit authorization; preflight is read-only")
    report = preflight(config)
    from evaluation.provenance import asset_manifest
    assets = asset_manifest(config)
    model = report.get("model", {})
    run = ExperimentStore(output_root).create("baseline_evaluation", config.to_dict(),
        model={"identifier": config.model_identifier, "architecture": config.adapter, "revision": config.model_revision,
               "config_sha256": model.get("config_sha256"), "checkpoint": config.model_path},
        dataset={"identifier": config.dataset_identifier, "revision": config.dataset_revision,
                 "fingerprint": report["dataset"]["fingerprint"], "evaluation": {"scorer": "normalized_exact_match_v1", "prompt_template": config.prompt_template,
                 "sample_ids": report["dataset"]["sample_ids"], "max_input_tokens": config.max_input_tokens, "max_new_tokens": config.max_new_tokens}},
        source_paths=["evaluation/config.py", "evaluation/runner.py", "evaluation/adapters.py", "evaluation/data.py"])
    write_json(run.path / "assets.json", assets)
    write_json(run.path / "preflight.json", report)
    run.transition("RUNNING")
    started = time.perf_counter()
    try:
        with record_output(run):
            if report["errors"]:
                raise ValueError("Preflight blocked: " + "; ".join(report["errors"]))
            examples, fingerprint = load_examples(config)
            if fingerprint != report["dataset"]["fingerprint"]:
                raise ValueError("Evaluation inputs changed after preflight")
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            previous_determinism = torch.are_deterministic_algorithms_enabled()
            previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                if config.device.startswith("cuda"):
                    torch.cuda.set_device(config.device)
                    torch.cuda.reset_peak_memory_stats(config.device)
                active = adapter or LocalAdapter()
                if isinstance(active, LocalAdapter):
                    active.authorize_gpu = authorize_gpu
                active.load(config)
                write_json(run.path / "effective_inference.json", active.effective() if hasattr(active, "effective") else {"status": "UNAVAILABLE_SYNTHETIC_ADAPTER"})
                if asset_manifest(config) != assets:
                    raise ValueError("Model/encoder assets changed during loading")
                layout = active.inspect()
                write_json(run.path / "architecture.json", layout)
                latencies, predictions, total_tokens, correct = [], [], 0, 0
                with torch.inference_mode():
                    first = active.prepare(examples[0])
                    for _ in range(config.warmup_iterations):
                        active.generate(first)
                    for index, record in enumerate(examples):
                        prepared = active.prepare(record)
                        if config.device.startswith("cuda"):
                            torch.cuda.synchronize(config.device)
                        begin = time.perf_counter()
                        output = active.generate(prepared)
                        if config.device.startswith("cuda"):
                            torch.cuda.synchronize(config.device)
                        latency = time.perf_counter() - begin
                        generation = active.decode(prepared, output)
                        if not math.isfinite(latency) or latency <= 0:
                            raise ValueError("Invalid generation timing")
                        if not generation.token_ids:
                            raise ValueError("No generated tokens; refusing a zero-token benchmark")
                        match = normalize(generation.text) in {normalize(a) for a in record["answers"]}
                        correct += int(match)
                        total_tokens += len(generation.token_ids)
                        latencies.append(latency)
                        result = {"id": record["id"], "prediction": generation.text, "references": record["answers"],
                                  "correct": match, "generated_token_ids": generation.token_ids, "generated_tokens": len(generation.token_ids),
                                  "prompt_tokens": prepared.prompt_tokens, "generation_latency_seconds": latency}
                        write_json(run.path / "predictions" / f"{index:06d}.json", result)
                        predictions.append(result)
                        run.log(f"Evaluated {record['id']}: {len(generation.token_ids)} tokens")
            finally:
                torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn)
            elapsed = time.perf_counter() - started
            method = "generate_only_perf_counter_cuda_sync" if config.device.startswith("cuda") else "generate_only_perf_counter_cpu"
            metadata = {"warmup_iterations": config.warmup_iterations, "samples": len(predictions), "batch_size": 1,
                        "preprocessing_included": False, "decoding_included": False, "eos_included_in_token_count": True,
                        "prompt_protocol_sha256": canonical_hash(config.to_dict())}
            metrics = {}
            def add(name, value, how, details=None):
                metrics[name] = measurement(name, value, status="MEASURED", method=how, metadata=details or {})
            add("inference_latency", statistics.mean(latencies), method, metadata)
            add("median_latency", statistics.median(latencies), method, metadata)
            add("p95_latency", nearest_rank(latencies, .95), "nearest_rank_p95:" + method, metadata)
            add("throughput", len(predictions) / sum(latencies), "samples/sum_generation_seconds", metadata)
            add("tokens_per_second", total_tokens / sum(latencies), "actual_tokens_including_eos/sum_generation_seconds", metadata)
            add("baseline_accuracy" if config.treatment == "baseline" else "compressed_accuracy", correct / len(predictions), "normalized_exact_match_v1", {"sample_ids": [r["id"] for r in predictions]})
            add("original_parameter_count" if config.treatment == "baseline" else "remaining_parameter_count", layout["logical_parameter_elements"], "unique_registered_parameters_plus_logical_packed_weights")
            sizes = [report.get("model", {}).get("artifact_bytes"), (report.get("vision") or {}).get("artifact_bytes")]
            if sizes[0] is not None:
                add("original_model_size" if config.treatment == "baseline" else "compressed_model_size", sum(s for s in sizes if s is not None), "model_and_vision_weight_files_stat")
            if config.device.startswith("cuda"):
                add("peak_gpu_memory", torch.cuda.max_memory_allocated(config.device), "cuda_peak_allocated_including_load_warmup_eval",
                    {"peak_reserved_bytes": torch.cuda.max_memory_reserved(config.device)})
            else:
                metrics["peak_gpu_memory"] = measurement("peak_gpu_memory", status="UNAVAILABLE", metadata={"reason": "CPU execution"})
            add("execution_duration", elapsed, "perf_counter_including_model_load_and_eval")
            quantization = None
            if layout.get("packed_bits"):
                if len(layout["packed_bits"]) != 1:
                    raise ValueError("Mixed packed bit widths are not a controlled treatment")
                quantization = {"method": "AWQ", "bits": layout["packed_bits"][0], "calibration": None}
            write_json(run.path / "evaluation_summary.json", {"samples": len(predictions), "correct": correct, "generated_tokens": total_tokens,
                "scorer": "normalized_exact_match_v1", "warnings": ["Exact match is not an official VQA scorer", "No optimization benefit is established by a single run"]})
        run.transition("COMPLETED", metrics=benchmark(metrics, quantization=quantization), artifacts=[str(p) for p in sorted(run.path.glob("predictions/*.json"))])
        return run
    except BaseException as exc:
        import traceback
        run.log(traceback.format_exc(), "stderr.log")
        run.transition("CANCELLED" if isinstance(exc, KeyboardInterrupt) else "FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
