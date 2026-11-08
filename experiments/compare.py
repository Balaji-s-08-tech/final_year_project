"""Comparisons report protocol differences rather than asserting equivalence."""
import csv
import io
from experiments.store import Run, read_json


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(left.keys() | right.keys()):
            result.update(differences(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else key))
        return result
    return {} if left == right else {prefix: {"baseline": left, "candidate": right}}


def compare(baseline, candidate):
    a, b = Run(baseline), Run(candidate)
    identities = [read_json(run.path / "experiment.json") for run in (a, b)]
    configs = [read_json(run.path / "config.json") for run in (a, b)]
    warnings = []
    for section in ("model", "dataset"):
        for field in (("identifier", "architecture", "revision", "config_sha256") if section == "model" else ("identifier", "revision", "fingerprint", "evaluation")):
            values = [identity[section].get(field) for identity in identities]
            if any(value is None for value in values) or values[0] != values[1]:
                warnings.append(f"{section}.{field} differs or is unavailable")
    config_diff = differences(*configs)
    for filename in ("assets.json", "effective_inference.json"):
        paths = [run.path / filename for run in (a, b)]
        if any(path.exists() for path in paths):
            if not all(path.exists() for path in paths) or read_json(paths[0]) != read_json(paths[1]):
                warnings.append(f"{filename} differs or is unavailable")
            if filename == "assets.json" and any(path.exists() and any(v.get("status") == "INCOMPLETE" for v in read_json(path).values() if isinstance(v, dict)) for path in paths):
                warnings.append("Asset identity is incomplete; large checkpoint content was not verified")
    if config_diff:
        warnings.append("Configurations differ; assess whether these are treatment or protocol changes")
    environments = [read_json(run.path / "environment.json") for run in (a, b)]
    for key in ("os", "pytorch", "cuda_runtime", "gpu_query", "cpu"):
        if environments[0].get(key) != environments[1].get(key):
            warnings.append(f"Execution environment {key} differs")
    states = [run.latest() for run in (a, b)]
    deltas = {}
    keys = ("compressed_accuracy", "inference_latency", "peak_gpu_memory", "compressed_model_size")
    for key in keys:
        measures = [state.get("metrics", {}).get("metrics", {}).get(key, {}) for state in states]
        fallback = {"compressed_accuracy": "baseline_accuracy", "compressed_model_size": "original_model_size"}.get(key)
        if fallback:
            measures = [m if m.get("status") == "MEASURED" else state.get("metrics", {}).get("metrics", {}).get(fallback, {}) for m, state in zip(measures, states)]
        comparable = all(state["status"] == "COMPLETED" for state in states) and all(m.get("status") == "MEASURED" for m in measures)
        comparable = comparable and measures[0].get("unit") == measures[1].get("unit") and measures[0].get("method") == measures[1].get("method")
        deltas[key] = {"difference": measures[1]["value"] - measures[0]["value"] if comparable else None,
                       "unit": measures[0].get("unit"), "status": ("CONTEXT_DIFFERS" if warnings else "COMPARABLE") if comparable else "MISSING_OR_INCOMPARABLE"}
    return {"schema_version": 1, "baseline_run": identities[0]["run_id"], "candidate_run": identities[1]["run_id"],
            "configuration_differences": config_diff, "provenance_differences": differences(identities[0]["model"], identities[1]["model"], "model") | differences(identities[0]["dataset"], identities[1]["dataset"], "dataset"),
            "warnings": warnings, "equivalent_protocol_verified": not warnings, "deltas": deltas}


def csv_report(report):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["baseline_run", "candidate_run", "metric", "difference", "unit", "status", "warnings"])
    for name, delta in report["deltas"].items():
        writer.writerow([report["baseline_run"], report["candidate_run"], name, delta["difference"], delta["unit"], delta["status"], "; ".join(report["warnings"])])
    return stream.getvalue()
