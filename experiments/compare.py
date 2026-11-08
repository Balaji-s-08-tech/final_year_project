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


def _diagnostic_compare(baseline, candidate):
    a, b = Run(baseline), Run(candidate)
    identities = [read_json(run.path / "experiment.json") for run in (a, b)]
    configs = [read_json(run.path / "config.json") for run in (a, b)]
    warnings = []
    for section in ("model", "dataset"):
        for field in (("identifier", "architecture", "revision", "config_sha256") if section == "model" else ("identifier", "revision", "fingerprint", "evaluation")):
            values = [identity[section].get(field) for identity in identities]
            if any(value is None for value in values) or values[0] != values[1]:
                warnings.append(f"{section}.{field} differs or is unavailable")
    missing_evidence=[]
    for section,field in [('model','checkpoint_sha256'),('dataset','evaluation')]:
        values=[identity[section].get(field) for identity in identities]
        if any(v is None for v in values): missing_evidence.append(f'{section}.{field}')
        elif values[0]!=values[1]: warnings.append(f'{section}.{field} differs')
    for filename in ('assets.json','effective_inference.json'):
        if not all((run.path/filename).exists() for run in (a,b)): missing_evidence.append(filename)
    for identity in identities:
        evaluation=identity['dataset'].get('evaluation') or {}
        for field in ('split','scorer','prompt_template','max_input_tokens','max_new_tokens'):
            if evaluation.get(field) is None: missing_evidence.append('dataset.evaluation.'+field)
        if not identity['source'].get('commit') or identity['source'].get('dirty') is not False:
            missing_evidence.append('clean_source_commit')
    if identities[0]['source'].get('commit')!=identities[1]['source'].get('commit'):
        warnings.append('Source implementation commit differs')
    config_diff = differences(*configs)
    for filename in ("assets.json", "effective_inference.json"):
        paths = [run.path / filename for run in (a, b)]
        if any(path.exists() for path in paths):
            if not all(path.exists() for path in paths) or read_json(paths[0]) != read_json(paths[1]):
                warnings.append(f"{filename} differs or is unavailable")
            if filename == "assets.json" and any(path.exists() and any(v.get("status") == "INCOMPLETE" for v in read_json(path).values() if isinstance(v, dict)) for path in paths):
                warnings.append("Asset identity is incomplete; large checkpoint content was not verified")
    for run in (a,b):
        effective=run.path/'effective_inference.json'
        if effective.exists() and not read_json(effective).get('generation'):
            missing_evidence.append('effective_generation_configuration')
    if config_diff:
        warnings.append("Configurations differ; assess whether these are treatment or protocol changes")
    environments = [read_json(run.path / "environment.json") for run in (a, b)]
    for key in ("os", "pytorch", "cuda_runtime", "gpu_query", "cpu", "dependencies"):
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
                       "unit": measures[0].get("unit"), "status": ("CONTEXT_DIFFERS" if warnings else "CONDITIONALLY_COMPARABLE" if missing_evidence else "COMPARABLE") if comparable else "MISSING_OR_INCOMPARABLE"}
    return {"schema_version": 1, "baseline_run": identities[0]["run_id"], "candidate_run": identities[1]["run_id"],
            "configuration_differences": config_diff, "provenance_differences": differences(identities[0]["model"], identities[1]["model"], "model") | differences(identities[0]["dataset"], identities[1]["dataset"], "dataset"),
            "warnings": warnings, "missing_evidence": sorted(set(missing_evidence)),
            "comparison_status": "NON_COMPARABLE" if warnings else "CONDITIONALLY_COMPARABLE" if missing_evidence else "COMPARABLE",
            "equivalent_protocol_verified": not warnings and not missing_evidence, "deltas": deltas}


def csv_report(report):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["baseline_run", "candidate_run", "metric", "difference", "unit", "status", "warnings"])
    for name, delta in report["deltas"].items():
        writer.writerow([report["baseline_run"], report["candidate_run"], name, delta["difference"], delta["unit"], delta["status"], "; ".join(report["warnings"] + report.get("evidence_errors", []) + [str(e) for e in report.get("exclusions", [])])])
    return stream.getvalue()


def compare(baseline, candidate, *, max_asset_hash_bytes=8*1024*1024):
    """Validate evidence and lifecycle at the public comparison boundary."""
    from experiments.evidence import assess
    if max_asset_hash_bytes is not None and (type(max_asset_hash_bytes)is not int or max_asset_hash_bytes<1):raise ValueError('Positive hash budget or explicit None required')
    evidence=[assess(p,max_asset_hash_bytes) for p in (baseline,candidate)]
    exclusions=[{'path':str(p),'reasons':e['exclusions']} for p,e in zip((baseline,candidate),evidence) if e['exclusions']]
    try: report=_diagnostic_compare(baseline,candidate)
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
        report={'schema_version':1,'baseline_run':None,'candidate_run':None,'warnings':[f'Invalid comparison record: {exc}'],
                'deltas':{},'configuration_differences':{},'provenance_differences':{},'missing_evidence':[]}
    errors=[f'{side}: {message}' for side,e in zip(('baseline','candidate'),evidence) for message in e['errors']]
    conditional=[f'{side}: {message}' for side,e in zip(('baseline','candidate'),evidence) for message in e['conditional']]
    # Preserve legacy diagnostic warnings; required evidence omissions do not become proof.
    if exclusions: verification='NON_COMPARABLE'
    elif report['warnings']: verification='NON_COMPARABLE'
    elif errors: verification='NOT_VERIFIED'
    elif conditional: verification='CONDITIONALLY_VERIFIED'
    else: verification='VERIFIED'
    report.update(verification_status=verification,evidence_errors=errors,conditional_evidence=conditional,exclusions=exclusions,
        eligible_for_measured_comparison=verification in ('VERIFIED','CONDITIONALLY_VERIFIED'),
        equivalent_protocol_verified=verification=='VERIFIED',
        comparison_status={'VERIFIED':'COMPARABLE','CONDITIONALLY_VERIFIED':'CONDITIONALLY_COMPARABLE','NOT_VERIFIED':'NOT_VERIFIED','NON_COMPARABLE':'NON_COMPARABLE'}[verification])
    for delta in report['deltas'].values():
        if exclusions: delta.update(difference=None,status='EXCLUDED_INVALID_RUN')
        elif verification not in ('VERIFIED','CONDITIONALLY_VERIFIED') and delta['status']!='MISSING_OR_INCOMPARABLE':
            delta['status']='DIAGNOSTIC_ONLY_'+verification
    return report


def compare_collection(baseline, candidates):
    """Aggregate only admitted result pairs and disclose every excluded pair."""
    accepted=[]; excluded=[]
    for candidate in candidates:
        report=compare(baseline,candidate)
        (accepted if report['eligible_for_measured_comparison'] else excluded).append({'path':str(candidate),'report':report})
    return {'comparisons':accepted,'exclusions':excluded,'included_count':len(accepted),'excluded_count':len(excluded)}
