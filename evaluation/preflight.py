"""Read-only feasibility gates. Never create a CUDA context or load weights."""
from pathlib import Path
from evaluation.inspection import compatibility
from evaluation.data import load_examples
from experiments.provenance import capture


def preflight(config, project_root=None, environment=None):
    config.validate()
    report = compatibility(config)
    try:
        examples, fingerprint = load_examples(config)
        dataset = {"fingerprint": fingerprint, "sample_ids": [e["id"] for e in examples]}
    except (OSError, ValueError, TypeError) as exc:
        report["errors"].append(f"Dataset: {exc}")
        dataset = {"fingerprint": None, "sample_ids": None}
    if environment is None:
        _, environment = capture(project_root or Path.cwd())
    ram = environment.get("ram", {}).get("available_bytes")
    if config.treatment_artifact_path:
        try:
            from experiments.store import read_json
            declaration=read_json(Path(config.treatment_artifact_path)/'treatment.json')
            if declaration.get('format')=='bunny-mlp-recipe-v1':
                from compression.recipe import reconstruction_ram
                required=reconstruction_ram(config)
                report['recipe_reconstruction_required_ram_bytes']=required
                if ram is None or ram<required:report['errors'].append('Insufficient/unknown RAM for source-bound reconstruction profile')
        except (OSError,ValueError,TypeError,AttributeError) as exc:report['errors'].append(f'Recipe admission: {exc}')
    gpu = (environment.get("gpu_devices") or [None])[0]
    mcfg = report.get("model", {}).get("config") or {}
    vcfg = (report.get("vision") or {}).get("config") or {}
    vcfg = vcfg.get("vision_config", vcfg)
    kv_estimate = None
    values = [mcfg.get(key) for key in ("hidden_size", "num_attention_heads", "num_hidden_layers")]
    if all(type(v) is int and v > 0 for v in values) and values[0] % values[1] == 0:
        hidden, heads, layers = values
        kv_heads = mcfg.get("num_key_value_heads", heads)
        image_size, patch_size = vcfg.get("image_size"), vcfg.get("patch_size")
        patches = (image_size // patch_size)**2 if type(image_size) is int and type(patch_size) is int and patch_size > 0 else 0
        if type(kv_heads) is int and kv_heads > 0:
            kv_estimate = 2 * layers * config.batch_size * kv_heads * (hidden // heads) * (config.max_input_tokens + config.max_new_tokens + patches) * (4 if config.dtype == "float32" else 2)
    if config.required_free_ram_bytes is not None and (ram is None or ram < config.required_free_ram_bytes):
        report["errors"].append("Insufficient/unknown available RAM for declared budget")
    if config.device.startswith("cuda"):
        if config.required_free_vram_bytes is None or config.required_free_ram_bytes is None:
            report["errors"].append("GPU execution requires explicit RAM and VRAM budgets")
        if not environment.get("cuda_runtime"):
            report["errors"].append("Installed PyTorch does not report a CUDA runtime")
        free = gpu.get("available_memory_bytes") if gpu else None
        if free is None or (config.required_free_vram_bytes is not None and free < config.required_free_vram_bytes):
            report["errors"].append("Insufficient/unknown free VRAM for declared budget")
    report.update(schema_version=1, dataset=dataset, environment=environment, execution_authorized=False,
                  status="BLOCKED" if report["errors"] else "PREFLIGHT_ONLY", memory_feasibility="NOT_PROVEN",
                  memory_planning={"declared_required_ram_bytes": config.required_free_ram_bytes,
                                   "declared_required_vram_bytes": config.required_free_vram_bytes,
                                   "kv_cache_bytes_estimate": kv_estimate,
                                   "estimate_status": "ANALYTICAL_NOT_MEASURED", "excludes": ["weights", "vision activations", "attention workspace", "allocator overhead", "temporary model copies"]},
                  warnings=report["warnings"] + ["Passing a declared memory budget is not proof that a model fits", "No GPU allocation or inference was performed"])
    return report
