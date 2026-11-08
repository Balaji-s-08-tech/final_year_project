"""Architecture/asset reports without loading weights or invoking remote code."""
from pathlib import Path
from experiments.provenance import canonical_hash, file_hash
from experiments.store import read_json


def local_checkpoint(path):
    path = Path(path).resolve()
    result = {"path": str(path), "config": None, "config_sha256": None, "weight_files": [], "artifact_bytes": None, "errors": []}
    if not path.is_dir():
        result["errors"].append("Local checkpoint directory is unavailable")
        return result
    try:
        config = read_json(path / "config.json")
        if not isinstance(config, dict):
            raise ValueError("config.json must be an object")
        result.update(config=config, config_sha256=canonical_hash(config))
        indices = sorted(path.glob("*.index.json"))
        names = set()
        if indices:
            for index in indices:
                mapping = read_json(index).get("weight_map")
                if not isinstance(mapping, dict) or not mapping or not all(isinstance(v, str) for v in mapping.values()):
                    raise ValueError("Invalid weight shard index")
                names.update(mapping.values())
        else:
            names.update(p.name for pattern in ("*.safetensors", "pytorch_model*.bin") for p in path.glob(pattern))
        if not names:
            result["errors"].append("No local model weights found")
        total = 0
        for name in sorted(names):
            item = (path / name).resolve()
            if not item.is_relative_to(path) or not item.is_file():
                result["errors"].append(f"Missing or unsafe shard: {name}")
                continue
            result["weight_files"].append(str(item))
            total += item.stat().st_size
        result["artifact_bytes"] = total if names and not result["errors"] else None
    except (OSError, ValueError, AttributeError) as exc:
        result["errors"].append(str(exc))
    return result


def compatibility(config):
    errors, warnings = [], []
    if config.adapter == "bunny_phi" and config.treatment != "baseline":
        errors.append("Dense Bunny adapter currently validates original baseline only; compressed/recovered lineage needs a separate bridge")
    if config.adapter == "portable_awq_text":
        path = Path(config.model_path)
        errors.extend([] if path.is_file() else ["Portable checkpoint is unavailable"])
        if not config.trusted_checkpoint:
            errors.append("Portable pickle loading requires trusted_checkpoint=true")
        if config.treatment not in {"awq_int4", "awq_int8"}:
            errors.append("Portable text evaluation requires an AWQ treatment; combined pruning lineage is not verified")
        return {"adapter": config.adapter, "model": {"path": str(path), "artifact_bytes": path.stat().st_size if path.is_file() else None},
                "vision": None, "errors": errors, "warnings": ["Portable packed storage does not imply accelerated kernels", "Image evaluation is unsupported"]}
    model = local_checkpoint(config.model_path)
    vision = local_checkpoint(config.vision_path)
    errors.extend(model["errors"] + vision["errors"])
    mcfg, vcfg = model["config"] or {}, vision["config"] or {}
    if mcfg.get('quantization_config') is not None:
        errors.append('Dense baseline cannot implicitly load an external quantized backend')
    if mcfg.get("model_type") != "bunny-phi":
        errors.append("Only the repository Bunny Phi-2 (bunny-phi) adapter is supported")
    if mcfg.get("mm_projector_type", "mlp2x_gelu") != "mlp2x_gelu":
        errors.append("Only the 2-layer GELU projector protocol is supported")
    if vcfg.get("model_type") not in {"siglip", "siglip_vision_model"}:
        errors.append("Vision checkpoint must be SigLIP")
    if "sig" not in str(config.vision_path).lower():
        errors.append("Legacy vision builder requires 'sig' in the local SigLIP path")
    for filename in ("tokenizer_config.json",):
        if not (Path(config.model_path) / filename).is_file():
            errors.append(f"Missing tokenizer asset: {filename}")
    if not any((Path(config.model_path) / f).is_file() for f in ("tokenizer.json", "vocab.json", "tokenizer.model")):
        errors.append("No local tokenizer vocabulary found")
    if not (Path(config.vision_path) / "preprocessor_config.json").is_file():
        errors.append("Missing local vision preprocessor configuration")
    warnings.append("File/config inspection does not prove tensor compatibility or memory feasibility")
    return {"adapter": config.adapter, "model": model, "vision": vision, "errors": errors, "warnings": warnings}


def architecture(model):
    """Count unique parameters and logical packed weights, including frozen tensors."""
    import torch
    from VLM.quantization.awq_quantize import PackedAWQLinear
    registered = sum(p.numel() for p in model.parameters())
    packed = [module for module in model.modules() if isinstance(module, PackedAWQLinear)]
    logical = registered + sum(m.in_features * m.out_features + (m.out_features if m.bias is not None else 0) for m in packed)
    layers = []
    for index, layer in enumerate(getattr(getattr(model, "model", None), "layers", [])):
        dimensions = {}
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.dense", "mlp.fc1", "mlp.fc2"):
            try:
                linear = layer.get_submodule(name)
                dimensions[name] = {"in_features": linear.in_features, "out_features": linear.out_features, "class": type(linear).__name__}
            except AttributeError:
                dimensions[name] = None
        layers.append({"index": index, "linears": dimensions})
    return {"class": type(model).__name__, "registered_parameter_elements": registered,
            "logical_parameter_elements": logical, "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "dtypes": sorted({str(p.dtype) for p in model.parameters()}), "packed_bits": sorted({m.bits for m in packed}), "layers": layers}
