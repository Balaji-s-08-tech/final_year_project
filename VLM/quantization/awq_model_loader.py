"""Explicit loading bridge for structured-pruned and recovered Bunny models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _enable_project_imports() -> None:
    root = _project_root()
    for path in (root, root / "VLM", root / "LLM-Pruner"):
        string_path = str(path)
        if string_path not in sys.path:
            sys.path.insert(0, string_path)


def _torch_load(path: Path):
    # weights_only=False is required because the base pruner stores the altered
    # ModuleList itself.  Checkpoints must therefore be treated as trusted files.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced
        return torch.load(path, map_location="cpu")


def _load_recovery_state(model, recovery_checkpoint: Path) -> None:
    state = _torch_load(recovery_checkpoint)
    if isinstance(state, dict) and "model" in state and hasattr(state["model"], "state_dict"):
        state = state["model"].state_dict()
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("Recovery checkpoint must contain a model state dictionary or a {'model': module} payload.")
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"Recovery must cover the full structured state; missing={missing}, unexpected={unexpected}")
    # Validate the entire payload before load_state_dict can partially mutate a
    # model and only then discover a later shape mismatch/nonfinite tensor.
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or tensor.shape != expected[name].shape:
            raise ValueError(f"Recovery tensor shape/type mismatch: {name}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite recovery tensor: {name}")
        target = expected[name]
        if tensor.is_complex() != target.is_complex() or tensor.is_floating_point() != target.is_floating_point():
            raise ValueError(f"Recovery tensor numeric type mismatch: {name}")
        if tensor.is_floating_point() and tensor.numel() and tensor.abs().max() > torch.finfo(target.dtype).max:
            raise ValueError(f"Recovery tensor overflows target dtype: {name}")
    model.load_state_dict(state, strict=True)


def load_bunny_recovered_model(
    *,
    base_model: Optional[str],
    pruned_checkpoint: Optional[str],
    recovery_checkpoint: Optional[str],
    device: str = "cpu",
) -> Tuple[object, torch.nn.Module, dict]:
    """Load the *structured-pruned + recovered* Bunny Phi-2 model.

    The IJCV Bunny pruner emits ``{'model': model, 'tokenizer': tokenizer}``,
    while recovery normally emits a LoRA directory.  A LoRA directory is not a
    standalone recovered model; it needs both the base model and this pruned
    checkpoint.  This function enforces that invariant.
    """
    _enable_project_imports()
    if recovery_checkpoint is not None and not Path(recovery_checkpoint).exists():
        raise FileNotFoundError(f"Explicit recovery checkpoint is unavailable: {recovery_checkpoint}")
    selected = Path(recovery_checkpoint or pruned_checkpoint or "")
    if selected.is_file():
        payload = _torch_load(selected)
        if isinstance(payload, dict) and "model" in payload and hasattr(payload["model"], "state_dict"):
            model = payload["model"]
            tokenizer = payload.get("tokenizer")
            if tokenizer is None:
                raise ValueError("Structured checkpoint has no tokenizer. Supply a compatible checkpoint with tokenizer metadata.")
            model.to(device)
            model.eval()
            return tokenizer, model, {"source_format": "serialized_structured_model", "source": str(selected),
                                      "recovery_provenance": "UNVERIFIED", "warning": "A serialized model does not establish training/recovery lineage"}

    if not base_model or not pruned_checkpoint:
        raise ValueError(
            "Adapter/state-dict recovery requires --base_model and --pruned_checkpoint. "
            "This prevents AWQ from silently quantizing the original dense Bunny model."
        )
    prune_path = Path(pruned_checkpoint)
    if not prune_path.is_file():
        raise FileNotFoundError(f"Pruned structural checkpoint not found: {prune_path}")

    from VLM.bunny.model.builder import load_pruned_bunny_model

    lora = None
    if recovery_checkpoint and Path(recovery_checkpoint).is_dir():
        if not (Path(recovery_checkpoint) / "adapter_config.json").is_file():
            raise ValueError("Recovery directory must be a merged LoRA directory containing adapter_config.json.")
        lora = recovery_checkpoint
    tokenizer, model = load_pruned_bunny_model(base_model, str(prune_path), lora=lora, device=device, device_map=None)
    if recovery_checkpoint and Path(recovery_checkpoint).is_file():
        _load_recovery_state(model, Path(recovery_checkpoint))
    model.to(device)
    model.eval()
    return tokenizer, model, {
        "source_format": "base_plus_structural_prune" + ("_plus_lora_recovery" if lora else "") + ("_plus_state_recovery" if recovery_checkpoint and not lora else ""),
        "base_model": base_model,
        "pruned_checkpoint": str(prune_path),
        "recovery_checkpoint": recovery_checkpoint,
        "recovery_state_coverage": "COMPLETE" if recovery_checkpoint and Path(recovery_checkpoint).is_file() else None,
        "recovery_provenance": "UNVERIFIED",
    }


def load_quantized_bunny_checkpoint(path: str, device: str = "cpu"):
    """Load a portable AWQ output made by :mod:`awq_quantize`."""
    payload = _torch_load(Path(path))
    if not isinstance(payload, dict) or payload.get("format") != "bunny-awq-portable-v1":
        raise ValueError("Not a Bunny portable AWQ checkpoint.")
    model = payload["model"].to(device).eval()
    return payload["tokenizer"], model, payload.get("metadata", {})
