"""Bunny Phi-2 activation-aware, weight-only AWQ integration.

This is an integration layer, not a replacement for the copied AWQ core.  The
core currently targets other transformer classes and depends on an unavailable
CUDA extension, so this module provides the smallest compatible Phi-2 adapter
and a portable packed-weight fallback.  It is correct on CPU/CUDA PyTorch but
does not claim kernel-level INT4 speedups.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from VLM.quantization.awq_calibration import build_calibration_samples, collect_linear_inputs, load_calibration_text
from VLM.quantization.awq_config import AWQConfig
from VLM.quantization.awq_model_loader import load_bunny_recovered_model

LOG = logging.getLogger("bunny_awq")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return result


def _cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    parent_name, _, leaf = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, module)


def _effective_group_size(in_features: int, requested_group_size: int) -> int:
    return in_features if requested_group_size == -1 else min(requested_group_size, in_features)


def _quant_params(weight: torch.Tensor, bits: int, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each row/group onto an unsigned grid that includes zero.

    Scales/zero points have shape (out_features, ceil(in_features/group_size)).
    Retain FP32 scales through packing: FP16 would erase small nonzero groups.
    Scale resolution is floored at FP32's smallest normal value; smaller
    magnitudes may round to zero. Half-step ties follow FP32 torch.round.
    """
    if weight.ndim != 2:
        raise ValueError("Only 2-D Linear weights can be quantized.")
    if bits not in (4, 8):
        raise ValueError("Only 4-bit and 8-bit weights are supported.")
    if group_size == 0 or group_size < -1:
        raise ValueError("group_size must be positive or -1.")
    if not weight.is_floating_point() or not weight.numel():
        raise ValueError("Weights must be a non-empty floating-point tensor.")
    if not torch.isfinite(weight.float()).all():
        raise ValueError("Weights must be finite and representable in FP32.")
    out_features, in_features = weight.shape
    group_size = _effective_group_size(in_features, group_size)
    n_groups = (in_features + group_size - 1) // group_size
    qweight = torch.empty_like(weight, dtype=torch.uint8)
    scales = torch.empty((out_features, n_groups), dtype=torch.float32, device=weight.device)
    zeros = torch.empty((out_features, n_groups), dtype=torch.uint8, device=weight.device)
    max_int = (1 << bits) - 1
    for group in range(n_groups):
        start, end = group * group_size, min((group + 1) * group_size, in_features)
        values = weight[:, start:end].float()
        minimum = values.amin(dim=1).clamp_max(0)
        maximum = values.amax(dim=1).clamp_min(0)
        # Divide before subtracting to avoid overflow for large mixed-sign ranges.
        scale = (maximum / max_int - minimum / max_int).clamp_min(torch.finfo(torch.float32).tiny)
        # Zero groups need no resolution; a unit scale reconstructs zero exactly.
        scale = torch.where(maximum == minimum, torch.ones_like(scale), scale)
        zero = torch.round(-minimum / scale).clamp(0, max_int)
        qweight[:, start:end] = torch.round(values / scale[:, None] + zero[:, None]).clamp(0, max_int).to(torch.uint8)
        scales[:, group], zeros[:, group] = scale, zero.to(torch.uint8)
    return qweight, scales, zeros


def _dequantize(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    out_features, in_features = qweight.shape
    weight = torch.empty((out_features, in_features), dtype=scales.dtype, device=qweight.device)
    group_size = _effective_group_size(in_features, group_size)
    for group in range(scales.shape[1]):
        start, end = group * group_size, min((group + 1) * group_size, in_features)
        weight[:, start:end] = (qweight[:, start:end].float() - zeros[:, group, None].float()) * scales[:, group, None]
    return weight


def _pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    padded = qweight
    if padded.shape[1] % 2:
        padded = F.pad(padded, (0, 1))
    return (padded[:, 0::2] | (padded[:, 1::2] << 4)).contiguous()


def _unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    unpacked = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device)
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = packed >> 4
    return unpacked[:, :in_features]


class PackedAWQLinear(nn.Module):
    """Portable groupwise weight-only AWQ Linear layer.

    The packed representation is real INT4/INT8 storage.  ``forward``
    dequantizes one layer at a time and uses PyTorch ``F.linear``; it is a
    portability fallback, not a claim of accelerated INT4 kernels.
    """

    def __init__(self, in_features: int, out_features: int, bits: int, group_size: int, bias: torch.Tensor | None, qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = _effective_group_size(in_features, group_size)
        if bits == 4:
            self.register_buffer("packed_weight", _pack_int4(qweight))
        elif bits == 8:
            self.register_buffer("packed_weight", qweight)
        else:
            raise ValueError("PackedAWQLinear supports 4- or 8-bit weights.")
        self.register_buffer("scales", scales.to(torch.float32))
        self.register_buffer("zero_points", zeros)
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.float16))

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int, group_size: int) -> "PackedAWQLinear":
        qweight, scales, zeros = _quant_params(linear.weight.detach(), bits, group_size)
        return cls(linear.in_features, linear.out_features, bits, group_size, linear.bias, qweight, scales, zeros)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qweight = _unpack_int4(self.packed_weight, self.in_features) if self.bits == 4 else self.packed_weight
        weight = _dequantize(qweight, self.scales.to(dtype=torch.float32), self.zero_points, self.group_size).to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}, group_size={self.group_size}"


def _pseudo_quantized(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    qweight, scales, zeros = _quant_params(weight, bits, group_size)
    return _dequantize(qweight, scales, zeros, group_size).to(weight.dtype)


def _search_shared_input_scale(linears: Mapping[str, nn.Linear], inputs: Mapping[str, torch.Tensor], config: AWQConfig) -> torch.Tensor:
    """Search AWQ channel scaling for Phi's shared layer-norm output.

    For every candidate, the equivalent transformed weights are pseudo
    quantized and compared with the original linear outputs on calibration
    activations.  No model clone or activation graph is retained.
    """
    first = next(iter(linears.values()))
    width = first.in_features
    if any(linear.in_features != width for linear in linears.values()):
        raise ValueError("Phi shared-input scale requires equal Linear input widths.")
    activation = torch.cat([inputs[name].float() for name in linears], dim=0)
    act_scale = activation.abs().mean(dim=0).clamp_min(1e-5)
    weight_scale = torch.stack([linear.weight.detach().float().abs().mean(dim=0) for linear in linears.values()]).mean(dim=0).clamp_min(1e-5).cpu()
    originals = {}
    for name, linear in linears.items():
        linear_device = linear.weight.device
        originals[name] = F.linear(
            inputs[name].float().to(linear_device),
            linear.weight.detach().float(),
            linear.bias.detach().float() if linear.bias is not None else None,
        ).cpu()
    best_loss, best_scale = float("inf"), torch.ones(width)
    for index in range(config.scale_grid):
        ratio = index / config.scale_grid
        scales = act_scale.pow(ratio) / weight_scale.pow(1.0 - ratio)
        scales = scales / torch.sqrt(scales.max() * scales.min()).clamp_min(1e-5)
        loss = 0.0
        for name, linear in linears.items():
            transformed = _pseudo_quantized(linear.weight.detach().float() * scales.to(linear.weight.device)[None, :], config.bits, config.group_size)
            reconstructed = F.linear(inputs[name].float().to(transformed.device) / scales.to(transformed.device), transformed, linear.bias.detach().float().to(transformed.device) if linear.bias is not None else None)
            loss += (reconstructed.cpu() - originals[name]).square().mean().item()
        if loss < best_loss:
            best_loss, best_scale = loss, scales.cpu()
    return best_scale


def _apply_layernorm_equivalent_scale(layer: nn.Module, linears: Iterable[nn.Linear], scale: torch.Tensor) -> None:
    norm = layer.input_layernorm
    scale = scale.to(norm.weight.device, norm.weight.dtype)
    norm.weight.data.div_(scale)
    if norm.bias is not None:
        norm.bias.data.div_(scale)
    for linear in linears:
        linear.weight.data.mul_(scale[None, :].to(linear.weight.device, linear.weight.dtype))


def _find_clip_values(weight: torch.Tensor, input_features: torch.Tensor, config: AWQConfig) -> torch.Tensor:
    """Activation-weighted MSE clipping, adapted from the copied AWQ logic."""
    qgroup = _effective_group_size(weight.shape[1], config.group_size)
    n_groups = (weight.shape[1] + qgroup - 1) // qgroup
    best = torch.empty((weight.shape[0], n_groups), device=weight.device, dtype=weight.dtype)
    x = input_features.float().to(weight.device)
    for group in range(n_groups):
        start, end = group * qgroup, min((group + 1) * qgroup, weight.shape[1])
        w = weight[:, start:end].float()
        feature = x[:, start:end]
        reference = F.linear(feature, w)
        original_max = w.abs().amax(dim=1)
        best_error = torch.full_like(original_max, float("inf"))
        best_max = original_max.clone()
        for index in range(max(1, int(config.clip_grid * config.max_clip_shrink))):
            maximum = original_max * (1.0 - index / config.clip_grid)
            candidate = w.clamp(-maximum[:, None], maximum[:, None])
            recovered = F.linear(feature, _pseudo_quantized(candidate, config.bits, qgroup))
            error = (recovered - reference).square().mean(dim=0)
            select = error < best_error
            best_error[select], best_max[select] = error[select], maximum[select]
        best[:, group] = best_max.to(best.dtype)
    return best


def _apply_clip(linear: nn.Linear, clip_values: torch.Tensor, group_size: int) -> None:
    group_size = _effective_group_size(linear.in_features, group_size)
    for group in range(clip_values.shape[1]):
        start, end = group * group_size, min((group + 1) * group_size, linear.in_features)
        maximum = clip_values[:, group].to(linear.weight.device, linear.weight.dtype)
        linear.weight.data[:, start:end].clamp_(-maximum[:, None], maximum[:, None])


def _phi_blocks(model: nn.Module) -> List[nn.Module]:
    try:
        blocks = list(model.model.layers)
    except AttributeError as exc:
        raise TypeError("Expected Bunny Phi-2 layout model.model.layers.") from exc
    if not blocks:
        raise ValueError("Model has no decoder blocks.")
    required = ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.dense", "mlp.fc1", "mlp.fc2")
    for index, block in enumerate(blocks):
        for name in required:
            try:
                module = block.get_submodule(name)
            except AttributeError as exc:
                raise TypeError(f"Block {index} is not Bunny Phi-2 compatible; missing {name}.") from exc
            if name != "input_layernorm" and not isinstance(module, nn.Linear):
                raise TypeError(f"Block {index}.{name} must be nn.Linear, got {type(module).__name__}.")
    return blocks


def _block_linears(block: nn.Module) -> Dict[str, nn.Linear]:
    return {name: block.get_submodule(name) for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.dense", "mlp.fc1", "mlp.fc2")}


def quantize_bunny_awq(model: nn.Module, samples: Sequence[torch.Tensor], config: AWQConfig) -> Dict[str, object]:
    """Apply calibration, Phi-2 AWQ scaling/clipping, and packed quantization."""
    config.validate()
    device = _device(config.device)
    blocks = _phi_blocks(model)
    model.to(device).eval()
    result = {"architecture": "Bunny Phi-2", "blocks": len(blocks), "scaled_groups": 0, "clipped_linears": 0, "quantized_linears": 0}
    for index, block in enumerate(blocks):
        LOG.info("Calibrating and quantizing Phi block %d/%d", index + 1, len(blocks))
        linears = _block_linears(block)
        features = collect_linear_inputs(model, linears, samples, device, config.max_tokens_per_linear)
        shared_names = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.fc1")
        shared = {name: linears[name] for name in shared_names}
        scale = _search_shared_input_scale(shared, features, config)
        _apply_layernorm_equivalent_scale(block, shared.values(), scale)
        for name in shared_names:
            features[name] = features[name] / scale.to(features[name].dtype)[None, :]
        result["scaled_groups"] += 1
        for name, linear in linears.items():
            clip_values = _find_clip_values(linear.weight.detach(), features[name], config)
            _apply_clip(linear, clip_values, config.group_size)
            result["clipped_linears"] += 1
        for name, linear in list(linears.items()):
            _set_module(block, name, PackedAWQLinear.from_linear(linear, config.bits, config.group_size))
            result["quantized_linears"] += 1
        del features
        gc.collect()
        _cleanup_cuda()
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_output(output_dir: Path, model: nn.Module, tokenizer, config: AWQConfig, source: Mapping[str, object], summary: Mapping[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"quantization": config.to_dict(), "source": dict(source), "summary": dict(summary), "model_config": model.config.to_dict()}
    metadata_path = output_dir / "awq_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    checkpoint = output_dir / f"bunny_awq_int{config.bits}.pt"
    torch.save({"format": "bunny-awq-portable-v1", "model": model.cpu(), "tokenizer": tokenizer, "metadata": metadata}, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply AWQ weight-only INT4/INT8 to recovered structured-pruned Bunny Phi-2.")
    parser.add_argument("--model_path", help="Recovered checkpoint: a serialized structured model, recovery adapter directory, or recovery state file.")
    parser.add_argument("--base_model", help="Bunny base model path/name; required for adapter/state recovery.")
    parser.add_argument("--pruned_checkpoint", help="LLM-Pruner Bunny pytorch_model.bin containing the altered ModuleList.")
    parser.add_argument("--recovery_checkpoint", help="Optional explicit recovery state/LoRA directory; overrides --model_path when supplied.")
    parser.add_argument("--calibration_data", required=True, help="Local .txt/.json/.jsonl calibration data.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--calibration_samples", type=int, default=32)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--max_tokens_per_linear", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto. CPU is the safe default for 6 GB GPUs.")
    parser.add_argument("--pruning_ratio", type=float)
    args = parser.parse_args()
    if not args.model_path and not args.recovery_checkpoint and not args.pruned_checkpoint:
        parser.error("Provide --model_path or --pruned_checkpoint/--recovery_checkpoint.")
    config = AWQConfig(bits=args.bits, group_size=args.group_size, calibration_samples=args.calibration_samples, sequence_length=args.sequence_length, max_tokens_per_linear=args.max_tokens_per_linear, seed=args.seed, device=args.device, pruning_ratio=args.pruning_ratio)
    config.validate()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _seed_everything(config.seed)
    recovery = args.recovery_checkpoint or args.model_path
    tokenizer, model, source = load_bunny_recovered_model(base_model=args.base_model, pruned_checkpoint=args.pruned_checkpoint, recovery_checkpoint=recovery, device=config.device)
    texts = load_calibration_text(args.calibration_data)
    samples = build_calibration_samples(tokenizer, texts, config.calibration_samples, config.sequence_length, config.seed)
    summary = quantize_bunny_awq(model, samples, config)
    checkpoint = _save_output(Path(args.output_dir), model, tokenizer, config, source, summary)
    LOG.info("Saved portable AWQ checkpoint to %s", checkpoint)


if __name__ == "__main__":
    main()
