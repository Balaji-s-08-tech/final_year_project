"""Calibration-data and activation collection for Bunny AWQ.

This module deliberately collects only inputs for one decoder block at a time.
Keeping feature tensors on CPU bounds calibration memory on small GPUs.
"""

from __future__ import annotations
from experiments.store import read_json
from experiments.jsonio import lines as bounded_lines
from experiments.jsonio import loads as strict_json_loads

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn


def load_calibration_text(source: str) -> List[str]:
    """Load text from .txt, .json, or .jsonl without downloading a dataset.

    JSON records may use ``text``, ``content``, ``prompt``, ``instruction``, or
    ``input``.  A list of strings is also accepted.  Remote dataset names are
    intentionally not implicit: a local calibration file makes the sampled
    data and the experiment reproducible.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration data must be a local .txt/.json/.jsonl file; not found: {source}"
        )
    if path.suffix.lower() == ".txt":
        records: Iterable[object] = [line.decode("utf-8",errors="strict").strip() for line in bounded_lines(path)]
    elif path.suffix.lower() == ".jsonl":
        records = [strict_json_loads(line) for line in bounded_lines(path) if line.strip()]
    elif path.suffix.lower() == ".json":
        data = read_json(path)
        records = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError("JSON calibration data must be a list or an object containing a 'data' list.")
    else:
        raise ValueError("Calibration data must have .txt, .json, or .jsonl extension.")

    text: List[str] = []
    for record in records:
        if isinstance(record, str):
            value = record
        elif isinstance(record, Mapping):
            value = next((record[key] for key in ("text", "content", "prompt", "instruction", "input") if record.get(key)), "")
        else:
            value = ""
        if isinstance(value, str) and value.strip():
            text.append(value.strip())
    if not text:
        raise ValueError("No non-empty calibration text was found.")
    return text


def build_calibration_samples(tokenizer, texts: Sequence[str], num_samples: int, sequence_length: int, seed: int) -> List[torch.Tensor]:
    """Tokenize a deterministic subset into one bounded sequence per sample."""
    if type(num_samples) is not int or num_samples < 1 or type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("Positive integer calibration bounds are required")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    samples: List[torch.Tensor] = []
    for index in indices:
        encoded = tokenizer(texts[index], return_tensors="pt", truncation=True, max_length=sequence_length)
        input_ids = encoded["input_ids"]
        if input_ids.numel() > 0:
            samples.append(input_ids)
        if len(samples) == num_samples:
            break
    if len(samples) != num_samples:
        raise ValueError("Insufficient nonempty calibration samples; refusing a silent count change")
    return samples


def _bounded_append(storage: List[torch.Tensor], value: torch.Tensor, remaining: int) -> int:
    if remaining <= 0:
        return 0
    flat = value.detach().reshape(-1, value.shape[-1])
    kept = flat[:remaining].to(device="cpu", dtype=torch.float32).contiguous().clone()
    if not torch.isfinite(kept).all():
        raise ValueError("Nonfinite calibration activations")
    if kept.numel():
        storage.append(kept)
    return remaining - kept.shape[0]


@torch.inference_mode()
def collect_linear_inputs(
    model: nn.Module,
    linears: Mapping[str, nn.Linear],
    samples: Sequence[torch.Tensor],
    device: torch.device,
    max_tokens_per_linear: int,
) -> Dict[str, torch.Tensor]:
    """Collect bounded input activations for selected linears on the current model.

    Call this for one Phi decoder block at a time.  Hooks retain CPU FP32
    tensors only and are always removed, including when a model forward fails.
    """
    features: Dict[str, List[torch.Tensor]] = {name: [] for name in linears}
    remaining = {name: max_tokens_per_linear for name in linears}
    handles = []

    def hook(_module, args, _output, *, name: str) -> None:
        if args and torch.is_tensor(args[0]):
            remaining[name] = _bounded_append(features[name], args[0], remaining[name])

    for name, linear in linears.items():
        handles.append(linear.register_forward_hook(lambda m, a, o, n=name: hook(m, a, o, name=n)))

    old_cache = getattr(model.config, "use_cache", None)
    if old_cache is not None:
        model.config.use_cache = False
    prior_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        consumed=0
        for input_ids in samples:
            consumed+=1
            model(input_ids=input_ids.to(device), use_cache=False)
            if all(count <= 0 for count in remaining.values()):
                break
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in prior_modes:
            module.training = mode
        if old_cache is not None:
            model.config.use_cache = old_cache

    model._awq_calibration_observation = {'requested_sequences':len(samples),'forwarded_sequences':consumed,
        'observed_tokens_per_linear':{name:max_tokens_per_linear-count for name,count in remaining.items()},
        'modality':'TEXT_ONLY','cap':max_tokens_per_linear}
    result = {name: torch.cat(values, dim=0) for name, values in features.items() if values}
    missing = sorted(set(linears) - set(result))
    if missing:
        raise RuntimeError(f"Calibration did not observe linear inputs: {', '.join(missing)}")
    return result
