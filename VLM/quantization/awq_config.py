"""Configuration shared by the Bunny Phi-2 AWQ integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class AWQConfig:
    """Reproducible, memory-conscious settings for one AWQ run."""

    bits: int = 4
    group_size: int = 128
    calibration_samples: int = 32
    sequence_length: int = 128
    max_tokens_per_linear: int = 512
    scale_grid: int = 20
    clip_grid: int = 20
    max_clip_shrink: float = 0.5
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float16"
    authorize_gpu: bool = False
    pruning_ratio: Optional[float] = None

    def validate(self) -> None:
        if type(self.authorize_gpu) is not bool:
            raise ValueError("authorize_gpu must be boolean")
        for field in ("bits", "group_size", "calibration_samples", "sequence_length", "max_tokens_per_linear", "scale_grid", "clip_grid", "seed"):
            if type(getattr(self, field)) is not int:
                raise ValueError(f"{field} must be an integer")
        if self.seed < 0 or self.dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("Invalid seed/dtype")
        if self.bits not in (4, 8):
            raise ValueError("Only 4-bit and 8-bit weight-only AWQ are supported.")
        if self.group_size == 0 or self.group_size < -1:
            raise ValueError("group_size must be positive or -1 (one group per row).")
        if self.calibration_samples < 1 or self.sequence_length < 1:
            raise ValueError("Calibration sample count and sequence length must be positive.")
        if self.max_tokens_per_linear < 1 or self.scale_grid < 1 or self.clip_grid < 1:
            raise ValueError("Calibration/token/grid settings must be positive.")
        if not 0.0 < self.max_clip_shrink <= 1.0:
            raise ValueError("max_clip_shrink must be in (0, 1].")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)
