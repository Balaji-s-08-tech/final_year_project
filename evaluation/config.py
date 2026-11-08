"""Strict configuration: paths resolve relative to the configuration file."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from string import Formatter


@dataclass(frozen=True)
class EvaluationConfig:
    model_path: str
    model_identifier: str
    model_revision: str
    dataset_path: str
    dataset_identifier: str
    dataset_revision: str
    prompt_template: str
    adapter: str = "bunny_phi"
    vision_path: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42
    max_samples: int = 32
    max_input_tokens: int = 512
    max_new_tokens: int = 32
    warmup_iterations: int = 2
    batch_size: int = 1
    treatment: str = "baseline"
    trusted_checkpoint: bool = False
    required_free_ram_bytes: int | None = None
    required_free_vram_bytes: int | None = None
    schema_version: int = 1
    dataset_split: str | None = None
    hash_large_assets: bool = False
    model_manifest_path: str | None = None
    vision_manifest_path: str | None = None

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported evaluation schema version")
        for key in ("model_path", "model_identifier", "model_revision", "dataset_path", "dataset_identifier", "dataset_revision", "prompt_template"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be specified explicitly")
        for key in ("model_identifier", "model_revision", "dataset_identifier", "dataset_revision"):
            if getattr(self, key).startswith("REPLACE_"):
                raise ValueError(f"Replace the template placeholder for {key} before inspection/execution")
        if self.dataset_split is not None and (not isinstance(self.dataset_split, str) or not self.dataset_split.strip() or self.dataset_split.startswith('REPLACE_')):
            raise ValueError('Invalid or placeholder dataset_split')
        if self.device.startswith('cuda') and self.dataset_split is None:
            raise ValueError('GPU evaluation requires an explicit held-out split')
        if type(self.hash_large_assets) is not bool:
            raise ValueError('hash_large_assets must be boolean')
        if self.adapter not in {"bunny_phi", "portable_awq_text"}:
            raise ValueError("Unsupported adapter; architecture support is explicit")
        if self.device not in {"cpu", "cuda:0"} or self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported device/dtype")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("The controlled CPU protocol uses float32 inputs")
        if self.adapter == "bunny_phi" and not self.vision_path:
            raise ValueError("Bunny multimodal evaluation requires local SigLIP assets")
        if self.adapter == "portable_awq_text" and (self.vision_path or "<image>" in self.prompt_template):
            raise ValueError("Portable AWQ text adapter cannot evaluate images")
        if self.adapter == "bunny_phi" and self.prompt_template.count("<image>") != 1:
            raise ValueError("Multimodal template must contain exactly one <image> token")
        if [name for _, name, _, _ in Formatter().parse(self.prompt_template) if name is not None] != ["question"]:
            raise ValueError("Template must contain exactly one {question} field")
        for key in ("seed", "max_samples", "max_input_tokens", "max_new_tokens", "warmup_iterations", "batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (0 if key in {"seed", "warmup_iterations"} else 1):
                raise ValueError(f"Invalid {key}")
        if self.batch_size != 1:
            raise ValueError("Version 1 supports batch_size=1; batch sweeps are not yet implemented")
        if self.treatment not in {"baseline", "structured_pruning", "unstructured_pruning", "awq_int4", "awq_int8", "pruning_awq"}:
            raise ValueError("Unknown treatment")
        if type(self.trusted_checkpoint) is not bool:
            raise ValueError("trusted_checkpoint must be boolean")
        for key in ("required_free_ram_bytes", "required_free_vram_bytes"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"Invalid {key}")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError("Configuration must be an object with known fields")
        for key in ("model_path", "dataset_path", "vision_path", "model_manifest_path", "vision_manifest_path"):
            if data.get(key):
                data[key] = str((path.parent / data[key]).resolve())
        return cls(**data).validate()
