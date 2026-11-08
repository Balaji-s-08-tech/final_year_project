"""Explicit local Bunny/Phi-2 and trusted portable text adapters."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import torch

from evaluation.inspection import architecture


@dataclass
class Prepared:
    inputs: dict[str, Any]
    prompt_tokens: int


@dataclass
class Generation:
    text: str
    token_ids: list[int]


class Adapter(Protocol):
    def load(self, config) -> None: ...
    def inspect(self) -> dict: ...
    def prepare(self, record: dict) -> Prepared: ...
    def generate(self, prepared: Prepared) -> torch.Tensor: ...
    def decode(self, prepared: Prepared, output: torch.Tensor) -> Generation: ...


class LocalAdapter:
    def load(self, config):
        from runtime_policy import admit_device
        admit_device(config.device, getattr(self, "authorize_gpu", False))
        self.config = config
        from VLM.quantization.awq_model_loader import _enable_project_imports
        _enable_project_imports()
        if config.adapter == "portable_awq_text":
            if not config.trusted_checkpoint:
                raise ValueError("Refusing untrusted pickle checkpoint")
            from experiments.artifacts import inspect_artifact
            metadata = inspect_artifact(config.model_path)
            if metadata["format"] != "portable_awq":
                raise ValueError("Portable evaluation requires a verified export sidecar")
            if config.treatment != f"awq_int{metadata['quantization']['bits']}":
                raise ValueError("Configured treatment contradicts verified artifact precision")
            from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint
            self.tokenizer, self.model, _ = load_quantized_bunny_checkpoint(config.model_path, config.device, authorize_gpu=getattr(self, "authorize_gpu", False))
            from VLM.quantization.awq_quantize import PackedAWQLinear
            bits = {m.bits for m in self.model.modules() if isinstance(m, PackedAWQLinear)}
            if bits != {metadata["quantization"]["bits"]}:
                raise ValueError("Artifact metadata contradicts loaded packed layers")
        else:
            from transformers import AutoTokenizer
            from bunny.model.language_model.bunny_phi import BunnyPhiConfig, BunnyPhiForCausalLM
            local = Path(config.model_path).resolve()
            model_config = BunnyPhiConfig.from_pretrained(local, local_files_only=True)
            if getattr(model_config, 'quantization_config', None) is not None:
                raise ValueError('Dense adapter refuses implicit quantized backend activation')
            model_config.mm_vision_tower = str(Path(config.vision_path).resolve())
            model_config.continuous_training = False
            self.tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
            self.model, loading = BunnyPhiForCausalLM.from_pretrained(local, config=model_config, local_files_only=True,
                torch_dtype=getattr(torch, config.dtype), low_cpu_mem_usage=True, output_loading_info=True)
            missing = [k for k in loading["missing_keys"] if "vision_tower" not in k]
            if missing or loading.get("mismatched_keys") or loading.get("unexpected_keys"):
                raise ValueError(f"Checkpoint tensor compatibility failed: {loading}")
            tower = self.model.get_vision_tower()
            if tower is None:
                raise ValueError("Checkpoint has no vision tower")
            if not tower.is_loaded:
                tower.load_model()
            self.processor = tower.image_processor
            self.model.to(device=config.device, dtype=getattr(torch, config.dtype))
        self.model.eval()
        if not callable(getattr(self.model, "generate", None)):
            raise ValueError("Checkpoint has no generation interface")

    def effective(self):
        from transformers import GenerationConfig
        self.generation_settings = GenerationConfig.from_dict(self.model.generation_config.to_dict())
        self.generation_settings.update(max_new_tokens=self.config.max_new_tokens, do_sample=False, num_beams=1, use_cache=True, pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        return {"generation": self.generation_settings.to_dict(), "model": self.model.config.to_dict(), "tokenizer": {"class": type(self.tokenizer).__name__, "name": self.tokenizer.name_or_path}, "processor": self.processor.to_dict() if hasattr(self, "processor") else None}

    def inspect(self):
        return architecture(self.model)

    def prepare(self, record, *, formatted=False):
        prompt = record["question"] if formatted else self.config.prompt_template.format(question=record["question"])
        if self.config.adapter == "bunny_phi":
            from PIL import Image
            from bunny.util.mm_utils import tokenizer_image_token, process_images
            ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt").unsqueeze(0)
            with Image.open(record["image"]) as source:
                pixels = process_images([source.convert("RGB")], self.processor, self.model.config)
            if not torch.is_tensor(pixels):
                raise ValueError("Version 1 requires uniform image tensor dimensions")
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                      "images": pixels.to(self.config.device, dtype=getattr(torch, self.config.dtype))}
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            inputs = {"input_ids": encoded["input_ids"], "attention_mask": encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))}
        length = inputs["input_ids"].shape[-1]
        if length > self.config.max_input_tokens:
            raise ValueError("Prompt exceeds max_input_tokens; implicit truncation is forbidden")
        context = getattr(self.model.config, "max_position_embeddings", None)
        visual_tokens = self.model.get_vision_tower().num_patches - 1 if self.config.adapter == "bunny_phi" else 0
        if context is not None and length + visual_tokens + self.config.max_new_tokens > context:
            raise ValueError("Expanded visual/text sequence exceeds model context window")
        return Prepared({key: value.to(self.config.device) for key, value in inputs.items()}, length)

    def generate(self, prepared):
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        # Bunny's multimodal helper mutates input_ids; give every invocation a clone.
        inputs = {key: value.clone() for key, value in prepared.inputs.items()}
        if self.config.adapter == "bunny_phi":
            # Expand image tokens once before generation. Feeding `images` into
            # legacy generate keeps a text-length attention mask against the
            # expanded KV cache on the second token. An expanded initial mask
            # plus inputs_embeds lets the inherited Phi cache slicing stay valid.
            _, _, mask, _, embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                inputs["input_ids"], None, inputs["attention_mask"], None, None, inputs.pop("images"))
            inputs["inputs_embeds"] = embeds
            inputs["attention_mask"] = mask
        self.effective()
        return self.model.generate(**inputs, generation_config=self.generation_settings, max_new_tokens=self.config.max_new_tokens,
                                   do_sample=False, num_beams=1, use_cache=True, pad_token_id=pad)

    def decode(self, prepared, output):
        if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] < prepared.prompt_tokens:
            raise ValueError("Unsupported generation output shape")
        tokens = output[0, prepared.prompt_tokens:].detach().cpu().tolist()
        eos = self.tokenizer.eos_token_id
        if eos in tokens:
            tokens = tokens[:tokens.index(eos) + 1]  # Include EOS, exclude post-EOS padding.
        return Generation(self.tokenizer.decode(tokens, skip_special_tokens=True), tokens)
