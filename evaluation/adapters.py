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
    effective_tokens: int | None = None


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
        if config.adapter == 'bunny_phi' and config.treatment != 'baseline' and not config.treatment_artifact_path:
            raise ValueError('Compressed Bunny treatment requires verified treatment artifact')
        from runtime_policy import admit_device, prepare_deterministic_cuda
        admit_device(config.device, getattr(self, "authorize_gpu", False))
        prepare_deterministic_cuda(config.device)
        self.config = config
        self.asset_verification = {}
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
            self.tokenizer, self.model, _ = load_quantized_bunny_checkpoint(config.model_path, config.device, authorize_gpu=getattr(self, "authorize_gpu", False), trusted=config.trusted_checkpoint)
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
            origin=getattr(model_config,'weight_origin',None)
            if origin != 'SYNTHETIC_TEST':
                from experiments.manifests import verify_model_manifest
                from experiments.store import read_json
                for source,manifest_path,identifier,revision in ((local,config.model_manifest_path,config.model_identifier,config.model_revision),(config.vision_path,config.vision_manifest_path,None,None)):
                    if not manifest_path:
                        raise ValueError('Unverified pretrained identity: supply complete model/vision content manifests or use explicitly marked SYNTHETIC_TEST fixtures')
                    document=read_json(manifest_path)
                    if identifier is not None and (document.get('identifier')!=identifier or document.get('revision')!=revision):
                        raise ValueError('Checkpoint manifest identity mismatch')
                    self.asset_verification[str(source)] = verify_model_manifest(source,document,hash_large=True)
            model_config.mm_vision_tower = str(Path(config.vision_path).resolve())
            model_config.continuous_training = False
            self.tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
            self.model, loading = BunnyPhiForCausalLM.from_pretrained(local, config=model_config, local_files_only=True,
                torch_dtype=torch.float32 if config.treatment_artifact_path else getattr(torch, config.dtype), low_cpu_mem_usage=True, output_loading_info=True, use_safetensors=True if not config.trusted_checkpoint else None)
            tower = self.model.get_vision_tower()
            if tower is None:
                raise ValueError("Checkpoint has no vision tower")
            if not tower.is_loaded:
                tower.load_model()
            from evaluation.checkpoint_layout import validate_bunny_checkpoint
            self.loading_report = {'bunny': validate_bunny_checkpoint(local, self.model, loading),
                                   'siglip': tower.layout_verification}
            self.asset_verification['checkpoint_loading'] = self.loading_report
            self.processor = tower.image_processor
            self.treatment_identity = None
            if config.treatment_artifact_path:
                from compression.multimodal import apply_treatment
                self.treatment_identity = apply_treatment(self, config.treatment_artifact_path, max_bytes=config.max_treatment_bytes)
            self.model.to(device=config.device, dtype=getattr(torch, config.dtype))
        from evaluation.integrity import validate_state, guard_logits
        validate_state(self.model)
        guard_logits(self.model)
        self.weight_origin = getattr(self.model.config, 'weight_origin', 'UNVERIFIED_LOCAL')
        self.model.eval()
        if not callable(getattr(self.model, "generate", None)):
            raise ValueError("Checkpoint has no generation interface")

    def effective(self):
        from transformers import GenerationConfig
        self.generation_settings = GenerationConfig.from_dict(self.model.generation_config.to_dict())
        self.generation_settings.update(max_new_tokens=self.config.max_new_tokens, do_sample=False, num_beams=1, use_cache=True, pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        self.asset_verification['treatment'] = getattr(self, 'treatment_identity', None)
        from evaluation.inference_policy import validate_generation
        validate_generation(self.generation_settings.to_dict())
        return {"asset_verification": self.asset_verification, "weight_origin": self.weight_origin, "pretrained_identity": "UNVERIFIED; local loading coverage is not acquisition provenance", "generation": self.generation_settings.to_dict(), "model": self.model.config.to_dict(), "tokenizer": {"class": type(self.tokenizer).__name__, "name": self.tokenizer.name_or_path}, "processor": self.processor.to_dict() if hasattr(self, "processor") else None}

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
        expanded = length + visual_tokens
        limit = getattr(self.model.config, 'tokenizer_model_max_length', None)
        if limit is not None and expanded > limit:
            raise ValueError('Multimodal prompt would truncate required content')
        if self.config.adapter == 'bunny_phi' and (inputs['input_ids'].eq(-200).sum().item()!=1 or length<2):
            raise ValueError('Invalid image-token boundary')
        return Prepared({key: value.to(self.config.device) for key, value in inputs.items()}, length, expanded)

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
            if embeds is None or embeds.shape[1] != prepared.effective_tokens or mask.shape[1] != prepared.effective_tokens:
                raise ValueError('Effective multimodal prompt differs: truncation/alignment failure')
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
