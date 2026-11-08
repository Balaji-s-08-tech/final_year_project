from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "VLM", ROOT / "LLM-Pruner"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _tree_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _has_weights(path: Path) -> bool:
    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    return any(any(path.glob(pattern)) for pattern in patterns)


def _runtime() -> tuple[object, dict[str, str], dict[str, str]]:
    required = (
        "torchvision",
        "transformers",
        "accelerate",
        "peft",
        "timm",
        "einops",
        "huggingface_hub",
        "PIL",
        "psutil",
        "VLM.bunny.model.builder",
        "VLM.quantization.awq_quantize",
    )
    torch = importlib.import_module("torch")
    loaded = {"torch": str(getattr(torch, "__version__", "available"))}
    missing = {}
    for name in required:
        try:
            module = importlib.import_module(name)
            loaded[name] = str(getattr(module, "__version__", "available"))
        except Exception as exc:
            missing[name] = f"{type(exc).__name__}: {exc}"
    return torch, loaded, missing


def _weight_files(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and _has_weights(path)


def _load_and_forward(args: argparse.Namespace, torch: object) -> dict[str, object]:
    if not getattr(args, "authorize_gpu_execution", False):
        raise PermissionError("Model/GPU execution requires separate explicit authorization")
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_dir = Path(args.model_dir).resolve()
    vision_dir = Path(args.vision_dir).resolve()
    image_path = Path(args.image_path).resolve()
    if not _weight_files(model_dir):
        raise RuntimeError(f"Model config or weight files are missing under {model_dir}")
    if not _weight_files(vision_dir):
        raise RuntimeError(f"Vision config or weight files are missing under {vision_dir}")
    if not image_path.is_file():
        raise RuntimeError(f"Probe image not found: {image_path}")

    free_vram, total_vram = torch.cuda.mem_get_info()
    psutil = importlib.import_module("psutil")
    memory_before = psutil.virtual_memory()
    free_ram = memory_before.available
    minimum_vram = int(args.minimum_free_vram_gib * 1024**3)
    minimum_ram = int(args.minimum_free_ram_gib * 1024**3)
    if free_vram < minimum_vram:
        raise RuntimeError(f"Only {free_vram / 1024**3:.2f} GiB free VRAM; require {args.minimum_free_vram_gib:.1f} GiB")
    if free_ram < minimum_ram:
        raise RuntimeError(f"Only {free_ram / 1024**3:.2f} GiB free host RAM; require {args.minimum_free_ram_gib:.1f} GiB")

    from PIL import Image
    from VLM.bunny.model.builder import load_pruned_bunny_model
    from VLM.bunny.util.mm_utils import process_images, tokenizer_image_token
    from bunny.constants import DEFAULT_IMAGE_TOKEN

    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_pruned_bunny_model(
        str(model_dir),
        pruned_model_path=None,
        device="cuda",
        device_map={"": "cuda"},
    )
    existing_tower = model.get_vision_tower()
    if existing_tower is not None and hasattr(existing_tower, "vision_tower_name"):
        existing_tower.vision_tower_name = str(vision_dir)
    model_args = SimpleNamespace(
        vision_tower=str(vision_dir),
        pretrain_mm_mlp_adapter=None,
        mm_projector_type=getattr(model.config, "mm_projector_type", None) or "mlp2x_gelu",
        use_s2=False,
    )
    model.get_model().initialize_vision_modules(model_args)
    model.to(device="cuda", dtype=torch.float16)
    image_processor = model.get_vision_tower().image_processor
    image = Image.open(image_path).convert("RGB")
    images = process_images([image], image_processor, model.config).to(device="cuda", dtype=torch.float16)
    prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\nDescribe this image briefly. ASSISTANT:"
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to("cuda")
    model.eval()
    with torch.inference_mode():
        output = model(input_ids=input_ids, images=images, use_cache=False, return_dict=True)
    if not torch.isfinite(output.logits).all().item():
        raise RuntimeError("Multimodal probe returned non-finite logits")

    free_vram_after, _ = torch.cuda.mem_get_info()
    memory_after = psutil.virtual_memory()
    process_rss = psutil.Process().memory_info().rss
    return {
        "result": "PASS",
        "model_directory_bytes": _tree_bytes(model_dir),
        "vision_directory_bytes": _tree_bytes(vision_dir),
        "host_total_gib": round(memory_before.total / 1024**3, 2),
        "host_available_before_gib": round(memory_before.available / 1024**3, 2),
        "host_available_after_gib": round(memory_after.available / 1024**3, 2),
        "process_rss_after_gib": round(process_rss / 1024**3, 2),
        "gpu_total_gib": round(total_vram / 1024**3, 2),
        "gpu_free_before_gib": round(free_vram / 1024**3, 2),
        "gpu_free_after_gib": round(free_vram_after / 1024**3, 2),
        "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "logits_shape": list(output.logits.shape),
        "logits_finite": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--vision-dir")
    parser.add_argument("--image-path")
    parser.add_argument("--minimum-free-vram-gib", type=float)
    parser.add_argument("--minimum-free-ram-gib", type=float)
    parser.add_argument("--authorize-gpu-execution", action="store_true",
                        help="Only AFTER billing and GPU execution are separately authorized")
    args = parser.parse_args()
    if args.load_model:
        if not args.authorize_gpu_execution:
            parser.error("--load-model requires --authorize-gpu-execution after separate billing confirmation")
        if any(v is None or not math.isfinite(v) or v <= 0 for v in (args.minimum_free_vram_gib, args.minimum_free_ram_gib)):
            parser.error("--load-model requires explicit positive finite RAM and VRAM budgets")
        if not args.model_dir or not args.vision_dir or not args.image_path:
            parser.error("--load-model requires local model/vision directories and an image path")
    try:
        torch, versions, missing = _runtime()
        from experiments.provenance import capture
        _, environment = capture(ROOT)
        devices = environment.get("gpu_devices") or []
        capability = bool(devices and torch.version.cuda)
        result = {
            "imports": versions,
            "missing_or_failed_imports": missing,
            "cuda_available": capability,
            "cuda_status": "CAPABILITY_ONLY_NOT_RUNTIME_VERIFIED",
            "model_loaded": False,
            "gpu_devices": devices or None,
        }
        if missing or not capability:
            result["result"] = "BLOCKED"
            print(json.dumps(result, indent=2))
            return 2
        result["result"] = "PREFLIGHT_ONLY"
        if args.load_model:
            if not args.model_dir or not args.vision_dir or not args.image_path:
                raise RuntimeError("--load-model requires --model-dir, --vision-dir, and --image-path")
            result["multimodal_forward"] = _load_and_forward(args, torch)
            result["model_loaded"] = True
            result["result"] = "PASS"
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"result": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
