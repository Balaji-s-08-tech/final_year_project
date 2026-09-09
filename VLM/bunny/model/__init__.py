"""Bunny language-model wrappers.

Only the Phi-2 backend (bunny_phi) is imported eagerly — it is the primary
target for Bunny-v1.0-3B.  All other backends (LLaMA, StableLM, Qwen2,
MiniCPM, Phi-3) are loaded lazily via __getattr__ so that missing or
incompatible dependencies (e.g. StaticCache in newer transformers) do not
prevent importing the Phi-2 pipeline.
"""

# --- Phi-2: always available (target model) ---------------------------------
from .language_model.bunny_phi import BunnyPhiForCausalLM, BunnyPhiConfig, BunnyDistillationModel

# --- Lazy imports for other backends -----------------------------------------
_LAZY_IMPORTS = {
    "BunnyStableLMForCausalLM": (".language_model.bunny_stablelm", "BunnyStableLMForCausalLM"),
    "BunnyStableLMConfig":      (".language_model.bunny_stablelm", "BunnyStableLMConfig"),
    "BunnyQwen2ForCausalLM":    (".language_model.bunny_qwen",     "BunnyQwen2ForCausalLM"),
    "BunnyQwen2Config":         (".language_model.bunny_qwen",     "BunnyQwen2Config"),
    "BunnyMiniCPMForCausalLM":  (".language_model.bunny_minicpm",  "BunnyMiniCPMForCausalLM"),
    "BunnyMiniCPMConfig":       (".language_model.bunny_minicpm",  "BunnyMiniCPMConfig"),
    "BunnyLlamaForCausalLM":    (".language_model.bunny_llama",    "BunnyLlamaForCausalLM"),
    "BunnyLlamaConfig":         (".language_model.bunny_llama",    "BunnyLlamaConfig"),
    "BunnyPhi3ForCausalLM":     (".language_model.bunny_phi3",     "BunnyPhi3ForCausalLM"),
    "BunnyPhi3Config":          (".language_model.bunny_phi3",     "BunnyPhi3Config"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __package__)
        value = getattr(mod, attr)
        # Cache in module globals so __getattr__ is not called again.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
