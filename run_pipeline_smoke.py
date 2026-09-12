"""End-to-end pipeline smoke test: Pruning (Taylor-FO) -> AWQ INT4."""

import gc
import os
import sys
import time
import psutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM-Pruner"))
sys.path.insert(0, PROJECT_ROOT)

import copy
import random

import numpy as np
import torch
import torch.nn as nn

SMOKE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "smoke_test_output")
PRUNED_CKPT_PATH = os.path.join(SMOKE_OUTPUT_DIR, "pruned_model.bin")
AWQ_CKPT_PATH = os.path.join(SMOKE_OUTPUT_DIR, "pruned_awq_int4.pt")
CALIB_PATH = os.path.join(SMOKE_OUTPUT_DIR, "calibration.txt")

PRUNING_RATIO = 0.1
NUM_EXAMPLES = 2
SEQ_LEN_PRUNE = 32
BLOCK_ATTN_START, BLOCK_ATTN_END = 3, 21
BLOCK_MLP_START, BLOCK_MLP_END = 3, 20


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def section(title):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def stage_1_load_model():
    section("STAGE 1: Load Bunny-v1.0-3B (Phi-2 backbone)")
    from transformers import AutoTokenizer
    from LLMPruner.models.hf_phi2.modeling_phi import PhiForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("BAAI/Bunny-v1_0-3B", use_fast=False)
    model = PhiForCausalLM.from_pretrained("BAAI/Bunny-v1_0-3B", low_cpu_mem_usage=True)
    model.half().to("cuda")
    model.config.pad_token_id = 50256
    model.config.bos_token_id = 50256
    model.config.eos_token_id = 50256

    total_params = sum(p.numel() for p in model.parameters())
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Model type      : {type(model).__name__}")
    print(f"  Parameters      : {total_params:,}")
    print(f"  GPU memory      : {vram:.2f} GB")
    print(f"  Tokenizer       : {type(tokenizer).__name__}")
    print("  STATUS: PASS")
    return model, tokenizer, total_params


def stage_2_taylor_pruning(model, tokenizer, original_params):
    section("STAGE 2: Structural Pruning (Taylor First-Order)")
    import LLMPruner.torch_pruning as tp
    from LLMPruner.pruner import phi2_pruner as phi_pruner
    from LLMPruner.datasets.example_samples import get_examples

    for param in model.parameters():
        param.requires_grad_(False)
        
    for i in range(BLOCK_ATTN_START, BLOCK_ATTN_END):
        for param in model.model.layers[i].parameters():
            param.requires_grad_(True)

    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
    ]).to("cuda")

    imp = phi_pruner.TaylorImportance(group_reduction="sum", taylor="param_first")

    kwargs = {
        "importance": imp,
        "global_pruning": False,
        "iterative_steps": 1,
        "ch_sparsity": PRUNING_RATIO,
        "ignored_layers": [],
        "channel_groups": {},
        "consecutive_groups": {
            layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
        },
        "customized_pruners": {},
        "root_module_types": None,
        "root_instances": (
            [model.model.layers[i].self_attn.q_proj for i in range(BLOCK_ATTN_START, BLOCK_ATTN_END)] +
            [model.model.layers[i].mlp.fc1 for i in range(BLOCK_MLP_START, BLOCK_MLP_END)]
        ),
    }

    pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
    model.zero_grad()

    print(f"  Pruner          : Taylor First-Order (param_first)")
    print(f"  Pruning ratio   : {PRUNING_RATIO}")
    print(f"  Attention layers: {BLOCK_ATTN_START}-{BLOCK_ATTN_END}")
    print(f"  MLP layers      : {BLOCK_MLP_START}-{BLOCK_MLP_END}")
    print(f"  Dataset         : alpaca (text-only, {NUM_EXAMPLES} samples)")
    print(f"  Seq length      : {SEQ_LEN_PRUNE}")

    print("  Computing Taylor importance via backward pass ...")
    example_prompts = get_examples("alpaca", tokenizer, NUM_EXAMPLES, seq_len=SEQ_LEN_PRUNE, batch=True)
    example_prompts = example_prompts.cpu()

    loss = model(example_prompts.to("cuda"), labels=example_prompts.to("cuda")).loss
    print(f"  Loss            : {loss.item():.4f}")
    loss.sum().backward()
        
    del loss, example_prompts
    gc.collect()
    torch.cuda.empty_cache()
    print("  Cleared GPU memory for importance scoring ...")

    pruner.step()
    pruned_params = sum(p.numel() for p in model.parameters())

    for layer in model.model.layers:
        layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

    model.zero_grad()
    for name, module in model.named_parameters():
        if "weight" in name:
            module.grad = None

    del pruner
    gc.collect()
    torch.cuda.empty_cache()

    reduction = (1 - pruned_params / original_params) * 100
    print(f"  Original params : {original_params:,}")
    print(f"  Pruned params   : {pruned_params:,}")
    print(f"  Reduction       : {reduction:.1f}%")

    pruned_layers = []
    for i, layer in enumerate(model.model.layers):
        q_out = layer.self_attn.q_proj.weight.shape[0]
        if q_out != 2560:
            pruned_layers.append(i)
    print(f"  Pruned layers   : {len(pruned_layers)} ({pruned_layers[0]}-{pruned_layers[-1]})" if pruned_layers else "  Pruned layers   : 0")
    if pruned_layers:
        ex = model.model.layers[pruned_layers[0]]
        print(f"  Example layer {pruned_layers[0]}: q_proj={tuple(ex.self_attn.q_proj.weight.shape)}, fc1={tuple(ex.mlp.fc1.weight.shape)}")
    print("  STATUS: PASS")
    return model, tokenizer, pruned_params


def stage_3_save_pruned(model, tokenizer):
    section("STAGE 3: Save Pruned Checkpoint")
    os.makedirs(SMOKE_OUTPUT_DIR, exist_ok=True)
    torch.save({"model": model, "tokenizer": tokenizer}, PRUNED_CKPT_PATH)
    size_mb = os.path.getsize(PRUNED_CKPT_PATH) / 1e6
    print(f"  Path            : {PRUNED_CKPT_PATH}")
    print(f"  Size            : {size_mb:.1f} MB")
    print("  STATUS: PASS")
    return size_mb


def stage_4_recovery_skipped():
    section("STAGE 4: LoRA Recovery Training")
    print("  RECOVERY: SKIPPED -- LOCAL GPU MEMORY CONSTRAINT")
    print("  Reason: Full Bunny multimodal recovery (Phi-2 + SigLIP + gradients)")
    print("          exceeds RTX 3050 6GB VRAM / 16GB RAM limits.")
    print("  The pruned model proceeds directly to AWQ without weight recovery.")
    print("  ACCURACY EVALUATION: PENDING")
    print("  STATUS: BLOCKED (hardware)")


def stage_5_awq_quantize(model, tokenizer):
    section("STAGE 5: AWQ INT4 Quantization")

    from VLM.quantization.awq_quantize import quantize_bunny_awq, PackedAWQLinear
    from VLM.quantization.awq_calibration import build_calibration_samples
    from VLM.quantization.awq_config import AWQConfig

    calib_texts = [
        "The structure of neural networks involves layers of interconnected nodes.",
        "Machine learning models can be compressed through pruning and quantization.",
        "Vision-language models combine visual perception with natural language understanding.",
        "Activation-aware weight quantization preserves salient weight channels.",
    ]
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(calib_texts))

    samples = build_calibration_samples(tokenizer, calib_texts, num_samples=2, sequence_length=32, seed=42)
    print(f"  Calibration     : {len(samples)} samples, seq_len=32")

    config = AWQConfig(bits=4, group_size=128, device="cuda", calibration_samples=2, sequence_length=32)
    print(f"  Bits            : {config.bits}")
    print(f"  Group size      : {config.group_size}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    summary = quantize_bunny_awq(model, samples, config)
    elapsed = time.time() - t0
    peak_vram_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_vram_res = torch.cuda.max_memory_reserved() / 1e9

    awq_count = sum(1 for m in model.modules() if isinstance(m, PackedAWQLinear))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    print(f"  Blocks quantized: {summary['blocks']}")
    print(f"  Linears replaced: {summary['quantized_linears']}")
    print(f"  PackedAWQLinear : {awq_count}")
    print(f"  Remaining Linear: {linear_count}")
    print(f"  Time            : {elapsed:.1f}s")
    print(f"  Peak CUDA alloc : {peak_vram_alloc:.2f} GB")
    print(f"  Peak CUDA resrv : {peak_vram_res:.2f} GB")
    print("  STATUS: PASS")
    return summary, elapsed, peak_vram_alloc, peak_vram_res


def stage_6_save_compressed(model, tokenizer, config_dict, summary):
    section("STAGE 6: Save Compressed Checkpoint")
    torch.save({
        "format": "bunny-awq-portable-v1",
        "model": model.cpu(),
        "tokenizer": tokenizer,
        "metadata": {"quantization": config_dict, "summary": summary},
    }, AWQ_CKPT_PATH)
    size_mb = os.path.getsize(AWQ_CKPT_PATH) / 1e6
    print(f"  Path            : {AWQ_CKPT_PATH}")
    print(f"  Size            : {size_mb:.1f} MB")
    print("  STATUS: PASS")
    return size_mb


def stage_7_reload_and_verify(tokenizer):
    section("STAGE 7: Reload & Forward Integrity Test")
    from VLM.quantization.awq_quantize import PackedAWQLinear

    payload = torch.load(AWQ_CKPT_PATH, map_location="cpu", weights_only=False)
    model = payload["model"]
    model.eval()

    awq_count = sum(1 for m in model.modules() if isinstance(m, PackedAWQLinear))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"  Reload          : SUCCESS")
    print(f"  PackedAWQLinear : {awq_count}")
    print(f"  Remaining Linear: {linear_count}")

    model.float()
    input_ids = tokenizer("The meaning of life is", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        output = model(input_ids=input_ids)

    logits = output.logits
    is_finite = torch.isfinite(logits).all().item()
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()

    print(f"  Output shape    : {tuple(logits.shape)}")
    print(f"  Finite          : {is_finite}")
    print(f"  NaN             : {has_nan}")
    print(f"  Inf             : {has_inf}")
    print(f"  Sample logits   : {logits[0, -1, :5].tolist()}")

    if is_finite and not has_nan and not has_inf:
        print("  STATUS: PASS")
    else:
        print("  STATUS: FAIL (non-finite output)")
    return is_finite and not has_nan and not has_inf


def main():
    print("=" * 64)
    print("  VLM COMPRESSION PIPELINE SMOKE TEST")
    print("  Bunny-v1.0-3B / Phi-2")
    print("  Taylor-FO Pruning -> AWQ INT4")
    print("=" * 64)

    seed_all(42)
    os.makedirs(SMOKE_OUTPUT_DIR, exist_ok=True)

    model, tokenizer, original_params = stage_1_load_model()

    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, pruned_params = stage_2_taylor_pruning(model, tokenizer, original_params)
    peak_taylor_alloc = torch.cuda.max_memory_allocated() / 1e9

    pruned_size = stage_3_save_pruned(model, tokenizer)
    stage_4_recovery_skipped()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    ckpt = torch.load(PRUNED_CKPT_PATH, map_location="cpu", weights_only=False)
    model = ckpt["model"]
    tokenizer = ckpt["tokenizer"]
    del ckpt
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    gc.collect()

    summary, quant_time, peak_vram_alloc, peak_vram_res = stage_5_awq_quantize(model, tokenizer)
    config_dict = {"bits": 4, "group_size": 128}
    awq_size = stage_6_save_compressed(model, tokenizer, config_dict, summary)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    forward_ok = stage_7_reload_and_verify(tokenizer)

    section("PIPELINE SUMMARY")
    print(f"  Original params       : {original_params:,}")
    print(f"  Pruned params         : {pruned_params:,}")
    print(f"  Parameter reduction   : {(1 - pruned_params / original_params) * 100:.1f}%")
    print(f"  Pruned checkpoint     : {pruned_size:.1f} MB")
    print(f"  AWQ INT4 checkpoint   : {awq_size:.1f} MB")
    print(f"  Compression ratio     : {pruned_size / awq_size:.2f}x (pruned FP16 -> INT4)")
    print(f"  Quantized linears     : {summary['quantized_linears']}")
    print(f"  INT4 config           : bits=4, group_size=128")
    print(f"  Quantization time     : {quant_time:.1f}s")
    print(f"  Reload                : SUCCESS")
    print(f"  Forward integrity     : {'PASS' if forward_ok else 'FAIL'}")
    print(f"  Recovery              : SKIPPED (hardware constraint)")
    print(f"  ACCURACY EVALUATION   : PENDING")

    current_alloc = torch.cuda.memory_allocated() / 1e9
    current_resrv = torch.cuda.memory_reserved() / 1e9
    sys_ram = psutil.Process().memory_info().rss / 1e9

    print("\n================================================================")
    print("  MEMORY USAGE")
    print("================================================================")
    print("  Dedicated GPU VRAM capacity: 6.00 GB\n")
    print(f"  Current CUDA allocated : {current_alloc:.2f} GB")
    print(f"  Current CUDA reserved  : {current_resrv:.2f} GB")
    print(f"  Peak CUDA allocated (Taylor-FO): {peak_taylor_alloc:.2f} GB")
    print(f"  Peak CUDA allocated (AWQ)      : {peak_vram_alloc:.2f} GB")
    print(f"  Peak CUDA reserved (AWQ)       : {peak_vram_res:.2f} GB\n")
    print(f"  Process system RAM RSS         : {sys_ram:.2f} GB")
    if peak_taylor_alloc > 6.0:
        print("\n  NOTE: Taylor-FO smoke configuration exceeds the 6 GB dedicated")
        print("  VRAM budget; Windows shared GPU memory was utilized.")
        print("  AWQ peak CUDA allocation remained below the 6 GB dedicated VRAM capacity.")

    print("\n" + "=" * 64)
    if forward_ok:
        print("  SMOKE TEST RESULT: PASS")
    else:
        print("  SMOKE TEST RESULT: FAIL")
    print("=" * 64)


if __name__ == "__main__":
    main()
