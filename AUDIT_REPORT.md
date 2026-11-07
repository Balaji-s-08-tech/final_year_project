# Bunny Phi-2 pruning, recovery, and AWQ audit

Audit date: 2026-10-26. Repository and the existing `.venv-vlm-awq` environment were inspected in place. This audit does not claim a full experiment ran.

## Source flow

### Upstream / base project code

- `LLM-Pruner/examples/Bunny.py` loads the causal language model via the local `LLMPruner.models.hf_phi2` implementation. It applies the configured Taylor/magnitude/random structured pruning path and, when `--save_model` is enabled (the parser default), saves `{'model': model, 'tokenizer': tokenizer}` to `logger.best_checkpoint_path`, under the LLM-Pruner pruning log directory. The object contains the altered model structure, not just a conventional state dict.
- `VLM/bunny/model/language_model/bunny_phi.py` defines `BunnyPhiForCausalLM`, built on the local Phi-2 implementation. `VLM/bunny/model/builder.py` selects the Phi-2 Bunny class for the intended Bunny-v1.0-3B configuration. `load_pretrained_model` has the inspected signature `(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False, device_map='auto', device='cuda', **kwargs)`; `load_pruned_bunny_model` is the structural checkpoint bridge used by recovery.
- `VLM/bunny/train/train_pruned.py` reads `model_name_or_path`, `pruned_model_path`, and `model_type`; training can use LoRA (`lora_enable`) or full tuning. LoRA recovery writes an adapter directory, model config and `non_lora_trainables.bin`; the trainer also saves checkpoints/state. Its output is not automatically a standalone full recovered model. For AWQ, the structural checkpoint and LoRA recovery directory must be supplied together. A full state recovery file must likewise be applied to the pruned model.
- Existing `smoke_test_output/pruned_model.bin` and `smoke_test_output/pruned_awq_int4.pt` are local artifacts. The existing `run_pipeline_smoke.py` explicitly skips recovery due to local VRAM constraints. Their presence is not proof of recovered-model lineage.

### Project-specific AWQ integration

- `awq_model_loader.py` loads a serialized model checkpoint directly if it contains a model and tokenizer. For adapter/state recovery it requires a base Bunny model plus the structural checkpoint; a LoRA directory must contain `adapter_config.json`. It does not silently fall back to the dense model if those are missing. State recovery is loaded with `strict=False` and rejected if no tensor names match. The adapter path calls Bunny's existing multimodal builder, which also loads the vision tower; that memory cost has not been validated on the 6 GB target.
- `awq_calibration.py` reads local TXT/JSON/JSONL, chooses a seeded subset, tokenizes bounded samples and captures bounded linear inputs on CPU in FP16 with hooks removed in `finally`.
- `awq_quantize.py` recognizes `model.model.layers` as Phi blocks and requires `input_layernorm`, `self_attn.{q_proj,k_proj,v_proj,dense}`, and `mlp.{fc1,fc2}` to be `nn.Linear`. It collects block inputs, searches equivalent scales, applies clipping and replaces those linears with `PackedAWQLinear`. It saves `awq_config.json` and a portable PyTorch checkpoint. This representation dequantizes for regular PyTorch execution; no custom high-performance INT4 CUDA kernel is provided.
- The integration core does not import `awq_inference_engine` or absent `quantization.utils` paths. The copied `awq/qmodule.py` and `w8a8_linear.py` do import `awq_inference_engine`, and other copied helpers contain model-specific imports; they are not the execution path used by `awq_quantize.py`.
- The quantizer supports AWQ INT4 and INT8. Output naming now reflects bits (`bunny_awq_int4.pt` or `bunny_awq_int8.pt`).

## Environment and dependencies

The existing environment is Python 3.11.9, PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, Transformers 4.37.2, Tokenizers 0.15.1, Accelerate 0.28.0, PEFT 0.10.0, SentencePiece 0.1.99, NumPy 1.26.4. CUDA is available on an RTX 3050 Laptop GPU with 6 GB dedicated VRAM. The unified `requirements.txt` includes optional packages such as DeepSpeed and BitsAndBytes that are not necessary for the CPU AWQ toy test. `requirements-bunny-awq.txt` pins the core Bunny/AWQ versions; PyTorch CUDA wheels are documented separately in `ENVIRONMENT_SETUP.md`.

The local HF cache contains Bunny tokenizer/config assets but no model weight files. No model was downloaded in this run. Current free system RAM observed was about 4 GiB; full model loading/reload was not attempted.

## Checkpoint requirements and integration commands

Pruning output is the serialized `pytorch_model.bin`-style object shown above. AWQ should receive either a serialized model that is already the intended recovered structural model, or all three recovery inputs: `--base_model`, `--pruned_checkpoint`, and `--recovery_checkpoint` (LoRA directory or matching state file). The loader cannot establish training provenance of a raw serialized model; record lineage externally and do not label the existing prune-only smoke checkpoint as recovered.

The quantizer interface is:

```powershell
python -m VLM.quantization.awq_quantize `
  --base_model BAAI/Bunny-v1_0-3B `
  --pruned_checkpoint <path-to-structural-checkpoint> `
  --recovery_checkpoint <path-to-recovery-adapter-or-state> `
  --calibration_data <path-to-local-calibration-text> `
  --bits 4 --group_size 128 --calibration_samples 2 --sequence_length 32 `
  --device cpu --output_dir <output-directory>
```

For a serialized full recovered model checkpoint, use `--model_path <recovered-model-checkpoint>` instead of the base/prune/recovery trio. The latter form must be confirmed by the experiment owner to be recovered; the file loader cannot infer that fact.

## Findings and blockers

- Import issue: `import LLMPruner` and the Bunny builder import both pass in the configured environment. The builder import verifies module availability only, not model weight loading.
- Actual model load: not run. Weight shards are absent from the local HF snapshot; only config/tokenizer files are cached.
- Pruning: the repository smoke script has an existing prune output, but a new pruning run was not started. This machine is constrained and no new costly pruning was needed for code validation.
- Recovery: not run. Existing smoke script explicitly skips it. A valid recovery output is required for the research pipeline.
- AWQ INT4/INT8: tiny Phi-like CPU tests exercise calibration, quantization, save/reload and finite forward for INT4, and INT8 quantization/save naming. They do not establish full Bunny model performance.
- Benchmark: artifact size is measured from supplied paths. GPU memory, latency, throughput and task quality remain `NOT MEASURED` unless an actual evaluator reports them.
- Full model checkpoint expansion was not attempted because only ~4 GiB RAM was free during audit. The 1.78 GB portable smoke artifact is present but was not reloaded in this run.

## Recommended order

1. Use the environment in `ENVIRONMENT_SETUP.md`; keep `transformers==4.37.2`.
2. Obtain/verify the Bunny base weights and project-specific pruning/recovery data.
3. Produce a structural checkpoint with the original pruning implementation.
4. Run recovery from that exact structural checkpoint and preserve the adapter plus `non_lora_trainables.bin` or an explicit full recovered state.
5. Run AWQ on those exact recovery inputs with a small local calibration file.
6. Reload and evaluate on an identified task protocol before reporting quality, latency, or memory comparisons.
