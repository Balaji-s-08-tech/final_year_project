# Final setup and validation report

Date: 2026-10-26

## Status summary

| Area | Status | Evidence |
|---|---|---|
| Environment | PASS | Existing Python 3.11.9, PyTorch 2.2.2+cu121, CUDA RTX 3050 6 GB |
| Dependencies | PASS | Core pinned Bunny/AWQ dependencies import; dashboard dependencies installed in project venv |
| LLM-Pruner import | PASS | `import LLMPruner` after adding `LLM-Pruner` to `sys.path` |
| Bunny builder import | PASS | Import works; inspected loader signature reported below |
| Bunny weights/model load | BLOCKED | Config/tokenizer cached, no Bunny model weights; no full model load attempted |
| Pruning | PARTIAL | Existing smoke artifact/script, no new pruning run in this validation |
| Recovery | NOT RUN | Existing smoke script explicitly skips recovery due VRAM; no recovered checkpoint validated |
| AWQ integration | PASS for tiny model | Two local tests cover toy Phi-like CPU INT4 save/reload/forward and INT8 export |
| Real Bunny AWQ | NOT RUN | No verified recovered Bunny model; full artifact reload avoided with ~4 GiB available RAM |
| Benchmark | PARTIAL | Live artifact-size job completed; accuracy, latency, throughput and peak memory are NOT MEASURED |
| Results import | PASS | JSON manifest import preserves checkpoint metadata and NOT MEASURED fields; model-free tests pass |
| Cloud execution | BLOCKED / PREPARED | Runbook and artifact inventory are ready; model weights, Bunny data, and recovery artifacts are missing; no cloud resources provisioned |
| Full end-to-end | BLOCKED | No verified recovery artifact; full Bunny weights are absent from local HF cache |

## Environment and imports

Observed versions: Python 3.11.9; PyTorch 2.2.2+cu121; torchvision 0.17.2+cu121; Transformers 4.37.2; Tokenizers 0.15.1; Accelerate 0.28.0; PEFT 0.10.0; SentencePiece 0.1.99; NumPy 1.26.4. CUDA is available and reports NVIDIA RTX 3050 Laptop GPU. The Bunny builder signature is `(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False, device_map='auto', device='cuda', **kwargs)`.

The local Hugging Face cache contains Bunny configuration and tokenizer assets but not model weights. No unrelated models or datasets were downloaded. No new Python environment was created because `.venv-vlm-awq` already existed. The adapter loading path uses the existing Bunny multimodal builder, which loads the vision tower as well; its memory requirement remains unverified on the 6 GB GPU.

## Work performed

- Added a local FastAPI dashboard API and connected the sibling React app to real local API responses; removed its active randomized mock-data path and made disconnected state explicit.
- Added job/checkpoint/calibration registration with approved-path checks, serialized local jobs, cancellation, system info, portable text inference, artifact-size benchmark manifests and explicit capability reporting. Pruning/recovery/cloud/image jobs remain unavailable in the dashboard.
- Added a JSON results-import endpoint/UI that preserves imported metadata and metrics, tags source/time, and rejects invalid sizes/non-finite values.
- Added focused API tests, tiny Phi-like AWQ tests, a verified artifact inventory, and cloud-execution/runbook documentation.
- Fixed AWQ export naming so INT8 results are labeled INT8 rather than INT4.
- Did not modify pruning, recovery algorithms, Bunny architecture, or LLM-Pruner core in this validation. Did not commit or push.

## Validation actually run

- `pytest backend/tests VLM/quantization/tests -q`: **11 passed**, one third-party deprecation warning.
- `python -m compileall -q VLM/quantization backend`: passed.
- LLM-Pruner import and Bunny builder import: passed.
- `python -m VLM.quantization.benchmark --help`: passed.
- AWQ config validation for INT4 and INT8: passed.
- Interface `npm run lint`: passed.
- Interface `npm run build`: passed.
- Live API and Vite HTTP checks: passed (health, system, models, checkpoints, CORS preflight, frontend root).
- A real artifact-size benchmark job against the existing AWQ checkpoint completed through the API and wrote a manifest. Its exact file size was 1,776,369,712 bytes. Accuracy, latency and memory were not measured.
- A live JSON import using an explicitly unmeasured fixture returned and displayed its source/checkpoint metadata without labeling it measured. The fixture file was removed after verification; the model-free test covers persistence in a temporary directory.
- A full-model checkpoint reload/forward was not run in this validation.

## Existing artifacts and blockers

See [ARTIFACT_INVENTORY.md](ARTIFACT_INVENTORY.md) for exact paths, byte sizes, types, roles, cache contents and missing artifacts. The local Bunny cache has config/tokenizer files only. The smoke script records recovery as skipped; two other AWQ outputs have unknown provenance. No recovery adapter/state, Bunny training data, or full model weight shards were found. About 4 GiB system RAM was available during the final resource check, so full checkpoint expansion was not attempted.

Cloud readiness is documentation-only at this point. `CLOUD_EXECUTION.md` records the pinned Python/PyTorch/Transformers stack, dataset layout, expected artifacts, Linux-only recovery dependencies and a resource estimate. The 3B FP16 weights alone have an arithmetic lower bound near 5.59 GiB, before vision weights or training memory. A 24 GB GPU is only a first LoRA smoke-test target, not a measured guarantee. No cloud VM was provisioned and no large model/data were downloaded.

## Next experiment command

First provide a verified recovery adapter/state output and the exact structural checkpoint used to train it. Then run:

```powershell
.\.venv-vlm-awq\Scripts\python.exe -m VLM.quantization.awq_quantize `
  --base_model BAAI/Bunny-v1_0-3B `
  --pruned_checkpoint <pruned-pytorch-model-bin> `
  --recovery_checkpoint <recovery-lora-directory-or-state-file> `
  --calibration_data <local-calibration-file.txt> `
  --bits 4 --group_size 128 --calibration_samples 2 --sequence_length 32 `
  --device cpu --output_dir checkpoints/bunny_awq_int4
```

Then reload/evaluate the produced checkpoint with an actual task protocol. Until model weights, Bunny data and a provenance-verified recovery output exist, the complete Bunny → pruning → recovery → AWQ INT4 experiment is **not executable end-to-end from the currently verified artifacts**.

## Run the local application

Backend window, from `D:\final year project\VLM_Compression`:

```powershell
$env:VLM_FRONTEND_ORIGIN = 'http://localhost:3000'
.\.venv-vlm-awq\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend window:

```powershell
Set-Location 'D:\final year project\Interface'
npm run dev
```

Next safe action: stage and verify the licensed base weight shards and Bunny dataset in the chosen cloud workspace, measure their sizes, then run a short environment/import and GPU-memory probe. Do not start pruning or recovery until those checks pass and the experiment hyperparameters are fixed.
