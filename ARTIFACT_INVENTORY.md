# Verified artifact inventory

Inventory checked: 2026-10-26. File sizes come from filesystem metadata. Large serialized model files were not loaded into RAM. Roles below are source-traced; a file's presence or name alone does not prove recovery provenance.

## Local project outputs

| Path | Size (bytes) | Type | Verified role / provenance |
|---|---:|---|---|
| `smoke_test_output/calibration.txt` | 310 | UTF-8 text | Small local smoke calibration file; used by `run_pipeline_smoke.py`, not a benchmark dataset. |
| `smoke_test_output/pruned_model.bin` | 5,295,119,617 | PyTorch pickle | `run_pipeline_smoke.py` stage 3 writes `{'model', 'tokenizer'}` after structural pruning. The same script explicitly skips recovery. Treat as prune-only smoke output; the artifact was not reloaded for this inventory. |
| `smoke_test_output/pruned_awq_int4.pt` | 1,776,369,712 | PyTorch pickle | Stage 6 AWQ INT4 output from that smoke script, after the prune-only input. Recovery was skipped, so this is not a recovered-model experiment result. |
| `test_awq_output/awq_config.json` | 3,444 | JSON | AWQ configuration/model metadata: 4 bits, group size 128, 2 calibration samples, 32 Phi blocks, 192 linears. It does not contain a recovery checkpoint reference or prove recovery. |
| `test_awq_output/bunny_phi2_awq_int4.pt` | 1,846,118,868 | PyTorch pickle | AWQ INT4 checkpoint paired with the above metadata. Recovery lineage not verified. |
| `test_awq_output/pruned_awq/pruned_bunny_awq_int4.pt` | 1,776,372,518 | PyTorch pickle | AWQ-looking output with no adjacent provenance sidecar found. Its prune/recovery lineage is unknown. |
| `checkpoints/b370ffa9ca0f46c69fc64bbf90387410/benchmark_manifest.json` | 1,023 | JSON | Locally generated benchmark job result. It contains measured file size for `smoke_test_output/pruned_awq_int4.pt` and `NOT MEASURED` for accuracy, latency, throughput, and peak memory. |
| `checkpoints/b7f12a1c84184f01aeadbdd0b0ef0ae9/benchmark_manifest.json` | 1,023 | JSON | Same actual file-size benchmark run through the live API during this continuation; it records the 1,776,369,712-byte AWQ artifact and keeps all other metrics `NOT MEASURED`. |

## Hugging Face cache

Snapshot directory: `C:\Users\Balaji\.cache\huggingface\hub\models--BAAI--Bunny-v1_0-3B\snapshots\0958a104bd39a448fc310b3d1b1fc3d398dc3fcb`. It contains 3,379,530 bytes of config/tokenizer assets, not model weights.

| File | Size (bytes) | Role |
|---|---:|---|
| `config.json` | 1,272 | Bunny Phi config; identifies `BunnyPhiForCausalLM`, hidden size 2560, 32 layers, vocab 50,295, FP16, and `google/siglip-so400m-patch14-384` vision tower. |
| `tokenizer.json` | 2,114,924 | Tokenizer asset. |
| `tokenizer_config.json` | 7,339 | Tokenizer configuration. |
| `vocab.json` | 798,156 | Tokenizer vocabulary. |
| `merges.txt` | 456,318 | Tokenizer merges. |
| `added_tokens.json` | 1,080 | Added token definitions. |
| `special_tokens_map.json` | 441 | Special token mapping. |

No `pytorch_model*.bin`, `model*.safetensors`, or other model weight shard exists in that snapshot.

## Missing artifacts

- **Bunny base weights:** missing from the inspected cache; only config/tokenizer files exist.
- **Recovery output:** no `adapter_config.json`, `adapter_model.*`, `non_lora_trainables.bin`, recovery state file, `trainer_state.json`, or full recovered checkpoint found in the inspected project trees.
- **Bunny training/pruning dataset:** no `bunny_695k.json` or dataset image directory found; `BUNNY_DATA_PATH` is not configured.
- **Validated AWQ result from recovered Bunny:** missing. Existing AWQ artifacts have prune-only or unknown provenance.
- **Evaluation result:** no task accuracy or quality result found. The benchmark harness does not implement an evaluator.

## Interpretation

The smoke script and filesystem sizes provide evidence that a structural pruning output and AWQ outputs were produced previously. They do not establish that the AWQ input was recovered. The only evidence for the smoke script's recovery state is its code, which calls a recovery-skipped stage before AWQ. None of the large payloads were deserialized during this inventory.
