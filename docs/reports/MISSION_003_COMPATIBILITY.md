# Mission 3 repository compatibility inspection

Inspected on 2026-11-08; source base is Mission 2 commit
`5a4b1bd6433924748de1e366b7791ddde336da6f` (which contains AWQ repair `e2fce72`).

## Source architecture

BunnyPhiForCausalLM integrates the local LLMPruner PhiForCausalLM/PhiModel with
BunnyMetaForCausalLM image insertion. The vision tower produces SigLIP features;
a configurable two-layer GELU projector maps them into the language residual
width. Phi decoder blocks expose shared input_layernorm, attention q/k/v/dense,
and MLP fc1/fc2. The AWQ adapter targets those six language linears, not an
optimized INT4 CUDA kernel. Actual full-model dimensions and parameter counts
cannot be verified without staged weights/configuration.

Multiple legacy builders have different local-only, dtype, device-map, vision
initialization and LoRA behavior. Training imports W&B and its save helpers use
DeepSpeed, while inference does not intrinsically need either dependency. These
paths should not be treated as interchangeable or consolidated without tests.
The new evaluator selects the dense multimodal class directly and checks loading
errors, avoiding the duplicate AutoModel registration for distillation.

## Current asset evidence

Filesystem inspection found **zero weight files** under the current checkout's
checkpoints/, smoke_test_output/, hf_cache/, or /workspace/.cache/huggingface.
No full Bunny, SigLIP, pruned, recovered or portable research checkpoint was
available there. Generated unit-test fixtures under temporary directories are not
research artifacts. The older ARTIFACT_INVENTORY.md describes another machine's
historical outputs and is not current availability evidence.

No held-out multimodal research dataset, official task evaluator, calibration
split manifest, recovery lineage, or remote Camber filesystem was verified. No
network model/data downloads were attempted. Current CPU development can proceed;
full baseline GPU compatibility/quality/performance remain unmeasured.

## Findings and gates

- A tiny local full Bunny+SigLIP forward works on CPU. Cached generation exposed
  a legacy expanded-image attention-mask mismatch on the second decoding token.
  The evaluator adapter supplies expanded embeddings/mask before generation;
  multi-token CPU regression coverage verifies this path. Legacy serving remains
  outside this validation.
- Structured pruning changes shapes and depends on serialized-object reconstruction;
  the dense pretrained loader cannot silently stand in for that checkpoint.
- Portable AWQ may be language-only. Its metadata and actual packed bit widths must
  agree; file names and request parameters are insufficient precision evidence.
- Full-model LoRA reconstruction can require large temporary copies. No claim is
  made that it fits on an L4. Existing recovery-state validation requires further
  hardening before recovered results are accepted as research evidence.
- Existing Bunny pruning sampler has duplicate-label, early-return, image-list
  conversion and sampling-loop defects; these are queued for a separate CPU-tested
  repair branch after Mission 3.

Read-only preflight and resource planning are ready for locally staged assets.
Actual GPU execution is blocked by unconfirmed credit/overage policy, missing
assets, and the pending held-out task/scoring decision. Nothing in this report
authorizes activating a node or spending cloud credits.
