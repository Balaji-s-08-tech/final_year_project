# Autonomous research program — consolidated handoff

## Executive summary and evidence boundary
CPU-compatible engineering is implemented and tested on stacked, unmerged branches.
The previous five audit findings have regression repairs. The framework now has
scoped pruning, portable AWQ validation and a bounded combined experiment suite.
**No full-model research comparison or L4 GPU readiness is established.** All real
research metrics remain Pending. The next external step is a separately authorized
dense baseline pilot, after billing, assets and task protocol are resolved.

## Completed engineering milestones
| Milestone | Branch | Code commit | CPU evidence |
| --- | --- | --- | --- |
| 3.5 integrity/safety | fix/mission-3-5-research-integrity | 6cd00f9e5e6ab0b643c0a2b6d076f76f2625aa72 | 245 tests |
| 3.6 architecture / 3.7 baseline specification | docs/mission-3-6-3-7-protocol | 7bd91943b7bddf9b9046860eacb30c64249dadd8 | 248 tests |
| 4 scoped pruning | feat/mission-4-pruning-framework | 96de78367184f2f90a87e47dfa7d187aa8c705c1 | 259 tests |
| 5 AWQ verification | feat/mission-5-awq-validation | cf40401e8c59e37b32a3b23831e6b9940cb9865c | 264 tests |
| 6 bounded pipeline | feat/mission-6-compression-pipeline | 77e79b4f55b38f469014f28e672663726fec0fe1 | 271 tests |
| 7 package and final offline guards | docs/mission-7-research-package | See branch tip / progress ledger | 276 tests |

All branch URLs use https://github.com/Balaji-s-08-tech/final_year_project/tree/ followed
by the branch name. Ancestry starts at a1b3272, preserving all original milestones.
Main remains 66c1d25252114451532786420b1e92e16de8a778. No merge/force push/branch deletion
or PR creation occurred. Existing PR read access returned Forbidden; PR inventory is
unverified, not assumed empty. All completed code branches were pushed.

## Architecture and functionality
See [architecture/dependency diagram](docs/ARCHITECTURE.md). Explicit local data and
model adapters feed controlled generation/scoring, immutable metrics and result
adapters. Model/encoder/tokenizer/processor fingerprints and effective generation
settings are recorded; missing/large-file identity is explicit and comparison warns.
Model manifests can deliberately verify large local files; default initialization
never performs unbounded full-weight hashing. Small-image fingerprints are exact.

Pruning ([methodology](docs/PRUNING_METHODOLOGY.md)): layer/global stable magnitude
masks on recognized Phi linears; actual zero/mask fractions; masked gradient behavior;
physical paired MLP fc1/fc2 slicing and reconstruction; complete tiny Bunny+SigLIP
forward/generation tests. Embeddings, LM head, vision and projector are preserved.
Physical attention/head/rotary/cache and cross-modal pruning are design-only. Masks
are not physical compression or acceleration evidence.

Quantization ([methodology](docs/AWQ_METHODOLOGY.md)): INT4/INT8 zero-inclusive affine
math, partial groups/odd widths, FP32 scales and source-precision bias, bounded FP32
calibration features, mode/cache/hook restoration, inner/outer metadata and packed
checkpoint verification. Tiny source bias underflow/overflow was repaired during
combined validation. Tensor error is distinct from task quality. Portable layers
dequantize in PyTorch; no optimized INT4 CUDA kernel is supplied.

Combined pipeline ([guide](docs/COMBINED_PIPELINE.md)): dense, masks, physical MLP,
AWQ INT4/INT8, masks→AWQ, MLP→AWQ. It clones small CPU inputs, enforces disjoint IDs,
modality/budgets and shared actual protocol identity, saves versioned artifacts with
hashes, predictions and JSON/CSV. All seven text variants and a tiny complete visual
combined path passed. Recovery is not silently added; prune+AWQ is not recovered
prune+AWQ. Full Bunny is excluded by default CPU fixture budgets.

## Reproduction and validation
The final source suite passed **276 tests, zero failures/skips**, with one existing
Starlette/AnyIO deprecation warning. Frontend TypeScript and Vite production build
passed. Python compilation, shell syntax and whitespace checks passed. Earlier
inspection identified defects; tests were added rather than deleting assertions.
One sampler truncation assertion was corrected to require rejection of a one-token
sequence (no causal targets); it still tests batch-axis preservation with valid data.

Use the existing Python 3.11 project environment. For an independently reviewed fresh
Linux CPU setup, requirements-cpu.lock records the actual dependency versions and
CPU Torch index; it is a version lock, not a wheel-hash supply-chain lock. Do not use
that CPU lock as a CUDA deployment environment. No packages were installed in this
program. docs/CPU_ENVIRONMENT.json records actual versions without secret values.

```bash
cd /workspace/final_year_project
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HOME=/workspace/.cache/huggingface .venv-vlm-awq/bin/python -m pytest -q
cd Interface
npm run lint
npm run build
```

Synthetic CLI (local random weights only, seven treatments):
```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-vlm-awq/bin/python -m compression --config configs/compression/synthetic_suite.json --output checkpoints/synthetic_experiments
```
Every call creates a unique suite. The final configured CLI check reloaded all seven
artifacts and verified finite CPU forward outputs. Do not present its random-fixture
accuracy/container sizes as research model benefits. No full weights/datasets were
downloaded. Test fixtures and manual evidence are outside source control.

## Artifact and configuration contracts
Model manifests: [baseline protocol](docs/BASELINE_PROTOCOL.md), local manifest tools
and JSON templates under configs/evaluation. Dataset manifests include explicit
revision/split/purpose, selected IDs, data/image fingerprints. Selected disjointness
checks cannot establish absence of semantic duplicates or pretraining contamination.
Full model/vision/tokenizer revisions remain unavailable; templates with REPLACE_*
identities are deliberately rejected. Fill them only from verified acquired assets.

Tracked runs retain initial config/environment, numbered lifecycle states, logs,
metrics/status/units/methods, effective inference and provenance. Final snapshots
are authoritative after crashes. Legacy and structured result records are read by
API/UI without turning nulls into zeros. Imported claims stay labeled imported.
Pipeline format vlm-compression-v1 is a trusted Python object container with separate
SHA manifest, not a portable cross-version secure deployment format. AWQ portable
exports keep their existing versioned format/content-bound sidecars. See
[tracking specification](docs/EXPERIMENT_TRACKING.md) and published JSON schemas.

## Resource policy and first L4 pilot
Zero personal spending remains absolute. No cloud SDK activation, paid API,
subscription change, resource increase, GPU workload or credential extraction ran.
Shared admission rejects unauthorized CUDA before context calls; API jobs are
CPU-only. Auto defaults to CPU. Local acknowledgement flags cannot prove billing
safety. Legacy remote model/encoder/dataset acquisition is now rejected unless
explicit local assets exist; the old full-model smoke path is disabled. These are
local controls, not a sandbox for arbitrary third-party Python or trusted pickles.

Before any external pilot, the user must:
1. Confirm Camber credit exhaustion/overage and persistent-storage charging policy.
2. Separately authorize node/GPU execution and any asset acquisition/storage costs.
3. Stage licensed, pinned Bunny, matching projector/tokenizer and local SigLIP
   config/weights/image processor; verify manifests. Actual assets are absent here.
4. Select held-out split, official scorer, prompts/answer extraction and leakage
   protocol; do not mix text-only and visual task scores.
5. Review disk headroom within the actual 50 GB quota and free host/GPU memory.
   Proposed 24 GiB RAM / 16 GiB VRAM admission minima are not allocation caps or fit
   evidence. Validate the installed CUDA Torch wheel, not nvidia-smi's CUDA label.

The following commands are specified for a human on an already authorized Camber
machine; **none were run here**. They do not activate infrastructure:
```bash
# First edit configs/evaluation/bunny_l4_baseline.json into /mnt/config/pilot.json
# with actual pinned identities, split/scorer protocol and existing local paths.
python -m evaluation --config /mnt/config/pilot.json --output /mnt/reports/new-preflight.json
# Only after separate billing/GPU authorization and a reviewed passing preflight:
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evaluation --config /mnt/config/pilot.json --output /mnt/runs/experiments --execute --authorize-gpu-execution
```
Use new report paths. The read-only report says PREFLIGHT_ONLY/NOT_PROVEN, never GPU
ready. Inspect a small dense fit/finite-output/multi-token pilot before expanding
sample counts or running compression. Save actual outputs, memory/timing samples,
software versions and any errors. The default protocol is normalized exact match,
not a substitute for an official named task scorer. Review that choice first.

## Remaining limitations and blocked work
- No real checkpoint/dataset, official scorer, full-model recovery manifest or L4
  compatibility/quality/performance evidence. Large asset hashes can remain incomplete. Set hash_large_assets=true only after a deliberate local I/O budget review; the runner verifies before and after loading. Dataset split is explicit and required before GPU admission.
- LoRA/non-LoRA state completeness and real pruned multimodal reconstruction require
  staged assets; strict full-state coverage alone is not verified training lineage.
- Physical head/vision/projector pruning and optimized sparse/INT4 kernels are unsupported.
- GPU utilization sampling, official perplexity/TTFT/task scoring and meaningful
  statistical comparisons require separate reviewed implementation/protocols.
- Local POSIX single-worker tracking is not distributed scheduling; PID reuse,
  orphan supervision and multi-file crash atomicity remain documented limits.
- Duplicate legacy loaders, inactive static UI presentation code and unsupported
  LLaVA/InternVL paths were not broadly rewritten. Do not treat them as validated.
- Third-party trusted pickle/callback code can execute arbitrary operations; local
  policy is not a security sandbox. Only reviewed entry points are in scope.

## Research comparison matrix — no real results
| Experiment | Quality | Latency | Throughput | VRAM | Size |
| --- | --- | --- | --- | --- | --- |
| Dense | Pending | Pending | Pending | Pending | Pending |
| Unstructured pruned | Pending | Pending | Pending | Pending | Pending |
| Structured pruned | Pending | Pending | Pending | Pending | Pending |
| AWQ INT4 / INT8 | Pending | Pending | Pending | Pending | Pending |
| Pruned + AWQ | Pending | Pending | Pending | Pending | Pending |

The next meaningful external experiment is the dense authorized pilot, not a claim
that all full-model compression/GPU validation is complete.
