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
| 7 package and final offline guards | docs/mission-7-research-package | 528fb3895e70f0f4fc88f4cdb620bb015722a419 | 276 tests |

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

## Mission 9 superseding integrity handoff

The Mission 8 NO-GO is retained pending independent review. See [repair ledger](docs/reports/MISSION_9_RESEARCH_INTEGRITY_REPAIRS.md) for original failures, root causes, fixes, regressions and limitations. This section supersedes earlier permissive loading/recovery claims.

Dedicated branch: fix/mission-9-research-integrity, based exactly on ba68888. Complete local model_manifest_path and vision_manifest_path are required for real dense evaluation, with matching model identity/revision and fully verified content. Test-generated configs explicitly mark SYNTHETIC_TEST; their measurements remain synthetic. Effective records retain verified manifests and unverified training origin. No assets were acquired.

Encoder loading fails on missing/mismatched tensors; nonfinite state/logits fail before scoring; both multimodal length limits and actual expanded mask/embedding length are enforced. Prediction records retain original/effective counts. Packed exports and loads validate executable dimensions; uniform physical MLP standard reload is covered. Comparison now explicitly reports missing evidence and three comparability states. Actual calibration usage and source/token hashes are retained; raw byte ratios are not research compression claims. Invalid history is isolated and nested suite children discovered.

Unsupported/unverified: full-model compressed multimodal evaluation bridge, heterogeneous standard pretrained reload, LoRA recovery with incomplete lineage, official benchmark scoring, semantic leakage/pretraining contamination, optimized sparse/quantized kernels, real L4 fit/quality/performance. Legacy acquisition sampler fails with a supported-helper instruction. Do not run historical GPU scripts as a substitute.

Next: independently review Mission 9 and rerun CPU regressions; resolve pinned licensed assets, held-out task/scorer, zero-overage/storage policy and separate GPU authorization. Only then consider a small dense diagnostic. No GPU pilot was launched or queued. All real research metrics remain Pending/Not measured.

Mission 9 implementation and validation commit: [92e69f84d74e37acf391cc4e4e409e30f6a483c0](https://github.com/Balaji-s-08-tech/final_year_project/commit/92e69f84d74e37acf391cc4e4e409e30f6a483c0), pushed to [fix/mission-9-research-integrity](https://github.com/Balaji-s-08-tech/final_year_project/tree/fix/mission-9-research-integrity). Final evidence: 305 CPU tests passed, one third-party warning; 179 Python files compiled; frontend type check/build passed. Ledger-only synchronization follows this source commit. NO-GO remains pending independent audit.

## Mission 11 superseding comparison/loader handoff

Branch: fix/mission-11-comparison-loader-integrity, exact parent 09a57c229af79e6c9caff15aa5e7051856f03fc3. See [repair report](docs/reports/MISSION_11_COMPARISON_LOADER_REPAIRS.md) for original reproductions, mandatory evidence, lifecycle admission, portable format contracts, tests and limitations.

Public compare() now validates core evidence, rechecks available asset content under an explicit hash budget, and reports four verification states. Failed/cancelled/incomplete/invalid runs cannot receive COMPARABLE/VERIFIED. compare_collection includes only eligible pairs and discloses every exclusion. Diagnostic-only differences are distinct from admitted research comparisons.

Supported portable exports include declarative architecture/precision headers. Public loader checks headers before trusted pickle construction and reconciles metadata/tensors/attention dependencies before returning a model. Headerless legacy portable files are unsupported; reviewed re-export is required. Pickle remains trusted executable code, not a secure format for arbitrary producers. Actual tensor checks follow deserialization; declared invalid dimensions fail before it.

All original CPU regression tests and seven synthetic treatments remain required. No real research results are introduced. Dense pilot and compression research remain NO-GO pending Mission 12 independent review, exact licensed assets and complete provenance, official held-out protocol/scorer, CUDA environment/fit evidence, zero-overage/storage confirmation and separate execution authorization. No GPU pilot was launched or queued.

Mission 11 implementation: [cbc0732be9afbf6ec3b6fcd6f15ca5120ab0b387](https://github.com/Balaji-s-08-tech/final_year_project/commit/cbc0732be9afbf6ec3b6fcd6f15ca5120ab0b387), successfully pushed to [fix/mission-11-comparison-loader-integrity](https://github.com/Balaji-s-08-tech/final_year_project/tree/fix/mission-11-comparison-loader-integrity). Final validation: 350 CPU tests passed, one third-party warning; 182 Python sources compiled; frontend type checking/build and whitespace checks passed. Documentation-only synchronization follows; NO-GO remains for Mission 12 review.

## Mission 13 superseding handoff

Branch: fix/mission-13-recursive-engineering; exact parent b999e492eed573c6361d15350169fda4c998640c. Eight remediation cycles completed, detailed in [Mission 13 report](docs/reports/MISSION_13_RECURSIVE_ENGINEERING.md). **NO-GO remains for GPU/research execution.**

The four Mission 12 defects were reproduced and repaired. Recursive finite/INT64/binary64 JSON validation prevents overflow/underflow/nested invalid evidence. Q/k normalization flags/shapes/epsilon and general LayerNorm invariants are reconciled. Protocol verification is separate from common valid measured coverage; unmeasured and invalid records are excluded. Pickle loading now requires literal trusted=True at supported public boundaries and has explicit CPU byte budgets (64 MiB default). Safetensors is preferred; weights_only is not treated as permission. CLI adds --trust-local-checkpoint; backend request adds strict trusted_checkpoint=false by default, with a UI acknowledgement that resets when the selected artifact changes. Do not supply trust for unknown files.

Additional repairs bind held-out data hashes/selected IDs to actual budgeted local JSONL/images and preserve per-record history errors. Invalid persisted jobs do not abort backend startup. Effective inference settings are validated against the supported greedy single-beam/single-continuation protocol. Headers reject invalid declarations before payload hashing; actual trusted-pickle objects are reconciled afterward, not sandboxed.

Validation: 409 CPU tests passed, zero failures/skips, one existing third-party warning; 189 Python sources compiled; frontend type check/build and whitespace checks passed. Existing assertions retained; owned-fixture loader invocations migrated to explicit trust and comparison fixtures strengthened with actual synthetic dataset evidence. Seven synthetic treatments remain tested. Evidence levels: IMPLEMENTED / CPU-VALIDATED / SYNTHETICALLY-VALIDATED only. No models/datasets were acquired and no GPU/paid service/infrastructure ran.

Commits so far: 6d27323567956be0fdf51d012f9edcf00a848242 (cycles 1–5), b9b68986d8b4001200b9c5d7df5cd64881e82d6d (cycle 6). Final cycle/synchronization inventory follows. Real model/encoder/tokenizer/projector provenance, official held-out scorer, recovery lineage, full-model compression bridge, CUDA/L4 feasibility and billing/storage/usage limits remain BLOCKED or UNVERIFIED.

Exact resume: verify this branch and recorded final SHA, inspect diff from b999e49 and run the existing CPU suite with CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. Independently review the adversarial regressions before declaring CPU readiness. A next bounded development session should quarantine/review remaining legacy ShortGPT/InternVL/training dynamic-code/acquisition entry points and add general input-size/node-count budgets; they are unsupported and were not executed here. Never run historical training scripts as a substitute for the controlled pipeline. Do not activate GPU/resources or acquire assets without separate authorization after independent audit and financial/protocol review.
