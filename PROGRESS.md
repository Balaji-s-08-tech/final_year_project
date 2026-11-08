# VLM Compression Optimizer engineering progress

## Preserved milestones

| Milestone | Branch | Commit | Local evidence |
| --- | --- | --- | --- |
| AWQ arithmetic | fix/awq-quantization-correctness-001 | e2fce72ded969a1220569275384860288b9c848c | 119 CPU tests passed |
| Immutable tracking | feat/experiment-tracking-002 | 5a4b1bd6433924748de1e366b7791ddde336da6f | 159 CPU tests passed |

Neither branch has been merged into the baseline. No resets, force pushes,
branch deletion, large asset downloads or GPU execution are part of this session.

## Autonomous session 2026-11-08

Scope: Mission 3 evaluation preparation, then bounded repairs of CPU-testable
research-blocking defects. Operating in Codex Cloud, not Camber. No node activation,
GPU inference, resource provisioning or paid API calls are authorized.

Mission 3 implementation is on feat/baseline-evaluation-003, based on Mission 2.
Hypothesis: explicit adapters, validated protocols and durable evidence can make
baseline execution reviewable and reproducible without pretending to validate
unavailable full models. The real tiny Bunny/SigLIP CPU test exposed a legacy cache
mask mismatch; the evaluator expands visual embeddings/mask once before decoding.

New modules: evaluation configuration, offline compatibility/architecture reports,
local-data validation/fingerprints, read-only memory preflight, typed adapters,
measured runner and opt-in CLI. No new dependencies. Actual full-model metrics
remain unmeasured. Final test results/commit references are recorded below as each
bounded increment completes.

Persistent blockers: Camber billing/overage policy unconfirmed; full model/vision
weights and held-out data absent locally; official task/scorer choice pending;
pruned/recovered multimodal evaluation bridge still requires validation. The
reported L4 capacity does not establish that a recovery or inference run fits.

Next research experiment: after separate authorization and staging, run a small
dense Bunny multimodal baseline with pinned inputs, inspect its actual finite
outputs and memory/timing records, then choose controlled compression comparisons.

### Mission 3 CPU validation completed

183 tests passed (159 preserved plus 24 evaluator tests), none failed or skipped;
one third-party deprecation warning. Full command: `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1 python -m pytest evaluation/tests experiments/tests backend/tests
VLM/quantization/tests -q`. Python compilation, shell syntax and whitespace checks
passed. No large weights/data downloaded. The tiny fixture is random and provides
no evidence of real model accuracy or acceleration.

Draft PR creation is blocked: a read-only GitHub API request returned Forbidden;
CLI authentication also reports its existing binding unusable. No token values were
printed or requested. Native Git push remains the synchronization route; branch
comparison links will be provided rather than claiming a draft PR exists.

### Mission 3 committed and pushed

`bce0b4c504ea82fdb4b3d41e0fe7814e081c12cd` on feat/baseline-evaluation-003.
No merge was performed. The next stacked branch is fix/research-input-validation-004.

### Bunny sampler increment

Repaired duplicate labels, premature return, list image conversion, unbounded
eligibility loops, EOS masking and text-axis slicing. Added a CPU loss/gradient
consumer check and fixed the channel-wise triple/duplicate-forward defect.
Targeted sampler suite: 10 tests. Complete suite: 193 tests passed; none failed or
skipped, one third-party warning. The algorithm has not been run on full Bunny.
See docs/RESEARCH_INPUT_REPAIRS.md for contract corrections and remaining limits.
Next independent task: malformed/stale artifact metadata and recovery-state
validation before model mutation.
