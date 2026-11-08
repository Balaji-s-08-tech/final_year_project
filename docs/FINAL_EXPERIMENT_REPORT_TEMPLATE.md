# Controlled experiment report — fill from inspected actual outputs

## Identity and execution evidence
- Experiment/run ID and immutable output directory:
- Source branch/commit, clean/dirty status, actual source/config hashes:
- Task track: visual question answering OR text generation (never mixed):
- Model/encoder/tokenizer/processor identities and verified manifest paths:
- Base, structural, recovery and quantized checkpoint identities/hashes:
- Dataset/split/revision, official scorer/version, calibration/recovery disjointness:
- Actual host/GPU/driver/Torch/CUDA environment record:
- Billing/overage and execution authorization reference (no secret values):
- Effective prompt, generation, seed, sample order/count, batch and context bounds:
- Warmup/repeat policy, synchronization and timing/memory boundaries:

## Results — no absent value is a zero
Use the versioned benchmark schema. Every measurement needs a value, unit, method,
status and explanatory metadata. Mark missing values Not measured and retain null.

| Experiment | Quality | Latency | Throughput | Peak VRAM | Artifact size |
| --- | --- | --- | --- | --- | --- |
| Dense | Pending | Pending | Pending | Pending | Pending |
| Unstructured | Pending | Pending | Pending | Pending | Pending |
| Structured | Pending | Pending | Pending | Pending | Pending |
| AWQ INT4 | Pending | Pending | Pending | Pending | Pending |
| AWQ INT8 | Pending | Pending | Pending | Pending | Pending |
| Pruning + AWQ | Pending | Pending | Pending | Pending | Pending |

## Interpretation and limitations
- Which outputs, raw samples and logs were actually inspected?
- Which measurements are CPU-validated, synthetic, analytical estimates or GPU-measured?
- Do task, split, effective inference and size definition match across controls?
- Are masks, physical parameter changes and serialized bytes reported separately?
- Are recovered checkpoints actually lineage-verified, or merely shape-compatible?
- Report variability across repetitions; do not select only the best timing.
- Record errors, warnings, failed/skipped cases and any incomplete run snapshots.
- State actual regressions and uncertainty; do not infer kernel speedup from packing.
- Engineering hypothesis, result, alternative explanations and next justified experiment:

Attach small reviewed JSON/CSV and hashes. Keep models, images, datasets, credentials
and large binaries outside Git. Do not claim a real improvement from a synthetic test.
