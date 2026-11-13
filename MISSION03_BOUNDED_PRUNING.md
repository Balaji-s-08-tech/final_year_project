# Mission 03: source-bound full-model structured pruning

Base `4c74729c6bfc2a5beb0f91c294ec5867a1e0c3cc`; development branch
`codex/mission03-bounded-pruning`. No pretrained multi-layer/full-model pruning
or GPU experiment was executed in this session. Commands below are proposed
operator actions, not experimental results.

## Evidence boundary and design

The user reports dense pretrained Bunny/SigLIP smoke latency 3.146983278 s,
allocated CUDA peak 7,141,774,848 bytes and prediction ` The square is red.` with
unchanged exact match 0/1. The layer-0 pretrained operator pilot reports width
10240→7680, masked-reference maximum error 3.337860107421875e-06, probe RMSE
0.026723869144916534, process lifetime peak RSS 887,635,968 bytes and saved artifact
157,327,680 bytes. These are supplied observations, not reruns or VQA quality
evidence. The operator probe is synthetic, not a held-out quality benchmark.

Implementation inspection covered `compression/pruning.py`, `execute.py`,
`multimodal.py`, `recovery.py`, AWQ quantization/calibration/portable loading,
checkpoint admission, evaluator config/preflight/adapters/runner, source manifests
and actual Bunny MLP/vision paths. General preparation retains an eight-times-
weight-file RAM gate and full FP32 stage snapshots; neither is suitable as a
drop-in full-model workflow on the supplied disk. Those gates remain unchanged.

The new path stores a **base-bound recipe**, not transformed weights. Compilation
reads one source MLP at a time using safetensors slices, applies the original
float64 mean-absolute fc1-row + fc2-column ranking with stable argsort, and records
sorted kept channels plus hashes of every selected tensor. Sparsity is fixed at
25%, layer distribution; attention, vision, projector, embedding and LM head are
untouched. No calibration data, recovery or AWQ is silently added.

The source manifest binds model identity/revision and all local source files;
the original config is hash-bound. Complete file hashes are verified before actual
compilation/reconstruction. As elsewhere, local hashes prove consistency, not
remote authenticity or safety of an adversarial filesystem; use reviewed owned
immutable assets from the pinned baseline.

## Format, completion and reconstruction

Output files:

- `plan.json`: immutable source/config/algorithm/layer-selection plan.
- `layers/0000.json` through `0031.json`: per-layer kept indices, original/remaining
  width, source dtype and content/shape/dtype-bound hashes of four resulting tensors.
- `recipe.json`: bounded references to all expected layer receipts and their hashes.
- `treatment.json`: final format/treatment declaration and canonical recipe digest.
- `compilation_resources.json`: process RSS/lifetime high-water observation;
  includes import/hash work and is not an isolated per-layer allocation delta.

Per-layer files preserve the existing 100,000-node JSON limit: 32×7680 indices
would exceed that limit as a single list. The format does not relax global parser
budgets. A 64 MiB directory profile and 64 MiB free-disk admission cover metadata,
receipts and pending writes. Exact directory bytes are inventoried at load time.

Compilation prevalidates dimensions and per-MLP FP32-equivalent size (≤256 MiB),
requires ≥3 GiB available RAM, verifies source content, then processes layers
incrementally. It keeps no whole-model tensor dictionary or FP32 model copy.
Source mappings close after each layer, and selected tensors own their storage;
mapped source pages are not retained across the entire compilation pass.
Receipts use exclusive atomic publication. Interruption leaves inspectable
receipts; `--resume` requires the identical plan/source and recomputes/verifies
each completed receipt before proceeding. Missing/corrupt completion markers or
inconsistent final files fail closed. No output or source is overwritten.

A `--pilot-layers` artifact is `PILOT_ONLY` and cannot be evaluated as a full VLM.
A full artifact requires every configured layer, exactly once, in order. Full
evaluation uses `treatment=structured_pruning` and `treatment_artifact_path` pointing
to the completed recipe directory.

Reconstruction retains the proven dense baseline loading and Bunny/SigLIP coverage
checks. It loads one original CPU model using the evaluation dtype, then replaces
MLP pairs one layer at a time from the original source tensors. It independently
recomputes kept indices and reconstructed hashes; edited selections or wrong
weights fail even if a recipe's outer hash is regenerated. fc1 rows/bias and fc2
columns are paired; fc2 bias is unchanged. Replacement parameters preserve source
dtypes. The final explicit runtime dtype/device conversion follows the existing
evaluation config and is reported separately.

The loader initially uses the **original** configuration. Setting intermediate
width to 7680 before reading dense 10240-wide weights would be incorrect. After
all replacements it records actual per-block widths and updates uniform
`intermediate_size`. The current supported policy is uniform 25% pruning of a
uniform original Phi configuration; arbitrary heterogeneous pruning policies are
rejected rather than guessed. Tests cover explicit module reconstruction across
all 32 blocks, including source dtypes and subsequent visual generation.

## Recipe versus sharded streaming weight export

| Property | Recipe (implemented) | Safe sharded weight export (future alternative) |
| --- | --- | --- |
| Additional disk | Kept indices/receipts/hashes, typically several MiB; admitted profile 64 MiB | About 4.20 GB / 3.91 GiB for FP16 pruned language blocks, plus headers/index/temporary shard space |
| Source dependence | Exact original Bunny and matching standalone vision assets remain required | Can replace language-block source dependence only with complete protected asset packaging/lineage |
| Working set | One MLP pair plus float64 ranking/slice/hash work | One input/output shard and ranking scratch if implemented carefully |
| Reload | Recompute and verify slices against pinned source; no full block payload load | Requires verified shard index, exact global coverage and atomic completion |
| Integrity | Hash-linked plan, receipts, selected tensors, source manifest | Must bind every shard, global tensor inventory, config and protected assets |
| Chosen now | Fits limited disk and preserves source precision | Deferred: 5.75 GiB free leaves limited headroom for retries and retained evidence |

No streaming weight export is claimed implemented. Existing full FP32 export is
still about 7.816 GiB for pruned language blocks and cannot fit the stated free disk.

## Analytical budgets and observations

Architecture inputs from the pinned baseline: H=2560, I=10240, L=32,
P=3,212,732,480 registered parameters. Source files total 9,876,575,208 bytes
(9.198 GiB), including redundant/unused checkpoint tensors. The 25% policy removes
2560 MLP channels per block and 419,512,320 parameters, leaving 2,793,220,160.
GiB means 2^30 bytes. Values below are estimates unless explicitly marked supplied
measurements. Buffers, allocator behavior, mappings/page cache, contexts and other
processes prevent tensor counts from proving an RSS/VRAM peak.

| Stage | Host RAM | VRAM | Additional disk |
| --- | --- | --- | --- |
| Metadata plan | Header/JSON work; no model tensors | 0 | 0, stdout only |
| Two-layer pilot / 32-layer compilation | One pair: original source FP16 ~0.098 GiB, float64 ranking scratch ~0.391 GiB, selected tensors/hash buffers; reserve 3 GiB available | 0 | Metadata under 64 MiB profile; no full weights |
| Full reconstruction admission | `max(config RAM budget, 2×base+vision weight-file bytes + 3 GiB)` = approximately **21.40 GiB available** for these assets | No CUDA context before ordinary authorized evaluation setup; CPU model rebuilt before transfer | No transformed weight payload |
| FP16 deployed pruned parameters | One CPU model incrementally rebuilt; runtime telemetry records parameter/buffer bytes separately | **5,586,440,320 parameter bytes ≈5.203 GiB**, plus unmeasured activations/workspace/cache; retain ≥10 GiB free-VRAM admission for pilot | Original assets + metadata + small run evidence |
| FP32 CPU evaluation alternative | 11,172,880,640 parameter bytes ≈10.406 GiB plus loading/ranking/runtime overhead | 0 | Same dependencies |
| Existing full FP32 snapshot path | Original general gate **73.59 GiB** remains; clones/reload overlap are unchanged | Separate old path | **7.816 GiB** pruned block payload alone |

The supplied dense GPU allocated peak is 6.651 GiB. Do not subtract parameter
savings from that number and call the result a measured pruned peak or assume a
6 GB GPU fits. Two T4 memories are not pooled; this path uses one configured device.
The earlier single-operator lifetime RSS of ~0.827 GiB is encouraging for bounded
compilation, but is not evidence of a 32-layer reconstruction peak.

Evaluation records `deployment_storage.json`: entire recipe-directory bytes,
base/vision dependency weight bytes and actual runtime parameter bytes. Treatment
identity also records runtime buffer bytes/dtypes. `compressed_model_size` retains
dependency-inclusive storage accounting. Recipe bytes cannot be reported as
deployed model bytes or an end-to-end compression ratio. CUDA allocated/reserved
peaks still require real execution and the existing evaluator's measurements.

## Staged Kaggle plan — each execution is a separate operator decision

Use the reviewed CUDA environment and already verified pinned assets. Commands
assume these existing paths; adjust them to the actual restored paths, never
redownload/copy a second model to make this work. Use a fresh process and new
output names. Neither this report nor CPU tests authorize a GPU run automatically.

Required sources remain `BAAI/Bunny-v1_0-3B` revision
`0958a104bd39a448fc310b3d1b1fc3d398dc3fcb` and
`google/siglip-so400m-patch14-384` revision
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`. The standalone vision directory is
`/kaggle/working/pinned-baseline-assets/siglip`; retain both original manifests
from `/kaggle/working/baseline-smoke-config`. Review actual free host RAM/disk
without allocating model weights or initializing CUDA:

```sh
python - <<'PY'
import json, shutil, psutil
print(json.dumps({
    'available_ram_bytes': psutil.virtual_memory().available,
    'free_output_disk_bytes': shutil.disk_usage('/kaggle/working').free,
    'compiler_required_available_ram_bytes': 3 * 1024**3,
    'compiler_required_free_disk_bytes': 64 * 1024**2,
}, indent=2))
PY
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

### Stage A: completed CPU fixture validation

Run the new regressions and existing complete suite as documented below. Fixtures
use narrow 32-block models and a separate real-width selection-count test
(32×7680 indices). They are not pretrained quality evidence.

### Stage B: next minimal pretrained CPU pilot, exactly layers 0 and 1

Read-only metadata/resource plan (does not hash payloads or write output):

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
python -m compression.recipe \
  --model-dir /kaggle/working/pinned-baseline-assets/bunny \
  --manifest /kaggle/working/baseline-smoke-config/bunny_manifest.json \
  --output /kaggle/working/mission03-layers01 --pilot-layers 0 1
```

After reviewing the plan and source paths, explicit CPU execution:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
python -m compression.recipe \
  --model-dir /kaggle/working/pinned-baseline-assets/bunny \
  --manifest /kaggle/working/baseline-smoke-config/bunny_manifest.json \
  --output /kaggle/working/mission03-layers01 --pilot-layers 0 1 --execute
```

Inspect both receipts and `compilation_resources.json`. Compare layer-0 kept
indices with the prior successful operator pilot's `pruning.layers[0].kept_neurons`.
Source file hashes and selection policy must agree. This pilot is deliberately
not a complete treatment; evaluation must reject it. Check RSS/headroom before
approving full compilation. Hashing all source files adds sequential disk I/O.

### Stage C: full recipe and reconstruction preflight, after Stage B review

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
python -m compression.recipe \
  --model-dir /kaggle/working/pinned-baseline-assets/bunny \
  --manifest /kaggle/working/baseline-smoke-config/bunny_manifest.json \
  --output /kaggle/working/mission03-full-recipe --execute
```

For interrupted compilation use the identical command with `--resume`; do not
delete completed receipts or rewrite the source manifest. Create a **new local
evaluation config** copied from the successful dense smoke config. Retain the
original model/vision paths/manifests/identities and prompt/settings; set:

```json
{
  "treatment": "structured_pruning",
  "treatment_artifact_path": "/kaggle/working/mission03-full-recipe",
  "evaluation_purpose": "smoke_test",
  "required_free_ram_bytes": 21474836480,
  "required_free_vram_bytes": 10737418240
}
```

These are edits to the copied config, not a complete standalone config. Keep
device `cuda:0`, dtype `float16`, batch 1, one red-square sample, max new tokens 16,
the original prompt and zero warmups for this diagnostic comparison. Preflight:

```sh
python -m evaluation --config /kaggle/working/mission03-pruned-smoke.json \
  --output /kaggle/working/mission03-preflight.json
```

It verifies complete recipe structure/source layout and reports the derived RAM
admission requirement without loading model weights or proving GPU fit. Payload
source validation and exact reconstructed values are checked on actual loading.
Inspect `recipe_reconstruction_required_ram_bytes`, all errors and free device
memory; do not lower thresholds just to get a passing report.

### Stage D: one GPU smoke only after explicit authorization

Fresh kernel/process; set deterministic cuBLAS before any CUDA initialization:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
python -m evaluation --config /kaggle/working/mission03-pruned-smoke.json \
  --output /kaggle/working/mission03-pruned-runs --execute --authorize-gpu-execution
```

Do not set CUDA_VISIBLE_DEVICES empty for this separately approved stage. Inspect
all 32 widths=7680, source/recipe identities, generation tokens, raw exact-match
score, runtime parameter bytes, GPU allocated/reserved peaks and dependency storage.
A valid execution can still have zero exact match. No accuracy retention follows.

## Held-out VQA protocol before quality research

Choose a licensed official VQAv2 validation subset with original question IDs and
ten human reference answers per question. Freeze an actual dataset revision or
annotation-content digest and deterministic list of at least 128 question/image
IDs for a **pilot**, with image hashes. A full benchmark/statistical power decision
remains separate. Do not invent a revision or fetch assets automatically.

Weight-only magnitude pruning needs no calibration or training data. Reserve any
later AWQ calibration/LoRA recovery examples from a disjoint training split; check
image IDs, question IDs and content hashes, and review semantic duplicates. Never
choose pruning thresholds or extraction rules after inspecting held-out answers.

Re-evaluate dense and pruned models using the same fixed subset, image processing,
prompt, greedy single-beam generation, dtype, seed, bounds and hardware. A proposed
short-answer instruction must be preregistered and applied to **both** models;
the earlier red-square baseline cannot be reused as its quality baseline.

Use the official VQAv2 evaluator/normalization with a pinned code revision: average
over reference annotators of min(matches among the other nine / 3, 1), with its
official punctuation, number/article and contraction processing. Preserve raw
answers and keep answer extraction fixed. Current repository normalized whole-
answer exact match is diagnostic and is **not** the official consensus score.
Official scorer integration/annotation validation remains a blocker; no new
official VQA score is fabricated by this engineering delivery.

Report absolute consensus scores and paired differences with image-aware paired
bootstrap intervals, explicit sample count and failures. Define retention only
after an adequately supported nonzero dense score; do not divide by the red-square
0/1 score. For latency use identical warmups and repeat runs, synchronized timing,
token counts and context bounds. Do not infer kernel acceleration from parameter
count. LoRA and AWQ are subsequent independently versioned treatments with their
own disjoint data/provenance, not silently included recovery steps.

## Rollback and evidence export

Every output directory must be new and outside source assets. Missing completion
markers, inconsistent receipts, source/config changes or insufficient resources
stop execution. Keep partial files and stderr; use `--resume` only for an identical
source/plan, otherwise use a fresh numbered directory. Never modify an existing
baseline config/run, delete assets to gain space, or call a failed run successful.
On reconstruction failure discard the in-memory adapter/process; source files and
recipe remain unchanged. Roll back operationally by using the original dense
config with no treatment path. No branch reset/merge is needed.

Keep the recipe directory, preflight, compilation resource observation and the
new evaluator run's config/source/environment/loading/effective/prediction/metric
files. Export only those small evidence directories; exclude pretrained weights,
datasets, credentials and caches. Example after successful stages, protecting an
existing archive via shell noclobber:

```sh
( set -C; tar -cz -C /kaggle/working mission03-layers01 mission03-full-recipe \
    mission03-preflight.json mission03-pruned-runs \
    > /kaggle/working/mission03-evidence-001.tar.gz )
```

Use only completed/appropriate paths; retain failures separately. Clean up only
reviewed new pilot output directories after preserving evidence, never broad
workspace/model/dataset/baseline roots. Artifact digest consistency is not an
independent signature of the producer.

## Remaining blockers

No pretrained two-layer/full-32 recipe compilation, actual full-model reconstruction
peak, new Kaggle GPU smoke, held-out VQA consensus score, latency benefit or
accuracy retention has been measured here. Native CUDA platform/version validation
and official scoring remain user-side prerequisites. Heterogeneous pruning
policies, sharded standalone weight export, optimized INT4 kernels and a
memory-reviewed full-model recovery/AWQ sequence remain future work.

## Delivered CPU validation

Executed in the CPU workspace with Python 3.12, Torch 2.2.2+cpu and Transformers
4.37.2. Hugging Face offline mode and an empty CUDA device list were enforced.
No pretrained assets were acquired or processed. The final complete regression
command was:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HOME=/workspace/final_year_project/.cache/huggingface OMP_NUM_THREADS=1 \
.venv-cpu/bin/python -m pytest backend/tests VLM/quantization/tests \
  evaluation/tests experiments/tests compression/tests cloud/tests \
  LLM-Pruner/tests -q -p no:cacheprovider
```

Result: **648 passed, 2 warnings, 67.73 seconds**. Warnings were the existing
Starlette/AnyIO deprecated alias and PEFT's tiny fixture vocabulary/config
assumption. The full suite includes 20 new recipe cases. Separate targeted
validation using the same offline environment and `python -m pytest
compression/tests/test_recipe.py -q -p no:cacheprovider` returned **20 passed in
8.78 seconds**. Tests cover full 32-block synthetic visual generation,
all four exact source selections and protected weights, source dtype retention,
real-width metadata counts, missing/bad dimensions, config/recipe/selection/source
corruption, memory-window lifetimes, interrupted resume, partial-treatment
rejection, disk/RAM admission before payload reads/model allocation, oversized
metadata rejection before hashing and dependency/residency metric separation.
These tests establish engineering properties, not pretrained VQA quality.

All **229 tracked/new Python files** compiled using `py_compile.compile` with
temporary output paths outside the repository. `python -m compression.recipe
--help` and `git diff --check` passed. Changed-file review found only seven Python
or Markdown files (each below 1 MiB), no model/dataset/checkpoint artifacts or
secret-pattern matches. The verified baseline checkpoint validator, vision loader,
source configuration files, existing experiments, general pruning/recovery/AWQ
implementations and historical evidence were not changed.

Delivered files: `compression/recipe.py`, `compression/tests/test_recipe.py`,
`evaluation/adapters.py`, `evaluation/inspection.py`, `evaluation/preflight.py`,
`evaluation/runner.py` and this report. Recipe-aware admission/reconstruction is
opt-in through the treatment declaration; existing treatment paths retain their
previous implementation.
