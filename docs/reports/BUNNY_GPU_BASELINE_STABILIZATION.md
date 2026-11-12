# Bunny GPU baseline stabilization implementation

Starting branch/commit: `codex/final-year-research-demo` /
`a43c2e369538ec7db6680333532987e348ea3afc`. Remote branch was fetched and matched this
checkpoint; starting worktree was clean. New branch:
`codex/bunny-gpu-baseline-stabilization`. Existing branches/work were preserved.

## Reviewed implementation

The supplied archive was inspected as external evidence, not executed as task
instructions. The patches identified two valid layout exceptions, but used fixed
421/440 counts. Those constants were not copied into acceptance logic.

`evaluation/checkpoint_layout.py` now reads actual safetensors headers and shard
maps. Single-file precedence matches Transformers. It rejects missing/unsafe shards,
duplicate keys and map/header inconsistencies. Shapes and floating dtypes are
checked against the configured model. Metadata work has finite tensor/shard budgets.

For full SigLIP, the expected complete model is constructed on the meta device,
without allocating its weights. Only its exact configured text/head tensor set
may be unused by SiglipVisionModel; arbitrary `text_model.*` keys are rejected.
Missing text metadata, missing vision tensors, wrong shapes and loader errors fail.
Both ordinary and S2 tower loaders use this validator. Vision-only exports remain
valid without an unused-key exception.

For Bunny's lazy tower, only a complete exact set of standalone vision keys can be
redundant. Shapes and values are checked against the actual selected SigLIP model
in bounded CPU row slices, with exact equality after FP16 conversion. Unknown
language/projector/vision keys or a different same-shape vision checkpoint fail.
The earlier blanket `vision_tower` missing-key filter was removed. The explicitly
trusted nonredundant `.bin` route retains strict Transformers coverage and gets
no safetensors redundancy exception.

Original Transformers warnings remain visible. Stable warning summaries disclose
accepted categories/counts after validation, and evaluation saves
`checkpoint_loading.json`. Missing/mismatched keys and `error_msgs` remain fatal.

`runtime_policy.prepare_deterministic_cuda` sets `:4096:8` before initialization,
preserves valid `:16:8`, rejects invalid values and requires restart if CUDA was
already initialized with no setting. Evaluation, direct adapter loading and live
API memory resets call it after device authorization. CPU calls do not query or
mutate CUDA settings. New `execution_policy.json` records the workspace choice and
requested deterministic policy.

New configs may declare `evaluation_purpose=smoke_test`. A successful finite,
nonempty execution gets a separate smoke summary; semantic correctness and
benchmark validity remain unassessed. Raw exact-match metrics retain their original
definition. A regression verifies that `The square is red.` versus `red` can yield
zero exact match while the execution smoke passes. No historical record is edited.

`scripts/reproduce_bunny_baseline.py` provides explicit pinned acquisition,
local-only preparation and opt-in dense GPU execution. It never acquires assets
or initializes CUDA on import/help/prepare. Preparation hashes local assets,
generates the teaching image, records source assertions and reviews disk/RAM/VRAM
admission. Existing outputs are never overwritten. No downloaded weights or dataset
files are committed.

## Exact verification in this session

All Python tests ran on Python 3.12.14 / torch 2.2.2+cpu with:

```sh
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/workspace/final_year_project/.cache/huggingface
export OMP_NUM_THREADS=1
```

| Command/check | Actual outcome |
| --- | --- |
| `.venv-cpu/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider` | **593 passed**, zero failures/skips, two warnings, 60.85s |
| `.venv-cpu/bin/python -m pytest evaluation/tests/test_checkpoint_layout.py -q` | **25 passed**, 4.91s; includes ordinary/S2, single/sharded, full/vision-only, trusted legacy, missing/extra keys, shapes, value mismatch and warning visibility |
| `.venv-cpu/bin/python -m pytest evaluation/tests/test_smoke_semantics.py evaluation/tests/test_cuda_determinism.py -q` | **7 passed**, 3.21s |
| Reproduction helper tests | Included in full suite: pins, no implicit downloads, local fixture preparation without CUDA, resource reports and output preservation |
| `npm run lint` in Interface | Passed |
| `npm run build` in Interface | Passed; Vite 6.4.6, 1677 modules, 2.48s |
| `DEMO_URL=http://127.0.0.1:3002 CHROMIUM_PATH=/usr/bin/chromium npm run test:demo` | Passed against production build served by Python launcher, with no page errors |
| `py_compile.compile(..., doraise=True)` for all tracked/new Python sources, output in temporary directory | **221 passed** |
| `bash -n` over all tracked shell scripts | **7 passed** |
| `.venv-cpu/bin/python -m pip check` | No broken requirements |
| Reproduction CLI `--help`, `prepare --help`, `run --help` | Passed |
| Generated PNG versus supplied PNG | Byte-for-byte equality confirmed locally |
| Git whitespace/diff checks | Passed |

Warnings retained: existing Starlette/AnyIO BlockingPortal deprecation and PEFT's
random-fixture vocabulary-config notice. API tests required managed-sandbox local
network permission, as in previous validation. No tests were disabled or weakened.

## Files changed

- `evaluation/checkpoint_layout.py`, `evaluation/adapters.py`
- `VLM/bunny/model/multimodal_encoder/siglip/siglip_encoder.py`
- `runtime_policy.py`, `backend/demo.py`
- `evaluation/config.py`, `evaluation/runner.py`
- `evaluation/tests/test_checkpoint_layout.py`
- `evaluation/tests/test_cuda_determinism.py`
- `evaluation/tests/test_smoke_semantics.py`
- `evaluation/tests/test_baseline_reproduction.py`
- `scripts/reproduce_bunny_baseline.py`
- `BUNNY_GPU_BASELINE_REPRODUCTION.md`
- `docs/reports/BUNNY_GPU_BASELINE_EVIDENCE.md`, this report
- `README.md`, `GPU_EXECUTION_GUIDE.md`

## Remaining evidence and next execution

The supplied Kaggle run is successful dense smoke evidence. This revised
implementation has only CPU fixture validation; it has **not been rerun on T4,
L4, or any other GPU here**. The archive has no Git identity and no model tensors,
so this session cannot independently recompute its full 421-tensor equality claim.
Future runs retain new source identity, structured loading checks and new timings.

The observed dense peak is not evidence of compression/training fit. The current
FP32 CPU preparation gate for these assets is 73.586 GiB, above the reported Kaggle
available RAM. Host peak RAM and free disk were not established by that GPU run.
Review space for caches, copies and every stage checkpoint before transformations.

Next: use a fresh CUDA process/kernel, reproduce the pinned dense smoke with this
branch, inspect `checkpoint_loading.json` and new memory/scoring records, then
choose a proper held-out evaluation/scorer and review one treatment's resource
plan. Actual pruning/LoRA/AWQ quality, speedup, official benchmark accuracy and
native Windows CUDA behavior remain unverified. No such experiments were run or
scheduled, and no scientific result is invented.

The user requested a local commit on a new branch. Publication is not performed
by this mission; the final task response provides the actual local commit SHA.
