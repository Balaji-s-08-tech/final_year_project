# Final-year engineering and demo delivery

Branch: `codex/final-year-research-demo`, based exactly on
`4b07bd7c3c593caffee345a5912327bff54fc832`. The final task response records commit
SHAs and publication outcome. Original development/review/main branches are preserved.

## Delivered implementation

- Declarative `bunny-treatment-v1` safetensors export/reconstruction for dense,
  masked, physically MLP-pruned, recovered, INT4/INT8 and combined language blocks.
  Base/tokenizer/processor/encoder/projector identities remain bound. Complete
  multimodal image-token expansion and generation are retained.
- Treatment labels are reconciled with actual widths, masks and packed storage.
  CPU regressions reject false compression labels, incorrect base/protected
  identity, corrupt weight hashes and incomplete module declarations.
- Optional loopback image-question API with server-owned registered images/model
  configs, bounded strict requests, GPU preflight/admission, single-flight execution,
  generation/error handling, actual identity and memory/timing responses.
- Existing React app now defaults to a backend-independent research presentation:
  architecture, interactive pruning/INT4 explanations, LoRA, unavailable research
  comparisons, actual separately labeled CPU text-fixture measurements, stored
  teaching image/question and provenance. Original engineering console remains.
- Cross-platform standard-library launcher serves a built demonstration without
  ML dependencies. Optional backend starts only when requested.
- Local preparation CLI provides ordered dense/prune/recover/AWQ stages with
  byte/host-memory admission, complete safetensors checkpoints, hashed plan/input
  identity, host observations, resume validation and optional CUDA evaluation.
  Preparation is CPU-only; GPU evaluation retains actual CUDA peak memory logging.

## Exact validation performed

CPU environment: Python 3.12.14, torch 2.2.2+cpu. Tests used
`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HF_HOME=/workspace/final_year_project/.cache/huggingface OMP_NUM_THREADS=1`.

| Command/check | Actual result |
| --- | --- |
| `.venv-cpu/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider` | **558 passed**, zero failures/skips, two warnings, 52.68s |
| `.venv-cpu/bin/python -m pytest evaluation/tests/test_multimodal_treatments.py compression/tests/test_execute.py backend/tests/test_demo.py -q` | **16 passed**, one warning, 9.27s |
| `.venv-cpu/bin/python -m pytest evaluation/tests/test_multimodal_treatments.py -q` after final declaration admission regression | **12 passed**, 6.50s |
| `.venv-cpu/bin/python -m pytest backend/tests/test_demo.py -q` after registry validation | **3 passed**, one warning, 3.96s |
| `npm run lint` from Interface | TypeScript passed |
| `npm run build` from Interface | Vite 6.4.6 passed; 1677 modules, 2.69s |
| `DEMO_URL=http://127.0.0.1:3001 CHROMIUM_PATH=/usr/bin/chromium node tests/demo.browser.cjs` from Interface | Production UI passed through Python launcher; offline navigation/sliders, unavailable metrics, nine measured teaching rows, image, disconnected inference, mocked live UI contract, no page errors |
| All tracked/new Python sources with `py_compile.compile(..., doraise=True)` to temporary bytecode | **215 files passed** |
| `bash -n` for all tracked shell scripts | **7 passed** |
| `.venv-cpu/bin/python -m compression.execute --help` | Passed |
| `.venv-cpu/bin/python scripts/launch_demo.py --help` | Passed |
| `.venv-cpu/bin/python scripts/local_preflight.py` | Passed; CPU runtime, no GPUs, memory feasibility NOT_PROVEN |
| `.venv-cpu/bin/python -m pip check` | No broken requirements |
| `git diff --check` | Passed |

Starlette's AnyIO BlockingPortal deprecation and PEFT's random-fixture vocabulary
configuration notice remain visible. No test was disabled or weakened.

During implementation the first treatment reload tests exposed stale Phi MLP
configuration after slicing; reconstructing `intermediate_size` fixed it. Resume
testing exposed use of config on a newly constructed adapter; preserving the
restored config before replacing the instance fixed it. Both cases remain covered.
The first browser launch lacked Playwright's bundled headless browser; the installed
system Chromium completed the tests. API tests require local socket permission in
the managed sandbox. These are recorded execution facts, not hidden test exclusions.

## Readiness and limits

| Area | Evidence/status |
| --- | --- |
| Code readiness | Supported integration and demo implemented; complete CPU regression suite passes |
| CPU validation | Complete tiny Bunny/SigLIP architecture fixtures, numerical/storage contracts, actual image API, resume and browser behavior validated |
| GPU execution preparation | Runnable local preflight/preparation/evaluation commands, budgets, checkpoints and guides provided; target runtime must still be validated |
| Actual GPU validation | **Not performed** |
| Research completion | **Not established**; no real pretrained-model accuracy, latency, VRAM fit or compression gains measured |

This machine reported approximately 9 GB available RAM and 6 GB free disk, no CUDA
runtime and no GPUs. Staging/running the full pretrained model is outside its
verified capacity. No model weights or datasets were acquired.

Remaining GPU/data-dependent work: licensed pinned real assets; official held-out
protocol/scorer and leakage review; measured L4/other CUDA fit and quality; actual
dense/compressed comparisons. Preparation recovery uses explicit **text-only**
supervision; image-supervised recovery, optimizer-step resume, physical attention/
vision pruning and optimized INT4 kernels are not supplied. Resume covers completed
stages, not arbitrary interrupted writes. Generic research comparison remains
conservative about transformed asset/effective metadata; no automatic scientific
comparability is claimed. Native Windows execution is unverified; the launcher
uses cross-platform Python primitives and was tested on Linux.

The bundled fixture snapshot contains real tiny random text measurements captured
with dirty source, disclosed in the UI. Its duration includes preparation; it is
not GPU latency or Bunny research quality. Content hashes establish consistency,
not artifact-producer authentication. Keep assets under reviewed local ownership.

## Delivery documents

Implementation commits:

- `f272912`: verified multimodal treatment reconstruction and image inference API.
- `c9ebe42`: bounded local preparation, stage checkpoints and resume.
- `c853af2`: offline React presentation, stored fixture snapshot and browser regression.

Changed files: `backend/demo.py`, `backend/main.py`, `backend/tests/test_demo.py`,
`compression/multimodal.py`, `compression/execute.py`, `compression/recovery.py`,
`compression/tests/test_execute.py`, `evaluation/adapters.py`, `evaluation/config.py`,
`evaluation/inspection.py`, `evaluation/provenance.py`, `evaluation/runner.py`,
`evaluation/tests/test_multimodal_treatments.py`, `configs/compression/local_preparation.json`,
`scripts/local_preflight.py`, `scripts/launch_demo.py`, `scripts/export_demo_fixture.py`,
`Interface/src/App.tsx`, `Interface/src/services/api.ts`,
`Interface/src/components/demo/ResearchDemo.tsx`, `Interface/public/demo-red-square.svg`,
`Interface/public/fixture-results.json`, `Interface/tests/demo.browser.cjs`,
`Interface/package.json`, `Interface/package-lock.json`, `README.md`,
`requirements-local-gpu.txt` and the five documents below. Pre-publication scan
reviewed all 32 changed paths and found no credential patterns, model/data payloads
or files above 2 MiB; the largest changed file is the 168 KB npm lockfile.

- DEMO_README.md: offline and optional live demonstration launch.
- GPU_EXECUTION_GUIDE.md: local CUDA prerequisites, commands and recovery limits.
- EXPERIMENT_REPRODUCTION.md: CPU verification and actual input contracts.
- MODEL_ARTIFACT_MANIFEST.md: treatment schema, reconstruction and identity checks.
- This file: implementation, exact validation and remaining evidence boundaries.
