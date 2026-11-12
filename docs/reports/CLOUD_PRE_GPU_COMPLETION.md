# Cloud CPU engineering validation

Base: `fee24fb50b142d00beb7ebb363167270eb45613b` (Mission 15).
Development branch: `codex/cloud-pre-gpu-completion`; `review/mission-15` is preserved.
Final commit identity is available from `git rev-parse HEAD`; this report does not
embed a self-referencing commit SHA.

## Implemented changes

- Global magnitude ranking no longer casts float64 weights to float32 and loses
  ordering. Physical MLP importance uses float64 accumulation to avoid FP16 overflow.
- AWQ clip-shrink configuration rejects booleans and nonnumeric values with ValueError.
- Tracking imports on Windows without `fcntl`, uses byte-range locking there and
  retains POSIX flock/fsync behavior. Spawned-process serialization and exclusive
  no-overwrite publication are tested on Linux. Native Windows remains unverified.
- Bounded CPU Phi LoRA recovery validates batches before hashing/allocation,
  freezes the base, checks losses/gradients/state, preserves source and RNG,
  and admits rank/parameter budgets. Safetensors adapter save/reload and merged
  prediction equivalence are tested against the same physically pruned fixture.
- Pipeline adds optional `structured_lora` and `structured_lora_awq`, with explicit
  recovery IDs/content hashes and held-out overlap checks. The default seven
  treatments remain available. `--with-recovery` executes nine synthetic treatments.
- npm package lock enables `npm ci`; Python 3.12 CPU development requirements and
  `LOCAL_GPU_EXECUTION_GUIDE.md` provide reproducible commands and scope boundaries.

Changed source/tests: `compression/pruning.py`, `compression/recovery.py`,
`compression/pipeline.py`, `compression/__main__.py`, `experiments/filesystem.py`,
`experiments/store.py`, `VLM/quantization/awq_config.py`,
`compression/tests/test_pruning_precision.py`, `compression/tests/test_recovery.py`,
`compression/tests/test_awq_config_types.py`, `experiments/tests/test_filesystem.py`.
Setup/docs: `Interface/package-lock.json`, `requirements-cpu-dev.txt`, `README.md`,
`LOCAL_GPU_EXECUTION_GUIDE.md`, this report.

## Actual commands and outcomes

Fresh environment initially lacked torch, transformers and pytest. Created
`.venv-cpu` and installed CPU dependencies using the official CPU PyTorch index.
Installed torch 2.2.2+cpu/torchvision 0.17.2+cpu and selected Python 3.12-compatible
dependency versions now recorded in `requirements-cpu-dev.txt`. No model or dataset
was downloaded. `python -m pip check` in that environment: no broken requirements.

Test environment for the commands below:

```sh
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=/workspace/final_year_project/.cache/huggingface
export OMP_NUM_THREADS=1
```

| Command | Actual result |
| --- | --- |
| `.venv-cpu/bin/python -m pytest compression/tests/test_pruning.py compression/tests/test_pruning_precision.py experiments/tests/test_filesystem.py -q` | 15 passed, 3.29s |
| `.venv-cpu/bin/python -m pytest compression/tests/test_recovery.py -q` (final) | 9 passed, 1 PEFT warning, 2.81s |
| `.venv-cpu/bin/python -m pytest VLM/quantization/tests evaluation/tests compression/tests cloud/tests LLM-Pruner/tests -q` (intermediate) | 222 passed, 1 warning, 14.68s |
| `.venv-cpu/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider` (final) | **541 passed, 0 failures/skips, 2 warnings, 41.56s** |
| `.venv-cpu/bin/python -m compression --with-recovery --config configs/compression/synthetic_suite.json --output checkpoints/cloud-pre-gpu-final` | Completed all nine treatments; unique suite `93674f59d0d949b3aadf9b4e9cb72915` |
| `.venv-cpu/bin/python -m experiments history --root checkpoints/cloud-pre-gpu-final` | Exit 0; suite and child lifecycle records COMPLETED |
| `load_artifact(path, trusted=True)` over the first nine-treatment CLI suite | All 9 SHA/architecture/reload checks passed |
| `python -m evaluation --help`, `python -m compression --help`, `python -m experiments --help`, `python -m VLM.quantization.awq_quantize --help` using `.venv-cpu/bin/python` | All exit 0 |
| `npm --cache /workspace/.npm-cache install --package-lock-only --ignore-scripts` then `npm --cache /workspace/.npm-cache ci --ignore-scripts` in Interface | Exit 0; 256 packages installed |
| `npm run lint` in Interface | TypeScript check passed |
| `npm run build` in Interface | Vite 6.4.6 build passed, 1676 modules, 2.84s |
| `py_compile.compile(path, cfile=temporary_path, doraise=True)` over all 206 tracked/staged Python files | All passed; bytecode outside source |
| `bash -n` for all 7 tracked shell scripts | All passed; historical workflows not executed |
| `git diff --check` and staged diff checks | Passed |

The original restricted-sandbox full-suite attempt stalled in the first API
TestClient request. A faulthandler trace isolated the wait to the asyncio portal.
The same single test passed in 2.66s with sandbox network permission, and the full
suite then passed with that permission. The stalled processes were terminated;
no tests were disabled or assertions weakened. The initial npm attempt failed
because `/home/agent/.npm/_cacache` was unwritable; using the workspace cache fixed it.

Warnings retained: Starlette's deprecated AnyIO BlockingPortal alias and PEFT's
missing pretrained vocabulary config notice for the generated random fixture.

## Evidence and remaining work

Code: implemented changes are CPU validated within their supported scope.
CPU validation: 541 passing tests, frontend type/build, compilation and CLI smoke.
GPU preparation: guide and prerequisite checks available; actual real-model
execution remains conditional. No assumption of 6 GB VRAM fit is made.
GPU validation: not performed. Research completion: not established.

Remaining blockers include unavailable pinned/licensed real model/vision/data
assets and official held-out protocol, full-model LoRA training/lineage and compressed
visual evaluation integration, measured local CUDA RAM/VRAM feasibility, native
Windows execution, and optimized INT4 kernels. The new recovery implementation
deliberately operates on bounded CPU fixtures; it does not claim those broader
capabilities. No research quality/speedup/compression benefit is inferred from
random fixtures. Synthetic generated artifacts and dependencies stay ignored.

GitHub authentication was rechecked and is now valid. Publication is restricted
to `codex/cloud-pre-gpu-completion`, with no force push or update to main/review.
The final task response records actual push outcome and final SHA.
