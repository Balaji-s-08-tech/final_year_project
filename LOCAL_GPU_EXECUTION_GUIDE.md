# Local GPU execution handoff

CPU engineering was performed from Mission 15 (`fee24fb`) on
`codex/cloud-pre-gpu-completion`. No GPU experiments or scientific measurements
were performed. Commands below use existing entry points. Asset-dependent commands
are conditional recipes, not evidence that those assets exist or that execution fits.

## CPU environment and reproducibility

Verified platform: Linux, Python 3.12.14, CPU torch 2.2.2, Node 24.22.1,
npm 11.9.0. Use a separate CUDA environment for local GPU execution. The CPU
requirements must not replace a CUDA-enabled PyTorch installation.

```sh
python -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-cpu-dev.txt
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=1
.venv-cpu/bin/python -m pip check
.venv-cpu/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider
cd Interface
npm ci
npm run lint
npm run build
```

The API tests require working local sockets/asyncio thread wakeups. In the managed
Codex sandbox, allow networking for the test command; without it, TestClient stalled
in the asyncio portal. This is not a test assertion failure. Set npm's `--cache` to
a writable workspace directory if the user cache is unavailable.

On Windows PowerShell use `.venv-cpu\Scripts\python.exe` and set offline variables
with `$env:HF_HUB_OFFLINE='1'` etc. Tracking now selects Windows byte-range locking
instead of importing `fcntl`. Native Windows execution is still unverified.
Windows file contents are flushed, but POSIX directory fsync durability is unavailable.

## Environment preflight, model assets, dataset, RAM and VRAM

Start from `configs/evaluation/bunny_l4_baseline.json`, making a separate local
configuration with actual paths, pinned model/vision/dataset revisions, held-out
split, reviewed prompt, declared memory budgets and manifest paths. Template
placeholders are intentionally rejected. No command acquires missing assets.

```sh
.venv-cpu/bin/python -m evaluation --help
.venv-cpu/bin/python -m evaluation --config /absolute/path/pilot.json --output /absolute/path/new-preflight.json
```

The preflight checks local model/encoder shards, tokenizer/processor configuration,
dataset records/images, declared free RAM/VRAM and a config-derived KV-cache estimate.
It does not load model weights or prove fit. The declared memory minima are
admission checks, not allocation limits. KV bytes exclude weights, vision activations,
attention workspace, allocator overhead and temporary model copies. FP32 dense
weights alone require roughly four bytes per parameter; FP16 two. Portable INT4
stores packed weights but dequantizes in PyTorch and does not guarantee low peak
VRAM or an optimized kernel. Do not assume Bunny plus SigLIP fits in 6 GB VRAM.

Manifests are built and verified by `experiments.manifests.model_manifest` and
`verify_model_manifest`; selected dataset identities/disjointness by
`dataset_manifest` and `require_disjoint`. Verified CPU coverage:

```sh
.venv-cpu/bin/python -m pytest experiments/tests/test_manifests.py evaluation/tests/test_baseline.py -q
```

Hashing large assets is an explicit local I/O decision (`hash_large_assets=true`);
default skipped hashes cannot establish complete identity. Keep calibration,
recovery and held-out records disjoint; ID/content checks do not establish absence
of semantic duplicates or pretraining contamination. Review the task/scorer before
calling normalized exact match an official benchmark.

## Baseline evaluation on an authorized local GPU

In the separately prepared CUDA environment, replace `python` with its interpreter.
Unhide the GPU only for the authorized local workload. First run preflight on that
same machine using a new output path. Inspect the report and actual available
memory before attempting a small dense run with bounded input/output tokens.

```sh
python -m evaluation --config /absolute/path/pilot.json --output /absolute/path/local-preflight.json
python -m evaluation --config /absolute/path/pilot.json --output /absolute/path/baseline-runs --execute --authorize-gpu-execution
```

The config must select `device=cuda:0` and the correct dtype for that installed
runtime. No GPU execution was tested here. Start with a minimal sample count;
record real errors, memory peaks, timing, predictions and exact asset hashes.

## Pruning, LoRA recovery, AWQ INT4 and combined evaluation

The following verified command runs nine tiny random CPU treatments: dense,
unstructured masks, physical Phi MLP pruning, INT4, INT8, mask+AWQ, MLP+AWQ,
MLP+LoRA and MLP+LoRA+INT4. It creates a fresh unique suite each time.

```sh
.venv-cpu/bin/python -m compression --with-recovery --config configs/compression/synthetic_suite.json --output checkpoints/local-pre-gpu-smoke
.venv-cpu/bin/python -m pytest compression/tests/test_pruning.py compression/tests/test_pruning_precision.py compression/tests/test_recovery.py VLM/quantization/tests -q
```

Recovery uses explicit labels/masks, ignores padding, rejects rows without shifted
targets, freezes the base parameters, trains only Phi linear LoRA adapters and
checks finite loss/gradients/state. The source is preserved. Recovery data IDs and
tensor content hashes are recorded, and held-out overlap is rejected. Adapters can
be saved as safetensors, reloaded against the same pruned base, and safely merged
before AWQ. Tests cover prediction equivalence after merge and reload. These tests
establish mechanics on fixtures; they do not establish recovery quality.

The recovery API and combined suite deliberately enforce CPU fixture bounds.
The dense Bunny evaluation adapter does not accept real compressed/recovered visual
treatments. Full-model recovery training, lineage manifests and a compressed
multimodal evaluation bridge remain implementation blockers. There is no verified
full-model pruning/recovery CLI to recommend; do not use quarantined historical
training scripts to bypass these gaps.

The AWQ CLI exists for explicitly trusted, content-bound compatible local serialized
models. Its options were verified with:

```sh
.venv-cpu/bin/python -m VLM.quantization.awq_quantize --help
```

Conditional CPU AWQ recipe after reviewing the source and its SHA256 (replace all
local paths and the digest):

```sh
python -m VLM.quantization.awq_quantize --model_path /absolute/path/trusted-model.pt --expected-source-sha256 VERIFIED_SHA256 --trust-local-checkpoint --calibration_data /absolute/path/calibration.jsonl --output_dir /absolute/path/new-awq-run --bits 4 --group_size 128 --calibration_samples 2 --sequence_length 32 --max_tokens_per_linear 64 --device cpu
```

`--device cpu` is intentional: that path does not assume 6 GB VRAM is sufficient.
The recipe was not executed on real assets. Existing recovery adapter directories
without validated lineage are rejected by the full-model loader.

## Artifact validation and failure recovery

The regression tests execute save/reload, SHA verification, finite forward outputs,
packed INT4 structure checks, LoRA adapter safetensors reload and merge equivalence:

```sh
.venv-cpu/bin/python -m pytest compression/tests/test_pipeline.py compression/tests/test_recovery.py compression/tests/test_quantization_validation.py VLM/quantization/tests/test_tiny_pipeline.py -q
.venv-cpu/bin/python -m experiments history --root checkpoints/local-pre-gpu-smoke
```

For programmatic pipeline reload, use `compression.pipeline.load_artifact(run_path,
trusted=True)` only on owned reviewed artifacts. It checks the content SHA and
architecture before returning the CPU model; the container is executable pickle.
Keep manifests/configs/predictions/logs with their artifacts.

After OOM or nonfinite loss, retain the failed run and logs, stop the workload,
reduce sample/token/batch bounds or move to a machine with measured adequate memory,
and start a new unique run. Do not overwrite completed output, promote a failed run
to successful metrics, silently substitute dense for compressed models, or claim
fixture quality as research evidence. Optimizer resume/full-model training recovery
is not supplied by this bounded LoRA API. Research completion requires actual
held-out experiments with the reviewed assets and protocol.
