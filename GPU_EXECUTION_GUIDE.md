# Portable local CUDA execution

For the now-supplied successful dense T4 smoke and reviewed loader fixes, use
[BUNNY_GPU_BASELINE_REPRODUCTION.md](BUNNY_GPU_BASELINE_REPRODUCTION.md).
It pins both checkpoints, configures deterministic cuBLAS, and reviews memory
before any compression. The earlier delivery's lack of a GPU rerun remains true.

Supported target: an already authorized NVIDIA L4 24 GB or another local CUDA
machine with measured adequate RAM, VRAM and disk. No GPU was used during this
delivery. A 24 GB card is a target, not proof of fit; 6 GB is not assumed sufficient.

## Prepare the runtime and assets

Use a separate Python 3.11/3.12 environment. Install the project-compatible
PyTorch/torchvision pair from the official CUDA index for that machine, then the
non-Torch dependencies pinned in `requirements-local-gpu.txt`. The validated CPU pair was torch 2.2.2 and torchvision
0.17.2; a CUDA build and its driver compatibility require local validation. Do not
install the CPU lock over a working CUDA runtime.

For a machine whose reviewed driver supports the official CUDA 12.1 builds, this
is a concrete setup recipe (not executed on a GPU here):

```sh
python -m venv .venv-local-gpu
# Use this environment's interpreter for every subsequent command.
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-local-gpu.txt
python -m pip check
```

On Windows invoke `.venv-local-gpu\Scripts\python.exe`; on Linux invoke
`.venv-local-gpu/bin/python`. The generic `python` above means that interpreter,
not an unrelated system installation. Confirm the installed wheel actually reports
a CUDA runtime with the following preflight.

```sh
python scripts/local_preflight.py
python -m evaluation --help
python -m compression.execute --help
```

Stage licensed pinned Bunny/Phi-2, matching tokenizer/projector, SigLIP weights and
processor locally. Do not invent revisions. Supply complete model/vision manifests
using `experiments.manifests.model_manifest(..., hash_large=True)` and verify them
with `verify_model_manifest`. Fill a local copy of
`configs/evaluation/bunny_l4_baseline.json` with actual identities, paths, held-out
split, prompts and reviewed memory budgets. See EXPERIMENT_REPRODUCTION.md.

```sh
python -m evaluation --config /absolute/path/dense.json --output /absolute/path/new-preflight.json
```

Preflight inspects local assets, dataset, software/runtime and declared free memory;
it estimates KV bytes without loading a model. It does not prove peak allocation.
CPU treatment preparation loads FP32 base weights and makes additional copies for
LoRA and export. Budget host RAM for those copies and disk for every stage. Safetensors
block files depend on the original base assets; they are not standalone full models.

## Baseline and treatment preparation

```sh
python -m evaluation --config /absolute/path/dense.json --output /absolute/path/baseline-runs --execute --authorize-gpu-execution
python -m compression.execute --config /absolute/path/dense.json --plan /absolute/path/local-plan.json --output /absolute/path/new-treatment-run
```

`local-plan.json` follows `configs/compression/local_preparation.json`. Paths inside
the plan resolve relative to that plan. Its stages run **on CPU**, in order: dense,
25% physical MLP pruning, bounded LoRA recovery, AWQ INT4. Maximum model bytes are
explicit. Calibration is two local text sequences of at most 32 tokens, bounded
to 64 captured tokens per linear; scaling/clipping grids are 3/4. Recovery performs
two FP32 CPU adapter steps using supplied tokenized text batches. These conservative
settings are operational defaults, not a reviewed research optimization protocol.

The recovery-batches JSON is a list of explicit `input_ids`, `attention_mask`,
`labels` arrays. Labels use -100 for padding/unsupervised targets. The helper checks
finite losses/gradients, rank, shapes and supervised causal targets. It does not
provide image-supervised recovery, automatic dataset preparation or optimizer-step
resume. Text recovery can be followed by full multimodal inference, but visual
recovery quality requires actual held-out evidence.

## Evaluate a compressed variant

Create a second evaluation config retaining exactly the same base/vision assets,
held-out dataset, prompt and generation protocol. Add the final stage directory
as `treatment_artifact_path`, its actual manifest `treatment`, and an explicit
`max_treatment_bytes` large enough for the reviewed artifact. Then:

```sh
python -m evaluation --config /absolute/path/compressed.json --output /absolute/path/compressed-preflight.json
python -m evaluation --config /absolute/path/compressed.json --output /absolute/path/compressed-runs --execute --authorize-gpu-execution
```

The loader reconstructs all language blocks against the base while preserving
tokenizer, processor, SigLIP and connector. It verifies hashes, structural dimensions,
coverage and declared treatment. CUDA evaluation logs actual peak allocated/reserved
memory and synchronized generation timings. Portable INT4 dequantizes in PyTorch;
no accelerated kernel or speedup is claimed.

## Checkpoints, resume and failures

```sh
python -m compression.execute --config /absolute/path/dense.json --plan /absolute/path/local-plan.json --output /absolute/path/existing-treatment-run --resume
```

Only complete stage receipts are resumable. Resume binds config, plan, calibration
and recovery file hashes, then revalidates each artifact against the base. A partial
stage is retained and rejected; use a fresh output root after inspecting it. Evaluation
completion is recorded separately. Stage receipts contain elapsed time and host RSS;
GPU evaluation records CUDA peaks. These are observations, not global allocation caps.

On OOM/nonfinite values stop, preserve logs and failed runs, review bounds and
resources, and start a new controlled attempt. Never relabel a dense fallback as
compressed or replace unavailable metrics with zero. Official scoring, statistical
comparison, full-model fit and quality remain GPU/data-dependent work.
