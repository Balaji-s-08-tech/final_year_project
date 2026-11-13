# Reproduce the pinned Bunny GPU smoke baseline

See [supplied evidence](docs/reports/BUNNY_GPU_BASELINE_EVIDENCE.md) for the original
Kaggle result and its limits. Commands here were CPU-tested where possible; no GPU
run or asset download was performed in the stabilization session.

## 1. Runtime and deterministic CUDA

Use a fresh process or notebook kernel with a reviewed CUDA environment. The
successful supplied run used Python 3.12.3, torch 2.2.2+cu121, torchvision
0.17.2+cu121 and Transformers 4.37.2. For a compatible CUDA 12.1 driver environment:

```sh
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-local-gpu.txt
python -m pip check
```

Set this **before any CUDA initialization**, including in a notebook:

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/kaggle/working/hf-cache
```

PowerShell: `$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'`. In a fresh notebook, set
`os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'` in the first cell before CUDA use.
Restart the kernel if CUDA was already initialized without a valid setting.
`:16:8` is also a documented supported deterministic workspace setting. The
runner/adapter check it before CUDA use; they never initialize CUDA just to check.
The reproduction `run` command also sets the default before importing Torch.

## 2. Review resources before acquiring assets

The supplied run's staged weight files occupy **9,876,575,208 bytes (9.198 GiB)**:
6,364,624,584 Bunny bytes plus 3,511,950,624 standalone SigLIP bytes. This is disk
storage, including redundant/unused tensors, not the resident model allocation.
Allow additional room for environment wheels, caches, temporary files and outputs.
The download command requires at least 16 GiB free on its destination volume;
that is a staging threshold, not a guarantee for every installation.

Observed single-T4 peaks were 6.651 GiB allocated and 6.969 GiB reserved. This
profile exceeds 6 GiB; do not recommend a 6 GB GPU based on it. Preparation declares
20 GiB free host RAM and 10 GiB free VRAM for smoke admission, retaining a margin
above that observation. The archive records about 31.6 GB available host RAM at
startup, not peak RAM. L4 24 GB remains a suitable target to check, not a proven
compression/training capacity claim.

## 3. Explicit pinned acquisition or reviewed local copies

Only on an authorized machine with sufficient storage/network access:

```sh
python scripts/reproduce_bunny_baseline.py download --destination /kaggle/working/pinned-baseline-assets --authorize-download
```

The script uses `snapshot_download` with these exact revisions and explicit file
lists; it never requests arbitrary executable remote model code:

- `BAAI/Bunny-v1_0-3B` @ `0958a104bd39a448fc310b3d1b1fc3d398dc3fcb`
- `google/siglip-so400m-patch14-384` @ `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`

The destination must be new. Failed or partial downloads are retained for review.
Alternatively stage verified copies yourself. A content manifest establishes local
consistency; the operator must establish source/revision authenticity. No secret
value is printed or placed in configs.

## 4. Generate a new smoke configuration and preflight

```sh
python scripts/reproduce_bunny_baseline.py prepare --model-dir /kaggle/working/pinned-baseline-assets/bunny --vision-dir /kaggle/working/pinned-baseline-assets/siglip --output /kaggle/working/baseline-smoke-config --acknowledge-pinned-assets
```

This hashes all local assets, makes complete manifests, generates the red-square
teaching image/annotation, and writes `smoke_config.json`, `source_assertion.json`,
`preflight.json` and `resource_review.json`. It loads no model and initializes no
CUDA context. It does substantial local hashing I/O. Output must be a new directory
outside the assets. Inspect the reports; missing assets/budgets block execution.

## 5. Execute only the dense baseline smoke

```sh
python scripts/reproduce_bunny_baseline.py run --config /kaggle/working/baseline-smoke-config/smoke_config.json --output /kaggle/working/baseline-smoke-runs --authorize-gpu-execution
```

Each run creates a unique experiment. Inspect predictions, raw exact-match score,
`smoke_test_summary.json`, `evaluation_summary.json`, `checkpoint_loading.json`,
`execution_policy.json`, environment and memory/timing metrics. A nonempty finite
continuation can pass the execution smoke while exact match remains zero.
Semantic correctness is not automatically inferred by the smoke summary.

Transformers may warn about using a full SigLIP checkpoint with a vision-only model
or unused checkpoint keys. Those original warnings remain visible. Additional
structured summaries confirm the exact configured unused-key set and verified
embedded vision equality. Unknown keys, missing tensors, shape mismatches, loader
errors or mismatched vision values still fail. Do not mute all Transformers logs.

The metadata checks support single-file and indexed/sharded safetensors, following
Transformers' single-file precedence. No fixed 421/440 acceptance counts exist.
The embedded vision layout is derived from Bunny's hidden-state selector. For
the pinned 27-block vision model, `-2` consumes the output of blocks 0..25;
block 26 and the pooling head need not be embedded. Their standalone weights
remain required by the full SigLIP loader. Post-layernorm remains mandatory in
the embedded export. See [subgraph correction](docs/reports/BUNNY_VISION_SUBGRAPH_FIX.md)
for the original 421/448 validation failure and the CPU regression evidence.
The explicitly trusted legacy Bunny `.bin` route keeps strict loading coverage;
it receives no redundant-key exception. Prefer the pinned safetensors route.

## 6. Before pruning, LoRA or AWQ

Do not extrapolate this dense smoke peak to gradients, optimizer state, calibration,
temporary dequantization or export copies. The current preparation implementation
uses FP32 CPU weights, LoRA copies and saved stage checkpoints. Its conservative
RAM admission is eight times the staged weight-file bytes: **79,012,601,664 bytes
(73.586 GiB)** for these assets, above the supplied Kaggle host's available RAM.
`resource_review.json` exposes that gate. Do not weaken it solely to get a run.

Budget disk separately for base assets, every complete stage checkpoint, temporary
export files and caches; free space sufficient for smoke is not sufficient for
multiple FP32 checkpoints. Review model-state tensor bytes and the selected plan
on the target machine before scheduling any transformation. A later memory-focused
implementation can reduce copies, but needs its own CPU tests and measured pilot.

Next GPU step: rerun this revised dense smoke in a fresh kernel, retain its new
layout/determinism evidence, then select a held-out dataset and official scorer.
Measure resource headroom for one proposed treatment before executing it. No
pruning, recovery, AWQ, quality retention or speedup result is established here.
