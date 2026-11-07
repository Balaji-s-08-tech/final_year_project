# Cloud experiment runbook

This is a reproducible manual runbook, not a cloud provisioning script. It does not create paid resources, download model/data, or start training. Run each expensive stage only after reviewing the inputs and resource budget.

## Resource and compatibility audit

- **Architecture:** the cached Bunny config is Phi-2 (`hidden_size=2560`, `num_hidden_layers=32`, `num_attention_heads=32`, vocabulary 50,295) with a SigLIP so400m vision tower and a 2-layer multimodal projector.
- **GPU memory:** a 3-billion-parameter FP16 model has an arithmetic weight-only lower bound around 6 GB decimal (about 5.59 GiB); this excludes vision weights, activations, CUDA workspaces, optimizer state, and temporary copies. The actual model parameter count/weight shard size is unavailable because weights are not cached. The 6 GB RTX 3050 is not a viable recovery-training target. A 24 GB GPU is a reasonable first cloud smoke-test capacity for memory-efficient LoRA, not a verified guarantee; full fine-tuning or teacher/student distillation may need substantially more. Measure peak allocation during the first short run before choosing a final allocation.
- **Software:** use Linux x86_64, Python 3.11, PyTorch 2.2.2 with a CUDA 12.1 wheel, Transformers 4.37.2, Tokenizers 0.15.1, Accelerate 0.28.0, PEFT 0.10.0, SentencePiece 0.1.99, and NumPy below 2. The existing Bunny code imports `wandb`; pruning and perplexity evaluation use `datasets`; LoRA recovery output calls a helper that imports DeepSpeed. The current local environment lacks `wandb`, DeepSpeed, BitsAndBytes and FlashAttention. `bitsandbytes` is only needed if the recovery run sets 4/8-bit training; it is not needed for the portable post-training AWQ path. Validate DeepSpeed compatibility on the chosen Linux/CUDA/PyTorch image before a long recovery run.
- **Dataset:** pruning with `--dataset bunny` requires `BUNNY_DATA_PATH` containing `bunny_695k.json` and `images/`; the helper reloads Bunny and processes multimodal examples. `train_pruned.py` requires a JSON `--data_path` plus `--image_folder` for multimodal training. The pruning script also runs WikiText-2 and Penn Treebank perplexity evaluation at the end; `datasets` downloads/cache access are needed unless those datasets are already staged. No such Bunny dataset is present locally.
- **Storage:** the existing prune-only smoke checkpoint is 5.30 GB and its AWQ output is 1.78 GB. The base model shard size and Bunny dataset size are unknown. Plan disk for base weights + dataset + structural checkpoint + recovery output + AWQ output + temporary checkpoint copies. Measure dataset storage after staging; do not assume a fixed total from these partial local artifacts.
- **Expected products:** pruning writes `LLM-Pruner/LLMPruner/prune_log/<run-name>/pytorch_model.bin` plus a run description/log. LoRA recovery writes a directory including `adapter_config.json`, adapter weights, `non_lora_trainables.bin`, config and trainer state. AWQ writes `awq_config.json` and `bunny_awq_int4.pt` (INT8 uses `bunny_awq_int8.pt`). The evaluation/benchmark JSON must retain explicit precision and `NOT MEASURED` for unavailable metrics.

The model builder used by recovery loads the multimodal vision tower. The AWQ block mapper quantizes Phi-2 linears and does not measure/evaluate the vision tower. Cloud memory limits must account for the loaded full Bunny model, even though AWQ targets the language backbone.

## Environment on the cloud VM

Start from a clean Linux VM image with CUDA driver support. Pin the project ML dependencies and install the matching PyTorch wheel explicitly. The local lock/pin intent is in `requirements-bunny-awq.txt`; `requirements.txt` additionally contains optional dependencies. For the upstream pruning/recovery stages, add and validate `datasets`, `wandb`, and compatible `deepspeed` on Linux. Avoid installing BitsAndBytes unless using quantized training. Set `WANDB_MODE=offline` if logs should remain local; the training script still imports and initializes wandb.

Example setup (run in the cloud project checkout after code/data transfer):

```bash
python3.11 -m venv .venv-vlm-awq
source .venv-vlm-awq/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-bunny-awq.txt
python -m pip install datasets wandb
python -m pip install deepspeed==0.14.4
python -c 'import torch,transformers; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(transformers.__version__)'
```

The DeepSpeed pin and GPU image must be smoke-tested first; the local Windows environment does not have DeepSpeed and cannot validate this Linux training dependency.

## Stage 1: structural pruning

Stage inputs: actual Bunny base-model weights, the Bunny dataset tree, and enough GPU/storage for the full model and calibration. `get_bunny()` in the pruner reads `BUNNY_DATA_PATH/bunny_695k.json` and `BUNNY_DATA_PATH/images`. The command below uses the existing script defaults for pruning ratio/method settings and writes under the local LLM-Pruner log tree. `--short` disables its optional ShortGPT overlay because its CLI action is `store_false`. In the current source, `--save_ckpt_log_name` is parsed but not used to name the output directory.

```bash
export BUNNY_DATA_PATH=/mnt/data/bunny
python LLM-Pruner/examples/Bunny.py \
  --base_model /mnt/models/Bunny-v1_0-3B \
  --short \
  --dataset bunny \
  --pruner_type taylor \
  --pruning_ratio 0.28 \
  --num_examples 10 \
  --seq_len_prune 64 \
  --device cuda \
  --eval_device cuda
```

With this base directory, ratio and dataset, the source derives `log_name=Bunny-v1_0-3B_0.28_bunny`; the structural artifact is `LLM-Pruner/LLMPruner/prune_log/Bunny-v1_0-3B_0.28_bunny/pytorch_model.bin`. It contains a Torch payload with `model` and `tokenizer`. This command was not run here. The script runs WikiText-2 and PTB perplexity evaluation at completion, so stage these datasets or ensure an authorized network/cache path before initiating it. Confirm that exact run path and inspect `description.txt` before recovery.

## Stage 2: recovery training

Required data files: a conversation JSON dataset accepted by `VLM/bunny/util/data_utils.py`, and its image directory. Do not invent recovery hyperparameters. Set the exact learning rate, epochs, batch size, accumulation, LoRA rank, seed, and save cadence from the approved experiment protocol before running. `wandb.init` is unconditional in this script; set `WANDB_MODE=offline` for a local log or configure an authorized W&B account.

Template, with placeholders intentionally requiring the experiment's approved settings:

```bash
export WANDB_MODE=offline
python -m VLM.bunny.train.train_pruned \
  --model_name_or_path /mnt/models/Bunny-v1_0-3B \
  --model_type phi-2 \
  --vision_tower /mnt/models/siglip \
  --pruned_model_path LLM-Pruner/LLMPruner/prune_log/Bunny-v1_0-3B_0.28_bunny/pytorch_model.bin \
  --data_path /mnt/data/bunny/bunny_695k.json \
  --image_folder /mnt/data/bunny/images \
  --output_dir /mnt/runs/bunny_recovery_lora \
  --lora_enable True \
  --fp16 True \
  --model_max_length 512 \
  --num_train_epochs <approved-epochs> \
  --per_device_train_batch_size <approved-batch-size> \
  --gradient_accumulation_steps <approved-accumulation> \
  --learning_rate <approved-learning-rate>
```

This template was not executed. For LoRA output, preserve both the adapter directory and the original structural checkpoint. Verify `adapter_config.json`, adapter weight files and `non_lora_trainables.bin` exist. Record the full training command, dataset revision, seed and checkpoint hash.

## Stage 3: AWQ INT4 (after recovery)

Run the local integration with the exact base, structural and recovery inputs. Keep calibration local and reproducible. Use CPU or a GPU only after measuring free memory; the current AWQ adapter loads through the Bunny multimodal builder for adapter checkpoints.

```bash
python -m VLM.quantization.awq_quantize \
  --base_model /mnt/models/Bunny-v1_0-3B \
  --pruned_checkpoint LLM-Pruner/LLMPruner/prune_log/Bunny-v1_0-3B_0.28_bunny/pytorch_model.bin \
  --recovery_checkpoint /mnt/runs/bunny_recovery_lora \
  --calibration_data /mnt/data/bunny_calibration.jsonl \
  --bits 4 --group_size 128 \
  --calibration_samples 32 --sequence_length 128 \
  --seed 42 --device cuda \
  --output_dir /mnt/runs/bunny_recovered_awq_int4
```

Run the INT8 AWQ control with the same checkpoint and calibration, changing `--bits 4` to `--bits 8` and output directory. The INT4/INT8 distinction must be explicit in all reports.

## Stage 4: evaluation and result import

Run an agreed task evaluator on the recovered FP16, recovered AWQ INT4 and (if available) AWQ INT8 outputs. Record the exact dataset/split, prompt/template, decoding settings, sample count, measured hardware memory and latency methodology. `VLM/quantization/benchmark.py` currently measures only file size; it does not measure model accuracy, latency or memory.

The dashboard's Benchmarks page accepts a cloud JSON manifest through **Import results**. Copy the manifest to the local machine first, then select it in the UI. Use the benchmark harness output as `results`, and include provenance/checkpoint details in `metadata`, such as run ID, base/pruned/recovery/AWQ hashes, dataset revision, hardware, and evaluator settings. Imported values are preserved and tagged with import time/source filename; non-finite values and invalid artifact sizes are rejected. Import does not independently validate the remote experiment or manufacture missing metrics. Keep original logs and hashes next to the cloud outputs.

## Safe stop conditions

Stop before starting a stage if base weights/data are missing, free disk is insufficient for source plus output copies, model/data revisions are ambiguous, the DeepSpeed environment has not passed a short import/save smoke test, or a short memory probe fails. Do not treat existing local prune/AWQ artifacts as recovered outputs; see `ARTIFACT_INVENTORY.md`.

## Cloud readiness audit addendum (2026-10-26)

### Account access and machine proposal

This workspace has no `gcloud` CLI, no authenticated GCP identity, and no connected GCP account integration. Project ID, billing credits, GPU quota, and account-specific live capacity therefore remain **unverified**. The public Google GPU zone table lists G2 in Mumbai (`asia-south1-a/b/c`) and A2 Standard in Singapore (`asia-southeast1-a`); that table is not a reservation or a guarantee that a machine can be allocated. Inspect the target project's quota and billing account before choosing a zone.

Proposed short dependency/model-load probe: `g2-standard-16` (16 vCPU, 64 GB host RAM, one L4 24 GB) in `asia-south1-c`, only if it is available and quota is approved. This can run the package check, but the included full Bunny plus SigLIP multimodal load probe intentionally gates on 32 GiB free VRAM and therefore requires a larger GPU. Proposed recovery candidate: `a2-highgpu-1g` (12 vCPU, 85 GiB host RAM, one A100 40 GB) in `asia-southeast1-a`. The A100 is a capacity proposal, not a measured fit; keep batch size at one with gradient checkpointing and stop on OOM. A 24 GB L4 is not the preferred recovery configuration. Public hardware/location references: [G2 machine types](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines), [GPU zones](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones), and [accelerator-optimized VM pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized).

Published accelerator-optimized price table values are used as **indicative list-price arithmetic**, not a region-specific quote: G2 standard-16 at USD 1.147208384/hour and A2 highgpu-1g at USD 3.673385/hour. This produces about USD 2.29 for a 2-hour L4 probe, USD 7.35 for a 2-hour A100 model-load probe, USD 88.16 for 24 A100-hours, or USD 176.32 for 48 A100-hours. The recovery durations are budgeting scenarios only; one epoch over 695K examples may take longer, and no duration has been measured. Disk, object storage, network egress, taxes, and regional price differences are excluded. Confirm the exact quote in the target project's Cloud Console calculator before approval.

Read-only account checks, after Google Cloud CLI installation and user login:

```bash
gcloud auth list --filter=status:ACTIVE --format='value(account)'
gcloud config get-value project
gcloud compute regions describe asia-southeast1 --format='yaml(quotas)'
gcloud compute machine-types list --zones=asia-southeast1-a --filter='name=a2-highgpu-1g'
gcloud compute accelerator-types list --filter='zone:(asia-southeast1-a)'
```

Credits and billing balance must be checked in Cloud Billing for the selected project/account; the commands above do not expose credit balance. Do not infer credit eligibility from a free-trial banner: Google documents that GPU attachment is unavailable to Free Trial accounts. These checks do not create resources.

### Data sources, license and storage

Authoritative model ID: [`BAAI/Bunny-v1_0-3B`](https://huggingface.co/BAAI/Bunny-v1_0-3B), whose model card identifies Apache-2.0 and whose current Hugging Face repository reports 6.37 GB total. The inspected local config snapshot is commit `0958a104bd39a448fc310b3d1b1fc3d398dc3fcb`. The required SigLIP source is [`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384), Apache-2.0, currently about 3.52 GB; pin its resolved revision and hashes when staging. Local Bunny cache inspection found config/tokenizer only, no model shards or SigLIP weights.

Authoritative Bunny data repository: [`BoyaWu10/Bunny-v1_0-data`](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data), dataset revision `2cf1067f6e3c6d717ada9e28fa58524ade62cd0d`. It reports 88.9 GB total; its image archives are split into parts that must be concatenated and unpacked. The pruning command's Bunny loader and multimodal recovery both require the dataset JSON and image tree. Recovery data is therefore **required**, not optional. Bunny's GitHub states its own code license is Apache-2.0 while included datasets/checkpoints remain under their respective original licenses. The dataset card's Apache-2.0 statement does not clear rights in source image collections; review the Bunny/SVIT and each image source's terms for the intended research use before copying data to a cloud bucket.

Minimum known on-disk inputs/outputs are approximately 6.37 GB base model + 3.52 GB SigLIP + 88.9 GB dataset + 5.30 GB structural checkpoint + 1.78 GB AWQ output = 105.87 GB, before recovery artifacts, OS/venv, evaluation data, checkpoints, and extraction working space. The uncompressed image-tree size and recovery checkpoint size are not known. Plan an initial 300 GB persistent disk, then verify extracted dataset size and available bytes before provisioning; increase it if unpacked images plus archives, checkpoints and outputs exceed the measured working set. Cloud disk is not present to inspect. Local D: has only 10,280,378,368 bytes free, so local staging is not feasible.

### Dependency proposal and compatibility checks

Keep the existing local environment untouched. The cloud setup uses the existing `requirements-bunny-awq.txt` plus the new `requirements-cloud-bunny.txt`; it does not install the broad `requirements.txt`, which includes unrelated application packages and an old `bitsandbytes==0.41.0` pin. CUDA PyTorch is installed separately from the official cu121 wheel index. The cloud extras pin Datasets, W&B and Hugging Face Hub. DeepSpeed is needed by the recovery script's LoRA state serialization helper even if ZeRO is not selected. BitsAndBytes and FlashAttention are not required for the proposed FP16 LoRA recovery path. A local dry-run found DeepSpeed's isolated build cannot import Torch when all packages are resolved together; install it separately after Torch, with ops disabled and build isolation disabled. This must still pass a Linux import/build check against the chosen image; no Linux VM is available here, so the candidate environment has **not** yet passed Linux `pip check` or import tests.

On the approved Linux host, use a clean Python 3.11 venv and run:

```bash
python3.11 -m venv .venv-vlm-cloud
source .venv-vlm-cloud/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-bunny-awq.txt -r requirements-cloud-bunny.txt
DS_BUILD_OPS=0 python -m pip install --no-build-isolation -r requirements-cloud-deepspeed.txt
python -m pip check
python - <<'PY'
import torch, torchvision, transformers, accelerate, peft, datasets, wandb, deepspeed
from VLM.bunny.model.builder import load_pruned_bunny_model
from VLM.quantization.awq_quantize import main
print({'python_cuda_torch': torch.__version__, 'torchvision': torchvision.__version__,
       'cuda_available': torch.cuda.is_available(),
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       'transformers': transformers.__version__, 'accelerate': accelerate.__version__,
       'peft': peft.__version__, 'datasets': datasets.__version__,
       'wandb': wandb.__version__, 'deepspeed': deepspeed.__version__})
PY
```

Set `WANDB_MODE=offline` before recovery if runs must not contact W&B; this project initializes W&B unconditionally. The inline import is a dependency check, not a model load or training run. Expected constraints are Python 3.11, Torch/Torchvision 2.2.2/0.17.2 cu121, Transformers 4.37.2, Accelerate 0.28.0, PEFT 0.10.0, and NumPy <2. A successful Windows import does not validate Linux CUDA/DeepSpeed compatibility.

### Proposed run configuration vs current code defaults

| Setting | Existing source default | Proposed experiment setting | Status |
|---|---|---|---|
| Pruning ratio / method | 0.28 / Taylor | 0.28 / Taylor | Retain current defaults |
| Pruning sample count / sequence length | 10 / 64 | 10 / 64 | Retain defaults; fixed seed 42 |
| Recovery data | `bunny_695k.json` + images | Bunny-695K at pinned repository revision | Required; not staged |
| LoRA rank / alpha / dropout | 64 / 16 / 0.05 | 64 / 16 / 0.05 | Retain current defaults |
| Recovery epochs | Transformers default 3 | 1 | Proposed change; requires experiment approval |
| Per-device batch / gradient accumulation | Transformers defaults 8 / 1 | 1 / 16 | Proposed memory-saving change; effective batch 16, not baseline-equivalent |
| Learning rate | Transformers default `5e-5` | `5e-5` | Retain code default pending protocol decision |
| Max sequence length | 512 | 512 | Retain default |
| Gradient checkpointing | False | True | Proposed memory-saving change |
| Recovery precision | not fixed by script | FP16 on A100 | Proposed; validate loss stability |
| AWQ calibration | INT4, group 128, 32 samples, 128 tokens, max 512 cached tokens per linear, seed 42 | Same settings, with local Bunny-derived calibration JSONL and recorded hash | Retain code defaults; generate a deterministic text-only sample from the training conversations |
| Evaluation | pruning script runs WikiText-2/PTB perplexity; benchmark lacks task evaluator | same perplexity plus agreed Bunny multimodal held-out task, memory, latency and throughput | Accuracy/performance pending; do not substitute perplexity for task accuracy |

The LoRA script's `Trainer` defaults are not the Bunny paper's stated global batch. The proposed batch/accumulation and one-epoch settings are engineering suggestions, not an approved reproduction of the original recovery recipe. Confirm the training protocol, split, learning rate, seed and evaluator before a paid training run.

If the proposed recovery settings are approved, use this concrete command after Stage 1 succeeds. It is included for reproducibility and was not run:

```bash
export WANDB_MODE=offline
python -m VLM.bunny.train.train_pruned \
  --model_name_or_path /mnt/models/Bunny-v1_0-3B \
  --model_type phi-2 \
  --vision_tower /mnt/models/siglip \
  --pruned_model_path LLM-Pruner/LLMPruner/prune_log/Bunny-v1_0-3B_0.28_bunny/pytorch_model.bin \
  --data_path /mnt/data/bunny/bunny_695k.json \
  --image_folder /mnt/data/bunny/images \
  --output_dir /mnt/runs/bunny_recovery_lora \
  --lora_enable True --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 \
  --fp16 True --gradient_checkpointing True --model_max_length 512 \
  --num_train_epochs 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 --learning_rate 5e-5 --seed 42
```

The AWQ loader cannot consume raw Bunny-695K conversation records directly. Once the dataset has been staged, create a deterministic local text-only calibration set from the Bunny training JSON without downloading anything else:

```bash
python cloud/prepare_awq_calibration.py \
  --source /mnt/data/bunny/bunny_695k.json \
  --output /mnt/data/bunny_calibration.jsonl \
  --samples 32 --seed 42
```

The adapter samples exactly 32 usable conversations by seed, joins their turn text, removes the `<image>` placeholder and writes JSONL records under the `text` key accepted by `VLM/quantization/awq_calibration.py`. Keep this training-derived calibration file separate from the held-out evaluation set and record its hash.

### Stage contracts and acceptance checks

1. **Structural pruning:** inputs are the exact base Bunny weights, tokenizer/config, SigLIP tower and Bunny data tree. Output is `LLM-Pruner/LLMPruner/prune_log/<run>/pytorch_model.bin`, a Torch serialization with `model` and `tokenizer` entries, plus run log/description. Accept only if the run completes, the file loads on CPU, model layer/attention dimensions reflect the pruning output, hashes and command are recorded, and perplexity logs exist. This is not recovery.
2. **Recovery:** inputs are the same base model, structural checkpoint, matching tokenizer/vision tower and Bunny JSON/image tree. The script loads the pruned transformer layers and trains LoRA. Output directory must contain `adapter_config.json`, an adapter weight file (`adapter_model.safetensors` or `adapter_model.bin`), `non_lora_trainables.bin`, model config, and `trainer_state.json`; preserve the pruning checkpoint and full logs. Accept only after the project loader reconstructs base + pruned layers + non-LoRA state + LoRA adapter and returns a merged model; record a deterministic multimodal forward pass. Do not identify the old prune-only smoke output as recovered.
3. **AWQ INT4:** inputs are the same base, structural file, recovery directory and a versioned calibration text set. Run scaling, clipping and quantization only after loader provenance checks pass. Accept only when `awq_config.json` identifies all source hashes/revisions, INT4 and group size; packed output saves successfully and has an independently recorded size.
4. **Reload/inference:** reload the exported quantized artifact in a fresh process without access to stale in-memory weights. Accept only with a fixed tokenized forward input producing finite logits and a documented model/load path.
5. **Evaluation:** compare recovered FP16, AWQ INT4 and an INT8 control under the same held-out task set, prompts, sample order, decoding, hardware and latency protocol. Report accuracy, artifact size, peak VRAM and latency/throughput as measured values. Missing measures stay `NOT MEASURED`; do not claim INT4 is better than INT8 without controlling the bit-width comparison.

### No-training model/GPU probe

After the inputs have been staged and disk capacity checked, first run `nvidia-smi` and the dependency check above. Do not attempt loading when free VRAM is below 32 GiB on the proposed A100, host RAM is below 48 GiB, or the full base+vision files are not present. The model-load-only probe checks for local config and weight files, then loads base Bunny plus SigLIP and runs one finite-logit image forward; it records free/peak GPU memory. This does not start training or download weights. It is safe only after the 40 GB GPU and disk gates pass. The probe has not been run because no cloud machine or model files are available.

```bash
nvidia-smi
df -h /mnt/data /mnt/runs
find /mnt/models/bunny /mnt/models/siglip -maxdepth 1 -type f -printf '%p %s bytes\n'
python cloud/bunny_gpu_probe.py
PROBE_IMAGE="$(find /mnt/data/bunny/images -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)"
test -n "$PROBE_IMAGE"
python cloud/bunny_gpu_probe.py --load-model --model-dir /mnt/models/bunny --vision-dir /mnt/models/siglip --image-path "$PROBE_IMAGE"
```

The first probe checks imports and reports available GPU memory without loading weights. The second is opt-in and refuses to load unless both local model directories contain config and weight files and at least 32 GiB GPU / 48 GiB host RAM is free. It calls `load_pruned_bunny_model` with the base directory and no prune or recovery paths, explicitly loads the configured vision tower, then runs one image forward. Do not pass the original dense model where recovery is expected. The eventual recovered AWQ run must pass the structural checkpoint and recovery directory explicitly as shown in Stage 3.
