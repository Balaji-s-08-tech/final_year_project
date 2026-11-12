# VLM Compression Optimizer

## Final-year demonstration delivery

A user-supplied Kaggle/T4 dense GPU smoke succeeded with pinned Bunny/SigLIP assets.
See [baseline reproduction](BUNNY_GPU_BASELINE_REPRODUCTION.md) and
[evidence review](docs/reports/BUNNY_GPU_BASELINE_EVIDENCE.md). Its red-square answer
identified red while whole-answer exact match remained 0/1. This supports smoke
execution, not benchmark accuracy or compressed-model validation.

Start with [DEMO_README.md](DEMO_README.md) for the offline React presentation and
optional image-question API. The [engineering status](FINAL_ENGINEERING_STATUS.md)
records current validation. Declarative safetensors treatments now connect dense,
pruned, recovered and packed language blocks to the complete Bunny multimodal
adapter. [GPU execution](GPU_EXECUTION_GUIDE.md) and
[artifact contracts](MODEL_ARTIFACT_MANIFEST.md) describe the supported path.
The older milestone records below remain historical evidence; no real GPU or
pretrained-model research results are claimed by the new integration.

A CPU-validated research engineering framework for Bunny/Phi-2 with a SigLIP vision
encoder. The research question is whether structured pruning, sparsity and AWQ can
improve the quality/resource trade-off under controlled evaluation. Full benchmark
quality, recovery success, compression speedups and fit beyond the reported dense
smoke profile remain unverified.

Start with [AUTONOMOUS_HANDOFF.md](AUTONOMOUS_HANDOFF.md) for the complete scope,
reproduction steps, branch inventory and external prerequisites. Progress and
engineering decisions are recorded in [AUTONOMOUS_PROGRESS.md](AUTONOMOUS_PROGRESS.md)
and [AUTONOMOUS_DECISIONS.md](AUTONOMOUS_DECISIONS.md).

## Implemented and validated scope

- Configuration-driven local baseline evaluation, actual generated-token counts,
  image/data fingerprints and effective inference provenance.
- Unique immutable experiment directories, versioned metrics, source/environment
  snapshots, logs, comparison JSON/CSV and compatible API result records.
- Phi language-linears magnitude masks and paired physical MLP neuron reduction;
  small complete Bunny/SigLIP forward and generation tests.
- Portable INT4/INT8 AWQ calibration/scaling/clipping/packing and checkpoint checks.
- Bounded CPU dense/pruned/AWQ/combined comparison suite, with actual saved artifact
  sizes and explicitly synthetic quality evidence.
- FastAPI research dashboard and React interface; API jobs remain CPU-only.

Physical attention/vision/projector pruning, full-model recovery reconstruction,
official task scorers and optimized sparse/INT4 kernels remain unsupported or
unverified. A mask does not prove compression or acceleration. Portable AWQ layers
dequantize in PyTorch. Real research checkpoints and held-out assets are absent.

## CPU development

For a fresh Python 3.12 CPU environment, use `requirements-cpu-dev.txt` and the
commands in [Local GPU execution guide](LOCAL_GPU_EXECUTION_GUIDE.md). The guide
records verified CPU commands and the remaining full-model GPU prerequisites.
`python -m compression --with-recovery --output checkpoints/synthetic_experiments`
also exercises bounded synthetic MLP+LoRA and MLP+LoRA+INT4 treatments.

Use Python 3.11 and the prepared `.venv-vlm-awq` environment. The installed Linux CPU
versions are recorded in `requirements-cpu.lock`; this is a version lock, not a
wheel-hash lock or a CUDA environment. Fresh dependency installation requires normal
approved package-registry access; no automatic model or dataset acquisition exists.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HOME=/workspace/.cache/huggingface .venv-vlm-awq/bin/python -m pytest -q

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-vlm-awq/bin/python -m compression --config configs/compression/synthetic_suite.json --output checkpoints/synthetic_experiments
```

The synthetic CLI constructs a tiny random model locally and creates unique output
runs. It has no model-download or GPU option. Its results must not be represented as
full Bunny research results. The prior environment/setup reports describe historical
machines; they are not current checkpoint availability evidence.

## Guides and components

| Area | Guide / implementation |
| --- | --- |
| Architecture and supported boundaries | [Architecture](docs/ARCHITECTURE.md), `evaluation/`, `compression/` |
| Baseline specification, manifests and acquisition checklist | [Protocol](docs/BASELINE_PROTOCOL.md) |
| Evaluator and external L4 handoff | [Evaluation](docs/BASELINE_EVALUATION.md) |
| Pruning | [Methodology](docs/PRUNING_METHODOLOGY.md) |
| AWQ and tensor/checkpoint validation | [Methodology](docs/AWQ_METHODOLOGY.md), `VLM/quantization/` |
| Combined experiments and artifact format | [Pipeline](docs/COMBINED_PIPELINE.md) |
| Tracking, metrics and schemas | [Contract](docs/EXPERIMENT_TRACKING.md), `experiments/`, `docs/schemas/` |
| Dashboard | [API/UI guide](DASHBOARD.md), `backend/`, `Interface/` |
| Final research report | [Template](docs/FINAL_EXPERIMENT_REPORT_TEMPLATE.md) |

Legacy LLM-Pruner, ShortGPT and copied AWQ modules remain for provenance and separate
research investigation. Do not assume every legacy entry point is a supported
pipeline. Local asset and device admission now reject implicit remote acquisition
and unauthorized CUDA on the supported Bunny paths; this is not a sandbox for
arbitrary Python or untrusted pickle code.

## Research results and resource policy

| Experiment | Quality | Latency | Throughput | VRAM | Size |
| --- | --- | --- | --- | --- | --- |
| Dense | Pending | Pending | Pending | Pending | Pending |
| Pruned | Pending | Pending | Pending | Pending | Pending |
| AWQ | Pending | Pending | Pending | Pending | Pending |
| Combined | Pending | Pending | Pending | Pending | Pending |

No GPU nodes, paid APIs, cloud resources or large model/data downloads are authorized
by these instructions. Camber credit-exhaustion/overage and persistent storage policies
must be confirmed separately. GPU flags are operator acknowledgements, not financial
approval or fit evidence. No feature branch is automatically merged into main.
