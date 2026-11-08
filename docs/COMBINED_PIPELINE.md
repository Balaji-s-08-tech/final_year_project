# Bounded combined compression pipeline

The implemented suite is CPU-only and explicitly bounded to small supplied models
(one million parameter elements and 64 MiB tensor storage by default), up to 128
held-out records and 32 calibration sequences of at most 128 tokens. It never loads
remote assets or provisions resources. The CLI creates only a random tiny Phi fixture:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m compression --output checkpoints/synthetic_experiments
```

Each invocation creates a unique suite and child runs for dense, unstructured masks,
physical MLP pruning, AWQ INT4, AWQ INT8, mask→AWQ and MLP→AWQ. Pruning-before-AWQ
is implemented; reversed ordering is rejected/unsupported because physical slicing
of packed groups would invalidate calibration and group metadata. No recovery
training is silently inserted. Such runs are prune-only, not recovered experiments.

The caller supplies a CPU model/tokenizer, explicit disjoint calibration IDs/token
tensors, held-out records and a fixed model/data/split/task/prompt protocol. The suite
binds actual small base state, calibration tokens, held-out data/images, tokenizer
vocabulary and reported effective generation settings. It rejects modality mismatches
and differing effective protocol identities. A VQA callback must use the complete
vision/projector path; tests run dense and structured+AWQ on a tiny real Bunny/SigLIP.
The default text predictor uses actual generated continuation IDs, not token limits.

Masks remain registered for prune-only artifacts. Before AWQ, materialize zeroed
dense weights explicitly, then recalibrate the actual pruned model. Physical MLP
widths remain in the serialized object and architecture metadata. Input base objects
are cloned and verified unchanged in tests; source/suite IDs are retained.

Every child saves predictions, compression configuration/architecture/provenance,
effective generation, logs, terminal metrics and a trusted local artifact. All variants
use the same `vlm-compression-v1` container (model, tokenizer, metadata), with SHA256/
size in artifact_manifest.json. This permits actual container-size comparisons;
serialization overhead and masks can increase size. It is not an optimized deployment
format. `load_artifact` checks hash, trust/load budget, format and reconstructed
architecture. Pickle is not safe for untrusted artifacts.

Suite comparison.json and comparison.csv report only actual CPU-fixture accuracy,
container bytes and end-to-end fixture execution duration. Latency, throughput, peak
GPU memory, compression benefit and real model quality remain Not measured; no
scientific gain is inferred. Normalized exact-match fixture scores are not official
VQA/task scores. Repeated calls preserve all prior results. Failures retain suite and
child failure records; no automatic resume or retry changes treatment semantics.

This implementation is a tested composition path and quality callback interface,
not permission to run full Bunny through it. The first real L4 task remains the
independently authorized dense baseline evaluator; extending compressed multimodal
recovery/reconstruction to real checkpoints needs staged assets and validation.
