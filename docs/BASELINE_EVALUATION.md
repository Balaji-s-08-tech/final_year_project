# Mission 3: offline baseline evaluation

This is preparation for an independently authorized experiment. Codex has no access
to Camber's GPU or filesystem. No model download, GPU inference, node activation,
or paid API is part of this milestone. The earlier 159-test Mission 2 baseline is
preserved on its own branch.

## Architecture and compatibility

`evaluation/config.py` validates explicit local paths, model/data identities and
revisions, seed, scoring template, sample count, decoding limits, memory budgets
and treatment. Unknown options and silent truncation are rejected. Relative paths
resolve from the configuration file. `inspection.py` checks local config/shard
indices and assets without unpickling weights; `preflight.py` checks declared
resource budgets using read-only environment queries. `adapters.py` explicitly
supports repository Bunny/Phi-2 + SigLIP and trusted portable AWQ text generation.
`runner.py` records each evaluation as a Mission 2 immutable experiment.

The dense adapter loads local Bunny weights and tokenizer, overrides the vision
reference with an explicit local SigLIP directory, checks loading incompatibilities,
loads the vision tower, and retains the checkpoint's projector. No remote-code
model loader is used. The legacy selector requires 'sig' in the local vision path.
No DeepSpeed/W&B or training dependency is needed for this evaluation path.
Loading-info checks reject missing nonvision tensors rather than evaluate random
replacement weights. Preflight cannot prove that actual tensor shapes match.

Generation expands image embeddings and the attention mask once before cached
decoding. CPU tests reproduced a legacy direct `generate(images=...)` second-token
mask/KV mismatch; the new adapter uses the inherited Phi generation path with
expanded initial embeddings/mask. This is covered with forced multi-token decoding.
Legacy serving entry points are unchanged and are not validated by this adapter.

Portable AWQ evaluation requires explicit trust for pickle files and a verified
content-bound export sidecar. Treatment bits must agree with loaded packed layers.
It is deliberately text-only; it cannot silently drop evaluation images. Registered
parameter counts omit packed buffers, so the architecture report separately records
registered parameters, logical packed weight/bias elements, buffer bytes, dtypes,
and per-layer projection dimensions. Logical counts are not inferred from file size.

Current execution matrix:

| Treatment | Evaluation path | Status |
| --- | --- | --- |
| Dense original Bunny | local multimodal adapter | Tiny random CPU path validated; full weights unavailable |
| Portable AWQ INT4/INT8 | trusted text-only adapter | CPU generation/control path; actual artifact bits required |
| Structured pruning | existing LLM-Pruner/ShortGPT | Evaluator reconstruction/provenance bridge pending |
| Unstructured pruning | no verified engine | Unsupported; no speedup claim for dense zero masks |
| Pruning + AWQ | packed storage exists | Combined recovery lineage/multimodal comparison pending |

Unsupported treatment combinations fail preflight. Existing model builder variants
are retained for compatibility; this milestone does not consolidate untested paths.

## Evaluation protocol version 1

Local JSONL records have exactly `id`, `question`, `answers`, and (for multimodal
evaluation) `image`. IDs must be unique; references nonempty; image paths must stay
within the dataset directory. Example format only, not an experimental dataset:

```json
{"id":"sample-id","question":"Is the object red?","answers":["yes"],"image":"images/example.png"}
```

`random.Random(seed).sample` fixes the selected order and exact requested sample
count. The raw JSONL and selected image contents are fingerprinted; a change after
preflight aborts execution. These fingerprints do not establish licensed origin,
training/evaluation disjointness, or an official benchmark split: record those
facts in the dataset identifier/revision and research protocol before GPU work.

The scorer is normalized_exact_match_v1: Unicode NFKC, case-fold, collapse
whitespace, match any reference. It does not remove punctuation or implement an
official POPE/MME/TextVQA/VQA consensus scorer. Accuracy from this harness is only
accuracy on its explicit records, never an official benchmark result.

Batch size is explicitly one; greedy decoding, one beam, KV cache enabled. Warmups
use the first selected sample and are excluded from scored/timed sample counts.
Timing includes model vision encoding/fusion and generation, excludes tokenizer,
image preprocessing and text decoding, and synchronizes before/after CUDA timing.
Mean/median latency and nearest-rank P95 are stored alongside every raw latency.
Throughput is samples divided by summed generation seconds; token throughput uses
actual continuation token IDs, including the first EOS and excluding trailing
post-EOS padding. It never substitutes max_new_tokens for an actual token count.
For meaningful tail latency, use a larger reviewed sample count than the smoke
configuration; one sample is not a reliable performance distribution.

`execution_duration` includes model loading/warmups/evaluation. CUDA peak allocated
memory includes loading, warmup and evaluation, with peak reserved bytes separately
annotated. CPU runs use null/UNAVAILABLE GPU memory, and GPU utilization remains
unmeasured (no utilization sampler has been validated). No first-token latency or
perplexity is inferred from generation. Ratios and accuracy retention require an
actual paired comparison; the single-run evaluator leaves those slots unmeasured.
Parameter/size/accuracy metric slots distinguish baseline from compressed runs.
Checkpoint weight-file size excludes tokenizer/config/processor storage; the
original and compressed experiments must use the same size definition.

The runner uses inference mode and deterministic Torch algorithms, restoring the
prior deterministic-algorithm mode afterward. Determinism across devices/versions
is not guaranteed. GPU execution may require an explicit CUBLAS_WORKSPACE_CONFIG
in the external shell before importing Torch; record it in the research protocol.
The configuration hash, commit, environment and source hashes accompany every run.

## Commands and resource gates

Use Python 3.11 and the pinned core/dashboard requirements prepared in onboarding.
The CUDA version displayed by nvidia-smi is driver capability, not the installed
PyTorch wheel's CUDA runtime; validate the actual wheel on Camber after authorization.
No new dependencies are introduced.

```bash
# Edit all identity/revision placeholders and local paths before use.
python -m evaluation --config /path/to/edited-config.json --output /path/to/new-preflight.json
# Equivalent read-only wrapper, from repository root:
bash cloud/evaluation_preflight.sh /path/to/edited-config.json /path/to/new-preflight.json
# CPU execution only, on deliberately supplied local assets:
CUDA_VISIBLE_DEVICES='' python -m evaluation --config /path/to/cpu-config.json --output checkpoints/experiments --execute
```

Default invocation only inspects. It never creates a CUDA context or loads weights.
Reports are exclusively created and cannot replace an earlier report. Missing
assets/budgets give a blocked report/nonzero status; a passing report is labeled
PREFLIGHT_ONLY and memory_feasibility=NOT_PROVEN.

`configs/evaluation/bunny_l4_baseline.json` is an **unexecuted proposal**, not a
validated training recipe. Its configurable 24 GiB free RAM / 16 GiB free VRAM
thresholds are conservative admission budgets, not measurements or a claim that
the workload fits on the reported L4/32 GB host. KV cache planning derives an
analytical byte estimate from local architecture/configuration and sequence bounds;
it excludes weights, vision activations, workspace, allocator overhead and copies.
Analytical values never populate measured benchmark slots. No assumed A100 gate
is used. Full asset storage, output space and the platform's 50 GB quota must be
checked externally before staging. No storage capacity has been verified on Camber.

Only after billing/overage policy, dataset/protocol, model assets and resource
authorization are settled may a human execute this opt-in command on Camber:

```bash
# NOT RUN in Codex; this flag is an operator acknowledgement, not a billing check.
python -m evaluation --config /path/to/edited-camber-config.json --output /mnt/runs/experiments --execute --authorize-gpu-execution
```

No script launches infrastructure. The flag cannot establish authorization or
guarantee financial safety; do not use it until the user separately authorizes GPU
execution. Use a new run for each baseline/control. Preserve experiment.json,
config.json, environment.json, preflight.json, architecture.json, predictions,
metrics, logs and terminal lifecycle snapshots. Transfer small reviewed JSON/CSV
through a results branch; model/data blobs stay in approved external storage.

## Limits and next research decision

The fixture test initializes a very small random Bunny/Phi-2 and SigLIP from code,
saves/reloads them locally and performs finite multimodal forward plus cached
generation on CPU. A second test quantizes a tiny real Phi model and runs portable
AWQ text evaluation. These prove software integration for the fixtures, not full
Bunny compatibility, useful quality, recovery success, GPU feasibility or speedup.

Before any comparative claim, choose a held-out dataset and its official scorer,
pin revisions, rule out calibration leakage, fix prompts and decoding/timing
boundaries, and preserve the complete vision/projector path for all VLM treatments.
Do not compare text-only portable AWQ scores with multimodal Bunny as equivalent.
Start with a small dense baseline fit/forward/generation run after authorization,
then inspect actual outputs before expanding to compression experiments.
