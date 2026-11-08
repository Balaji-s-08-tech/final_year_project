# Baseline protocol and asset acquisition specification

## Identity and assets
Target model: BAAI/Bunny-v1_0-3B, repository Bunny/Phi-2 implementation.
Target encoder: google/siglip-so400m-patch14-384. Exact immutable remote revisions,
actual checkpoint files, tokenizer files and image processor configuration are
**unavailable**, not guessed. A local model directory must include the matching
projector and complete shards. Use experiments.manifests.model_manifest and
verify_model_manifest for explicit content evidence. Default hashing is bounded;
large files remain INCOMPLETE until deliberately hashed locally. No download code
is supplied. Verify licenses, total disk needs and staging source before acquisition.

Model manifest v1: kind=model, identifier, revision, files mapping local relative
path to sha256/status/size_bytes, status. It covers tokenizer, processor, model and
encoder configuration alongside weights; keep separate model and encoder manifests.
Dataset manifest v1: kind=dataset, identifier, revision, split, purpose, fingerprint,
sample_ids, image_sha256. Build selected-sample manifests with dataset_manifest.
require_disjoint rejects selected ID/image overlap; it cannot prove no semantic
duplicates or contamination in unknown pretraining data.

## Independent task tracks
Visual question answering: held-out image+question+reference records, one image
sentinel, pinned Bunny conversation template. Dataset and official scorer selection
remain an explicit research blocker. For a controlled smoke run only, use normalized
exact match; do not call that POPE/MME/VQA consensus scoring. Short yes/no answers
may need task-specific extraction; free captions need a different scorer or reviewed
human evaluation. Never mix these scores with text-only generation quality.

Text generation: use a distinct text-only held-out split, tokenizer and prompt
protocol. Portable language AWQ is a text-control track and cannot replace a full
multimodal compressed model on the visual track. Official perplexity/corpus scoring
and named benchmark adapters remain unsupported rather than invented.

## Effective inference and measurements
Batch size one initially; explicit greedy decoding/one beam/KV cache. Save all effective
GenerationConfig fields, EOS behavior, tokenizer, image processor and asset manifests.
Initial bounds: 256 text tokens, 32 generated tokens; account for inserted vision
tokens and reject context overflow or silent truncation. Three initial warmups are
pilot settings, not a validated convergence rule. Repeat the same complete held-out
protocol in at least three unique runs for a reviewed study; compare distributions
only after verifying identities/protocols and reporting variability, not just best time.
The current runner supports one run at a time; repeat invocations produce unique IDs.

Generation timing includes model image encoding/fusion and decoding kernels, excludes
tokenizer/pixel preprocessing and text decoding. Synchronize GPU before/after timing
only after authorization. Save raw sample latencies, mean/median/nearest-rank P95,
actual continuation token count (first EOS included), samples and tokens per measured
generation second. No TTFT or perplexity is inferred. GPU peak allocated includes
load/warmup/evaluation; reserved peak is separate metadata. GPU utilization is Not
measured until a sampling protocol is implemented. Compare the same size definition
(weight files vs whole export container), precision and modality; do not compare
directory bytes with unrelated serialized-model bytes as an architectural speed claim.

## Reproduction / acquisition checklist
- Confirm zero-overage/credit exhaustion behavior and obtain separate GPU permission.
- Choose official task, split, scorer/version, prompt and answer extraction policy.
- Acquire licensed pinned Bunny, encoder/tokenizer/processor and held-out assets
  outside Codex only after storage/resource authorization. Do not download from a test.
- Keep calibration/recovery disjoint from held-out evaluation; save manifests.
- Verify all content hashes deliberately, record unavailable fields explicitly.
- Pin Python/core dependencies and the actual compatible CUDA Torch wheel; driver
  CUDA capability is not the wheel runtime. Save the environment snapshot.
- Check full disk headroom, declared RAM/VRAM budgets, image/context bounds; admission
  thresholds are not process memory caps or fit evidence.
- Run read-only evaluation preflight first; edit all REPLACE_* placeholders.
- After explicit GPU permission, run only a small dense pilot, inspect actual outputs,
  then plan repeat counts and compression controls. Preserve every immutable run.

The existing L4 JSON is an unexecuted template. Billing, assets and official scoring
are blocked. None of this document authorizes node activation or cloud spending.
