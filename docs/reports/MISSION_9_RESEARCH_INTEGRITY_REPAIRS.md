# Mission 9 — research integrity remediation

Starting commit: ba68888bba08ec903e34e45e1098108fd698fd8b.
Branch: fix/mission-9-research-integrity. CPU engineering only. **NO-GO remains** until an independent review and external prerequisites are satisfied.

## Reproduction and evidence

Mission 8's eleven independent local probes and report remain outside the repository under /workspace/audits/mission8-ba68888. Mission 9 first added ten negative regression cases and ran them against the unchanged starting implementation: all ten failed, reproducing missing encoder coverage, three nonfinite checkpoint cases, internal truncation, stale physical dimensions, nonfinite reconstruction bias, scale conversion loss and two overflow cases. Log: /workspace/audits/mission9-before.log. Additional adversarial regressions cover comparisons, malformed exports, runtime logits, corrupt and mismatched encoder files, duplicate calibration content, exact prompt boundaries, history, calibration usage and unsupported recovery. No existing assertions were weakened or removed. The original tiny Bunny fixture now explicitly declares SYNTHETIC_TEST; this labels its random assets rather than pretending they are pretrained.

## Finding-to-repair ledger

| Audit finding | Root cause and repair | Regression evidence |
| --- | --- | --- |
| H1 encoder random replacement | SigLIP loader discarded loading diagnostics. Both normal/S2 routes now use local-only loading and reject missing/unexpected/mismatched/error keys; floating state is validated. Dense evaluation requires complete model/vision manifests unless config explicitly marks SYNTHETIC_TEST. | missing encoder tensor, corrupt file, mismatched shape, unmarked random assets rejected |
| H2 nonfinite output | Tokens were accepted without validating source state/logits. Shared finite-state and output hooks now fail before generation argmax/scoring; runner preserves FAILED status and diagnostics. Pipeline models and portable loads install the same guard. | NaN/+Inf/-Inf state and runtime-logit tests; FAILED run assertion |
| H3 prompt integrity | Only context maximum was checked, while Bunny helper had its own truncation bound. Validate both; check exactly one sentinel and expected expanded embeddings/mask length. Save text/effective lengths and NONE truncation status in predictions. | internal truncation, exact expanded-limit boundary, sentinel boundary; prior long-prompt tests retained |
| H4 comparison equivalence | Optional evidence was treated as sufficient. Required content hash, effective inference, assets, scorer/split/prompt/bounds and clean source evidence are now explicit; source/dependency mismatches are reported. Comparable/conditionally comparable/non-comparable states are distinct. | adversarial different SHA, missing evidence and self-comparison without evidence never verify equivalence |
| H5 invalid packed bias | Validator omitted bias and loaded structural checks. Shared packed validation checks dimensions, bias, groups, storage/range, positive scales, finite reconstruction, MLP dependencies and pruning widths at export/load boundaries. | malformed packed-bias export rejected; existing INT4/INT8 round trips retained |
| M1 physical MLP configuration | Slicing changed tensors alone. Uniform intermediate width and per-layer pruning metadata now follow actual tensors; custom reconstruction also updates metadata. | standard save_pretrained/from_pretrained round trip and matching logits; custom reconstruction tests retained |
| M2 reconstruction state | Only selected weights were finite-checked. Every incoming tensor is type/finite checked before candidate load; candidate state validated afterward. | nonfinite fc1 bias rejected |
| M3 half scales | nn.Module._apply cast FP32 master scales. Packed layer preserves a master copy at FP32 while moving devices. | INT8 tiny weights survive .half() with FP32 scales |
| M4 extreme range | Rounded affine endpoints overflow FP32 reconstruction. Dequantization now rejects nonfinite grids. | +/- FP32 maximum rejects INT4 and INT8 |
| M5 disconnected real compression bridge | This is an unsupported capability, not a validated research route. Existing dense/portable adapter restrictions remain; no full-model combined bridge is claimed. | existing compressed-treatment/modality rejection and seven synthetic treatment tests retained |
| M6 unverified recovery | Partial LoRA coverage was permissive. The recovered-model bridge now explicitly rejects LoRA directories as UNSUPPORTED until coverage/lineage has a validated manifest. Serialized models retain UNVERIFIED recovery provenance and receive structural validation. | unsupported LoRA regression; original complete-state and malformed recovery tests retained |
| M7 calibration/leakage | Caps hide actual usage and IDs can be renamed. Record forwarded sequences and observed tokens per linear; reject exact held-out text token overlap in the bounded suite. | cap stops after one of four sequences and records it; renamed exact text rejected |
| M8 sizes | Directory discovery counted unused exports. Inspection now chooses standard loader-preferred safetensors/bin/index files; raw filesystem ratios carry explicit UNVERIFIED research equivalence/scope metadata. | unused duplicate file excluded; raw ratio explicitly labeled |
| M10 history/legacy | Invalid records could break the entire result reader; nested children were undiscovered. Isolate bad records with diagnostics and recursively discover runs. Legacy acquisition sampler explicitly fails with an actionable supported-helper message. | malformed history isolated; nested child discovered; existing API tests retained |
| L1 warning suppression | Legacy builder changed global warning policy. Global suppression removed. | complete suite warnings remain visible |

M9 (official scorer, repeated research statistics and GPU measurement) and remaining L1 limits are explicit research prerequisites rather than claimed implemented experiments. Backend interactive latency retains its explicit CPU/no-warmup method and must not be compared to controlled evaluator timing. Physical head/vision pruning, heterogeneous standard pretrained reload, a full-model compressed multimodal bridge, optimized kernels and recovery training remain unsupported/unverified. No unit test proves these capabilities.

## Contracts and usage

A real dense evaluation config must now supply model_manifest_path and vision_manifest_path, resolving relative to the config file. Build reviewed manifests using experiments.manifests.model_manifest(..., hash_large=True) against already staged local files; model identifier/revision must match the evaluation config. The loader verifies complete content and saves the verified documents in effective_inference.json. No acquisition occurs. Content verification cannot prove who pretrained a checkpoint: weight_origin and pretrained_identity explicitly retain that distinction. Synthetic config markers are a trust declaration for tests, not cryptographic evidence or a sandbox.

No automatic evaluation truncation is supported. Exceeding text, expanded multimodal or context bounds fails; predictions retain original text and effective expanded counts. A future intentionally truncated protocol must be separately specified and versioned.

Comparison's raw differences remain available for diagnostic backward compatibility, with context warnings/status. They are not scientific gain claims; equivalent_protocol_verified requires complete evidence and no mismatches. Compression treatment lineage is not inferred merely because identifiers match. Legacy records with missing evidence remain conditionally comparable at best.

Packed checkpoint verification is trusted local-object validation, not safe unpickling or training authentication. Full state recovery coverage is distinct from training provenance. Uniform physical MLP reload is demonstrated; heterogeneous widths require the custom reconstruction helper and compatible factory.

## Validation and limitations

Use the existing CPU environment with CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. Full-suite and frontend results are recorded in AUTONOMOUS_PROGRESS.md after final validation. Dependencies were not installed or downloaded. The suite retains real tiny Bunny/SigLIP generation and all seven synthetic treatments. Mock adapters remain explicitly fixtures; they do not prove pretrained quality or GPU compatibility.

Remaining questions require exact licensed pinned Bunny/SigLIP/tokenizer/projector assets, an official held-out task/scorer decision, complete acquisition/recovery provenance and independently authorized GPU execution after zero-overage/storage confirmation. Semantic leakage/pretraining contamination, CUDA precision stability, full-model reload/quality, L4 fit and performance remain UNVERIFIED. All real result cells remain Pending/Not measured.

Final validation: **305 CPU tests passed, zero failures/skips**, one pre-existing Starlette/AnyIO deprecation warning (30.81 seconds). Python compilation passed for 179 sources. Frontend TypeScript and production build passed; diff whitespace check passed. The bounded suite's seven treatments remain covered. Original/release logs are retained outside the repository. No GPU/model acquisition or paid service was used.
