# Mission 11 — comparison and portable loader integrity

Branch: fix/mission-11-comparison-loader-integrity.
Exact base: 09a57c229af79e6c9caff15aa5e7051856f03fc3.
GPU baseline status: **NO-GO pending independent Mission 12 review and external prerequisites.**

## Original failures and root causes

Four new regression cases were run before implementation against the unchanged base. All four failed (log /workspace/audits/mission11-before.log):

1. Empty assets.json could yield verified equivalence: comparison checked existence/equality rather than evidence contents.
2. Failed runs could receive COMPARABLE: lifecycle checks only guarded individual deltas.
3. Portable INT4 tensors with declared INT8 metadata loaded: the loader checked module shapes but not the metadata contract.
4. An incompatible q_proj width loaded and failed on forward: checks validated each linear against itself, without Phi head/residual dependencies.

Mission 10's external independent probes remain the additional original evidence. Tests use local random synthetic models and actual comparison/loading interfaces, never real benchmark assets.

## Comparison boundary

`experiments.evidence.assess` performs strict JSON object decoding, rejects duplicate keys and nonfinite JSON, and validates records at `experiments.compare.compare`, not just presentation/CLI. Mandatory evidence:

- Matching lifecycle/identity and config digest; COMPLETED plus a valid complete benchmark document for measured eligibility.
- Nonempty model identifier/revision/architecture, valid configuration/checkpoint SHA256, dataset identifier/revision/content fingerprint, held-out split, scorer, prompt and positive input/output token limits.
- Nonempty model/encoder inventories with verified statuses, relative bounded paths, byte sizes and SHA256. Comparison rehashes the available local files, checks inventory identity, binds checkpoint hash to a file or aggregate inventory, and reconciles model configuration hash with config.json or the portable header.
- Typed effective generation fields (token bound, beam count, sampling/cache booleans and pad/EOS), consistent requested/effective output bound; effective tokenizer and, for visual protocols, image processor. An explicitly text-only protocol may declare the encoder unavailable.
- Clean source commit identity and typed Python/PyTorch/dependency versions.

Default asset hashing budget is 8 MiB **per file**. Missing, changed or larger assets are unverifiable and cannot be VERIFIED. Callers may deliberately supply `max_asset_hash_bytes=<positive bytes>` or explicit None for unrestricted local hashing; this performs local I/O and grants no resource/cloud authorization. Actual deployment assets are absent in this mission.

Verification states:

| State | Meaning |
| --- | --- |
| VERIFIED | Complete valid core evidence, compatible records, eligible successful runs; optional context available |
| CONDITIONALLY_VERIFIED | Core evidence and run eligibility pass; only optional OS/CPU metadata is missing on both sides |
| NOT_VERIFIED | Mandatory evidence is missing/malformed/unverifiable, without a stronger explicit conflict |
| NON_COMPARABLE | Failed/cancelled/incomplete/invalid lifecycle or material identity/configuration/environment conflict |

Conflicting evidence takes precedence over omissions. `comparison_status` preserves legacy COMPARABLE/CONDITIONALLY_COMPARABLE terminology only for admitted results. `equivalent_protocol_verified` is true only for VERIFIED. Diagnostic deltas remain available for backward compatibility but are explicitly marked diagnostic when evidence is not admitted. Invalid run deltas are null. JSON/CSV retain explanations. These records verify the supported content/protocol contract, not producer authenticity, pretraining history, task quality or hardware suitability.

`compare_collection` admits only eligible successful pairs and returns every excluded path/report/reason and counts. Failed runs remain excluded even if their state contains measured metrics. No scientific averages are fabricated from excluded runs. The original pairwise CLI stays compatible; collection comparison is a Python API.

The current framework may lack required identity evidence (for example a directory checkpoint digest). That yields NOT_VERIFIED/NON_COMPARABLE rather than a relaxed equivalence claim. Material compression-treatment differences are not automatically approved as scientifically fair; a future explicit shared-base comparison protocol is separate work.

## Portable admission

Canonical portable precision strings are **AWQ INT4** and **AWQ INT8**, paired with method AWQ and integer bits 4/8. Boolean/unknown/conflicting precision values fail. New exports preserve the existing trusted portable-v1 payload interface and add a mandatory declarative sidecar with:

- Content-bound file hash/size and canonical precision.
- Model configuration/hash, decoder block dimensions, and actual packed-layer bits/effective groups/dimensions.
- Requested quantization/calibration configuration, kept separate from each layer's actual effective group size.

`portable_integrity.header` validates content/format, typed positive integer dimensions, layer count, hidden/head divisibility, supported KV relationships, projection/residual/MLP pairing, rotary width, pruning widths, packed paths/dimensions and precision **before trusted pickle loading**. Legacy portable files lacking this header are explicitly unsupported, not silently accepted. Existing supported writer/public loader calls retain their interfaces; re-export a reviewed in-memory legacy model to obtain a supported artifact.

After deserialization, `reconcile` checks inner/outer precision, complete quantization configuration, model configuration, actual packed modules and actual attention dependencies before returning a model or moving it to an execution device. Export validates before serialization. Shared packed admission also applies actual dependency checks to combined artifacts. A headless `tiny-phi2` fixture descriptor is explicitly synthetic linear-test scope; it does not imply full Phi attention or GPU validation.

Supported Phi attention has consistent q/k/v/output/residual widths and equal query/KV head counts for this vendored implementation. Physical attention pruning, unsupported GQA changes and heterogeneous standard pretrained reload are not inferred to work. Uniform physical MLP treatment remains supported.

**Trust limitation:** PyTorch object pickle remains trusted code. A declarative header validates declared dimensions before construction; actual object/tensor reconciliation necessarily follows deserialization. This is not safe untrusted-pickle ingestion, a sandbox, or proof that a malicious producer cannot lie. Invalid declarations fail before construction, and contradictory loaded tensors cannot return partial success. No claim of a wholly declarative state-dict reconstruction system is made.

## Regression coverage and validation

New tests reproduce all four original manifestations and cover empty/null/list/duplicate-key evidence, missing inventories/fields, invalid hashes/sizes, changed/missing files, path escape, explicit hash budget, contradictory configs/hashes, positive and conditional verification, failed/cancelled/created/running states with injected metrics, mixed collection exclusions, malformed terminal metrics, canonical precision, zero/negative/floating/boolean head dimensions, incompatible KV/hidden/projection dimensions, rejection before pickle via a fail-if-called hook, export rejection before torch.save, inner metadata conflicts, and valid INT4/INT8 load→forward→comparison.

No existing test was weakened or deleted. Initial integration exposed JSON int-key normalization (model id2label) and legacy missing-metric CSV status compatibility; these were corrected in implementation. JSON-canonical hashes reconcile equivalent serialized configurations. Existing missing metric statuses remain intact.

Full suite, compilation/frontend/whitespace evidence and final commit inventory are recorded in AUTONOMOUS_PROGRESS.md after final validation. All seven synthetic treatments and prior model/evaluation regression suites remain required. No package installs, downloads, GPU execution, paid APIs or infrastructure were used.

## Disposition and handoff

- Empty evidence equivalence: repaired at the public comparison boundary; local content revalidation now required.
- Failed-run comparability: repaired with lifecycle eligibility, null invalid deltas, explicit exclusion diagnostics and collection admission.
- Portable precision contradictions: repaired across header, payload configuration and loaded module checks.
- Portable attention dependencies: repaired for the explicitly supported Phi contract, with header preflight and post-load reconciliation.

These are CPU implementation conclusions awaiting independent Mission 12 verification. Full Bunny/SigLIP checkpoint compatibility, actual calibration suitability, official held-out scoring, acquisition/recovery lineage, L4 memory/performance and billing/storage policy remain unverified. All real research measurements remain Pending/Not measured. No GPU pilot is authorized or queued.

Final acceptance evidence: **350 CPU tests passed, zero failures/skips**, one existing third-party Starlette/AnyIO deprecation warning (34.51 seconds). Python compilation: 182 sources. Frontend TypeScript and Vite production build passed; whitespace check passed. All seven synthetic treatments remain functional through the original suite. Final logs: /workspace/audits/mission11-acceptance.log and mission11-frontend.log. No old test assertions were changed.
