# Autonomous engineering decisions

## Shared admission, not a billing system
Context: independent CUDA gates missed backend/AWQ paths. One local device admission
function rejects CUDA before runtime operations. The HTTP API stays CPU-only; local
CLI/operator acknowledgement is explicit and is not proof of billing authorization.
Unapproved auto selects CPU without querying CUDA. Alternative: environment-wide
implicit grants; rejected because they can silently authorize unrelated entry points.
Validation: patched CUDA probes, loader/API regression tests, complete CPU suite.
Risk: arbitrary third-party Python can call CUDA directly; this is not a sandbox.

## Supervision and serialization contracts
Reject/resample samples without shifted targets, rather than force a zero loss.
Reject nonfinite losses before gradients. Serialized measurements require explicit
units/metadata, while constructor defaults remain convenient for internal creation.
Existing text-axis truncation test now also verifies rejection of one-token inputs;
its expectation changed because a one-token sequence has no causal target.

## Bounded provenance and effective settings
Hash small local model/tokenizer/processor/encoder assets; explicitly label large
weight hashes incomplete. Save the model's effective generation configuration rather
than only requested kwargs. Compare actual saved assets/settings and warn when identity
is incomplete. Alternative: full hashing every initialization; rejected due large
checkpoint cost. No requested revision string is treated as content authentication.

## Architecture review and protocol boundaries
Reviewed Phi shared LayerNorm/attention/MLP, SigLIP/projector and duplicate legacy
loaders. Preserve legacy vendor paths instead of an unvalidated consolidation.
Support physical MLP slicing only when both dependent linears can be validated;
head/vision/projector changes remain design-only until their interfaces are proven.
Use local manifests and selected ID/image overlap checks rather than claiming
absence of semantic/pretraining leakage. No official scorer or revision is guessed.
Validation: tiny real multimodal integration, manifest roundtrip/change rejection,
CPU suite. Risk: unavailable full assets and large-file hashing remain explicit.

## Pruning implementation boundary
Use stable magnitude masks for exact shape preservation and paired fc1/fc2
slicing for physical MLP changes. Alternatives: generic head/channel removal or
mask-as-speedup claims; rejected without rotary/cache/cross-modal validation.
No seed-dependent random ranking; ties are stable. Reconstruction strict-loads a
candidate copy. Risks: masks add storage; physical slicing changes quality and
needs recovery; no latency claim. Validation: gradients, roundtrips and tiny
complete multimodal interface on CPU.

## Calibration storage and artifact inspection
Slice before CPU transfer and clone bounded FP32 activation storage. FP16 feature
retention was rejected because tiny values can disappear and full activation storage
can survive via a view. Restore mixed training/cache state on exceptions. Require
trusted pickle plus a bounded load budget for artifact verification; no hidden large
model load. Tensor reconstruction error is separate from task quality. Validation:
INT4/INT8 export/reload, hooks/modes failure recovery, tiny values and full suite.

## Combined suite and evidence boundary
Use explicit small CPU inputs, clone the dense base, bind actual state/calibration/
data/effective settings and preserve every treatment run. Choose prune then recalibrate
AWQ; packed-then-physical slicing is unsupported. Keep masks in prune-only exports,
materialize before AWQ. All variants use one generic trusted container; measure its
actual bytes but make no speed or real-quality claim. Mandatory modality-specific
predictor/effective settings prevent accidental text/vision score mixing. CLI has
no real-model or GPU load option. Validation: all seven text treatments, tiny real
multimodal combined path, serialization, failure and overlap/budget rejection.

## Final offline and packaging boundary
Legacy remote HF references could acquire weights without GPU execution. Require
local sources before transformer calls; prevent automatic device-map/quantized
backend activation. The old hardcoded full-model smoke script is disabled rather
than implying its single flag also authorizes acquisition. Alternative: silently
rely on network policy; rejected because policy can change and cache misses should
not initiate acquisition. CPU fixtures verify early rejection and local operation.

Expose dataset split explicitly and opt-in large-file hashing with a reviewed I/O
budget; missing identity remains null/INCOMPLETE rather than a fabricated hash.
The exact installed CPU version lock is Linux/Python-specific, not a wheel-hash or
CUDA lock. Keep the first real dense GPU pilot distinct from the bounded synthetic
compression composition path. Stop further work that needs assets, task judgment,
GPU execution or unverified cross-modal reconstruction. These limits preserve
scientific credibility instead of claiming full-model success from unit tests.

## Mission 9 — fail closed and preserve evidence
- Decision: require complete local model/vision manifests for non-synthetic dense evaluation and reject all encoder coverage gaps. Synthetic fixtures explicitly declare SYNTHETIC_TEST; local hashes do not establish training provenance.
- Context/alternative: Transformers warning-only random replacement was accepted as a baseline. Continuing with warnings risks invalid research; silently allowing missing tensors is rejected.
- Decision: validate finite state and every generated logits output, reject internal prompt truncation, and preserve FAILED runs rather than aggregate partial successes.
- Decision: retain diagnostic raw comparison differences while requiring complete evidence for equivalence. Missing evidence is conditional; material contradictions are non-comparable.
- Decision: update uniform MLP configuration and retain per-layer widths. Heterogeneous pretrained reload and LoRA recovery without validated coverage/lineage remain unsupported; custom state reconstruction remains explicit.
- Decision: retain filesystem byte ratios with scope/UNVERIFIED metadata; never imply task-quality or runtime benefits. Record actual calibration caps/usage and reject exact text-token leakage.
- Risks: trusted pickle and synthetic markers are trust declarations, not authentication; full hashes add deliberate local I/O; strict vision coverage may reject full SigLIP exports with unused heads until independently reviewed.
- Validation: original ten negative regressions failed before repair; final CPU suite and frontend evidence in progress/report. GPU questions remain unanswered and NO-GO remains in force.

## Mission 11 — comparison admission and declarative portable headers
- Decision: validate evidence at compare(), rehash local asset content, and bind checkpoint/config hashes to the inventory. Empty/malformed/unavailable evidence is never proof. Default per-file hashing is capped at 8 MiB, with an explicit caller override for reviewed local I/O.
- Context/alternatives: file existence/equality produced false equivalence. Trusting equal empty objects or default settings was rejected. This is record/content consistency verification, not producer authentication or pretraining proof.
- Decision: expose VERIFIED, CONDITIONALLY_VERIFIED, NOT_VERIFIED and NON_COMPARABLE. Only missing optional OS/CPU context permits conditional eligibility. Core omissions prevent admission; material contradictions and unsuccessful/invalid lifecycle states prevent comparison. Retain clearly marked diagnostic deltas for old consumers, but exclude them from compare_collection measured admission.
- Decision: add hash-bound declarative architecture/packed-layer headers to portable-v1 exports. Validate declared dimensions/precision before trusted deserialization; reconcile actual module dependencies and metadata before device placement or return. Unsupported headerless legacy objects fail explicitly.
- Alternatives: a new safe state-dict-only model factory would remove more pickle risk but requires a separately validated model/tokenizer/vision reconstruction contract. Current work does not claim that capability or safe untrusted deserialization.
- Risks: strict supported Phi attention contract rejects unvalidated GQA/head-pruned models; unavailable historical assets prevent VERIFIED; complete model/encoder/scorer/source evidence and real hardware remain external requirements.
- Validation: four pre-fix regressions failed; adversarial and complete CPU evidence is in Mission 11 report/progress. No existing tests weakened, packages installed, assets downloaded or GPU jobs run. NO-GO retained for Mission 12.

## Mission 13 initial repair decisions (cycles 1–5)
- Use one recursive JSON numeric contract rather than selected numeric leaf checks. INT64/binary64 limits are explicit; parsing underflow/overflow is rejected.
- Validate the vendored Phi normalization interface as implemented; reject unsupported layouts rather than reshape silently. Real model assets remain unverified.
- Separate protocol identity from common measured result coverage. Fully unmeasured/failed results cannot enter admitted measured collections.
- Require literal explicit pickle trust at every supported public boundary. Safetensors is preferred; restricted tensor pickle still requires trust. Legacy unported binary training paths fail closed.
- Deliberate compatibility change: migrate fixture calls and explicit CLI invocations to pass trust while retaining assertions. No grandfathered automatic consent.
- Default checkpoint load budget is 64 MiB; trust and cheap declarative checks precede artifact reads/deserialization. Consent and hashes are not authentication or sandboxing.

## Mission 13 cycle 6 — content-derived held-out provenance
Dataset identity alone is insufficient. Verification now recomputes the evaluator's actual local data/image selection fingerprint and sample IDs, respecting the explicit hashing budget. Missing assets are NOT_VERIFIED, not assumed equal. Comparison fixtures were strengthened to generate real synthetic input evidence rather than dummy hashes; assertions were retained. This does not prove licensing, official split validity or semantic/pretraining disjointness. Validation: 392 CPU tests passed.

## Mission 13 cycles 7–8 — persistence isolation and effective policy
- Invalid persisted rows are explicit diagnostics rather than service-wide exceptions. Invalid owner IDs are rejected before process liveness inspection; automatic job resume remains unsupported.
- Validate effective generation semantics using the implemented greedy, single-continuation protocol at both execution and comparison boundaries. Unknown or invalid material fields are rejected, not silently defaulted into evidence.
- LayerNorm tensor shape and epsilon are model-compatibility inputs. Q/k normalization flags, shape and epsilon must agree with the vendored forward and config; unsupported variants fail admission.
- Completed malformed metric schemas cannot be processed again by coverage reporting; invalid rows retain exclusions.
- Stop at eight cycles with a clean committed handoff. Known unsupported legacy acquisition/dynamic-code workflows and external assets/GPU validation remain review work; do not claim repository-wide security or research readiness.
