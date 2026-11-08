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
