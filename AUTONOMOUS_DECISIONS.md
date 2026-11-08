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
