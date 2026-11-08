# Mission 13 — recursive engineering log

Exact parent b999e492eed573c6361d15350169fda4c998640c; branch fix/mission-13-recursive-engineering. CPU-only; no acquisitions, external services or GPU execution. Evidence levels in this session: IMPLEMENTED, CPU-VALIDATED and SYNTHETICALLY-VALIDATED only. GPU/research readiness remains NO-GO pending independent review and external prerequisites.

## Initial reproduction
Five regression cases failed on unchanged source (four audit areas; numeric area has two manifestations). Evidence: /workspace/audits/mission13/before.log. They reproduced VERIFIED exponent-overflow evidence, collection abort, wholly unmeasured admission, absent trust gating, and q/k normalization/configuration disagreement.

## Cycle 1 — recursive numeric evidence contract (high)
Root cause: parse_constant missed exponent overflow, and only selected leaf fields were validated. Added shared recursive JSON ingress/in-memory validation: signed INT64, finite binary64, explicit rejection of binary64 underflow, duplicates, unsupported values, ambiguous keys, excessive depth and cycles. Store reads/writes, canonical hashing, metric metadata, evidence, configuration/data/calibration and API imports use the contract. Malformed candidates produce per-record errors and do not abort valid peers. Focused evidence: 13 numeric tests passed; boundary/nested cases included. Commit recorded in progress ledger.

## Cycle 2 — q/k normalization dependencies (high)
Descriptor/reconciliation omitted normalization enablement and shape. It now compares actual module/config flags and declares query/key shapes. Supported enabled normalization must be LayerNorm over the actual pre-reshape residual width used by the vendored forward. Unsupported head-width normalization is rejected. A deterministic tiny Phi enabled-normalization forward/reload test passes; it does not prove Bunny asset compatibility. Headers missing new required normalization evidence require reviewed re-export. No silent architecture repair.

## Cycle 3 — measured-result eligibility (medium)
Completed lifecycle is separate from a usable measurement. Comparison now lists common measured metrics with matching method/unit, and explicit COMPLETE/PARTIAL/NONE coverage per run. A pair with no common valid measurement is excluded, retaining protocol verification separately. Invalid/failed runs remain excluded. Partial results do not imply all metrics are measured; diagnostic legacy differences are not scientific aggregates.

## Cycle 4 — executable checkpoint trust (high)
One central checkpoint I/O policy requires literal trusted=True before any executable pickle load. Portable/recovery/state/combined/verifier boundaries and backend/CLI callers propagate explicit acknowledgement; CLI adds --trust-local-checkpoint. Backend request schema defaults false with strict boolean typing. UI consent defaults false and resets on checkpoint selection. Dense/vision supported loads default to safetensors. Legacy object overlays require trusted_checkpoint=True; remaining legacy tensor-pickle paths also require consent (weights_only is defense in depth, not sufficient trust on the installed older Torch). Unported historical training helpers now fail closed for binary pickle paths. Dynamic code is disabled in the supported Bunny builder. No untrusted file was deserialized.

Existing tests creating their own trusted local artifacts were migrated to pass the new explicit keyword/CLI flag. Assertions were retained; no test was removed or weakened. This is a deliberate API contract change, not implicit grandfathered trust. Acknowledgement is not authenticity, isolation or a guarantee that malicious trusted code is safe.

## Cycle 5 — adjacent trust ordering and resource budget (medium)
Adversarial review reproduced three combined-loader failures: missing consent reached manifest I/O before policy rejection. Another regression showed invalid dimensions triggered full payload hashing first. Trust now precedes artifact I/O. Model loads have positive explicit CPU byte budgets (64 MiB default); headers reject invalid declarations before payload hashing, enforce a 1 MiB sidecar limit and recheck header stability. No large file was staged. Comparison header hashing honors its caller's deliberate I/O budget; explicit None is an unrestricted hashing choice, not model execution permission.

## Validation to date
The initial integrated repair suite passed 368 tests; further public-boundary tests exposed cycle 5 issues before repair. Full source validation and commit inventory are updated in AUTONOMOUS_PROGRESS.md. Remaining cycles will inspect adjacent provenance, lifecycle and evaluation-policy paths; eight cycles is a limit, not a completeness proof.
