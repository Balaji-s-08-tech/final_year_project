# Mission 14 findings register

Authoritative detail, reproductions, affected interfaces, limits and prerequisite matrix: [independent audit](docs/reports/MISSION_14_INDEPENDENT_READINESS_AUDIT.md). Parent `12530f504eba2112360ac4b8bdb90037e0bf6e72`; audited source `86834bce2a7eecc6cbc4125003723e317b0e9c3c`.

| ID | Severity | Independently observed failure | Disposition |
| --- | --- | --- | --- |
| F1 | HIGH | Unbounded JSON/file/body/collection admission | Scoped repairs PASS; asset discovery still OPEN as R2; no whole-process sandbox claim |
| F2 | MEDIUM | Ten historical modules had no pre-import isolation | Ten AST guards PASS; repository-wide isolation OPEN as R3 |
| F3 | HIGH | Different warmups with equal units/methods received admitted timing delta | PASS; measurement metadata reconciled at compare boundary; conservative under-admission documented |
| F4 | HIGH | Incompatible Phi embedding width exported as portable checkpoint | PASS on actual small random Phi; header/actual residual/vocabulary/norm checks; real assets UNVERIFIED |
| F5 | MEDIUM | Safetensors allocation skipped shared byte admission | PASS; real owned safetensors round trip requires no pickle consent |
| F6 | HIGH | Stale content consent accepted; combined artifact could be replaced after manifest check | PASS under trusted immutable local-storage assumptions; same-handle digest check; not author authentication |
| F8 | HIGH | Backend silently truncated prompts, coerced settings and forwarded unsupported segment IDs | PASS on small real implementation path; nine inference regressions |
| F7 | HIGH when network-exposed | Nonlocal job submission returned 202 to a mock executor | PASS for enforced local peer/Host/Origin policy; public/multi-user/proxy deployments unsupported |
| S1 | MEDIUM, introduced here | Validation overhead plus pretty stdout exceeded unchanged job deadline | FIXED; compact JSON output retains contents; unchanged regression passes |
| S2 | HIGH, introduced here | Recursive bounded discovery let artifact child state shadow lifecycle | FIXED; direct-directory lookup and negative regression; final full suite rerun |
| R1 | MEDIUM | One corrupt lifecycle aborts whole CLI history (exit 1, valid peer absent) | FAIL / OPEN at eight-scope ceiling; original and final source reproduced |
| R2 | MEDIUM | Both asset-inventory functions admit 10001 zero-byte file entries without discovery cap | FAIL / OPEN; two external assertions; no real assets/exhaustion attempt |
| R3 | MEDIUM | Standalone hf_prune and historical shell routes retain acquisition/mutation/CUDA paths | FAIL static / OPEN; never executed; not reachable from supported app in inspected call chains |

Unverified concerns remain around third-party config/tokenizer/image parser budgets, process/log/disk/time quotas, same-inode mutation, distributed/PID recovery, dependency vulnerabilities, complete legacy code, S2/alternative architectures and all full-model/GPU behavior. Do not promote them to reproduced exploits or tested guarantees.

Final repository suite: **477 passed**, zero failed/skipped, one third-party warning. New repository probes: **68 passed**. External known-open probes: **3 failed** (R1 and two R2 readers), explicitly excluded from any green-suite claim. Remaining source, protocol and financial gates keep GPU/research **NO-GO**.
