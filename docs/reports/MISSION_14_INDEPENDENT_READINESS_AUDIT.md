# Mission 14 — independent research-readiness audit

## 1. Executive verdict

**NO-GO for real-asset, GPU and research execution.** The controlled CPU implementation improved, but passing tests do not close the remaining input, legacy, asset, protocol and financial gates. Eight remediation scopes were completed; final adversarial review found three additional reproducible medium-severity gaps that remain open at the cycle ceiling. A copied legacy pruner also remains unguarded; it was inspected statically and never executed.

Mission 13's unchanged source independently passed **409 tests**. Mission 14's final repository suite passed **477 tests, zero failures/skips**, including **68 new adversarial regressions**. Separately, **five external known-open assertions fail**, covering CLI history, two asset-inventory readers and two extreme-exponent validation/collection cases. Those failures are disclosed, not counted as passing regressions. No real accuracy, speed, VRAM, compression benefit or real checkpoint compatibility was established.

Evidence is IMPLEMENTED, CPU-VALIDATED or SYNTHETICALLY-VALIDATED only. No feature acquired REAL-ASSET-VALIDATED, GPU-VALIDATED or RESEARCH-VALIDATED status.

## 2. Starting state, scope and preserved Git state

- Verified clean starting checkout: `fix/mission-13-recursive-engineering`, exactly `12530f504eba2112360ac4b8bdb90037e0bf6e72`.
- Created only `audit/mission-14-independent-readiness` from that commit.
- Inspected the complete saved Mission 13 patch against `b999e492eed573c6361d15350169fda4c998640c`, its report and all three AUTONOMOUS records before remediation. Previous reports were not used as test evidence.
- Last audited implementation commit: **`86834bce2a7eecc6cbc4125003723e317b0e9c3c`**. The following documentation commit contains this report; resolve its exact ID using `git log -1 --format=%H -- docs/reports/MISSION_14_INDEPENDENT_READINESS_AUDIT.md`. The session handoff also records the final branch HEAD.
- All pre-existing local branch refs were compared against the initial saved ref inventory and remain unchanged. Read-only `git ls-remote` independently verified remote `main=66c1d25252114451532786420b1e92e16de8a778`, Mission 9 `09a57c229af79e6c9caff15aa5e7051856f03fc3`, Mission 11 `b999e492eed573c6361d15350169fda4c998640c` and Mission 13 `12530f504eba2112360ac4b8bdb90037e0bf6e72`.
- No merges, PRs, resets, force pushes, branch deletion or writes to existing branches were performed. No model/dataset acquisition, GPU execution, historical training execution, paid service, credentials inspection or untrusted pickle deserialization occurred.

Initial state, original file hashes, full patch, exact source snapshot and failure logs are under `/workspace/audits/mission14/`. These paths are session evidence, not assumed to exist on another machine. Durable reproductions and conclusions are recorded below and in eight new repository test files.

## 3. Independent Mission 13 verification

| Previous claim | Independent inspection/evidence | Result and limit |
| --- | --- | --- |
| Numeric overflow/underflow cannot establish verification | Shared `experiments.jsonio`, public compare/evidence paths, negative exponent/INT64/nested tests and new independent strict-input probes | PASS for ordinary numeric/overflow cases; FAIL for extreme-underflow exception isolation (R4). Resource exhaustion was a separate reproduced gap. |
| Query/key normalization matches vendored Phi forward | Inspected normalization before head reshape, descriptor flag/shape/epsilon reconciliation; actual tiny enabled-normalization forward/export/reload test | PASS on synthetic Phi. Full embedding/residual interfaces were incomplete; repaired in F4. |
| Completed but unmeasured runs are excluded | Inspected schema/lifecycle/evidence/common-metric coverage; empty, failed, cancelled and malformed regression paths | PASS for measured admission tested. A single common measured metric is partial coverage, not proof of useful task quality. |
| Pickle requires literal acknowledgement | Traced public portable/recovery/state/combined/verifier and legacy tensor paths to `checkpoint_io`; rejection spies run before deserialization | PASS within explicit trusted-local policy. Consent was not bound to changed content and safetensors lacked a budget; repaired in F5/F6. |
| Malformed history does not abort peers | Backend readers/recovery pass; final independent CLI probe fails on unchanged and final source | PARTIAL. CLI omission remains R1. |
| Dataset selection is bound to local contents | Recomputed JSONL/selected images, IDs, seed, prompt/input limits and fingerprints in comparison | PASS on fixtures; semantic leakage and true held-out provenance remain unverified. |
| Inference policy is consistent | Evaluator checks effective expansion and logits. Backend had truncation/coercion/tokenizer-interface gaps | PARTIAL at baseline; bounded backend repairs in F8. Real assets remain unverified. |

The baseline suite was run from an exact external archive without changing refs. The first 20 new failing probes included ten static legacy-isolation assertions; these are one isolation finding, not ten executed exploits. Additional original-source failures covered network admission, collection limits, backend inference and artifact replacement. Failure logs identify actual assertions rather than inferring defects from test names.

## 4. Reproduced defects and bounded fixes

All fixes below have original-source negative evidence and final positive regressions. Historical code was checked by AST only, as execution was forbidden.

### F1 — HIGH: incomplete input/resource admission — PARTIALLY REPAIRED

**Affected:** `experiments/jsonio.py` (`loads`, `validate`, file reads), `experiments/store.py` (`read_json`, `runs`, `Run.latest`), configuration/data/calibration readers, `experiments.compare.compare_collection`, and HTTP request parsing in `backend/main.py`.

**Expected / original actual:** reject bounded overloads before unbounded reads/parsing. Original source accepted 100001 nodes, a scalar larger than 1 MiB, a 4097-character key and a lone surrogate. A sparse 8 MiB+1 JSON fixture reached `Path.read_text` before size admission. A 2 MiB+1 body reached framework validation (422) instead of early 413. 129 comparison candidates and 1001 persisted history records had no admission limit. No memory exhaustion attack was attempted.

**Fix:** strict UTF-8, bounded regular-file reads and lexical JSON preflight followed by graph validation; bounded JSONL/line/count handling; ASGI body admission including streamed requests without Content-Length; bounded history/discovery and comparison candidates. Invalid bounded comparison records retain per-record errors and valid peers.

**Regressions:** `test_mission14_inputs.py` (19 cases), including duplicate keys, malformed encodings, overflow/underflow, nesting, cycles, explicit custom budgets, stream body size, mixed valid/oversized records and direct-directory state selection. Original logs: `before.log`, `collection-before.log`. **Remaining:** asset-inventory readers are not covered by discovery limits (R2); third-party HF JSON parsing is not a universally bounded parser.

**Budget defaults/rationale:**

| Boundary | Default | Review/override |
| --- | --- | --- |
| JSON source and persisted document | 8 MiB, 100000 nodes including keys, 64 levels | Explicit typed `JsonLimits` for library reads; persistence intentionally uses the standard profile |
| Scalar / key / primitive token | 1 MiB string, 4096-byte decoded key, 128-character primitive | Conservative raw lexical string limit also applies; escaped large Unicode strings can be rejected earlier than their decoded size |
| Dataset/calibration JSONL or text | 64 MiB total, 1 MiB line, 100000 lines | Explicit positive `lines` arguments; partition larger corpora |
| HTTP body | 2 MiB, checked before framework JSON decoding | Fixed local API profile; deliberately smaller than file-based workflows |
| History / discovery | 1000 runs, 10000 entries | Explicit `runs`/`files` parameters; narrow the store rather than silently omitting runs |
| Comparison collection | 128 candidates | Explicit reviewed `max_candidates` |
| Checkpoint tensor/model load | 64 MiB | Explicit reviewed CPU load budget; this is a fixture limit, not a full Bunny budget |
| Checkpoint listing | At most 64 MiB per hashed artifact, 256 MiB admitted source bytes per listing | Larger/unhashed artifacts are unavailable for executable UI consent; no automatic enlargement |

These substantially exceed 32-example pilot configuration/metrics workloads, but do not constitute OS RSS, CPU-time, log, concurrency or disk quotas. Duplicate parsing at ASGI/framework boundaries is bounded, not allocation-free. Exceeding a collection limit fails explicitly rather than returning an undisclosed partial collection.

### F2 — MEDIUM: executable historical routes lacked quarantine — PARTIALLY REPAIRED

**Affected:** four ShortGPT modules (`prune`, `short_hf`, `short_vlm`, `short_llama`), three LLM-Pruner examples (`InternVL`, `Bunny`, `llava-vicuna_prune`), `LLM-Pruner/post_training.py`, and both Bunny training entry modules.

**Evidence:** AST/control-flow inspection found imports, acquisition and/or dynamic model code reachable without an unsupported-workflow admission error. Existing reports already excluded these from the controlled protocol, but that exclusion was not executable. **Fix:** unconditional top-level rejection before heavy imports and acquisition. Dashboard no longer recommends the historical Bunny pruning script. All ten static regression cases pass; none executes these modules.

**Remaining:** this is not repository-wide quarantine. `LLM-Pruner/hf_prune.py` remains unguarded, with pretrained acquisition at lines 40–41 and a script main; `LLM-Pruner/scripts/llama_prune.sh` reaches it. `scripts/evaluate.sh` contains file mutations and an external CUDA evaluation invocation. These were inspected only and must not be run. See R3. No supported evaluation/backend/compression call chain was found to launch them.

### F3 — HIGH: incompatible measurement metadata admitted as comparable — REPAIRED in tested paths

**Affected:** `experiments.compare._diagnostic_compare`, `compare`.

**Reproduction:** use the real persisted `pair` fixture, set latency metadata to warmup 0 versus 3 while retaining equal units/methods and valid protocol evidence. Original source returned admitted comparison and numerical latency difference. This can misrepresent timing controls as equivalent.

**Fix:** reconcile unit, method and complete metadata at the actual comparison boundary; expose `metric_protocol_conflicts`, reject incompatible pairs and withhold that delta. Five independent cases cover warmup, batch size, synchronization and repetition differences. No GPU/assets required.

**Limit:** strict whole-metadata equality can reject legitimate comparisons; evaluator timing metadata hashes the whole requested config (including treatment/path). Asset equality also conservatively rejects changed compressed checkpoints. Do not interpret rejection as proof of scientific inequality or use the diagnostic output as a research aggregate. A lineage-aware real-treatment comparison bridge remains unavailable.

### F4 — HIGH: portable architecture omitted residual interfaces — REPAIRED on supported synthetic Phi

**Affected:** `VLM.quantization.portable_integrity.describe`, `validate_architecture`, reconciliation used by export/load.

**Reproduction:** create the actual tiny vendored Phi fixture (hidden size 16), replace the embedding with width 17, pack one MLP layer and export. Original export succeeded despite a broken residual interface.

**Fix:** validate actual embedding tensor/attributes, vocabulary/config, LM head dimensions and residual input/final LayerNorm shapes/epsilon; add declarative interfaces to the header and reconcile them after trusted load. Seven tests include malformed embedding/head/normalization cases, an invalid header rejected before pickle, and unchanged valid portable paths. Old headers without the newly required interfaces require reviewed re-export; no silent migration or architecture repair.

**Limit:** full Bunny/SigLIP/projector assets, unusual Phi variants, multimodal S2 settings and CUDA execution were not verified. This is not generic architecture admission. Real-asset validation is required to resolve those questions.

### F5 — MEDIUM: safetensors allocation lacked byte admission — REPAIRED

**Affected:** `checkpoint_io.load_tensors` safetensors branch.

**Reproduction:** spy on shared admission while loading an owned two-element safetensors file; original branch bypassed it. **Fix:** call explicit byte admission before safe-format allocation. Two tests cover rejection ordering and actual safe tensor round trip without pickle consent. Safetensors is preferred but not a resource sandbox; installed parser internals were not independently security-audited.

### F6 — HIGH: consent/recovery identity could drift — REPAIRED under trusted immutable local-storage policy

**Affected:** `backend.jobs.submit`/worker, request schema, checkpoint listing, UI selection/consent, portable loader, raw AWQ source path and `compression.pipeline.load_artifact`.

**Reproductions:** (1) submit trusted inference with digest `0…0` while the current owned sentinel has a different digest: original manager accepts it; a mock executor prevents any deserialization. (2) after a valid combined artifact's manifest hash check, replace it with another owned, explicitly trusted tiny model inside the import hook: original loader returns the changed payload. Both are content-binding failures; no hostile pickle is used.

**Fix:** explicit SHA256 consent at backend admission and execution; identity reset/derivation from ID+digest in UI; propagate expected identity into CLI/model loaders; hash the admitted open file descriptor and deserialize through that same handle. Combined artifacts also pass their manifest digest into that boundary. Eight tests verify stale/missing/malformed identity, queued replacement, pre-deserialization rejection, real owned portable forward and combined replacement.

**Limit:** the hash binds bytes, not author identity or training provenance. Executable pickle still runs trusted Python. Same-inode concurrent mutation is not sandboxed; use owned immutable storage. Frontend behavior was source-reviewed and type/build validated, not exercised by a browser interaction test. Public raw callers may explicitly trust their source without supplying a digest; UI/server executable jobs require the digest. Real recovery lineage remains UNVERIFIED.

### F8 — HIGH: backend inference policy diverged from evaluator — REPAIRED on synthetic assets

**Affected:** `backend.jobs.JobManager._infer`, `backend.schemas.JobRequest`.

**Original failures:** (1) an actual tokenizer produces 2051 tokens from a 2051-character punctuation prompt; `truncation=True,max_length=2048` silently discards content; (2) bool/float/string token limits are coerced by the request schema; (3) an owned tiny Phi portable checkpoint fails generation because the tokenizer emits unsupported `token_type_ids`.

**Fix:** no implicit truncation; explicit token/context admission; strict token limit typing; suppress segment IDs at tokenization; validate effective greedy settings using the shared policy; record original/effective lengths and truncation false. Nine regressions include actual owned export/load/generate. Backend latency remains a no-warmup text diagnostic and is not the controlled VQA benchmark.

### F7 — HIGH if network-exposed: CORS was the only apparent API boundary — REPAIRED only for local-operator use

**Affected:** API admission and frontend listener.

**Reproduction:** TestClient peer `203.0.113.20` submits a benchmark to a mock executor; original endpoint returns 202. No model or external service runs. **Fix:** enforce loopback peer, local Host and configured Origin independently of CORS, before parsing. Forwarded headers do not grant application access. Frontend dev listener binds loopback. Eight ASGI tests cover remote peer, hostile Host, hostile/null Origin, allowed Origin and IPv6 loopback. The installed TestClient has an IPv6 URL parsing bug; the IPv6 test supplies the ASGI peer and Host explicitly through a localhost transport.

**Limit:** this is network admission, not user authentication. Do not expose the service publicly or put it behind an untrusted reverse proxy. A privileged local process is within the trust domain. Use `--no-proxy-headers` and a loopback server bind if operating the local prototype. No end-to-end HTTP server, browser or proxy deployment was tested.

## 5. Self-review regressions and remaining findings

### S1/S2 — Mission 14-introduced regressions — FIXED

These were not defects in Mission 13. Bounded JSON validation made hundreds of durable state snapshots caused by pretty benchmark stdout exceed an unchanged 10-second backend test deadline. Investigation reproduced the failure in isolation. Compact single-record JSON stdout retains JSON values and reduces history amplification; the unchanged regression passes (1.16 seconds in its isolated rerun).

The first bounded discovery implementation made `Run.latest` recurse into artifacts. A final adversarial test demonstrated `zzz_artifacts/state_999999.json` shadowing current lifecycle state. Discovery now explicitly stays in the run directory; the regression passes. Full tests were rerun after both corrections.

### R1 — MEDIUM / FAIL / OPEN: CLI history still loses valid peers

**File/function:** `experiments/__main__.py:main`, history list comprehension.

**Reproduce:** create one valid and one malformed persisted run, replace the latter `state_000000.json` with `{broken`, invoke `python -m experiments history --root <owned-fixture-root>`. Expected structured valid record plus per-record INVALID/error. Actual exit **1**, no valid peer in stdout. Reproduced on unchanged Mission 13 and final code. It does not overwrite records or admit invalid measurements, but collection availability/diagnostics are inconsistent with the backend.

**Fix proposed:** bounded per-record error isolation, explicit INVALID diagnostics and preserved valid entries, regression through the real CLI. No GPU/assets needed. Not implemented because the eight-scope ceiling was reached. External `test_cli_history_keeps_valid_peer` intentionally fails.

### R2 — MEDIUM / FAIL / OPEN: model/asset inventories bypass discovery budget

**File/functions:** `experiments.manifests.model_manifest`, `evaluation.provenance.asset_manifest`.

**Reproduce:** an owned directory with **10001 zero-byte files** and no model/data assets. Expected explicit discovery admission at the documented 10000-entry profile or a reviewed override. Both original and final functions return all **10001** file records using unbounded recursive discovery. No actual exhaustion was attempted. Two external assertions intentionally fail.

**Impact:** a large or growing local asset tree can consume uncontrolled traversal/hash CPU and intermediate memory before JSON graph admission. JSON document budgets do not fix discovery. **Fix proposed:** use bounded discovery with explicit profile/override and aggregate hashing budget, preserve symlink confinement, test admission before hashing/materialization. No GPU/real assets needed. Process-wide quotas and third-party tokenizer/config/image parser budgets also remain outside the tested guarantees.

### R3 — MEDIUM / FAIL (static) / OPEN: incomplete legacy isolation

`LLM-Pruner/hf_prune.py` has a reachable script main and unguarded pretrained acquisition, and a shipped shell wrapper invokes it. The historic evaluation shell also contains file mutation and CUDA execution. This is a static reachability finding, not a reproduced runtime acquisition or exploit; execution was forbidden. These are unsupported standalone routes, not found reachable from the supported app. Proposed next bounded fix: inventory remaining standalone acquisition/training launchers and quarantine before imports/side effects; retain pure model definitions used by the supported path. Do not execute the scripts to test isolation.

### R4 — MEDIUM / FAIL / OPEN: extreme underflow bypasses validation exception isolation

**Affected:** `experiments/jsonio.py:_float`, reached through `loads`/`read_json`, `experiments.evidence.assess`, `compare_collection` and JSON API admission.

**Reproduction:** `loads('1e-999999999999999999999999999999999999999999')` should reject with the documented validation error. Float conversion produces zero, but `Decimal(text).is_zero()` raises **decimal.InvalidOperation**, not ValueError. Put that literal in a candidate's config.json and compare `[bad_candidate, valid_candidate]`: the exception aborts the entire collection before valid-peer processing. Both assertions fail against unchanged Mission 13 and final audited code; `extreme-exponent-baseline.log` records the original source line. This is exception/availability failure, not evidence of VERIFIED nonfinite measurements.

**Root cause:** the underflow check assumes Decimal construction cannot fail for a JSON token within the character budget; callers isolate ValueError, not Decimal exceptions. The API middleware also lacks this exception normalization.

**Recommended next fix:** normalize bounded decimal-construction failures to a clear ValueError at the JSON trust boundary, or use a reviewed bounded mantissa-based underflow check. Test enormous signed exponents, nonzero/zero mantissas, all public JSON readers, API rejection and mixed comparison peers. Do not merely swallow exceptions in presentation or convert invalid values to zero. No GPU or assets required. Not repaired because remediation had reached the eight-scope ceiling. **Core comparison robustness remains NO-GO until independently fixed/retested.**

### Other limitations / UNVERIFIED concerns

- All trusted pickle producers and files must be verified externally; `weights_only=True` on Torch 2.2.2 is not a sandbox. No package upgrade or comprehensive vulnerability-advisory audit was performed.
- Third-party Transformers config/tokenizer parsers and Pillow image decoders do not inherit every application JSON/resource limit. Full directory hashing with explicit unlimited I/O is deliberate, but not a memory/time guarantee.
- Local POSIX atomic publication is not tamper-proof/distributed transaction storage. Logs/revisions have no complete disk/time quota. PID reuse and orphan-worker recovery are not a distributed scheduler. No multi-process stress/security proof is claimed.
- Copied builder branches still contain remote acquisition helpers, model-family alternatives and adapter loading. Local path/device/trust gates exist, but these branches were not executed and are not the controlled evaluator. Historical training scripts are not a safe substitute for it.
- Unused React components/data still contain historical “VERIFIED” figures; the current App does not import them. They are not research evidence. Reusing those components requires removing or explicitly labeling unsupported figures.
- The AWQ source docstring's CPU/CUDA correctness wording exceeds available validation; this report supersedes that implication. CUDA compatibility remains unverified.

## 6. Architecture and legacy entry-point inventory

Supported data flow: local evaluation config → bounded dataset selection/content fingerprints → inspection/preflight → asset manifests → LocalAdapter (Bunny/Phi + local SigLIP + projector, or trusted portable text) → effective prompt/generation and finite-logit guard → synchronized generation-only timing/scoring → immutable run records → evidence-based comparison. Backend jobs are a separate CPU local-operator path, not the full research runner.

| Route | Reachability / execution potential | Disposition |
| --- | --- | --- |
| `python -m evaluation` | Config/local assets; explicit execution and GPU authorization flags. Default is preflight only | Supported interface; CPU-fixture validated; real pilot BLOCKED |
| `python -m compression` | Small synthetic CPU factory and seven treatments, parameter/model budgets, no acquisition | Supported CPU fixture path, not full-model research execution |
| AWQ CLI / portable loader | Local calibration/checkpoints, explicit pickle trust/device policy; packed storage with dense dequantization fallback | Supported narrow path; real calibration/performance UNVERIFIED |
| `python -m experiments` | Bounded store/compare; CLI history lacks corrupt-peer isolation | Partially supported, R1 OPEN |
| Backend FastAPI / current React App | Single local worker, argv subprocess with no shell, approved paths, digest/trust checks | Conditional local-operator use only; no production/multi-user auth |
| `cloud/bunny_gpu_probe.py` | GPU diagnostic behind explicit flag and memory checks; lower-level builder, not full evaluator/protocol | No execution performed; not baseline research evidence |
| Ten guarded historical modules | Standalone CLI/import blocked before imports/acquisition; underlying text still retained | Quarantined, AST-validated only |
| `LLM-Pruner/hf_prune.py`, old shell launchers | Independent acquisition/training/CUDA paths; no supported caller found | Unsupported, insufficiently isolated, do not execute |
| Copied builder variants/ShortGPT/InternVL utilities | Some remote-code/acquisition references remain behind guards or outside supported paths | Unsupported/unverified outside explicitly audited routes |
| Full-model compressed VQA adapter | Dense Bunny adapter rejects compressed treatment labels; portable adapter is language-only | Not implemented as a validated research bridge |

AST inspection compiled 198 sources and indexed **166** acquisition/loading/subprocess call sites using the recorded final selector. The initial broader selector indexed 179; counts are not coverage percentages. Only two non-test direct `torch.load` calls remain, both in `checkpoint_io.py` behind explicit acknowledgement; one uses restricted tensor loading. No builtin `eval`, `exec` or `os.system` call was found by the AST selector. `model.eval()` is not dynamic execution. Residual `trust_remote_code`/Hub/subprocess references were inspected by source/call chain; source absence/guards do not prove arbitrary user-supplied code safe.

## 7. Pruning, AWQ, combined pipeline and research methodology

- **Unstructured magnitude pruning:** deterministic stable masks, layer/global selection, actual zero/masked counts, masked gradients and round trips are CPU-tested. Tensor dimensions stay intact; masks may increase container size. No physical compression or speedup follows from masking.
- **Physical Phi MLP pruning:** fc1 rows and fc2 columns are sliced together, residual widths preserved, intermediate configuration updated for uniform widths. Synthetic vendored Phi forward/save/reload and strict state reconstruction are tested. Heterogeneous standard-HF reload and real Bunny recovery are not established.
- **Attention/vision pruning:** not supported by the controlled pruning framework. Attention matrices may be masked as weights, which is not head removal. SigLIP channel/head pruning is not implemented.
- **AWQ mathematics:** INT4/INT8 zero-inclusive asymmetric grids, group boundaries, constants/zeros/tiny values, FP32 scales, odd-width packing, reconstruction and owned serialization are tested. Extreme nonrepresentable grids fail. Activation search/clipping and hooks are exercised on small actual modules. These tests do not prove real multimodal calibration, quality preservation, kernel speedup or GPU feasibility.
- **Calibration:** local text records with deterministic sample selection, token hashes and observed consumption. Text-only calibration does not activate SigLIP/projector and is not VQA calibration. Selected IDs are unavailable for the generic text loader; provenance says so.
- **Seven treatments:** dense; unstructured; structured MLP; AWQ INT4; AWQ INT8; unstructured→AWQ INT4; structured MLP→AWQ INT4. Existing tests actually generate/export/reload all seven owned artifacts and preserve the base state. A small random multimodal dense/structured+AWQ interface path is also exercised. These are synthetic comparisons only.
- **Fairness:** suite records fixed base-state/held-out/calibration/protocol/generation identities. IDs and selected exact-token/image overlap checks are useful but do not prove absence of semantic or pretraining contamination. Core persisted comparison is deliberately more conservative and not a real cross-treatment analysis bridge.
- **Scoring:** normalized exact match is a proxy. No official VQA/POPE/MME scorer or human assessment is implemented/validated. Keep visual evaluation and portable text generation separate.
- **Timing:** runner excludes CPU preprocessing and text decoding, includes model generation including visual encoding, uses synchronization on the authorized CUDA branch, records raw per-sample timings and mean/median/nearest-rank P95. Warmup is configurable; three warmups is not empirically established convergence. Repeated runs/variability are a protocol requirement, not automated real evidence. Backend no-warmup latency is a different method.
- **Memory/size:** runner records peak allocated with reserved metadata; not full process/driver VRAM. Actual CUDA branch and RAM/VRAM fit are unverified. Weight-file bytes and serialized export bytes are different size definitions; raw container ratios do not establish fair compression. Parameter counts distinguish registered parameters, packed logical weights and buffers.
- **Trust:** no current real quality, latency, throughput, VRAM, utilization, accuracy retention or compression improvement can be trusted as measured research evidence because none was acquired in this session. All comparison-matrix research cells remain Pending / Not measured.

## 8. Tests, builds and exact validation evidence

Existing dependencies only: Torch **2.2.2+cpu**, Transformers **4.37.2**, no CUDA runtime reported by this installed Torch. No installs/upgrades or asset downloads.

CPU/offline prefix for every Python test/probe:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
```

Final repository suite (prefix applied as environment assignments):

```sh
.venv-vlm-awq/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider --junitxml=/workspace/audits/mission14/verified-tests.xml
```

**477 passed, zero failures/skips, one warning, 36.05 s.** Independent new suite:

```sh
.venv-vlm-awq/bin/python -m pytest experiments/tests/test_mission14_*.py -q -p no:cacheprovider --junitxml=/workspace/audits/mission14/adversarial-tests.xml
```

**68 passed, zero failures/skips, one warning, 3.97 s.** The warning is Starlette's use of deprecated AnyIO `BlockingPortal`; it was not hidden.

External remaining-gap probes: `PYTHONPATH=/workspace/final_year_project .venv-vlm-awq/bin/python -m pytest /workspace/audits/mission14/test_open_findings.py -q -p no:cacheprovider --junitxml=/workspace/audits/mission14/open-tests.xml` → **5 failed intentionally**, matching R1, both R2 functions and both R4 numeric/collection cases (4.15s). They also reproduce against unchanged source. These are outside the passing repository suite; the project is not asserted defect-free.

Other actual checks:

- `cd Interface && npm run lint && npm run build`: both passed; Vite 6.4.3, 1676 modules, final build 1.46 s. No browser interaction test performed.
- Compiled **198** tracked/new Python files using `py_compile.compile(..., doraise=True)` with outputs outside source. This parses quarantined code without executing it.
- `bash -n` passed for all **7** tracked shell scripts; no historical script executed.
- `git diff --check` and diff against exact parent passed after a trailing blank line found early was corrected in a later commit.
- AST comparison verified all existing test assertions unchanged. Four TestClient setups were changed only to specify the supported loopback peer/Host. No existing test was removed, skipped or weakened.
- AST loader/acquisition inventory and source search performed; only central production direct Torch loads found. Installed third-party internals and untrusted producers are outside this claim.

Baseline logs: `baseline-tests.log`/`baseline.xml`. Negative logs: `before.log`, `network-before.log`, `collection-before.log`, `inference-before.log`, `generation-before.log`, `token-interface-before.log`, `pipeline-replacement-before.log`, `extreme-exponent-baseline.log`. Intermediate known regression: `after-resource.log`, `timing-investigation.log`, `self-review-before.log`; corrected isolated run `timing-corrected.log`. Final logs: `verified-tests.log`, `adversarial-tests.log`, `open-tests.log`, `frontend.log`, `static-checks.json`, `final-call-inventory.json`.

## 9. Commit and changed-file inventory

| Commit | Change / audit scope |
| --- | --- |
| `2c0a431e6ed7f8018a0f3e9a6dd66b264400ade1` | Cycle 1 resource admission, bounded readers/collections/body and regressions |
| `aff25d918f29bd74d003ac30bbd9dc95d97adc13` | Cycle 2 enumerated legacy quarantine and static regressions |
| `bfcbe2a70719190572624254686145dbf6d4f562` | Cycle 3 measurement metadata equivalence |
| `30e12128a1f9a890ba68af21bd8e872c2f7eeeee` | Cycle 4 Phi residual/vocabulary/normalization admission |
| `51d42fb7b7a525288c1ad91e25a38ea4d71d9341` | Cycle 5 safe-format byte budget |
| `5fb491c445dcd62cdf98355cbb728c43da76ff46` | Cycles 6–7 content-bound consent/lineage and backend inference |
| `0524bf17953642af98b092b1a79e033869ee0352` | Cycle 8 local API network boundary and test transport setup |
| `86834bce2a7eecc6cbc4125003723e317b0e9c3c` | Cycle 1 self-review corrections: direct state discovery / stdout amplification |

Final source changes span shared JSON/store/compare/evidence readers; evaluation/calibration/CLI ingestion; checkpoint/portable integrity and combined reload; local API/schema/registry/jobs; frontend consent/listener; ten legacy guards; eight new test modules; four existing test transport setups. Five required Mission 14 records document the outcome. Dependency versions, main and prior milestone branches were not changed. Full changed-file names are obtained reproducibly with `git diff --name-status 12530f504eba2112360ac4b8bdb90037e0bf6e72..HEAD`.

## 10. GPU pilot prerequisite matrix

This is an assessment only. No instructions here authorize acquisition, activation or spending.

| Prerequisite | Status | Missing evidence / required action |
| --- | --- | --- |
| Bunny/Phi-2 | BLOCKED | Candidate BAAI/Bunny-v1_0-3B only. Exact licensed revision, staged files, complete verified manifest/projector are unavailable. |
| SigLIP | BLOCKED | Candidate google/siglip-so400m-patch14-384 only. Matching licensed revision/config/weights and checksums unavailable. |
| Tokenizer / processor / projector | BLOCKED | Bind all files to the actual model/encoder manifests, validate vocabulary, processor dimensions, projector tensors and prompt contract. |
| Dataset | PARTIALLY VERIFIED | Manifest/selection/fingerprint machinery is CPU-validated; actual held-out dataset/split/selected content is BLOCKED. |
| Scorer | UNVERIFIED | Choose official task/version/answer extraction; justify any proxy. Exact match is not official VQA scoring. |
| Leakage controls | PARTIALLY VERIFIED | Selected ID/image/token overlap controls tested. Actual split review and semantic/pretraining contamination remain UNVERIFIED. |
| Recovery | NOT APPLICABLE to pristine dense pilot; BLOCKED for recovery studies | Tiny synthetic reloads do not establish real training/recovery source lineage or artifact compatibility. |
| GPU runtime | BLOCKED | Installed Torch is CPU-only. Camber CUDA-enabled wheel/runtime, actual hardware availability and compatibility not inspected on target. |
| RAM / VRAM | BLOCKED | No measured dense-model peak RAM or VRAM. Template free-memory thresholds and analytical KV estimate are not fit evidence. |
| Billing / storage / exhaustion | BLOCKED | Conclusive zero-overage, storage charges, credit exhaustion behavior and hard limits missing. A student allowance is insufficient. |
| Authorization | BLOCKED | Separate explicit approval for asset acquisition/storage and GPU/node execution has not been given. |

**L4 template assessment:** 32 samples, batch 1, 256 text tokens, 32 continuation tokens and three warmups are reasonable initial *protocol bounds*, not a proven fit/quality/performance budget. It still contains REPLACE placeholders, no concrete manifest paths and `hash_large_assets=false`, which cannot establish complete large-asset verification. Declared free thresholds are 24 GiB RAM and 16 GiB VRAM; actual available capacity and load peaks are unknown. Vision tokens/workspace/temporary copies add memory and may dominate. The template must remain non-executable as provided. Bound disk staging/checkpoint copies within the user's actual quota; do not assume stopped storage is free.

## 11. Separate final verdicts

| Category | Verdict | Reason |
| --- | --- | --- |
| Repository CPU engineering integrity | CONDITIONAL GO | 477 passing repository tests and actual negative/positive evidence; R1/R2/R4 remain medium, external assertions fail; independent review still needed. |
| Supported-path security readiness | CONDITIONAL GO for owned local-operator fixtures only | Explicit trust, identity and scoped limits; no authentication/sandbox/OS quotas, incomplete asset discovery and third-party parsing bounds. Public deployment NO-GO. |
| Repository-wide legacy isolation | NO-GO | Ten modules quarantined, remaining independent acquisition/mutation scripts insufficiently isolated. |
| Experiment tracking/comparison integrity | NO-GO for general comparison robustness | Ordinary invalid/failed/unmeasured/mismatched records are rejected, but R4 can abort whole collections, CLI history is inconsistent, and real-treatment bridge/scientific protocol remains unvalidated. |
| Real-asset evaluation readiness | NO-GO | Assets, scorer, actual split and complete manifests missing; real compressed multimodal bridge unsupported. |
| GPU pilot readiness | NO-GO | Independent asset/code review, target fit/runtime, billing limits and separate authorization missing. |
| Research execution readiness | NO-GO | No approved real dataset/scorer or measured baseline; synthetic success is not research validation. |

No known remaining critical/high failure was reproduced in the enumerated supported CPU fixtures after fixes; that narrow observation is not a repository-wide security certification. The open resource/legacy findings and unverified model branches prevent a general readiness claim.

## 12. Exact resume and prioritized handoff

1. Verify audit branch HEAD, clean status and this report's documentation commit; preserve all original refs. Read `MISSION_14_HANDOFF.md` and external evidence if available. Do not reset/rewrite existing work.
2. Start a fresh bounded CPU remediation session. First reproduce/fix **R4**: normalize the extreme-underflow validation failure at the shared JSON boundary, then verify API rejection and unaffected valid comparison peers. Next fix **R1** through the real history CLI with a malformed peer; add its failing regression and preserve valid records/diagnostics.
3. Fix **R2**: bound both model/asset inventory traversal and aggregate hashing before materialization, with explicit reviewed profiles, symlink handling and valid large-manifest tests. The 10001-empty-file reproduction requires no models/datasets.
4. Extend static quarantine review to **R3** and remaining standalone training/acquisition/evaluation launchers; never execute them for testing. Preserve the vendored Phi definitions and pure helpers needed by supported paths.
5. Review core comparator's strict metadata/asset equivalence before implementing a real cross-treatment study. Do not loosen evidence to make synthetic comparisons look research-valid. Require base lineage, task/split/scorer/effective settings and comparable size/timing definitions.
6. Independently re-audit the Mission 14 fixes, including frontend consent and server/proxy assumptions. Keep tests offline and CPU-only, use existing dependencies, and rerun full suites/checks after actual code changes.
7. Only after an external research/billing/asset decision, prepare a pristine dense pilot package: licensed immutable revisions and complete manifests, verified held-out task/scorer/prompt, matching non-S2 Bunny/SigLIP/projector interfaces, acquisition/storage approval, hard financial/resource/stop conditions and a compatible target runtime. A read-only preflight is not execution permission.
8. Do not start a node, download assets, run GPU inference or produce benchmark claims in this session. The next research experiment, after all gates close, is dense baseline only; inspect its actual outputs before planning compression.
