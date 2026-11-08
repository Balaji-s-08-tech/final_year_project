# Mission 15 — final stabilization and engineering exit

## Verdict and checkpoint

**GO for the bounded CPU stabilization exit.** All three specified Mission 14 defects are repaired, their before-fix reproductions are recorded, and complete available CPU verification passes. This verdict covers the three repairs and preserves the established supported CPU interfaces. It is not a repository-wide security certification or a research/GPU readiness claim.

**GPU execution remains unauthorized.** Real assets, official task/scorer selection, target runtime/fit, financial guarantees and separate execution authorization remain external blockers. No GPU pilot, acquisition, training or research benchmark was performed. No additional mission or recursive remediation programme was started.

- Exact starting branch: `audit/mission-14-independent-readiness`.
- Exact starting HEAD: `ee3170c8c6634055e6539f69ebfff2cf44f5bc1b`; working tree verified clean.
- New branch: `fix/mission-15-final-stabilization`, created directly from that commit.
- Final validated implementation HEAD: **`26ccba2a2a21eeff4e2c50a53cbc905521429c1f`** — `fix(stabilization): close numeric history and asset inventory exit blockers`.
- A documentation-only commit contains this report and the final handoff. Its exact final branch HEAD is recorded in the task output and external final-state evidence; retrieve it with `git rev-parse HEAD` or `git log -1 --format=%H -- MISSION_15_HANDOFF.md`. Source is unchanged after the validated implementation commit.

## Original failures and repairs

All nine original regression cases were written before changing implementation and failed on the exact starting source: **9 failed, one warning, 2.97s**. This is actual negative evidence, not inferred from previous reports. Test: `experiments/tests/test_mission15_stabilization.py`; logs: `/workspace/audits/mission15/before.log` and `before.xml`.

### 1. Extreme numeric exponents — repaired (Mission 14 R4)

**Reproduction:** parse nonzero or zero mantissas with enormous signed exponents, such as `1e-999999999999999999999999999999999999999999`. The original `_float` obtained zero from `float()` and then raised `decimal.InvalidOperation` while constructing Decimal. That exception escaped the ValueError contract, aborted mixed comparison collections and produced HTTP500 instead of a validation response.

**Fix:** `experiments.jsonio` normalizes `DecimalException`, `OverflowError` and conversion/range `ValueError` in numeric parser hooks into `JsonValidationError`, a ValueError subtype. It provides `as_dict()` with `code=INVALID_JSON_NUMBER` and a meaningful message. Integer-range and nonfinite-constant rejections use the same representation. Existing ValueError isolation remains compatible; invalid numbers are never substituted with zero or promoted to successful measurements.

`backend.admission.RequestAdmission` returns **HTTP422** with a structured `detail` object for these errors. Other existing validation responses retain their contracts. Comparison readers reject the malformed candidate with per-record diagnostics and preserve unrelated valid candidates.

**Tests:** actual extreme signed exponents, API rejection before endpoint processing, mixed valid/bad comparisons, overflow, underflow, nonfinite constants, INT64 bounds, zero/small/subnormal/maximum-finite positive controls. Successful parsing of an ordinary representable zero is preserved; unsafe exponent conversions are explicitly rejected.

### 2. CLI history isolation — repaired (Mission 14 R1)

**Reproduction:** create good and bad local runs; put `{broken` in the bad run's state; invoke the real `python -m experiments history`. Previously the command exited 1 and printed no valid peer.

**Fix:** the CLI reads each bounded discovered run independently, validates the basic lifecycle object/run identity and preserves valid state rows unchanged. A corrupt run produces:

```json
{
  "status": "INVALID",
  "path": "/local/experiment/run",
  "stage": "lifecycle",
  "error": {"type": "JSONDecodeError", "message": "..."}
}
```

Numeric validation errors additionally preserve their machine-readable code. The command completes with a JSON list containing valid rows and explicit invalid diagnostics. It does not claim every row is valid and does not mutate or repair corrupt persisted records. Whole-store discovery/record budget exhaustion remains an explicit failure; it does not silently omit records.

**Tests:** real subprocess CLI, valid-peer preservation, malformed/null/list/invalid-status state variants, path/stage/type/message diagnostics and byte-for-byte preservation of corrupt input. Existing creation/history CLI tests remain unchanged and pass.

### 3. Asset-discovery budgets — repaired (Mission 14 R2)

**Affected readers:** `experiments.manifests.model_manifest` (including verification) and `evaluation.provenance.asset_manifest`. Both originally returned inventories for 10001 owned empty files without admission.

**Fix:** both readers use the small shared `experiments.assets` inventory implementation. Streaming `os.scandir` avoids materializing an unbounded directory listing. It enforces entry count, file count, traversal depth, relative-path length, individual/aggregate source bytes and aggregate planned hashing bytes. All regions are admitted before any payload hashing. Model and encoder share one operation budget.

Only regular files are hashed. Directory symlinks are not traversed; file symlinks inside a directory source must remain within that source. Internal file aliases remain supported, as do directly selected portable files. Hashing uses one regular descriptor, reads at most admitted size plus a growth sentinel, and rejects changed/shrunken/growing files rather than publishing misleading evidence. These are consistency checks under owned immutable-storage assumptions, not a hostile-filesystem sandbox.

**Default explicit profile:**

| Limit | Default |
| --- | --- |
| Traversal entries / files | 10000 / 10000 across all regions |
| Directory depth | 32 below source root |
| Relative-path bytes | 4096 (UTF-8) |
| Individual source-file size | 16 GiB |
| Aggregate source bytes | 32 GiB |
| Aggregate hashing bytes | 32 GiB |
| Default per-file hashing threshold | 8 MiB |

Count/depth/path limits bound metadata work; byte caps accommodate multi-GB model shards while retaining finite admission. They are policy caps, not measured model sizes, RAM requirements or proof of fit. Existing `hash_large=False` still marks files over 8 MiB `SKIPPED_SIZE_LIMIT` and retains INCOMPLETE status. Explicit `hash_large=True` hashes only within the finite profile; it no longer implies unlimited I/O. Reviewed callers can supply typed `AssetLimits` via the additive `limits=` keyword on both readers/manifest verification. Boolean and non-positive/noninteger limits are rejected. A larger profile grants no acquisition, GPU or spending authorization.

Valid manifest field shapes, SHA256 content definitions, deterministic file keys, hidden-file inclusion and missing/skipped region diagnostics are preserved. Exhaustion raises a descriptive ValueError; there is no silent partial verified inventory.

**Tests:** original 10001-file cases for both public readers; small explicit count/file/aggregate/hash/depth boundaries; admission spies before any hashing; model+encoder aggregate budget; valid manifest verification/hash equality; hidden files; skipped files; internal aliases/external link rejection/directory-loop non-traversal; file growth; single portable source and missing regions. Tests use tiny or empty owned fixtures only.

## Exact verification

All Python tests/probes ran with:

```sh
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
```

Existing dependencies only; no packages were installed or upgraded, and no assets were acquired. Git synchronization is the explicit exception to offline execution and targets only the new branch.

| Check actually executed | Result |
| --- | --- |
| Nine original regressions before fix | 9 failed, 2.97s, one warning |
| Relevant existing/new initial suite | 70 passed, 5.04s, one warning |
| Complete Mission 15 regression module | **45 passed**, 3.37s, one warning |
| Complete available repository CPU suite | **522 passed**, 38.71s, zero failures/skips, one warning |
| Original five external Mission 14 failing probes | **5 passed**, 1.87s |
| Python compilation | **200 files passed** |
| Shell syntax | **All 7 scripts passed**; no historical script executed |
| Frontend type checking | Passed |
| Frontend production build | Passed; Vite 6.4.3, 1676 modules, 1.98s |
| Git whitespace/diff checks | Passed |
| Existing tests | No existing test file/assertion changed or removed |
| Original local branch refs | All compared with saved starting inventory; unchanged |

Warning: Starlette's deprecated AnyIO `BlockingPortal` alias. It was reported, not hidden or silenced.

Full suite command:

```sh
.venv-vlm-awq/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider --junitxml=/workspace/audits/mission15/full-tests.xml
```

Focused command:

```sh
.venv-vlm-awq/bin/python -m pytest experiments/tests/test_mission15_stabilization.py -q -p no:cacheprovider --junitxml=/workspace/audits/mission15/regressions.xml
```

Original Mission14 probes (when session evidence exists):

```sh
PYTHONPATH=/workspace/final_year_project .venv-vlm-awq/bin/python -m pytest /workspace/audits/mission14/test_open_findings.py -q -p no:cacheprovider
```

Frontend commands, from `Interface`: `npm run lint` and `npm run build`. Compilation used `py_compile.compile(..., doraise=True)` over all tracked/new Python files with bytecode output outside the repository. Syntax used `bash -n` for each tracked `.sh`; none was executed. Final source/diff review was scoped to the three repairs, without beginning another audit programme.

Evidence directory: `/workspace/audits/mission15/` contains starting refs, `before`, `initial-fixes`, `regressions`, `full-tests`, `mission14-open-recheck` logs/XML, frontend output and static-check metadata. These are session-local evidence; the repository regression module is the durable reproduction for future workspaces.

## Changed files and commit inventory

Implementation commit: `26ccba2a2a21eeff4e2c50a53cbc905521429c1f`.

- `experiments/jsonio.py`: numeric conversion/validation exception normalization and structured diagnostics.
- `backend/admission.py`: structured numeric HTTP422 response.
- `experiments/__main__.py`: per-run CLI history isolation.
- `experiments/assets.py`: shared bounded inventory and admitted regular-file hashing.
- `experiments/manifests.py`: bounded model inventory/verification with explicit profile.
- `evaluation/provenance.py`: bounded shared model+encoder inventory.
- `experiments/tests/test_mission15_stabilization.py`: 45 focused cases, including nine original reproductions.
- `MISSION_15_PROGRESS.md`: persistent execution/validation ledger.
- `MISSION_15_HANDOFF.md` and this report: final documentation-only handoff commit.

No model-loading, pruning, quantization algorithms, scoring, treatment definitions, GPU authorization, dependency versions or frontend source were changed. Complete source history is `git log ee3170c8c6634055e6539f69ebfff2cf44f5bc1b..HEAD`; no existing history was rewritten.

## Remaining limitations and final backlog

These are not new remediation scope and do not reopen this stabilization mission:

- Unsupported standalone legacy training/acquisition scripts remain outside the supported routes; repository-wide isolation/security is not certified. Do not execute historical workflows as a workaround.
- Pickle requires explicit trusted-source acknowledgement; it is not sandboxed. Real immutable asset storage and source authenticity remain external responsibilities.
- Budgets are per operation, not OS CPU/RAM/disk quotas, global experiment I/O quotas or a distributed scheduler. Third-party config/tokenizer/image parser internals are not comprehensively resource-audited.
- Strict comparison metadata/asset equality may conservatively reject legitimate transformed treatments. Full-model compressed multimodal comparison/recovery lineage remains unvalidated; seven CPU treatments are synthetic evidence only.
- Physical Phi MLP pruning, mask sparsity and portable INT4/INT8 remain CPU/synthetic-validated; attention-head/SigLIP physical pruning is unsupported. No optimization benefit is asserted.
- Exact licensed/pinned Bunny/SigLIP/tokenizer/processor/projector assets, complete local manifests, approved held-out split/scorer and leakage review remain unavailable/unverified.
- Target CUDA-enabled PyTorch, measured RAM/VRAM/disk feasibility, Camber zero-overage/storage/exhaustion policies and separate acquisition/GPU authorization remain **BLOCKED**. CPU tests do not supply that evidence.

**Engineering exit:** the three specified blockers are closed with passing reproduction/regression/full verification. Stop stabilization after clean committed/pushed synchronization. The next meaningful work is separately authorized real research validation and prerequisite review, not an automatically invented engineering mission. **GPU execution remains unauthorized.**
