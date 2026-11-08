# Autonomous research development program

Starting source: a1b3272ebbf81d2b8515e832a017c859cba82ba3. Remote main remains
66c1d25252114451532786420b1e92e16de8a778. Branches and ancestry verified on
2026-11-08. Open PR inspection returned GitHub API Forbidden; no PR was created.

## Mission 3.5 — research integrity

Branch: fix/mission-3-5-research-integrity. The five audit findings are repaired:
shifted supervised-target eligibility/finite loss; shared explicit GPU admission;
strict serialized units; legacy/structured result adaptation and evaluator history;
small asset manifests, effective generation/tokenizer/processor settings and
comparison warnings for missing/changed/incomplete content identity.

Baseline 236 tests; current validated pass: 245 CPU tests. No failures/skips; one
third-party deprecation warning. Frontend validation recorded at commit completion.
Changed areas: runtime_policy, sampler/loss, AWQ/model loaders, evaluator provenance,
metric/result adapters, backend status/history and result display; new regressions.
No paid resources, downloads or GPU execution. Large weight hashes remain explicitly
incomplete unless supplied by later verified asset manifests. Next: 3.6/3.7 architecture
and baseline protocol, then CPU pruning, quantization verification and pipeline.

Commit SHAs are appended after each milestone commit in its dependent milestone.

Mission 3.5 committed/pushed: 6cd00f9e5e6ab0b643c0a2b6d076f76f2625aa72.
Frontend type/build passed. Audit resolution is in docs/reports/RESEARCH_INTEGRITY_REPAIRS.md.

## Missions 3.6 / 3.7 — architecture and baseline protocol
Branch: docs/mission-3-6-3-7-protocol, stacked above 3.5. Architecture/dependency
diagram and baseline protocol distinguish full visual evaluation from portable
text controls. Added local model manifest construction/verification and selected
split-disjointness checks, with three CPU tests. Full suite: 248 passed, no failures
or skips, one existing warning. Protocol/model/dataset templates intentionally do
not invent unavailable pinned revisions or official scorer decisions. Current code
changes are manifests and tests; documentation does not claim unavailable assets
or full reconstruction/GPU validation. Next: architecture-specific pruning.

## Mission 4 — pruning framework
Protocol prerequisite committed/pushed: 7bd91943b7bddf9b9046860eacb30c64249dadd8.
Branch feat/mission-4-pruning-framework. Implemented deterministic layer/global
magnitude masks, physical paired Phi MLP slicing, explicit mask materialization,
shape/state reconstruction and sparsity reporting. Added eleven tests including
tiny real Bunny+SigLIP full forward/generation. Full suite: 259 passed, no failures
or skips, one existing third-party warning. Changed compression/pruning.py, its
tests and methodology. Attention/vision/projector physical pruning remains explicitly
unsupported. No GPU/downloads/spending. Next: AWQ calibration and artifact verification.
