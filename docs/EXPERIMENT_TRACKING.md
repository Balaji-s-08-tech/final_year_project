# Experiment tracking and benchmark contract (version 1)

This layer records evidence; it does not supply a quality evaluator or GPU performance
profiler. No full Bunny result is established by these infrastructure tests.

## Architecture and storage

`experiments/store.py` provides a local POSIX filesystem store. No database or new
dependency is required (psutil was already required). The store uses UUID4 IDs,
exclusive ID claims across experiment names, `flock` for lifecycle serialization,
and fsynced temporary files published with an atomic no-replace hard link. Run
initialization occurs in a hidden staging directory, renamed after all initial
files are complete. Failed staging directories/ID reservations remain evidence;
history excludes hidden incomplete directories. Storage must support POSIX locks,
hard links and atomic rename; NFS/object stores and Windows are not supported.

Default backend layout, already covered by the checkpoints Git ignore rule:

```text
checkpoints/experiments/.ids/<uuid>        # permanent exclusive ID reservation
checkpoints/experiments/dashboard_benchmark/<uuid>/
    experiment.json                      # immutable identity/provenance
    config.json                          # immutable requested/effective configuration
    environment.json                     # immutable environment snapshot
    state_000000.json                    # CREATED, revision zero
    state_000001.json                    # subsequent immutable lifecycle snapshots
    execution.log                       # append-only structured execution messages
    stdout.log                          # retained raw output
    stderr.log                          # retained stderr / exception trace
    metrics.json                        # terminal benchmark schema
    artifacts.json                      # terminal output paths
    failure_report.json                 # FAILED/CANCELLED only
    artifacts/                          # outputs belonging exclusively to this run
```

State transitions: CREATED → QUEUED or RUNNING; QUEUED → RUNNING; RUNNING →
COMPLETED, FAILED or CANCELLED. CREATED/QUEUED may also fail or cancel. Terminal
runs cannot transition or accept log/snapshot updates. Earlier state revisions
and initial metadata never change. Immutability is an application contract, not
an OS security boundary against a user editing files directly.

The highest numbered complete state revision is authoritative. Terminal state
contains metrics, artifact paths and failure details. Terminal JSON convenience
files are published before that revision. A crash between those operations leaves
an explicitly nonterminal record; recovery retains existing evidence and appends a
FAILED revision. Consult the revision, not a convenience file alone, after a crash.
This is atomic per-file publication, not a multi-file transactional database.

## Identity and provenance

The JSON Schema is [experiment-v1.json](schemas/experiment-v1.json). Unknown model,
dataset and hardware fields are null; they are not inferred from filenames. Each
identity includes commit, branch, working-tree dirty status, source hashes, a
canonical sorted-JSON config hash, seed, model/checkpoint information, dataset
revision/fingerprint and calibration/evaluation configuration. Specify model
and dataset revisions in programmatic run creation; unavailable fields remain null.

Environment snapshots contain Python/OS, dependency versions, CPU/RAM, PyTorch,
CUDA runtime when Torch is importable, and optional `nvidia-smi` name/driver/free-MiB
output. Collection never loads a model, creates a CUDA context, dumps environment
variables, or reads credential files. Missing tools give null/UNAVAILABLE fields.

Initialization hashes only explicitly supplied sources and small local checkpoint/
dataset files (8 MiB default maximum). `max_hash_bytes=None` opts into full hashing;
oversized inputs retain SKIPPED_SIZE_LIMIT rather than a fictional digest. Canonical
configuration hashes reject nonfinite JSON. Model config hashes can be supplied
before loading; AWQ additionally records the loaded model config hash in its
artifact sidecar. Local hashes establish content identity, not recovery lineage.
Configurations are retained verbatim: callers must not put credentials in them.

## Metrics and units

[benchmark-v1.json](schemas/benchmark-v1.json) describes all required metric slots.
Every metric has `value`, `unit`, `method`, `status`, and optional explanatory
`metadata` (serialized as an object). Status is MEASURED, UNMEASURED, UNAVAILABLE,
or FAILED. Only MEASURED permits a numeric value and requires a nonempty method;
all other statuses require null. Real measured zero is allowed. Nonfinite values,
negative measurements and incompatible units are rejected.

| Fields | Unit / definition |
| --- | --- |
| original_model_size, compressed_model_size | bytes, integer artifact size under the stated filesystem method |
| original_parameter_count, remaining_parameter_count | parameters, integer tensor-element count; never inferred from file size |
| compression_ratio | ratio, original bytes / compressed bytes; undefined if compressed size is zero |
| quantization_error | weight_units; method/metadata must identify MAE, max error, etc. |
| inference_latency, median_latency, p95_latency | seconds; identify warmups, samples, synchronization and percentile method |
| throughput | samples/second; identify batch size and timing boundary |
| tokens_per_second | tokens/second; count actual generated tokens |
| peak_gpu_memory | bytes; distinguish allocated/reserved and reset boundary in method |
| gpu_utilization | percent in [0,100]; identify sampling interval and aggregation |
| execution_duration | seconds from perf_counter, excluding experiment initialization |
| baseline_accuracy, compressed_accuracy | fraction in [0,1]; identify task, split, prompts and scoring |
| accuracy_retention | ratio, compressed / baseline accuracy; undefined for zero baseline |
| perplexity | dimensionless; identify tokenizer, corpus and token-loss aggregation |

Quantization includes method (AWQ or null), bits (4/8 or null) and calibration
configuration. Ratios are not invented from unavailable inputs. The current CLI
measures artifact size and duration only. Backend text generation also records its
existing CPU generate timing, explicitly labeled without warmup; it is not a
controlled performance comparison. Other slots remain unmeasured.

## Artifact precision and compatibility

Benchmark arguments and dashboard `bits` never establish artifact precision. New
AWQ exports write `<checkpoint>.metadata.json` containing schema_version=1,
filename, byte size, SHA256, format=`portable_awq`, precision=`AWQ INT4` or
`AWQ INT8`, quantization method/bits/calibration, and model_config_sha256. Export
reserves `awq_config.json` exclusively and publishes the checkpoint atomically.
Repeated or concurrent exports into that namespace fail rather than replacing it.
The exporter checks actual PackedAWQLinear bit widths before declaring precision;
dense models or contradictory packed layers are rejected.
CLI exports now serialize packed layers under their stable importable module path;
a fresh-process reload regression covers the prior `__main__` pickle failure.

Dense artifacts may use the same sidecar contract with format=`dense`, precision
FP16/FP32/BF16 and quantization=null. The exporter/operator must establish those
facts; a caller-provided label alone is not verification. Inspection validates
metadata, checks contradictions and verifies file size and SHA256 without
unpickling model objects. This trusts the producing exporter, not arbitrary claims
about the internal tensors of an untrusted pickle. A hash does not authenticate
the producer. Legacy checkpoints still load through the existing loader, but
legacy files lacking bound metadata get UNKNOWN precision in benchmarking and
are not automatically registered for portable inference. Re-export trusted legacy
models to obtain metadata; do not manufacture sidecars from filename guesses.

Full export/verification hashing is intentionally separate from bounded run
initialization. Verification and dashboard discovery currently read each bound
artifact in full, which can be costly for large files. Directory artifacts have
measured aggregate byte size but UNKNOWN precision. An artifact being modified
concurrently is unsupported: finish export before registration/benchmarking.

## CLI and Python use

Run commands from the repository root, with the project Python environment:

```bash
python -m experiments create --name awq_int4 --config /path/to/config.json
python -m experiments history --root checkpoints/experiments
python -m VLM.quantization.benchmark --artifact /path/to/checkpoint.pt --output /path/to/new_manifest.json
python -m experiments compare /path/to/baseline_run /path/to/candidate_run --output /path/to/new_comparison.json --csv /path/to/new_comparison.csv
```

The benchmark CLI retains the legacy --original_fp16/--pruned_fp16/--pruned_int8/
--pruned_awq_int4 arguments, but those argument names are roles, not precision
evidence. The JSON output is now a versioned envelope with `metadata` and `results`;
each legacy result entry keeps its previous fields and adds a structured `benchmark`.
When multiple roles are supplied, terminal summary metrics use the last supplied
role as candidate, retaining original size and the measured artifact-size ratio
when an original artifact was provided. Every role remains in the manifest.
The existing dashboard/import endpoints already understand that envelope. Standalone
benchmark runs live under `<output parent>/experiments/benchmark/<uuid>`; the
requested output path and its `.reservation` are exclusive. Retry with a new path.

Standalone AWQ commands retain their arguments but now put outputs beneath
`<output_dir>/experiments/awq_intN/<uuid>/artifacts/`, rather than reusing filenames
directly beneath output_dir. The private --experiment-dir option attaches child
CLI execution to an already RUNNING backend run; the parent owns finalization.

```python
from experiments.store import ExperimentStore
from experiments.schema import benchmark, measurement

run = ExperimentStore('checkpoints/experiments').create(
    'baseline', {'seed': 42, 'batch_size': 1},
    model={'identifier': 'tiny', 'architecture': 'phi', 'revision': 'pinned-revision'},
    dataset={'identifier': 'heldout', 'revision': 'pinned-revision',
             'evaluation': {'split': 'test', 'scorer': 'exact_match'}})
run.transition('RUNNING')
run.log('Beginning controlled evaluation')
# Record actual evaluator outputs with measurement(..., status='MEASURED', method=...).
run.transition('COMPLETED', metrics=benchmark())  # explicitly unmeasured example
```

Comparison JSON/CSV identifies baseline/candidate, configuration and model/dataset
differences, warnings, and candidate-minus-baseline accuracy, latency, VRAM and
size differences. Unmeasured/incompatible units/methods or incomplete runs yield
null deltas. Differing/unknown protocols and model identity are warned about;
treatment configuration differences remain visible and deltas are marked
CONTEXT_DIFFERS. Use this for pruning ratios, model variants, INT4/INT8 and
combined treatments; it does not assert causality or statistical significance.

## Backend persistence and imports

Existing job routes retain lowercase statuses and response structures. Durable
snapshots retain job records/logs and uppercase experiment statuses. A new manager
restores history; queued/running records whose owner PID has exited become FAILED,
without automatic retries. Use one API worker; cross-process scheduling, PID reuse
detection and orphan subprocess supervision are not implemented. A still-live
owner is left alone. Do not start multiple workers against the same run store.
`GET /api/experiments` exposes the local dashboard/CLI store's experiment history.

Results import retains the supplied manifest and existing response shape in a
separate immutable import run under `checkpoints/imported_results/import/<uuid>`.
Imports preserve source labels/time; supplied values remain unverified external
claims. Versioned nested benchmark documents are validated. Imported values are
not silently promoted into locally measured experiment metrics. `/api/benchmarks`
continues discovering all manifests beneath checkpoints, including prior layouts.

## Camber handoff

Codex Cloud does not access Camber's filesystem/GPU. Use this feature branch and
commit as the source revision. No infrastructure activation is part of this change.
After separate GPU authorization, stage licensed weights/data on Camber, pin
model/dataset revisions and evaluation protocol, and create a new run there.
Collect environment.json on that actual machine; do not copy Codex's hardware
snapshot as Camber evidence. The reported L4/23 GB VRAM/32 GB RAM capacity is a
target configuration, not proof that a model/training workload fits.

Archive the completed run's JSON/config/logs plus comparison CSV. Commit only small
reviewed outputs under a separate research-results branch/directory, using
`git add -f` only for deliberately reviewed ignored files. Inspect files for paths
or sensitive information before sharing. Keep checkpoints/datasets outside Git;
use approved artifact storage for large outputs and preserve their identifiers and
hashes. Never activate a Camber node or start an expensive experiment as part of
tracking setup. Compare only after actual outputs and logs have been inspected.

CPU validation:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest experiments/tests backend/tests VLM/quantization/tests -q
```

## Milestone 002 validation record

Base: AWQ repair commit `e2fce72ded969a1220569275384860288b9c848c`.
Hypothesis: exclusive publication plus append-only lifecycle records prevents
result replacement, and content-bound precision metadata removes request-based
mislabeling while preserving the existing API contract.

Local CPU evidence: **159 tests passed** (119 existing, 40 new), zero failed or
skipped, with one third-party Starlette/AnyIO deprecation warning. Tests exercise
thread/process concurrency, atomic write failure, duplicate protection, all
terminal states, bounded provenance, metric validation, protocol differences,
JSON/CSV commands, restart/interruption history, imports, contradictory/stale
metadata, export collisions and repeated CLI runs. Fresh-process CLI artifact
reload exposed and verified a repair to the earlier `__main__` pickle failure.
Syntax compilation and whitespace checks passed. These are infrastructure tests,
not evidence of model quality or GPU performance improvements.

Compatibility changes are deliberate: new CLI runs use unique subdirectories,
benchmark JSON uses an envelope, and metadata-free artifacts report UNKNOWN.
Next increment: implement a controlled baseline evaluator with fixed held-out
data, actual token counts, warmup/synchronization rules and resource profiling.
Prepare its CPU tests first; Camber execution still requires separate authorization.
