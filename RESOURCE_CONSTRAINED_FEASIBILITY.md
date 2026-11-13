# Resource-constrained Bunny compression feasibility

Base commit: `bfaa0e41aa30835cf6a320a4a7122fe571a8306f`.
Working branch: `codex/resource-constrained-feasibility`.
This assessment prepares one bounded pilot. It does not execute pretrained-model
compression, launch GPU work, publish code or alter existing assets/results.

## Evidence and conclusion

The user reports successful pretrained dense execution on Kaggle, run
`0d62dc4e38bb4939b078703cabad7cd6`, at the base commit: prediction
` The square is red.`, unchanged exact match 0/1, generation latency
3.146983278 seconds, peak allocated CUDA memory 7,141,774,848 bytes (6.651 GiB),
421 embedded vision tensors verified against 448 standalone tensors. These are
supplied GPU observations; this session has not independently rerun them. Reserved
VRAM and peak host RSS for this particular run were not supplied.

Available resources supplied for planning: approximately 29.8 GiB host RAM,
14.56 GiB on each of two T4s, and 5.75 GiB free disk after restoration. The two
GPUs' memories are separate. Existing code does not shard the model across them.

The current end-to-end preparation command is **not admitted** on this machine.
Its 73.59 GiB RAM gate is conservative, but full FP32 stage exports and overlapping
reload allocations are separate real blockers. Removing the gate does not solve
them. An AWQ-only endpoint has promising tensor/storage estimates, but its full
execution has not been demonstrated within this resource envelope.

The smallest proposed experiment is **one pretrained Phi MLP pair, layer 0,
25% paired physical pruning, CPU-only**, with a saved/reloaded operator artifact
and deterministic synthetic-input equivalence check. This is scientifically
interpretable as an operator correctness and resource-feasibility pilot. It cannot
answer whether pruned Bunny retains image-question accuracy or improves speed.
No full-VLM quality experiment is justified by the existing single red-square
smoke alone; a reviewed representative held-out protocol/scorer is also needed.

## Accounting inputs and meanings

Pinned architecture: hidden width H=2560, MLP width I=10240, L=32 decoder blocks.
The earlier supplied same-checkpoint architecture record reports P=3,212,732,480
registered parameters. These dimensions/count are architecture inputs, not a new
memory measurement. Check them against the latest Kaggle `architecture.json`.
Known staged weight files occupy 9,876,575,208 bytes (9.198 GiB), including redundant
Bunny vision and unused SigLIP text/head tensors. Disk bytes are not resident model
bytes. All figures below use GiB = 2^30 bytes.

Reproduce the tensor accounting without loading any weights:

```sh
python -m compression.resource_plan --hidden 2560 --intermediate 10240 --layers 32 --parameters 3212732480
```

The tool is read-only and does not change admission policies. Its tensor formulas
are checked against actual tiny Phi dense/pruned/packed state sizes in CPU tests.
They exclude nonparameter buffers, Python/runtime memory, allocator fragmentation,
source mappings/page cache, unmeasured serializer buffers and training activations.
They are not measured RSS/VRAM peaks or hard allocation limits.

For each block, six supported linear weight counts are 4H²+2HI. Other block
parameters total 7H+I: four attention biases, two MLP biases and input LayerNorm
weight/bias. Pruning removes floor(.25I)=2560 neurons and leaves I'=7680. The
remaining model count is P − L(2H+1)(I−I'). INT4 accounting includes two codes/byte,
FP32 scales and uint8 zero points per 128-column group, source FP32 biases and
uncompressed normalization. It does not infer kernel speed from storage.

## Stage budgets and feasibility

Only the dense GPU peak in the first row is a supplied measurement. Other numbers
are analytical tensor counts or explicitly provisional operating budgets. An
available-memory admission threshold is a precondition, not a process quota.

| Stage | Host RAM components / provisional planning budget | VRAM | Additional output disk | Verdict on current machine |
| --- | --- | --- | --- | --- |
| Existing dense FP16 smoke | Host peak unmeasured; existing smoke admission requires 20 GiB free | **Measured allocated 6.651 GiB**; reserved peak not supplied; admission 10 GiB free | Small prediction/config/log files; source weights already restored | Supplied smoke passed; no compression evidence |
| CPU FP32 restoration | Parameters 11.968 GiB + buffers/load temporaries/mappings; provisional 18–20 GiB reserve | 0 for CPU preparation | No new weights until export | Plausible individually, not proved; general 73.59 GiB gate remains |
| Physical MLP pruning after this change | About 12.359 GiB parameters + worst per-matrix score temporaries at start; reserve 20 GiB including runtime/mapping allowance | 0 | Pruning itself can emit small index metadata | Algorithm plausibly fits alone; general execution remains blocked |
| Full pruned export | Pruned parameters 10.406 GiB + cloned block state 7.816 GiB; possible serializer copy adds another 7.816 GiB, before overhead; provision 30–34 GiB | 0 on current preparation path | **7.816 GiB** payload + header | Disk exceeds 5.75 GiB; RAM not safely established |
| LoRA after pruning | Source + copied model **20.811 GiB**; rank-2 adapter/grad/Adam state about **20 MiB**; training activations/workspaces additional and unmeasured; provision at least 32 GiB for review | Current recovery is CPU-only | Adapter-only storage can be small, but current pipeline exports a full 7.816 GiB merged stage | No safe full-stage admission; not a claim that LoRA inherently requires 73.59 GiB |
| AWQ after pruning, before export | Starts with 10.406 GiB FP32 parameters; feature capture raw cap 5 MiB plus concatenations; scale/clipping/packing temporary matrices require further hundreds of MiB; provisional reserve 20–24 GiB | CPU plan uses 0; CUDA alternative puts the **whole** model on one device and needs a separate measured budget | Final pruned block INT4 payload **1.056 GiB** | Endpoint storage fits; execution and preceding stages remain unvalidated/blocked |
| AWQ only, without pruning/recovery | Dense FP32 parameters 11.968 GiB plus analogous search/calibration temporaries; provisional 20–24 GiB reserve | Same whole-model CUDA caveat | Final block INT4 payload **1.267 GiB** | Potential future candidate after pilot and reviewed stage admission; not authorized/executed |
| Full pruned reload/evaluation within current execute() | Prepared model remains alive while a fresh FP32 base and replacement linears are built: roughly **30.2 GiB** tensor residency can overlap, before mapped artifact pages/runtime | CUDA transfer/inference comes later; unmeasured compressed peak | Base dependencies plus existing 7.816 GiB artifact | Not admitted on 29.8 GiB; a fitting endpoint does not prove reload fit |
| Proposed one-MLP pilot | Source FP32 0.1954 GiB; pruned pair 0.1465 GiB; tensor planning envelope about **0.635 GiB**; require **3 GiB free host RAM** for runtime/source/export allowance | **0**, CPU-only code | About **0.147 GiB** + small JSON/header; require **0.5 GiB free disk** | Best-scoped pilot; still needs actual pretrained execution and RSS evidence |

The provisional reserves above are design review targets, not certified upper
bounds. Record process peak RSS on target hardware. Page cache is reclaimable;
RSS, available host RAM and tensor storage should not be added as interchangeable
measurements. GPU reserved memory, allocated tensors and device free memory are
also distinct. Other processes can consume headroom after admission.

## Allocation and semantics audit

### Structured pruning

The layerwise physical policy ranks neuron j using
mean(abs(fc1[j,:])) + mean(abs(fc2[:,j])) in float64, with stable sorting; it slices
the same kept indices from fc1 rows/bias and fc2 columns, preserving fc2 bias and
the hidden residual dimension. There is no attention/vision pruning.

Before this change, the prevalidation map retained every original MLP after
replacement. At the end it could retain an additional **6.252 GiB** of original
MLP weights/biases alongside the 10.406 GiB pruned model (about 16.657 GiB total).
These are retained old allocations, not new bytes caused by creating the map.
The map is now dropped after successful prevalidation for physical pruning.
Only the current pair needs to remain alive while being replaced.

`_slice` previously allocated random initialized destination weights, multiple
indexing results and a clone before replacing the initialized parameter. It now
constructs layer metadata on `meta` and uses owned index_select outputs, with
single-axis fast paths for the actual fc1/fc2 operation. Values, ordering, dtype,
flags, mode and independent storage are preserved. Discarded initialization no
longer consumes RNG state. CPU indices explicitly follow the source device.

No importance calculation, sparsity, distribution or precision was changed.
Global magnitude masks still require very large score/order arrays: the MLP-only
FP64 global score and int64 order arrays alone each approach 12.5 GiB, before sort
workspace/model/masks. Masks preserve dense tensor shapes and do not provide the
physical storage reduction of paired slicing.

### Recovery

`recover_lora` deliberately deep-copies and preserves its input. It does not use
QLoRA, CPU offload, activation checkpointing or quantized optimizer state. The
copied base is frozen; optimizer state is for adapters, not all three billion
base parameters. The copied vision tower is still resident even though recovery
batches are text-only. Gradients need activation paths through the language model.
Replacing deepcopy with an in-place API, dropping source residency or changing
training precision would require an explicit new contract and validation; none
was done here. The two short CPU recovery steps are mechanical validation, not
evidence of useful research recovery.

### AWQ

`quantize_bunny_awq` moves the whole model to its configured device. Capturing one
block's CPU FP32 inputs does not imply blockwise weight offload. It reruns the
current model for successive blocks and quantizes progressively; caching a single
dense-model hidden-state trace would change those semantics. Hooks/cache/modes are
restored by the collector. Raw features for six linears are bounded by
64(5H+I')×4 = 5 MiB for the preparation plan, with additional list/cat copies.

Shared-scale search keeps original outputs and constructs FP32 transformed and
dequantized matrix candidates. Clipping loops over groups; packing keeps original
linears until replacement advances to the next block. There is no full-model
deepcopy here, but many matrix temporaries. Portable inference dequantizes each
linear into FP32 then casts for matmul; optimized INT4 throughput is not supplied.
Reducing sample counts/grids, changing calibration modality or reordering/pruning
after calibration would change the experiment; none was silently substituted.

### Checkpoints and admission

`export_treatment` clones **all language-block tensors** into CPU-owned state and
writes a full safetensors payload. Full exports do not become sparse files when
weights are zero. General execution saves every requested stage, retains prior
stage artifacts, and does not free its prepared model before fresh evaluation.
`apply_treatment` retains the base while building all replacement linears.

Safetensors output is written to its own new file; the generic `publish()` helper
uses a temporary file and a hard link, which reference the same data rather than
requiring two physical payload copies. Older retained stage files and partial
failed output still consume disk. Do not delete them automatically to make room.

The **eight-times-staged-weight-file** gate in `compression.execute` is an explicit
conservative heuristic (about 73.59 GiB here), not a measured minimum or an
unavoidable algorithmic requirement. It is unchanged. Model byte budgets, device
authorization, manifest checks and trusted-checkpoint requirements are unchanged.
No general override or force flag was added. The micro-pilot is a different,
single-component admission profile with its own small caps; it never calls full
model preparation with a reduced threshold.

## One proposed Kaggle pilot — not executed on pretrained weights here

**Question:** can layer 0's pretrained MLP be physically reduced by exactly 25%,
match the corresponding zeroed-neuron operator within FP32 tolerance, and produce
a reloadable small artifact inside a measured resource budget?

This is a correctness/resource hypothesis. Synthetic Gaussian probe activations
are explicitly separate from calibration data or a held-out VQA dataset.

1. After code review, copy the reviewed branch changes to the Kaggle checkout.
   This branch has not been published. Reuse the existing pinned assets and their
   complete Bunny manifest from the successful baseline. Do not acquire a second
   checkpoint copy. Use a fresh CPU process, not a live full-model training session.
2. Adjust the paths below to the actual existing restored assets. Dry-run first:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
python -m compression.mlp_pilot \
  --model-dir /kaggle/working/pinned-baseline-assets/bunny \
  --manifest /kaggle/working/baseline-smoke-config/bunny_manifest.json --layer 0
```

The default reads configuration/safetensors headers, validates exact pair shapes
and declared source identity, checks the component cap, and prints a plan. It
does not load model tensors, hash payloads, write outputs or initialize CUDA.
Confirm H=2560, I=10240, kept width 7680 and expected bytes before proceeding.

3. Only after explicit operator approval, run **one CPU pilot**:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
python -m compression.mlp_pilot \
  --model-dir /kaggle/working/pinned-baseline-assets/bunny \
  --manifest /kaggle/working/baseline-smoke-config/bunny_manifest.json --layer 0 \
  --output /kaggle/working/mlp-layer0-pilot-001 --execute
```

Execution rechecks at least 3 GiB available RAM and 512 MiB free output disk,
requires a new output outside assets and caps FP32 source pairs to 256 MiB.
Dimension caps also bound probe buffers. It verifies the selected shard(s) and
config/index hashes against the manifest before loading only four MLP tensors.
Hashing may read a several-GB shard sequentially; this adds I/O, not a model copy.
Source authenticity remains dependent on the previously reviewed pinned acquisition.

4. The pilot uses the source activation function, fixed layerwise 25% ranking,
   and seed-42 CPU input shape [1,16,2560]. It checks the paired-slicing result
   against original dense fc1 with removed activations zeroed before fc2, using
   rtol=atol=1e-4. It reports dense-to-pruned probe RMSE separately; that diagnostic
   is not accuracy or a calibration objective.
5. It saves `mlp.safetensors` plus `pilot.json`, verifies bitwise state equality
   after reload, and records RSS/lifetime high-water observations at boundaries.
   The Linux process high-water includes imports and all prior work in that
   process; it is not an isolated per-stage allocation measurement. No GPU is used.
6. Accept only finite results, passing equivalence, reproducible kept indices,
   matching source hashes and successful reload. Review lifetime RSS against the
   3 GiB reservation and artifact bytes against the 256 MiB output cap. Stop if
   any guard/check fails, if actual headroom is inadequate, or if the source changed.
   Keep failed/partial output and logs; choose a fresh path for any later retry.

Expected output paths for the command above:

- `/kaggle/working/mlp-layer0-pilot-001/mlp.safetensors`: only fc1/fc2 weights
  and biases; expected payload about 157,327,360 bytes plus a small header.
- `/kaggle/working/mlp-layer0-pilot-001/pilot.json`: selected source-file hashes,
  kept indices, dimensions, artifact hash/bytes, probe diagnostics and resource
  observations. A successful status is `CPU_OPERATOR_PILOT_COMPLETED`.

Failure and cleanup procedure: a nonzero exit or absent `pilot.json` is not a
completed pilot. Preserve stderr and any partial output; do not rerun into the
same directory. Inspect disk/RAM, source hashes and the failing check, then choose
a new numbered pilot directory for an approved retry. No automatic deletion or
cleanup is performed. After reviewing and retaining the diagnostic evidence, the
operator may archive/remove only that exact newly created pilot directory if
space is needed. Never use the model directories, manifest directory, baseline
run directory, repository root or `/kaggle/working` as a cleanup target.

Artifact format is **phi-mlp-micro-pilot-v1**, not a deployable Bunny treatment.
It cannot be used by the UI/adapter as evidence of a compressed full VLM. Both
T4s remain unused during this first pilot. The standalone vision tower, model
checkpoint, baseline artifacts and existing records remain unchanged.

## Follow-up strategies requiring separate review

After a successful operator pilot, possible full-model work includes releasing
prepared models before reload, constructing replacements without random full-size
initialization, bounded/sharded export with complete integrity/atomicity semantics,
and a clearly named in-place recovery API. A base-bound kept-index recipe could
avoid full weight snapshots for pruning-only studies, but must reconstruct the
exact same tensors and be validated as a new artifact format. These were not
introduced as ad hoc deployment shortcuts.

An AWQ-only stage may be a better disk fit than the multi-stage sequence; qualify
its true CPU peak and review a stage-specific policy before changing admission.
GPU offload, activation checkpointing, mixed precision, LoRA rank/targets or grid
changes require explicit protocols and equivalence/quality tests. Two independent
GPUs do not supply automatic pooled VRAM. None of these experiments is queued.

## Implementation and CPU validation

Files changed/prepared:

- `compression/pruning.py`: release physical-pruning prevalidation references,
  avoid discarded initialization/redundant slicing copies, explicit CPU/source
  index placement. No pruning policy or full-model admission gate was changed.
- `compression/resource_plan.py`: read-only analytical storage/accounting CLI.
- `compression/mlp_pilot.py`: separately bounded, default-plan-only CPU operator
  pilot; requires explicit execution, source verification, RAM/disk admission,
  numerical checks and independent artifact format.
- `compression/tests/test_pruning_resources.py`, `test_resource_plan.py` and
  `test_mlp_pilot.py`: 17 new regressions covering module lifetimes, owned storage,
  value/flag/RNG behavior, actual tiny dense/pruned/packed storage counts, source
  tampering and admission/output preservation.
- This report: supplied evidence, stage budgets, audit and proposed pilot.

The initial pruning resource reproductions failed before implementation:
**2 failed / 1 passed, 1.42s**. The original modules remained alive and slicing
consumed RNG through discarded initialization. Both are repaired; all existing
tests and assertions remain intact.

The initial feasibility CPU verification used Python 3.12.14 / torch 2.2.2+cpu and:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HOME=/workspace/final_year_project/.cache/huggingface OMP_NUM_THREADS=1 \
.venv-cpu/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider
```

**627 passed, zero failures/skips, two warnings, 58.84s.** The two warnings are
the existing Starlette/AnyIO alias deprecation and PEFT fixture-vocabulary notice.
The focused three new regression modules passed **16 tests, 2.20s**; the final
complete run also includes the additional source-hash assertion. All **227 Python
files compiled** into a temporary directory; `git diff --check` passed. Both CLI
help/planning commands ran successfully. API tests used local network permission
in the managed sandbox as in earlier CPU validation.

Only tiny random CPU fixtures were executed. No pretrained compression pilot,
GPU experiment, acquisition, full-model checkpoint export or historical metric
rewrite was performed. No frontend source or dependency was changed.

## Final review for authorized publication

The subsequent user request authorizes committing and pushing this branch only.
Final review confirms that the verified baseline, loaders, configuration and
existing result code are unchanged from the base. The pilot obtains all four
parameter tensors through safetensors `get_tensor`, converts them to owned CPU
FP32 values and installs them into metadata-only Linear layers. It never fills
parameter values randomly. The random input probe is separately labeled synthetic.
An independent regression compares source tensors directly with loaded values,
then verifies fc1 kept rows/bias, fc2 corresponding columns, unchanged fc2 bias
and the same-input zeroed-neuron reference. Source authenticity still depends
on the reviewed pinned acquisition and local content manifest, not the repo name
or revision string alone.

RAM, disk, shape/component caps and output-path admission precede shard payload
hashing and tensor loading. Tests explicitly trap either expensive operation
after an admission failure. No full-model admission threshold was relaxed.
The final task response records the actual commit/push result; no GPU execution,
merge, force push or publication to another branch is authorized.

Publication review validation (same offline CPU environment and full-suite command
above): **628 passed, zero failures/skips, two existing warnings, 57.94s**.
The focused command below passed **30 tests, 4.43s**:

```sh
.venv-cpu/bin/python -m pytest compression/tests/test_mlp_pilot.py compression/tests/test_pruning_resources.py compression/tests/test_resource_plan.py compression/tests/test_pruning.py compression/tests/test_pruning_precision.py -q
```

All **227 Python sources compiled** again; whitespace checks passed. The seven
changed paths were reviewed for credential patterns, model/data payloads and large
artifacts with no findings. A diff against the base confirmed no changes to
`evaluation/`, `VLM/bunny/`, `backend/`, `runtime_policy.py`, `configs/` or
`experiments/`. Existing branch refs remained unchanged before publication.
Only local CPU fixtures were executed; the pretrained Kaggle micro-pilot remains
an explicit future operator action.
