# CPU-tested research input repairs

These repairs follow Mission 3 on a stacked feature branch. They do not establish
full-model pruning, recovery or GPU compatibility.

## Bunny calibration sampler

Hypothesis: a bounded, unique sample-selection pass and one shape-checked multimodal
expansion helper eliminate the sampler's label/list errors and make its consumer
contract testable without loading a VLM or downloading data.

`prepare_bunny_samples` accepts an existing model, tokenizer and dataset. It reads
each candidate record once, caps text length before image insertion, rejects
insufficient eligible samples, preserves actual EOS tokens in masks when EOS is
also padding, and avoids mutating input records. Equal-shaped images are stacked;
variable shapes remain a list of NCHW tensors on the declared/model device and
dtype. No `.half()` is called on a Python list. Multimodal preparation passes labels
once and runs without retaining the source model's autograd graph; detached
embeddings remain usable by the pruning language model's backward pass.

Non-batched output is a list of `(embeddings, labels)` pairs, matching the actual
Taylor param_second/param_mix consumer. Batched output remains `(embeddings,
labels, attention_mask)`. This corrects the previously inconsistent non-batched
tuple/early-return behavior. Sampling follows the caller's seeded Python RNG and
cannot loop forever when valid records are unavailable. Candidate traversal can
still read many records when eligibility is sparse.

The Bunny channel-wise loss consumer no longer calls `.to()` on a triple or
overwrites the multimodal loss with a second token-only forward. A shared loss
helper validates the triple; the parameter-second accumulation skips absent
gradients. This does not validate the rest of the structured pruning graph.
Optional text dataset imports are lazy; the CPU sampler helper needs no datasets
package and cannot download anything on import.

Tests cover all requested unbatched examples, variable/equal image shapes,
EOS/padding, invalid bounds, finite traversal, seeded subset selection, text-axis
truncation, source immutability and downstream gradient propagation. No dataset
recipe or accuracy-retention claim follows from these fixture tests. Legacy
LLaVA/InternVL samplers and multi-GPU execution are not repaired in this increment.

## Artifact headers and recovery state

Hypothesis: reject ambiguous metadata and validate recovery tensors before copying
any state so invalid inputs cannot silently create mislabeled or mixed-weight runs.

Artifact metadata now requires an object, integer version/size (not booleans),
supported format, lowercase SHA256, consistent precision/format/bits and object/null
calibration data. Duplicate JSON keys, nonfinite JSON constants and headers over
1 MiB are rejected. Discovery reports UNKNOWN for invalid metadata instead of an
uncaught attribute/type error. Valid artifact contents are still hash-verified;
this does not authenticate an external producer or prove internal tensor semantics.

Explicit missing recovery paths raise FileNotFoundError immediately, rather than
fall back to the pruned model and label the input as recovered. A full-state recovery
file must cover the structured model's exact state keys with compatible tensor
shapes/types and finite values that fit the destination dtype. Prevalidation occurs
before load_state_dict; copying one matching tensor is no longer accepted as a
full recovery. Partial LoRA adapter handling is separate and is not forced into the
full-state contract. Raw serialized objects and all reconstructed recovery inputs
are marked recovery_provenance=UNVERIFIED; complete tensor coverage does not prove
that training occurred on the claimed pruning/data revision.

This deliberately rejects formerly accepted partial full-state files. Legacy
builder initialization, adapter/non-LoRA state completeness and a content-bound
recovery training manifest remain future work. In particular, a full multimodal
state may need its vision modules initialized before strict loading; no fallback
to partial copying is provided. Serialized files must remain trusted; no attempt
is made to sandbox Python pickle execution.

Versioned benchmark documents also require the documented object/measurement
fields and an integer schema version. Malformed list/null documents, unknown or
incomplete measurement fields, and invalid quantization/calibration structures
produce validation failures/HTTP 422 before any imported run is created, rather
than an API attribute error or a silently defaulted measurement.

## Legacy GPU probe admission

The old probe no longer assumes an A100-like 32 GiB free VRAM / 48 GiB free RAM.
Loading requires explicit positive finite memory budgets, local asset paths and
--authorize-gpu-execution; policy checks occur before runtime imports. Its private
load function also rejects an unauthorized call. Default inspection queries
nvidia-smi metadata and installed PyTorch CUDA capability without calling CUDA
context APIs, and explicitly labels that information as capability-only. It does
not prove runtime compatibility or authorize billing. Training-only imports
(datasets, W&B, DeepSpeed, trainer) were removed from this inference probe's
prerequisites; training dependency validation is still necessary before recovery.

Six CPU tests cover forbidden execution, missing/nonfinite budgets, missing paths,
the private gate and avoidance of CUDA context APIs. Use the Mission 3 tracked
evaluation CLI for actual future baseline experiments; the older probe is only a
manual admission/finite-forward diagnostic and has not run on the L4.
