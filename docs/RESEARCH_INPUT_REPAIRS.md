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
