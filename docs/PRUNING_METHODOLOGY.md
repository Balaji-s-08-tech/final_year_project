# Pruning framework — CPU-validated scope

`compression.pruning` supports magnitude masks on explicit Phi block linears and
physical MLP neuron reduction on the dependent fc1/fc2 pair. No model loading or
network calls occur. Inputs must already be on CPU. Config validates finite sparsity
in [0,1), distribution and recognized targets; all targets are prechecked before
mutation. The caller owns its working model; clone it for experimental controls.

Magnitude masks select floor(N*sparsity) smallest absolute weights with stable
tie-breaking in module/flattened order, globally or per layer. Forward multiplies
by a boolean registered mask; masked gradients are zero. State records masked and
actual-zero fractions, which may differ when the source already contains zeros.
Dense shapes/parameter counts remain unchanged and masks can increase serialized
size. No runtime or physical compression benefit is claimed.

`materialize_masks` explicitly makes zeroed ordinary linears before AWQ so scale
search cannot accidentally quantize unmasked source values. It preserves forward
results and removes ongoing gradient constraints. Do not resume masked training
after materialization without reapplying the documented mask.

Physical MLP reduction ranks a neuron by mean absolute fc1 incoming and fc2 outgoing
weights, removes floor(intermediate*sparsity), then slices fc1 rows/bias and fc2
columns together. Residual widths, attention, embeddings, LM head, projector and
vision tower remain unchanged. This heuristic is not a quality-retention claim.
The state reconstruct helper prechecks residual/MLP dependencies, builds a candidate
copy, then strict-loads it; whole structured objects also serialize with trusted
PyTorch. Standard pretrained config alone cannot reconstruct heterogeneous widths.
Always retain the per-layer shape/state manifest and use a compatible model factory.

Validated: tiny Phi shapes, masks, actual sparsity, deterministic ordering, gradients,
state/whole-object roundtrip and tiny real Bunny+SigLIP complete forward/generation.
This does not validate full research checkpoints or GPU memory/accuracy retention.
Physical attention head pruning needs synchronized q/k/v rows, output columns,
head counts, rotary/KV/cache handling; it is design-only. Vision or projector channel
pruning requires an additional cross-region reconstruction contract and is rejected.
