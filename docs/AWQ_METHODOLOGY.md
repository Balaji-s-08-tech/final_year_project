# AWQ validation and calibration protocol

The portable Phi adapter is architecture-specific: shared LayerNorm scaling for
q/k/v/fc1, activation-weighted clipping, then groupwise unsigned INT4/INT8 packing
of six language linears. Calibration, channel search and clipping are distinct
from generic zero-inclusive affine quantization mathematics. No optimized CUDA
kernel, full Bunny quality gain or recovery experiment is claimed.

Calibration v1: seed 42, explicit local training-derived text records, separate
held-out evaluation; select exactly the requested nonempty sample count. Record
dataset revision, record IDs where available, file hash, sequence bound, token
cap, grids and max clip shrink. Tokenization truncation here is deliberate and
separate from held-out evaluation's no-silent-truncation rule. Text calibration
does not substitute for a validated image-conditioned Bunny calibration protocol.

Activation collection slices before CPU transfer, retains bounded independent FP32
storage (tiny values must not disappear via FP16), and restores cache, hooks and
mixed module training states even on failure. Calibration repeatedly forwards the
full current model per block; this is a performance cost, not an optimized loader.

Use AWQConfig(bits=4 or 8, group_size=128, calibration_samples=32,
sequence_length=128, max_tokens_per_linear=512, scale_grid=20, clip_grid=20,
seed=42, device='cpu') as an unexecuted full-model proposal. Small fixtures reduce
grids/counts explicitly. Nondivisible/odd widths use partial final groups. Scales
stay FP32; explicit half-conversion can reintroduce underflow. Subnormal scale
resolution and ordinary quantization rounding remain limitations.

`compression.quantization.reconstruction_error` records CPU max/mean absolute
error and RMSE, with relative L2 unavailable for a zero reference. These are tensor
measurements, not task quality. `verify_checkpoint` requires a trusted local pickle
and an explicit load-size budget (64 MiB default for synthetic tests). It validates
content-bound sidecar, inner metadata, actual packed bits/shapes/groups/scales/zero
points. No file is downloaded. Large research artifacts require separately reviewed
CPU resources and trust; verification cannot authenticate a producer or training.

Save both source/pruning/recovery manifests, actual calibration configuration,
artifact hashes and effective inference settings. Run the same held-out scorer after
reload; quality evaluation uses the evaluator adapter interface and remains Pending
for real assets. Never compare language-only portable scores to image-conditioned
baseline scores as equivalent.
