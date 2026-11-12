# Bunny treatment artifact contract

Format `bunny-treatment-v1` is a directory containing:

- `blocks.safetensors`: complete language-block tensors, including normalization,
  dense/masked linears or packed INT4/INT8 buffers. No pickle code is loaded.
- `treatment.json`: SHA256 of that file, base asset identity, protected state hash,
  original and resulting MLP widths, exact language-linear declarations, actual
  treatment classification and optional recovery ledger.

The base identity hashes pinned model identity/revision and complete model/vision
file inventories. The protected hash covers embedding, LM head, encoder and
multimodal connector tensors. Tokenizer/processor configurations are bound through
the base inventory. Treat identities as content evidence, not producer signatures.

Supported reconstruction: dense, magnitude masks, paired physical Phi MLP pruning,
merged LoRA recovered weights, INT4/INT8 and physical pruning plus quantization.
Attention head reduction and vision/projector compression are unsupported.
Recovery is an attached training ledger, not proof of successful research recovery.

Load through `LocalAdapter` using `EvaluationConfig.treatment_artifact_path`.
The loader verifies byte budget and SHA before deserializing safetensors, checks
base/protected identity, restricts modules to the six supported linears per Phi
block, reconciles attention/residual dimensions, loads complete block state and
validates all tensors. The label is recomputed from loaded structure/storage;
a dense artifact cannot be called pruned or INT4 by editing its label.

GPU loading first restores against the FP32 CPU base, then transfers to the
requested dtype/device. This requires adequate host RAM. Source manifests must be
complete for pretrained weights; explicitly marked tiny random fixtures use test
identity only. No real weights or datasets are included in Git.

Artifacts retain dependencies on original base/vision assets. Report required
storage as base + vision + treatment bytes, or report treatment payload separately
with that dependency disclosed. Do not compare payload-only bytes to full dense
deployment bytes as an end-to-end compression claim.

Validation tests exercise all supported layouts with a tiny complete Bunny/SigLIP
model, visual generation equivalence after save/reload, altered base/protected
identities, corrupt payload hashes, incomplete module coverage and false labels.
This is architecture-fixture validation; real pretrained-model compatibility and
performance remain unverified until execution on the staged assets.
