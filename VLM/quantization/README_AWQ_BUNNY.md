# Bunny structured pruning + AWQ integration

The base IJCV code performs structural pruning and then recovery training.  For
Bunny, `LLM-Pruner/examples/Bunny.py` saves the *structurally modified* model
as `pytorch_model.bin` with this Python payload:

```python
{"model": model, "tokenizer": tokenizer}
```

The recovery path in `VLM/bunny/train/train_pruned.py` reloads that structural
checkpoint through `load_pruned_bunny_model`.  Its normal LoRA recovery output
is an adapter directory, not a complete model by itself.  Therefore AWQ must
receive the pruned checkpoint as well as the base model whenever recovery is an
adapter.  This avoids accidentally quantizing the original dense model.

## Integration flow

```text
Bunny Phi-2 base model + structural prune checkpoint + recovery weights
    -> recovered/pruned Bunny model
    -> deterministic local-text calibration
    -> per-Phi-block activation collection (CPU bounded)
    -> Phi-2 shared LayerNorm AWQ scale search
    -> equivalent LayerNorm/Linear scaling
    -> activation-weighted MSE clipping
    -> groupwise packed INT4 (or INT8 control) linear replacement
    -> portable checkpoint + awq_config.json
```

The added Phi-2 adapter matches the actual Bunny-3B layout in this repository:
`input_layernorm`, `self_attn.{q,k,v}_proj`, `self_attn.dense`, and
`mlp.{fc1,fc2}`.  It does not alter Bunny, LLM-Pruner, ShortGPT, or the copied
AWQ core.

## Run

Use the dedicated Python 3.11 environment described in
[`ENVIRONMENT_SETUP.md`](../../ENVIRONMENT_SETUP.md).  The portable AWQ path
does not require BitsAndBytes or `awq_inference_engine`.

Create a small, local calibration file first, for example
`D:/data/calibration.txt` (one text sample per line).  Then run from the
repository root:

```powershell
python -m VLM.quantization.awq_quantize `
  --base_model BAAI/Bunny-v1_0-3B `
  --pruned_checkpoint D:/runs/bunny_prune/pytorch_model.bin `
  --model_path D:/runs/bunny_recovery `
  --calibration_data D:/data/calibration.txt `
  --bits 4 --group_size 128 `
  --calibration_samples 32 --sequence_length 128 `
  --device cpu `
  --output_dir D:/runs/bunny_pruned_awq_int4
```

`--model_path` above is the LoRA recovery directory and must contain
`adapter_config.json`.  For a recovered full state-dictionary file, pass it as
`--model_path` with `--base_model` and `--pruned_checkpoint`.  For a trusted
self-contained serialized structured model (`{"model", "tokenizer"}`), it can
be supplied directly as `--model_path`.

To produce the AWQ INT8 control, change `--bits 4` to `--bits 8`.  It is useful
for separating the AWQ calibration/scaling effect from the INT4-vs-INT8
bit-width difference.

```powershell
python -m VLM.quantization.benchmark `
  --original_fp16 D:/models/bunny-original `
  --pruned_fp16 D:/runs/bunny_prune/pytorch_model.bin `
  --pruned_int8 D:/runs/bunny_pruned_int8 `
  --pruned_awq_int4 D:/runs/bunny_pruned_awq_int4 `
  --output D:/runs/bunny_benchmark_manifest.json
```

The benchmark manifest reports artifact sizes and precision. Accuracy,
generation latency, tokens/sec, and peak GPU memory say `NOT MEASURED` until an
actual Bunny evaluation protocol records them. It does not claim that AWQ is
better than INT8. Exports are named `bunny_awq_int4.pt` or `bunny_awq_int8.pt`
according to the selected bit width.

## Existing copied AWQ core

`VLM/quantization/awq/` contains copied components for scale search, clipping,
and quantization, but is not plug-compatible with this Bunny path as checked:

- `pre_quant.py` has no Bunny Phi-2 block selection and only recognizes
  LLaMA/Qwen/OPT/Bloom/MPT/Falcon/NeoX/LLaVA/InternVL layouts.
- `auto_scale.py` does not implement `PhiDecoderLayer` rules.
- Multiple core files import `VLM.quantization.utils.*`, which is absent here.
- `qmodule.py` and `w8a8_linear.py` require `awq_inference_engine`, which is
  neither bundled nor listed in requirements.

The portable `PackedAWQLinear` integration is consequently intentional.  It
stores actual packed 4-bit weights but dequantizes one linear layer at a time
with standard PyTorch.  It verifies functionality without custom CUDA kernels;
it must not be used to claim INT4 kernel latency improvements.

## Hardware limits and status

The defaults use CPU because a full Bunny-3B FP16 LVLM plus calibration working
memory can exceed a 6 GB RTX 3050, especially with the vision tower present.
Calibration keeps only a capped number of activations for one block and uses
inference mode.  CUDA can be requested after measuring available memory; no
claim is made that the full recovery model fits on 6 GB.

The small Phi-like CPU tests pass for the portable INT4 save/reload/forward path
and the INT8 control. Full Bunny model loading, recovery, full-model AWQ, and
task evaluation remain pending: the local Hugging Face cache has tokenizer and
config files but no model weights, and the existing local smoke run skipped
recovery because of hardware constraints. Existing prune/AWQ smoke artifacts
must not be described as a recovered-model result.
