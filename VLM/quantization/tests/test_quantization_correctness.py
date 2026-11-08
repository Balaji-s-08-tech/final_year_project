"""CPU numerical regressions; no pretrained models or GPU are required."""

import io

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from VLM.quantization.awq_config import AWQConfig
from VLM.quantization.awq_quantize import (
    PackedAWQLinear,
    _apply_clip,
    _dequantize,
    _find_clip_values,
    _pack_int4,
    _pseudo_quantized,
    _quant_params,
    _save_output,
    _unpack_int4,
)


CASES = {
    "positive": [[1., 2., 3., 4., 5.], [.1, .2, .3, .4, .5], [2., 1., 4., 3., 6.]],
    "negative": [[-1., -2., -3., -4., -5.]],
    "mixed": [[-3., -.2, 0., .4, 2.], [1., -2., 3., -4., 0.]],
    "constant_positive": [[1.] * 5, [2.] * 5],
    "constant_negative": [[-1.] * 5, [-2.] * 5],
    "zero": [[0.] * 5],
    "small": [[-1e-12, 0., 2e-12, 3e-12, -4e-12], [1e-10] * 5],
}


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [1, 2, 4, 128, -1])
@pytest.mark.parametrize("case", CASES)
def test_group_reconstruction_and_packed_forward(case, bits, group_size):
    weight = torch.tensor(CASES[case], dtype=torch.float32)
    rows, width = weight.shape
    effective = width if group_size == -1 else min(width, group_size)
    q, scales, zeros = _quant_params(weight, bits, group_size)
    restored = _dequantize(q, scales, zeros, group_size)
    assert q.dtype == zeros.dtype == torch.uint8
    assert scales.shape == zeros.shape == (rows, (width + effective - 1) // effective)
    assert scales.dtype == torch.float32
    assert torch.isfinite(scales).all() and (scales > 0).all()
    assert q.max() <= 2**bits - 1 and zeros.max() <= 2**bits - 1
    for start in range(0, width, effective):
        group = start // effective
        original = weight[:, start:start + effective]
        error = (restored[:, start:start + effective] - original).abs()
        # Rounding plus endpoint saturation must stay within half a grid step.
        tolerance = scales[:, group, None] / 2 + original.abs() * 2e-6
        assert torch.all(error <= tolerance)
    assert torch.equal(restored[weight == 0], weight[weight == 0])

    linear = nn.Linear(width, rows, bias=False)
    linear.weight.data.copy_(weight)
    packed = PackedAWQLinear.from_linear(linear, bits, group_size)
    codes = _unpack_int4(packed.packed_weight, width) if bits == 4 else packed.packed_weight
    assert torch.equal(codes, q)
    assert torch.equal(packed.scales, scales)
    # Identity inputs expose every reconstructed weight without summation error.
    assert torch.equal(packed(torch.eye(width)).T, restored)
    assert torch.equal(_pseudo_quantized(weight, bits, group_size), restored)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("values", [[1., 2.], [-2., -1.], [1., 1.], [-1., -1.], [0., 0.]])
def test_original_failures_have_expected_reconstruction(values, bits):
    weight = torch.tensor([values])
    restored = _pseudo_quantized(weight, bits, 128)
    levels = 2**bits - 1
    if values == [1., 2.]:
        torch.testing.assert_close(restored[:, 1], torch.tensor([2.]))
        # One lies halfway between codes; FP32 arithmetic may select either.
        assert abs(restored[0, 0].item() - 1.) <= 1 / levels + 2e-7
    elif values == [-2., -1.]:
        torch.testing.assert_close(restored[:, 0], torch.tensor([-2.]))
        assert abs(restored[0, 1].item() + 1.) <= 1 / levels + 2e-7
    else:
        torch.testing.assert_close(restored, weight, atol=0, rtol=1e-6)


def test_int4_pack_round_trip_all_codes_with_odd_rows_and_columns():
    codes = (torch.arange(51).reshape(3, 17) % 16).to(torch.uint8)
    packed = _pack_int4(codes)
    assert packed.shape == (3, 9)
    assert torch.equal(_unpack_int4(packed, 17), codes)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("bits", [4, 8])
def test_input_dtypes(dtype, bits):
    weight = torch.tensor([[1., 1., 0., -2., -2.]], dtype=dtype)
    result = _pseudo_quantized(weight, bits, 2)
    assert result.dtype == dtype
    torch.testing.assert_close(result, weight, atol=0, rtol=1e-6)


@pytest.mark.parametrize("bits", [4, 8])
def test_legacy_fp16_scale_checkpoints_still_load(bits):
    packed = PackedAWQLinear.from_linear(nn.Linear(5, 3), bits, 2)
    packed.scales = packed.scales.half()  # Original checkpoint representation.
    inputs = torch.eye(5)
    expected = packed(inputs)
    buffer = io.BytesIO()
    torch.save(packed, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    assert torch.equal(loaded(inputs), expected)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("bias", [False, True])
def test_serialization_preserves_small_scales_and_reconstruction(bits, bias):
    linear = nn.Linear(5, 3, bias=bias)
    linear.weight.data.copy_(torch.tensor([
        [1., 2., 3., 4., 5.],
        [-1., -2., -3., -4., -5.],
        [1e-12, -2e-12, 0., 3e-12, 4e-12],
    ]))
    if bias:
        linear.bias.data.copy_(torch.tensor([.5, -.25, 0.]))
    packed = PackedAWQLinear.from_linear(linear, bits, 2)
    inputs = torch.eye(5)
    expected = packed(inputs)
    buffer = io.BytesIO()
    torch.save(packed, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    assert torch.equal(loaded(inputs), expected)
    replacement = PackedAWQLinear.from_linear(linear, bits, 2)
    replacement.load_state_dict(loaded.state_dict())
    assert torch.equal(replacement(inputs), expected)
    assert loaded.scales.dtype == torch.float32
    assert (loaded.scales[-1] > 0).all()
    # Small row must retain signal through actual packed forward and reload.
    assert loaded(inputs)[:, -1].abs().max() > 0


@pytest.mark.parametrize("bits", [4, 8])
def test_portable_checkpoint_preserves_quantized_model_outputs(tmp_path, bits):
    from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint
    from VLM.quantization.tests.test_tiny_pipeline import TinyPhiModel, TinyTokenizer

    torch.manual_seed(17)
    model = TinyPhiModel().eval()
    # Exercise the public portable export/loader, including metadata and tokenizer.
    for block in model.model.layers:
        block.mlp.fc1 = PackedAWQLinear.from_linear(block.mlp.fc1, bits, 3)
    inputs = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = model(input_ids=inputs).logits.clone()
    checkpoint = _save_output(tmp_path, model, TinyTokenizer(), AWQConfig(bits=bits), {"source_format": "test-fixture"}, {})
    tokenizer, loaded, metadata = load_quantized_bunny_checkpoint(str(checkpoint), trusted=True)
    assert isinstance(tokenizer, TinyTokenizer)
    assert metadata["quantization"]["bits"] == bits
    with torch.inference_mode():
        assert torch.equal(loaded(input_ids=inputs).logits, expected)


@pytest.mark.parametrize("bits", [4, 8])
def test_clipping_search_does_not_worsen_group_calibration_loss(bits):
    weight = torch.tensor([
        [1., 2., 10., 1., 1.],
        [-1., -2., -10., -1., -1.],
        [-2., 0., 3., .1, -.3],
        [0., 0., 0., 0., 0.],
    ])
    features = torch.eye(5)
    config = AWQConfig(bits=bits, group_size=3, clip_grid=10)
    clips = _find_clip_values(weight, features, config)
    linear = nn.Linear(5, 4, bias=False)
    linear.weight.data.copy_(weight)
    _apply_clip(linear, clips, config.group_size)
    assert torch.isfinite(clips).all() and (clips >= 0).all()
    for group, start in enumerate(range(0, 5, 3)):
        before = weight[:, start:start + 3]
        after = linear.weight[:, start:start + 3].detach()
        assert (after.abs() <= clips[:, group, None]).all()
        baseline_error = (F.linear(features[:, start:start + 3], _pseudo_quantized(before, bits, 3)) - F.linear(features[:, start:start + 3], before)).square().mean(0)
        clipped_error = (F.linear(features[:, start:start + 3], _pseudo_quantized(after, bits, 3)) - F.linear(features[:, start:start + 3], before)).square().mean(0)
        assert torch.all(clipped_error <= baseline_error + 1e-7)


@pytest.mark.parametrize("weight,bits,group", [
    (torch.ones(2, 2), 3, 2),
    (torch.ones(2, 2), 4, 0),
    (torch.ones(2, 2), 4, -2),
    (torch.ones(0, 2), 4, 2),
    (torch.ones(2, 0), 4, 2),
    (torch.ones(2), 4, 2),
    (torch.ones(2, 2, dtype=torch.int64), 4, 2),
    (torch.tensor([[float("nan")]]), 4, 2),
    (torch.tensor([[float("inf")]]), 8, 2),
])
def test_invalid_quantization_inputs_fail_explicitly(weight, bits, group):
    with pytest.raises(ValueError):
        _quant_params(weight, bits, group)
