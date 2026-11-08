from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from VLM.quantization.awq_calibration import build_calibration_samples
from VLM.quantization.awq_config import AWQConfig
from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint
from VLM.quantization.awq_quantize import PackedAWQLinear, _save_output, quantize_bunny_awq


class TinyTokenizer:
    def __call__(self, text, return_tensors, truncation, max_length):
        values = [ord(char) % 31 + 1 for char in text[:max_length]] or [1]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}

    def decode(self, values, skip_special_tokens=True):
        return str(values.tolist())


class TinyConfig:
    use_cache = True

    def to_dict(self):
        return {"model_type": "tiny-phi2"}


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.dense = nn.Linear(8, 8)

    def forward(self, hidden):
        values = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        return self.dense(values / 3)


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, hidden):
        return self.fc2(torch.relu(self.fc1(hidden)))


class TinyPhiBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = TinyAttention()
        self.mlp = TinyMlp()

    def forward(self, hidden):
        normalized = self.input_layernorm(hidden)
        hidden = hidden + self.self_attn(normalized)
        hidden = hidden + self.mlp(normalized)
        return (hidden,)


class TinyPhiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 8)
        self.model.layers = nn.ModuleList([TinyPhiBlock()])
        self.config = TinyConfig()

    def forward(self, input_ids, use_cache=False):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=hidden)


def test_tiny_calibration_awq_save_reload_and_forward(tmp_path: Path):
    torch.manual_seed(7)
    tokenizer = TinyTokenizer()
    samples = build_calibration_samples(tokenizer, ["alpha beta", "gamma delta"], 2, 8, 19)
    model = TinyPhiModel().eval()
    result = quantize_bunny_awq(model, samples, AWQConfig(bits=4, group_size=128, calibration_samples=2, sequence_length=8, max_tokens_per_linear=16, scale_grid=3, clip_grid=4, device="cpu"))
    assert result["blocks"] == 1
    assert result["quantized_linears"] == 6
    assert sum(isinstance(module, PackedAWQLinear) for module in model.modules()) == 6
    output_dir = tmp_path / "portable-output"
    checkpoint = _save_output(output_dir, model, tokenizer, AWQConfig(bits=4, group_size=128, calibration_samples=2, sequence_length=8, device="cpu"), {"source_format": "test-fixture"}, result)
    loaded_tokenizer, loaded_model, metadata = load_quantized_bunny_checkpoint(str(checkpoint), device="cpu", trusted=True)
    assert isinstance(loaded_tokenizer, TinyTokenizer)
    assert metadata["summary"]["quantized_linears"] == 6
    with torch.inference_mode():
        logits = loaded_model(input_ids=samples[0])
    assert torch.isfinite(logits.logits).all()


def test_quantizer_accepts_int8_control(tmp_path: Path):
    torch.manual_seed(2)
    model = TinyPhiModel().eval()
    samples = [torch.randint(1, 31, (1, 4))]
    result = quantize_bunny_awq(model, samples, AWQConfig(bits=8, group_size=128, calibration_samples=1, sequence_length=4, max_tokens_per_linear=8, scale_grid=2, clip_grid=2, device="cpu"))
    assert result["quantized_linears"] == 6
    assert all(module.bits == 8 for module in model.modules() if isinstance(module, PackedAWQLinear))
    output_dir = tmp_path / "output-int8"
    checkpoint = _save_output(output_dir, model, TinyTokenizer(), AWQConfig(bits=8, group_size=128, calibration_samples=1, sequence_length=4, device="cpu"), {"source_format": "test-fixture"}, result)
    assert checkpoint.name == "bunny_awq_int8.pt"
    assert (output_dir / "awq_config.json").is_file()
