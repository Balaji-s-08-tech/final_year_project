from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from evaluation.config import EvaluationConfig
from evaluation.adapters import LocalAdapter, Prepared, Generation
from evaluation.inspection import compatibility, local_checkpoint, architecture
from evaluation.data import load_examples, normalize
from evaluation.preflight import preflight
from evaluation.runner import evaluate, nearest_rank
from experiments.store import read_json, ExperimentStore


@pytest.fixture
def tiny_bunny(tmp_path):
    from VLM.quantization.awq_model_loader import _enable_project_imports
    _enable_project_imports()
    from bunny.model.language_model.bunny_phi import BunnyPhiConfig, BunnyPhiForCausalLM
    from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    vision, model_path = tmp_path / "siglip", tmp_path / "bunny"
    vcfg = SiglipVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                             num_attention_heads=2, image_size=4, patch_size=2)
    torch.manual_seed(19)
    SiglipVisionModel(vcfg).save_pretrained(vision)
    SiglipImageProcessor(size={"height": 4, "width": 4}).save_pretrained(vision)
    cfg = BunnyPhiConfig(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    cfg.mm_vision_tower = str(vision)
    cfg.mm_hidden_size = 8
    cfg.mm_projector_type = "mlp2x_gelu"
    cfg.image_aspect_ratio = "pad"
    BunnyPhiForCausalLM(cfg).save_pretrained(model_path)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "USER": 4, ":": 5,
        "ASSISTANT": 6, "is": 7, "this": 8, "red": 9, "yes": 10, "no": 11}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]").save_pretrained(model_path)
    from PIL import Image
    Image.new("RGB", (4, 4), "red").save(tmp_path / "red.png")
    data = tmp_path / "eval.jsonl"
    data.write_text(json.dumps({"id": "red", "question": "is this red", "answers": ["yes"], "image": "red.png"}) + "\n")
    return EvaluationConfig(str(model_path), "tiny-random-bunny-test-fixture", "fixture-r1", str(data), "fixture-data", "fixture-d1",
        "USER: <image>\n{question}\nASSISTANT:", vision_path=str(vision), max_samples=1, warmup_iterations=1,
        max_new_tokens=2, max_input_tokens=32)


def test_real_tiny_multimodal_load_generate_and_track(tiny_bunny, tmp_path):
    config = tiny_bunny
    assert compatibility(config)["errors"] == []
    adapter = LocalAdapter()
    adapter.load(config)
    records, fingerprint = load_examples(config)
    prepared = adapter.prepare(records[0])
    original = prepared.inputs["input_ids"].clone()
    adapter.model.generation_config.eos_token_id = None  # force a second cached decoding step
    with torch.inference_mode():
        logits = adapter.model(**{key: value.clone() for key, value in prepared.inputs.items()}, use_cache=False).logits
        assert torch.isfinite(logits).all()
        result = adapter.generate(prepared)
    assert result.shape[1] == prepared.prompt_tokens + config.max_new_tokens
    assert torch.equal(prepared.inputs["input_ids"], original)
    assert len(adapter.decode(prepared, result).token_ids) > 0
    run = evaluate(config, tmp_path / "runs")
    state = run.latest()
    assert state["status"] == "COMPLETED"
    metrics = state["metrics"]["metrics"]
    assert metrics["baseline_accuracy"]["status"] == "MEASURED"
    assert metrics["original_parameter_count"]["value"] == sum(p.numel() for p in adapter.model.parameters())
    assert metrics["peak_gpu_memory"]["value"] is None and metrics["peak_gpu_memory"]["status"] == "UNAVAILABLE"
    assert metrics["gpu_utilization"]["value"] is None
    assert metrics["tokens_per_second"]["value"] > 0
    assert read_json(run.path / "architecture.json")["layers"][0]["linears"]["mlp.fc1"]["out_features"] == 32
    assert read_json(run.path / "experiment.json")["dataset"]["fingerprint"] == fingerprint


class ControlledAdapter:
    """Fixture output, not evidence about Bunny quality."""
    def load(self, config):
        self.calls = 0
    def inspect(self):
        return {"logical_parameter_elements": 10, "packed_bits": []}
    def prepare(self, record):
        return Prepared({"input_ids": torch.ones(1, 3, dtype=torch.long)}, 3)
    def generate(self, prepared):
        self.calls += 1
        return torch.tensor([[1, 2, 3, 7, 2]])
    def decode(self, prepared, output):
        return Generation(" YES ", [7, 2])


def test_scoring_token_counts_warmups_and_reproducibility(tiny_bunny, tmp_path):
    adapter = ControlledAdapter()
    before = torch.are_deterministic_algorithms_enabled()
    run = evaluate(tiny_bunny, tmp_path / "runs", adapter=adapter)
    assert adapter.calls == 2  # one warmup, one measured sample
    assert torch.are_deterministic_algorithms_enabled() == before
    summary = read_json(run.path / "evaluation_summary.json")
    assert summary["correct"] == 1 and summary["generated_tokens"] == 2
    metrics = run.latest()["metrics"]["metrics"]
    assert metrics["baseline_accuracy"]["value"] == 1
    assert metrics["tokens_per_second"]["value"] == pytest.approx(2 * metrics["throughput"]["value"])
    assert normalize("ＹＥＳ \t\n") == "yes"
    second = evaluate(tiny_bunny, tmp_path / "runs", adapter=ControlledAdapter())
    assert second.path != run.path
    assert read_json(second.path / "experiment.json")["dataset"] == read_json(run.path / "experiment.json")["dataset"]


@pytest.mark.parametrize("changes", [
    {"adapter": "unsupported"}, {"max_samples": 0}, {"max_new_tokens": True}, {"seed": -1},
    {"batch_size": 2}, {"dtype": "float16"}, {"model_revision": ""}, {"warmup_iterations": -1},
    {"prompt_template": "{wrong}"}, {"vision_path": None}, {"required_free_ram_bytes": -1},
    {"schema_version": True}, {"model_revision": "REPLACE_WITH_REAL_REVISION"},
])
def test_invalid_config(tiny_bunny, changes):
    with pytest.raises(ValueError):
        replace(tiny_bunny, **changes).validate()


def test_read_only_preflight_and_gpu_gate(tiny_bunny, tmp_path, monkeypatch):
    config = replace(tiny_bunny, device="cuda:0", dtype="float16", required_free_ram_bytes=10, required_free_vram_bytes=20)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Read-only preflight must not initialize CUDA"))
    report = preflight(config, environment={"ram": {"available_bytes": 100}, "cuda_runtime": "fixture-only", "gpu_devices": [{"available_memory_bytes": 30}]})
    assert report["status"] == "PREFLIGHT_ONLY" and report["memory_feasibility"] == "NOT_PROVEN"
    assert not report["execution_authorized"]
    with pytest.raises(PermissionError):
        evaluate(config, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
    assert preflight(config, environment={"ram": {"available_bytes": 1}})["status"] == "BLOCKED"
    assert preflight(replace(config, required_free_vram_bytes=None), environment={})["status"] == "BLOCKED"


def test_missing_assets_architecture_and_shard_safety(tiny_bunny, tmp_path):
    report = compatibility(replace(tiny_bunny, model_path=str(tmp_path / "missing")))
    assert report["errors"]
    path = Path(tiny_bunny.model_path)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../escape.safetensors"}}))
    assert "unsafe" in " ".join(local_checkpoint(path)["errors"])
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "absent.safetensors"}}))
    assert local_checkpoint(path)["artifact_bytes"] is None


def test_dataset_integrity_and_no_silent_truncation(tiny_bunny):
    first, fingerprint = load_examples(tiny_bunny)
    assert load_examples(tiny_bunny) == (first, fingerprint)
    from PIL import Image
    Image.new("RGB", (4, 4), "blue").save(first[0]["image"])
    assert load_examples(tiny_bunny)[1] != fingerprint
    with pytest.raises(ValueError, match="Fewer"):
        load_examples(replace(tiny_bunny, max_samples=2))
    adapter = LocalAdapter()
    adapter.load(replace(tiny_bunny, max_input_tokens=1))
    with pytest.raises(ValueError, match="truncation"):
        adapter.prepare(first[0])


def test_duplicate_examples_and_image_escape_rejected(tiny_bunny):
    path = Path(tiny_bunny.dataset_path)
    original = path.read_text()
    path.write_text(original + original)
    with pytest.raises(ValueError, match="unique"):
        load_examples(tiny_bunny)
    record = json.loads(original)
    record["image"] = "../outside.png"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Image"):
        load_examples(tiny_bunny)


def test_failures_leave_a_durable_run(tiny_bunny, tmp_path):
    class Failing(ControlledAdapter):
        def generate(self, prepared):
            raise RuntimeError("intentional fixture failure")
    with pytest.raises(RuntimeError, match="intentional"):
        evaluate(tiny_bunny, tmp_path / "runs", adapter=Failing())
    run = ExperimentStore(tmp_path / "runs").runs()[0]
    assert run.latest()["status"] == "FAILED"
    assert "intentional" in (run.path / "stderr.log").read_text()
    assert read_json(run.path / "failure_report.json")["error"]


def test_percentile_definition():
    assert nearest_rank([1., 9., 2., 3.], .95) == 9.
    assert nearest_rank([1.], .95) == 1.
    with pytest.raises(ValueError):
        nearest_rank([], .95)


def test_config_cli_defaults_to_inspection(tiny_bunny, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_bunny.to_dict()))
    output = tmp_path / "report.json"
    result = subprocess.run([sys.executable, "-m", "evaluation", "--config", str(path), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read_json(output)["status"] == "PREFLIGHT_ONLY"
    assert not list(tmp_path.glob("**/experiment.json"))
    assert EvaluationConfig.load(path) == tiny_bunny
    invalid = tiny_bunny.to_dict() | {"unknown_option": True}
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError):
        EvaluationConfig.load(path)


def test_portable_text_evaluation_uses_actual_packed_precision(tiny_bunny, tmp_path):
    from transformers import PhiConfig, AutoTokenizer
    from LLMPruner.models.hf_phi2.modeling_phi import PhiForCausalLM
    from VLM.quantization.awq_config import AWQConfig
    from VLM.quantization.awq_quantize import quantize_bunny_awq, _save_output
    tokenizer = AutoTokenizer.from_pretrained(tiny_bunny.model_path, local_files_only=True)
    model = PhiForCausalLM(PhiConfig(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    dense_count = sum(p.numel() for p in model.parameters())
    qconfig = AWQConfig(bits=4, group_size=3, calibration_samples=1, sequence_length=4, max_tokens_per_linear=4, scale_grid=2, clip_grid=2)
    samples = [torch.tensor([[1, 7, 8, 9]])]
    summary = quantize_bunny_awq(model, samples, qconfig)
    checkpoint = _save_output(tmp_path / "awq", model, tokenizer, qconfig, {}, summary)
    data = tmp_path / "text.jsonl"
    data.write_text(json.dumps({"id": "text", "question": "is this red", "answers": ["yes"]}))
    config = replace(tiny_bunny, adapter="portable_awq_text", model_path=str(checkpoint), vision_path=None,
        dataset_path=str(data), prompt_template="{question}", treatment="awq_int4", trusted_checkpoint=True)
    run = evaluate(config, tmp_path / "runs")
    assert run.latest()["metrics"]["quantization"]["bits"] == 4
    assert run.latest()["metrics"]["metrics"]["remaining_parameter_count"]["value"] == dense_count
    adapter = LocalAdapter()
    with pytest.raises(ValueError, match="treatment"):
        adapter.load(replace(config, treatment="awq_int8"))
    with pytest.raises(ValueError, match="untrusted"):
        adapter.load(replace(config, trusted_checkpoint=False))
    assert compatibility(replace(config, treatment="pruning_awq"))["errors"]


def test_input_changes_and_unknown_treatment_cannot_pass(tiny_bunny, tmp_path, monkeypatch):
    assert compatibility(replace(tiny_bunny, treatment="structured_pruning"))["errors"]
    from evaluation import runner
    original = runner.load_examples
    def changed(config):
        examples, _ = original(config)
        return examples, "changed-after-inspection"
    monkeypatch.setattr(runner, "load_examples", changed)
    with pytest.raises(ValueError, match="changed"):
        evaluate(tiny_bunny, tmp_path / "runs", adapter=ControlledAdapter())
    assert ExperimentStore(tmp_path / "runs").runs()[0].latest()["status"] == "FAILED"
