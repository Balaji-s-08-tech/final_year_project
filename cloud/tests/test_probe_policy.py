"""No GPU code runs in these tests; authorization checks precede all runtime work."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

PATH = Path(__file__).resolve().parents[1] / "bunny_gpu_probe.py"


@pytest.mark.parametrize("args,message", [
    (["--load-model"], "separate billing"),
    (["--load-model", "--authorize-gpu-execution"], "explicit positive finite"),
    (["--load-model", "--authorize-gpu-execution", "--minimum-free-vram-gib", "nan", "--minimum-free-ram-gib", "10"], "finite"),
    (["--load-model", "--authorize-gpu-execution", "--minimum-free-vram-gib", "16", "--minimum-free-ram-gib", "24"], "local model/vision"),
])
def test_probe_load_requires_explicit_admission_before_imports(args, message):
    result = subprocess.run([sys.executable, str(PATH), *args], capture_output=True, text=True)
    assert result.returncode == 2 and message in result.stderr
    assert result.stdout == ""


def module():
    spec = importlib.util.spec_from_file_location("probe_fixture", PATH)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_private_load_function_also_refuses_unauthorized_calls():
    with pytest.raises(PermissionError):
        module()._load_and_forward(SimpleNamespace(authorize_gpu_execution=False), None)


def test_read_only_probe_does_not_call_cuda_context_apis(monkeypatch, capsys):
    import torch
    probe = module()
    monkeypatch.setattr(probe, "_runtime", lambda: (torch, {"torch": torch.__version__}, {}))
    from experiments import provenance
    monkeypatch.setattr(provenance, "capture", lambda *args: ({}, {"gpu_devices": None}))
    monkeypatch.setattr(sys, "argv", [str(PATH)])
    for name in ("mem_get_info", "get_device_name", "get_device_properties", "is_available"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: pytest.fail("No CUDA initialization during read-only probe"))
    assert probe.main() == 2  # CPU PyTorch / no GPU; a block is the honest outcome.
    import json
    report = json.loads(capsys.readouterr().out)
    assert report["model_loaded"] is False and report["result"] == "BLOCKED"
