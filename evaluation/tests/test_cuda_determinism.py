import pytest
import torch
from runtime_policy import prepare_deterministic_cuda


def test_workspace_set_without_initializing_cuda(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('Must not initialize CUDA'))
    assert prepare_deterministic_cuda('cuda:0') == ':4096:8'


def test_late_configuration_requires_restart(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    with pytest.raises(RuntimeError,match='restart'):prepare_deterministic_cuda('cuda:0')


@pytest.mark.parametrize('value',[':4096:8',':16:8'])
def test_valid_operator_configuration_preserved(monkeypatch,value):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',value)
    assert prepare_deterministic_cuda('cuda:0') == value


def test_invalid_configuration_rejected(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG','invalid')
    with pytest.raises(ValueError):prepare_deterministic_cuda('cuda:0')


def test_cpu_does_not_read_or_mutate_cuda_configuration(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:pytest.fail('CPU must not inspect CUDA'))
    assert prepare_deterministic_cuda('cpu') is None
