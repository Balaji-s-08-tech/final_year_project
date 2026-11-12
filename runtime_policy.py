"""Local device admission only; never provisions infrastructure or grants billing permission."""
def prepare_deterministic_cuda(device):
    """Configure cuBLAS before first CUDA use, without initializing CUDA."""
    if not str(device).startswith('cuda'):
        return None
    import os
    import torch
    value = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if value is None:
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA is already initialized without CUBLAS_WORKSPACE_CONFIG; restart with :4096:8 before CUDA initialization')
        value = ':4096:8'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = value
    if value not in {':4096:8', ':16:8'}:
        raise ValueError('Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before CUDA initialization')
    return value


def admit_device(device: str, authorized: bool = False) -> str:
    if not isinstance(device, str) or not (device in {"cpu", "auto", "cuda"} or device.startswith("cuda:")):
        raise ValueError("Unsupported local device")
    if device == "auto" and not authorized:
        return "cpu"  # Safe, deterministic default; do not even query CUDA.
    if device.startswith("cuda") and authorized is not True:
        raise PermissionError("GPU execution is disabled; separate explicit authorization is required")
    return device


def admit_model(model, authorized=False):
    for tensor in list(model.parameters()) + list(model.buffers()):
        admit_device(str(tensor.device), authorized)


def require_local_asset(path):
    from pathlib import Path
    if not isinstance(path, (str, Path)) or not str(path).strip() or not Path(path).exists():
        raise FileNotFoundError('Only explicitly staged local assets are allowed; remote acquisition is disabled')
    return str(Path(path).resolve())


def admit_placement(mapping, authorized=False):
    if mapping is None or mapping == '':
        return
    if isinstance(mapping, dict):
        for value in mapping.values():
            if type(value) is int:
                admit_device(f'cuda:{value}', authorized)
            elif str(value) not in {'cpu', 'disk'}:
                admit_device(str(value), authorized)
    elif str(mapping) in {'auto','balanced','balanced_low_0','sequential'}:
        if authorized is not True:
            raise PermissionError('Automatic device placement requires GPU authorization')
    else:
        admit_device(str(mapping), authorized)
