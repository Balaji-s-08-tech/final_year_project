"""Executable checkpoint admission. Acknowledgement is not authentication or a sandbox."""
from pathlib import Path


def require_trusted(trusted=False):
    if trusted is not True:
        raise ValueError('Executable pickle requires explicit trusted=True after verifying source; untrusted artifacts are rejected')


def admit_size(path, max_load_bytes=64*1024*1024):
    if type(max_load_bytes)is not int or max_load_bytes<1:raise ValueError('Explicit positive integer checkpoint byte budget required')
    if Path(path).stat().st_size>max_load_bytes:raise ValueError('Checkpoint exceeds explicit CPU load budget')


def load_pickle(path, *, trusted=False, max_load_bytes=64*1024*1024):
    require_trusted(trusted)  # Must precede importing/loading any artifact payload.
    admit_size(path,max_load_bytes)
    import torch
    return torch.load(Path(path),map_location='cpu',weights_only=False)


def load_tensors(path, *, trusted=False, max_load_bytes=64*1024*1024):
    """Safetensors needs no pickle consent; even restricted .bin loading needs trust.

    weights_only is defense in depth, not a sandbox, especially on legacy Torch.
    """
    path=Path(path)
    if path.suffix=='.safetensors':
        admit_size(path,max_load_bytes)
        from safetensors.torch import load_file
        return load_file(str(path),device='cpu')
    require_trusted(trusted)
    admit_size(path,max_load_bytes)
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)
