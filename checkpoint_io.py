"""Executable checkpoint admission. Acknowledgement is not authentication or a sandbox."""
from pathlib import Path


def require_trusted(trusted=False):
    if trusted is not True:
        raise ValueError('Executable pickle requires explicit trusted=True after verifying source; untrusted artifacts are rejected')


def admit_size(path, max_load_bytes=64*1024*1024):
    if type(max_load_bytes)is not int or max_load_bytes<1:raise ValueError('Explicit positive integer checkpoint byte budget required')
    if Path(path).stat().st_size>max_load_bytes:raise ValueError('Checkpoint exceeds explicit CPU load budget')


def validate_digest(value):
    import re
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('Explicit SHA256 checkpoint identity required')
    return value


def verify_identity(path, expected_sha256, *, max_load_bytes=64*1024*1024):
    from experiments.provenance import file_hash
    validate_digest(expected_sha256)
    admit_size(path,max_load_bytes)
    actual=file_hash(path,max_load_bytes)
    if actual.get('status')!='HASHED' or actual.get('sha256')!=expected_sha256:
        raise ValueError('Checkpoint content hash changed since trust acknowledgement')


def load_pickle(path, *, trusted=False, max_load_bytes=64*1024*1024, expected_sha256=None):
    require_trusted(trusted)  # Before artifact access, including metadata/hash reads.
    if expected_sha256 is not None:validate_digest(expected_sha256)
    admit_size(path,max_load_bytes)
    import os
    import stat
    import hashlib
    fd=os.open(Path(path),os.O_RDONLY|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>max_load_bytes:
            raise ValueError('Checkpoint is not regular or exceeds CPU load budget')
        if expected_sha256 is not None:
            digest=hashlib.sha256();count=0
            while True:
                chunk=stream.read(min(1024*1024,max_load_bytes-count+1))
                if not chunk:break
                count+=len(chunk)
                if count>max_load_bytes:raise ValueError('Checkpoint exceeds CPU load budget')
                digest.update(chunk)
            if digest.hexdigest()!=expected_sha256:raise ValueError('Checkpoint hash changed before deserialization')
            stream.seek(0)
        import torch
        return torch.load(stream,map_location='cpu',weights_only=False)


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
