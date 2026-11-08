"""Local device admission only; never provisions infrastructure or grants billing permission."""
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
