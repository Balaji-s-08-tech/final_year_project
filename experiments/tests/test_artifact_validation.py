import json
import pytest
from experiments.artifacts import inspect_artifact
from experiments.provenance import file_hash


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"fixture-only")
    digest = file_hash(path, None)
    data = {"schema_version": 1, "filename": path.name, "size_bytes": digest["size_bytes"], "sha256": digest["sha256"],
            "format": "portable_awq", "precision": "AWQ INT4", "quantization": {"method": "AWQ", "bits": 4}}
    return path, path.with_name(path.name + ".metadata.json"), data


@pytest.mark.parametrize("patch", [
    {"schema_version": True}, {"size_bytes": True}, {"size_bytes": -1}, {"sha256": "no-digest"},
    {"format": "dense"}, {"format": "unknown-format"}, {"quantization": {"method": "AWQ", "bits": True}},
    {"quantization": {"method": "AWQ", "bits": 4, "calibration": []}}, {"precision": "INT4"},
    {"precision": []}, {"format": {}},
])
def test_invalid_header_fails_explicitly(artifact, patch):
    path, sidecar, data = artifact
    sidecar.write_text(json.dumps(data | patch))
    with pytest.raises(ValueError):
        inspect_artifact(path)


@pytest.mark.parametrize("payload", ["[]", "null", '{"schema_version":1,"schema_version":1}', '{"bad":NaN}', '{"bad":Infinity}'])
def test_bad_json_cannot_crash_discovery_with_attribute_errors(artifact, payload):
    path, sidecar, _ = artifact
    sidecar.write_text(payload)
    with pytest.raises(ValueError):
        inspect_artifact(path)


def test_oversized_metadata_and_backend_unknown_fallback(artifact, monkeypatch):
    path, sidecar, _ = artifact
    sidecar.write_text(" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        inspect_artifact(path)
    from backend import checkpoints
    monkeypatch.setattr(checkpoints, "approved_roots", lambda: (path.parent,))
    records = checkpoints.list_checkpoints()
    assert len(records) == 1 and records[0]["precision"] == "UNKNOWN"
