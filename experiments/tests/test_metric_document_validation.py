import pytest
from experiments.schema import benchmark, validate_benchmark


@pytest.mark.parametrize("document", [None, [], {"schema_version": True}, {"schema_version": 1, "metrics": []}])
def test_malformed_top_level_documents_fail_explicitly(document):
    with pytest.raises(ValueError):
        validate_benchmark(document)


@pytest.mark.parametrize("change", ["missing_value", "extra_field", "metric_list", "quantization_list", "method_list", "calibration_list"])
def test_invalid_nested_measurements_rejected(change):
    document = benchmark()
    if change == "missing_value":
        del document["metrics"]["inference_latency"]["value"]
    elif change == "extra_field":
        document["metrics"]["inference_latency"]["undocumented"] = True
    elif change == "metric_list":
        document["metrics"]["inference_latency"] = []
    elif change == "quantization_list":
        document["quantization"] = []
    elif change == "method_list":
        document["quantization"]["method"] = []
    else:
        document["quantization"]["calibration"] = []
    with pytest.raises(ValueError):
        validate_benchmark(document)


def test_api_rejects_bad_versioned_document_with_422(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import main
    monkeypatch.setattr(main, "IMPORTED_RESULTS_ROOT", tmp_path / "imports")
    client = TestClient(main.app,base_url="http://localhost:8000",client=("127.0.0.1",50000))
    for malformed in ([], None, {"schema_version": True}):
        response = client.post("/api/results/import", json={"manifest": {"results": {"x": {"precision": "UNKNOWN", "benchmark": malformed}}}})
        assert response.status_code == 422
    assert not (tmp_path / "imports").exists()
