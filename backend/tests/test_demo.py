import json
from dataclasses import replace
from fastapi.testclient import TestClient
from evaluation.tests.test_baseline import tiny_bunny
from backend.main import app

client = TestClient(app, base_url='http://localhost', client=('127.0.0.1', 1234))


def test_optional_backend_without_registry(monkeypatch):
    monkeypatch.delenv('VLM_DEMO_REGISTRY', raising=False)
    assert client.get('/api/demo/models').json()['models'] == []
    assert client.post('/api/demo/inference', json={'model_id':'missing','image_id':'red','question':'hello'}).status_code == 404


def test_cpu_image_api_reports_actual_identity(tiny_bunny, tmp_path, monkeypatch):
    from evaluation.adapters import LocalAdapter
    from compression.pruning import prune, PruningConfig
    from compression.multimodal import base_identity, protected_identity, export_treatment
    adapter=LocalAdapter(); adapter.load(tiny_bunny)
    source=base_identity(tiny_bunny); protected=protected_identity(adapter.model)
    prune(adapter.model, PruningConfig(method='mlp_neurons', sparsity=.25))
    artifact=tmp_path/'treatment'
    export_treatment(adapter, artifact, source_identity=source, protected_sha256=protected, original_widths=[32])
    (tmp_path/'dense.json').write_text(json.dumps(tiny_bunny.to_dict()))
    compressed=replace(tiny_bunny,treatment='structured_pruning',treatment_artifact_path=str(artifact))
    (tmp_path/'compressed.json').write_text(json.dumps(compressed.to_dict()))
    registry=tmp_path/'registry.json'; registry.write_text(json.dumps({'dense':'dense.json','compressed':'compressed.json'}))
    monkeypatch.setenv('VLM_DEMO_REGISTRY',str(registry))
    assert len(client.get('/api/demo/models').json()['models']) == 2
    for model in ('dense','compressed'):
        response=client.post('/api/demo/inference',json={'model_id':model,'image_id':'red','question':'is this red','max_new_tokens':2})
        assert response.status_code == 200, response.text
        body=response.json()
        assert body['generated_tokens'] > 0 and body['peak_vram_bytes'] is None
        assert body['identity']['treatment'] == ('baseline' if model=='dense' else 'structured_pruning')
        assert bool(body['identity']['artifact_sha256']) == (model=='compressed')


def test_request_bounds_and_no_image_path_input():
    for changes in ({'max_new_tokens':True},{'max_new_tokens':10000},{'image_path':'/etc/passwd'}):
        assert client.post('/api/demo/inference',json={'model_id':'dense','image_id':'red','question':'hello',**changes}).status_code == 422
