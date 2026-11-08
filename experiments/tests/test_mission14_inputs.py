import ast
import json
from pathlib import Path
import pytest
from experiments.store import read_json
@pytest.mark.parametrize('kind',['nodes','string','key','surrogate'])
def test_json_missing_resource_bounds(kind):
    from experiments.jsonio import loads,validate
    if kind=='nodes':value=json.dumps([None]*100001)
    elif kind=='string':value=json.dumps('x'*(1024*1024+1))
    elif kind=='key':value=json.dumps({'k'*4097:1})
    else:
        with pytest.raises(ValueError):validate({'value':'\ud800'})
        return
    with pytest.raises(ValueError):loads(value)


def test_file_budget_before_unbounded_read(tmp_path,monkeypatch):
    path=tmp_path/'large.json'
    with path.open('wb') as f:f.truncate(8*1024*1024+1)
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('Unbounded text read before byte admission'))
    with pytest.raises(ValueError,match='budget|limit'):read_json(path)


def test_large_http_request_stops_before_validation(monkeypatch):
    from backend import main
    from fastapi.testclient import TestClient
    client=TestClient(main.app)
    response=client.post('/api/results/import',content=b' '*(2*1024*1024+1),headers={'Content-Type':'application/json'})
    assert response.status_code==413


def test_comparison_collection_budget(monkeypatch):
    import experiments.compare as module
    monkeypatch.setattr(module,'compare',lambda *a:{'eligible_for_measured_comparison':True})
    with pytest.raises(ValueError,match='budget|limit'):
        module.compare_collection('baseline',['candidate']*129)


def test_history_collection_budget(tmp_path):
    from experiments.store import ExperimentStore
    for i in range(1001):
        p=tmp_path/'name'/str(i)/'experiment.json';p.parent.mkdir(parents=True);p.write_text('{}')
    with pytest.raises(ValueError,match='budget|limit'):ExperimentStore(tmp_path).runs()

