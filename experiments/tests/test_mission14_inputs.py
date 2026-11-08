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
    client=TestClient(main.app,base_url="http://localhost:8000",client=("127.0.0.1",50000))
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


def test_state_selection_never_descends_into_artifacts(tmp_path):
    from experiments.store import ExperimentStore
    run=ExperimentStore(tmp_path/'runs').create('scope',{})
    child=run.path/'zzz_artifacts'/'state_999999.json';child.parent.mkdir();child.write_text('{"status":"FAILED"}')
    assert run.latest()['status']=='CREATED'


@pytest.mark.parametrize('text',['{"x":1,"x":2}','{"x":NaN}','{"x":1e999}','{"x":1e-999}','{"x":9223372036854775808}',b'"\xff"','['*70+'0'+']'*70])
def test_independent_strict_json_boundary_cases(text):
    from experiments.jsonio import loads
    with pytest.raises(ValueError):loads(text)


def test_graph_and_custom_budget_boundaries(tmp_path):
    from experiments.jsonio import JsonLimits,loads,validate,lines
    cycle=[];cycle.append(cycle)
    with pytest.raises(ValueError,match='Circular'):validate(cycle)
    profile=JsonLimits(max_bytes=64,max_nodes=3,max_string_bytes=8,max_key_bytes=2,max_depth=2,max_number_chars=4)
    assert loads('{"a":1}',limits=profile)=={'a':1}
    with pytest.raises(ValueError):loads('[1,2,3]',limits=profile)
    p=tmp_path/'bounded.jsonl';p.write_text('{}\n{}\n{}\n')
    with pytest.raises(ValueError,match='record'):list(lines(p,max_records=2))
    assert len(list(lines(p,max_records=3)))==3


def test_malformed_oversized_record_preserves_valid_collection_peer(tmp_path,monkeypatch):
    from experiments.tests.test_mission11 import pair
    from experiments.compare import compare_collection
    runs=pair(tmp_path,monkeypatch);bad=tmp_path/'bad';bad.mkdir()
    with (bad/'config.json').open('wb') as stream:stream.truncate(8*1024*1024+1)
    report=compare_collection(runs[0].path,[bad,runs[1].path])
    assert report['included_count']==report['excluded_count']==1
    assert report['exclusions'][0]['report']['evidence_errors']


def test_http_stream_budget_without_content_length():
    import asyncio
    from backend.admission import RequestAdmission
    async def exercise():
        sent=[];called=[]
        async def app(*args):called.append(True)
        async def receive():return {'type':'http.request','body':b'x'*1024*1024,'more_body':True}
        async def send(message):sent.append(message)
        scope={'type':'http','method':'POST','path':'/api/jobs','client':('127.0.0.1',1),'headers':[(b'host',b'localhost')]}
        await RequestAdmission(app)(scope,receive,send)
        assert not called
        assert sent[0]['status']==413
    asyncio.run(exercise())
