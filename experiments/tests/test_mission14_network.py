from pathlib import Path
import pytest

def test_nonlocal_mutation_requires_authorization(tmp_path,monkeypatch):
    from backend import main,jobs
    from fastapi.testclient import TestClient
    monkeypatch.setattr(jobs,'get_checkpoint',lambda _:{'name':'owned-fixture','path':'unused','format':'bin'})
    manager=jobs.JobManager(executor=lambda *a:{'mock_only':True},experiment_root=tmp_path/'runs');monkeypatch.setattr(main,'jobs',manager)
    client=TestClient(main.app,base_url='http://localhost:8000',client=('203.0.113.20',1234))
    response=client.post('/api/jobs',json={'operation':'benchmark','checkpoint_id':'owned-fixture'})
    assert response.status_code in (401,403)


@pytest.mark.parametrize('peer,host,origin,status',[
    ('127.0.0.1','localhost',None,200),('::1','[::1]',None,200),
    ('127.0.0.1','localhost','http://localhost:3000',200),
    ('203.0.113.20','localhost',None,403),
    ('127.0.0.1','rebind.attacker.invalid',None,403),
    ('127.0.0.1','localhost','http://attacker.invalid',403),
    ('127.0.0.1','localhost','null',403)])
def test_peer_host_origin_all_required(peer,host,origin,status):
    from backend.main import app
    from fastapi.testclient import TestClient
    client=TestClient(app,base_url='http://localhost',client=(peer,1234))
    headers={'Host':host}
    if origin is not None:headers['Origin']=origin
    headers['X-Forwarded-For']='127.0.0.1'
    assert client.get('/api/health',headers=headers).status_code==status
