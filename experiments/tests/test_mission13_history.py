import pytest
from experiments.store import ExperimentStore


def test_corrupt_history_keeps_valid_peer(tmp_path,monkeypatch):
    from backend import main
    monkeypatch.setattr(main,'PROJECT_ROOT',tmp_path);monkeypatch.setattr(main,'OUTPUT_ROOT',tmp_path/'checkpoints')
    store=ExperimentStore(tmp_path/'checkpoints'/'experiments');good=store.create('good',{});bad=store.create('bad',{})
    (bad.path/'state_000000.json').write_text('{broken')
    history=main.experiment_history()
    assert any(r['path']==str(good.path) for r in history)
    assert any(r.get('state',{}).get('status')=='INVALID' for r in history)
    report=main.benchmark_results();assert report['invalid_records']


def test_backend_restart_records_invalid_job(tmp_path):
    from backend.jobs import JobManager
    store=ExperimentStore(tmp_path);bad=store.create('bad',{})
    from experiments.store import read_json
    import json
    p=bad.path/'state_000000.json';state=read_json(p);state['job']={'id':'invalid','status':'UNKNOWN'};p.write_text(json.dumps(state))
    manager=JobManager(experiment_root=tmp_path)
    assert manager.invalid_runs and manager.list()==[]
