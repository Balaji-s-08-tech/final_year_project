from pathlib import Path
import pytest
import torch
from experiments.tests.test_mission11 import pair, rewrite
from experiments.store import read_json
from experiments.compare import compare

def test_measured_protocol_metadata_conflict(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch)
    for count,r in enumerate(runs):
        p=sorted(r.path.glob('state_*.json'))[-1];d=read_json(p)
        d['metrics']['metrics']['inference_latency']['metadata']={'warmup_iterations':count*3,'batch_size':1};rewrite(p,d)
    report=compare(*[r.path for r in runs])
    assert not report['eligible_for_measured_comparison']
    assert report['deltas']['inference_latency']['difference'] is None



@pytest.mark.parametrize('field,value',[('warmup_iterations',3),('batch_size',2),('synchronize',False),('repetitions',4)])
def test_each_measurement_setting_is_material(tmp_path,monkeypatch,field,value):
    runs=pair(tmp_path,monkeypatch)
    for i,run in enumerate(runs):
        path=sorted(run.path.glob('state_*.json'))[-1];doc=read_json(path)
        doc['metrics']['metrics']['inference_latency']['metadata']={field:value} if i else {}
        rewrite(path,doc)
    report=compare(*(r.path for r in runs))
    assert report['verification_status']=='NON_COMPARABLE'
    assert 'inference_latency' in report['metric_protocol_conflicts']
    assert report['deltas']['inference_latency']['difference'] is None
