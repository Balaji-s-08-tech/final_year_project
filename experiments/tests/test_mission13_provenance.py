from experiments.tests.test_mission11 import pair,rewrite
from experiments.store import read_json
from experiments.compare import compare

def test_unbound_dataset_identity_never_verifies(tmp_path,monkeypatch):
    from experiments.provenance import canonical_hash
    runs=pair(tmp_path,monkeypatch)
    for r in runs:
        config=read_json(r.path/'config.json');config.pop('dataset_path',None);rewrite(r.path/'config.json',config)
        identity=read_json(r.path/'experiment.json');identity['config_sha256']=canonical_hash(config);rewrite(r.path/'experiment.json',identity)
    assert not compare(*[r.path for r in runs])['equivalent_protocol_verified']


def test_changed_dataset_is_rejected(tmp_path,monkeypatch):
    from pathlib import Path
    runs=pair(tmp_path,monkeypatch)
    config=read_json(runs[0].path/'config.json');p=Path(config['dataset_path'])
    p.write_text(p.read_text().replace('fixture answer','changed answer'))
    q=compare(*[r.path for r in runs])
    assert not q['eligible_for_measured_comparison']
    assert any('fingerprint' in e for e in q['evidence_errors'])


def test_selected_ids_must_match_actual_data(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch)
    for r in runs:
        p=r.path/'experiment.json';identity=read_json(p);identity['dataset']['evaluation']['sample_ids']=['not-selected'];rewrite(p,identity)
    assert not compare(*[r.path for r in runs])['eligible_for_measured_comparison']
