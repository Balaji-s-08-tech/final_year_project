"""Three Mission 14 exit blockers: public CPU-only reproductions and regressions."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.jsonio import loads
from experiments.store import ExperimentStore
from experiments.manifests import model_manifest
from evaluation.provenance import asset_manifest

EXTREME = '9' * 42


@pytest.mark.parametrize('token', ['1e-' + EXTREME, '-1e-' + EXTREME, '0e-' + EXTREME, '0e+' + EXTREME])
def test_extreme_exponent_is_validation_error(token):
    with pytest.raises(ValueError):
        loads(token)


def test_bad_exponent_preserves_valid_comparison_peer(tmp_path, monkeypatch):
    from experiments.tests.test_mission11 import pair
    from experiments.compare import compare_collection
    runs = pair(tmp_path, monkeypatch)
    bad = tmp_path / 'bad'
    bad.mkdir()
    (bad / 'config.json').write_text('{"seed":1e-' + EXTREME + '}')
    report = compare_collection(runs[0].path, [bad, runs[1].path])
    assert report['included_count'] == report['excluded_count'] == 1
    assert report['exclusions'][0]['report']['evidence_errors']


def test_numeric_api_error_is_structured():
    from backend.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app, base_url='http://localhost:8000', client=('127.0.0.1', 50000), raise_server_exceptions=False)
    response = client.post('/api/results/import', content='{"manifest":{"x":1e-' + EXTREME + '}}', headers={'Content-Type': 'application/json'})
    assert response.status_code == 422
    assert response.json()['detail']['code'] == 'INVALID_JSON_NUMBER'
    assert response.json()['detail']['message']


def test_history_cli_isolates_corrupt_run(tmp_path):
    store = ExperimentStore(tmp_path / 'runs')
    good = store.create('good', {})
    bad = store.create('bad', {})
    (bad.path / 'state_000000.json').write_text('{broken')
    result = subprocess.run([sys.executable, '-m', 'experiments', 'history', '--root', str(store.root)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)
    assert any(row.get('run_id') == good.path.name and row['status'] == 'CREATED' for row in records)
    invalid = next(row for row in records if row.get('status') == 'INVALID')
    assert invalid['path'] == str(bad.path)
    assert invalid['error']['type'] == 'JSONDecodeError'
    assert invalid['error']['message']


@pytest.fixture(scope='module')
def excessive_inventory(tmp_path_factory):
    root = tmp_path_factory.mktemp('owned-empty-inventory')
    for index in range(10001):
        (root / str(index)).touch()
    return root


@pytest.mark.parametrize('reader', ['model_manifest', 'asset_manifest'])
def test_original_unbounded_inventory_is_rejected(excessive_inventory, reader):
    with pytest.raises(ValueError, match='budget|limit'):
        if reader == 'model_manifest':
            model_manifest(excessive_inventory, 'OWNED_EMPTY_FILES', 'fixture-v1')
        else:
            asset_manifest(SimpleNamespace(model_path=str(excessive_inventory), vision_path=None, hash_large_assets=False))


@pytest.mark.parametrize('token', ['1e999', '-1e999', '1e-999', 'NaN', 'Infinity', '9223372036854775808'])
def test_numeric_rejections_keep_machine_readable_context(token):
    from experiments.jsonio import JsonValidationError
    with pytest.raises(JsonValidationError) as error:
        loads(token)
    assert error.value.as_dict()['code'] == 'INVALID_JSON_NUMBER'
    assert error.value.as_dict()['message'] == str(error.value)


@pytest.mark.parametrize('token,value', [('0e-999', 0.0), ('-0.0', -0.0), ('1e-12', 1e-12), ('5e-324', 5e-324), ('1.7976931348623157e308', 1.7976931348623157e308), ('9223372036854775807', 9223372036854775807)])
def test_representable_numeric_edges_still_work(token, value):
    assert loads(token) == value


@pytest.mark.parametrize('damage', [None, [], {'status': 'UNKNOWN'}, {'status': ['COMPLETED']}])
def test_history_cli_rejects_malformed_lifecycle_without_losing_good_peer(tmp_path, damage):
    store = ExperimentStore(tmp_path / 'runs')
    good, bad = store.create('good', {}), store.create('bad', {})
    path = bad.path / 'state_000000.json'
    path.write_text(json.dumps(damage))
    before = path.read_bytes()
    result = subprocess.run([sys.executable, '-m', 'experiments', 'history', '--root', str(store.root)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert good.latest() in rows
    invalid = next(row for row in rows if row.get('status') == 'INVALID')
    assert invalid['path'] == str(bad.path) and invalid['stage'] == 'lifecycle'
    assert invalid['error']['type'] == 'ValueError'
    assert path.read_bytes() == before


def test_inventory_valid_content_and_explicit_skip_profile(tmp_path):
    from dataclasses import replace
    from experiments.assets import AssetLimits
    from experiments.provenance import file_hash
    from experiments.manifests import verify_model_manifest
    limits = AssetLimits(small_hash_bytes=4)
    (tmp_path / '.gitattributes').write_bytes(b'abc')
    (tmp_path / 'weights').write_bytes(b'12345678')
    doc = model_manifest(tmp_path, 'OWNED', 'v1', limits=limits)
    assert doc['status'] == 'INCOMPLETE'
    assert doc['files']['weights'] == {'sha256': None, 'status': 'SKIPPED_SIZE_LIMIT', 'size_bytes': 8}
    full = model_manifest(tmp_path, 'OWNED', 'v1', hash_large=True, limits=limits)
    assert full['files']['weights'] == file_hash(tmp_path / 'weights', None)
    assert full['files']['.gitattributes'] == file_hash(tmp_path / '.gitattributes', None)
    assert verify_model_manifest(tmp_path, full, hash_large=True, limits=limits) == full
    config = SimpleNamespace(model_path=str(tmp_path), vision_path=None, hash_large_assets=True)
    assert asset_manifest(config, limits=limits)['model']['files'] == full['files']
    with pytest.raises(ValueError, match='budget'):
        model_manifest(tmp_path, 'OWNED', 'v1', hash_large=True, limits=replace(limits, max_total_hash_bytes=10))


@pytest.mark.parametrize('reader', ['model_manifest', 'asset_manifest'])
@pytest.mark.parametrize('limit', ['max_entries', 'max_files', 'max_file_bytes', 'max_total_bytes', 'max_total_hash_bytes'])
def test_inventory_limits_precede_any_hashing(tmp_path, monkeypatch, reader, limit):
    from experiments import assets
    (tmp_path / 'a').write_bytes(b'12345678')
    (tmp_path / 'b').write_bytes(b'12345678')
    limit_value = 1 if limit in ('max_entries', 'max_files') else 7 if limit == 'max_file_bytes' else 15
    limits = assets.AssetLimits(**{limit: limit_value})
    monkeypatch.setattr(assets, '_hash_file', lambda *a: pytest.fail('Read payload before operation admission'))
    with pytest.raises(ValueError, match='budget'):
        if reader == 'model_manifest':
            model_manifest(tmp_path, 'OWNED', 'v1', hash_large=True, limits=limits)
        else:
            asset_manifest(SimpleNamespace(model_path=str(tmp_path), vision_path=None, hash_large_assets=True), limits=limits)


def test_shared_region_budget_and_traversal_depth(tmp_path, monkeypatch):
    from experiments import assets
    model, encoder = tmp_path / 'model', tmp_path / 'encoder'
    model.mkdir(); encoder.mkdir()
    (model / 'w').write_bytes(b'123456')
    (encoder / 'w').write_bytes(b'123456')
    monkeypatch.setattr(assets, '_hash_file', lambda *a: pytest.fail('Payload read before aggregate admission'))
    with pytest.raises(ValueError, match='budget'):
        asset_manifest(SimpleNamespace(model_path=str(model), vision_path=str(encoder)), limits=assets.AssetLimits(max_total_bytes=11))
    nested = model / 'one' / 'two'
    nested.mkdir(parents=True)
    with pytest.raises(ValueError, match='depth budget'):
        model_manifest(model, 'OWNED', 'v1', limits=assets.AssetLimits(max_depth=1))


def test_inventory_confines_links_and_skips_directory_loops(tmp_path):
    from experiments.assets import AssetLimits
    root = tmp_path / 'assets'; root.mkdir()
    (root / 'w').write_bytes(b'owned')
    (root / 'alias').symlink_to(root / 'w')
    (root / 'loop').symlink_to(root, target_is_directory=True)
    doc = model_manifest(root, 'OWNED', 'v1', limits=AssetLimits(max_depth=1))
    assert doc['files']['w'] == doc['files']['alias']
    assert set(doc['files']) == {'w', 'alias'}
    outside = tmp_path / 'outside'; outside.write_bytes(b'owned external fixture')
    (root / 'escape').symlink_to(outside)
    for call in (lambda: model_manifest(root, 'OWNED', 'v1'), lambda: asset_manifest(SimpleNamespace(model_path=str(root), vision_path=None))):
        with pytest.raises(ValueError, match='escapes'):call()


def test_asset_growth_after_admission_is_rejected(tmp_path, monkeypatch):
    from experiments import assets
    path = tmp_path / 'w'; path.write_bytes(b'123')
    original = assets._hash_file
    def grow(file, info):
        file.write_bytes(b'1234')
        return original(file, info)
    monkeypatch.setattr(assets, '_hash_file', grow)
    with pytest.raises(ValueError, match='changed'):model_manifest(tmp_path, 'OWNED', 'v1')


@pytest.mark.parametrize('value', [0, -1, True, None, 1.5])
def test_asset_limits_require_explicit_positive_integers(value):
    from experiments.assets import AssetLimits
    with pytest.raises(ValueError):AssetLimits(max_entries=value)


def test_single_portable_file_and_missing_region_remain_explicit(tmp_path):
    path = tmp_path / 'owned.pt'; path.write_bytes(b'owned sentinel; never deserialized')
    doc = asset_manifest(SimpleNamespace(model_path=str(path), vision_path=None))
    assert list(doc['model']['files']) == [path.name]
    assert doc['model']['status'] == 'CONTENT_HASHED'
    assert doc['encoder'] == {'status': 'UNAVAILABLE'}
    missing = asset_manifest(SimpleNamespace(model_path=str(tmp_path / 'missing'), vision_path=None))
    assert missing['model']['status'] == 'INCOMPLETE' and not missing['model']['files']
