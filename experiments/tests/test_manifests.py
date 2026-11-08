import pytest
from experiments.manifests import model_manifest, verify_model_manifest, require_disjoint


def test_local_manifest_roundtrip_and_changes(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    doc = model_manifest(tmp_path, 'fixture', 'r1')
    assert verify_model_manifest(tmp_path, doc)['status'] == 'VERIFIED_LOCAL_CONTENT'
    (tmp_path/'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError): verify_model_manifest(tmp_path, doc)


def test_empty_and_placeholder_are_not_verified(tmp_path):
    with pytest.raises(ValueError): model_manifest(tmp_path, 'fixture', 'REPLACE_ME')
    with pytest.raises(ValueError): model_manifest(tmp_path, 'fixture', 'r1')


def test_split_overlap_is_rejected():
    evaluation = {'purpose':'evaluation','sample_ids':['one'],'image_sha256':['image'],'fingerprint':'eval'}
    calibration = {'purpose':'calibration','sample_ids':['two'],'image_sha256':['different'],'fingerprint':'cal'}
    assert require_disjoint(evaluation, calibration)['status'].startswith('CHECKED')
    with pytest.raises(ValueError): require_disjoint(evaluation, calibration | {'sample_ids':['one']})
    with pytest.raises(ValueError): require_disjoint(evaluation, calibration | {'image_sha256':['image']})
