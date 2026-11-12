from pathlib import Path
import pytest
import torch

from evaluation.tests.test_baseline import tiny_bunny
from scripts import reproduce_bunny_baseline as reproduction
from experiments.store import read_json


def test_revision_pins_are_the_supplied_baseline():
    assert reproduction.BUNNY == ('BAAI/Bunny-v1_0-3B','0958a104bd39a448fc310b3d1b1fc3d398dc3fcb')
    assert reproduction.SIGLIP == ('google/siglip-so400m-patch14-384','9fdffc58afc957d1a03a25b10dba0329ab15c2a3')


def test_download_requires_explicit_authorization(tmp_path):
    with pytest.raises(PermissionError):reproduction.download(tmp_path/'assets')
    assert not (tmp_path/'assets').exists()


def test_prepare_only_local_fixtures_does_not_initialize_cuda(tiny_bunny,tmp_path,monkeypatch):
    monkeypatch.setattr(reproduction,'BUNNY',(tiny_bunny.model_identifier,tiny_bunny.model_revision))
    monkeypatch.setattr(reproduction,'SIGLIP',('fixture-siglip','fixture-r1'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('Prepare must not initialize CUDA'))
    output=tmp_path/'prepared'
    config_path=reproduction.prepare(tiny_bunny.model_path,tiny_bunny.vision_path,output,True)
    config=read_json(config_path)
    assert config['evaluation_purpose']=='smoke_test'
    assert config['max_samples']==1 and config['warmup_iterations']==0
    assert read_json(output/'preflight.json')['execution_authorized'] is False
    assert read_json(output/'resource_review.json')['compressed_gpu_fit']=='NOT_PROVEN'
    assert read_json(output/'source_assertion.json')['acquisition_provenance']=='OPERATOR_ASSERTED_PINNED_LOCAL_ASSETS'
    original=Path(config_path).read_bytes()
    with pytest.raises(FileExistsError):reproduction.prepare(tiny_bunny.model_path,tiny_bunny.vision_path,output,True)
    assert Path(config_path).read_bytes()==original
