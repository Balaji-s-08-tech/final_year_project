from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from safetensors import SafetensorError

from evaluation.tests.test_baseline import tiny_bunny
from compression import mlp_pilot
from experiments.manifests import model_manifest
from experiments.store import write_json, read_json


@pytest.fixture
def source(tiny_bunny,tmp_path,monkeypatch):
    monkeypatch.setattr(mlp_pilot,'BUNNY',(tiny_bunny.model_identifier,tiny_bunny.model_revision))
    manifest=tmp_path/'manifest.json'
    write_json(manifest,model_manifest(tiny_bunny.model_path,tiny_bunny.model_identifier,tiny_bunny.model_revision,hash_large=True))
    return Path(tiny_bunny.model_path),manifest


def test_plan_never_loads_tensors_or_writes_output(source,tmp_path,monkeypatch):
    monkeypatch.setattr(mlp_pilot,'_load_pair',lambda *a:pytest.fail('Planning loaded weights'))
    root,manifest=source
    plan=mlp_pilot.pilot(root,manifest,output=tmp_path/'unused')
    assert plan['status']=='PLAN_ONLY_NOT_EXECUTED'
    assert not (tmp_path/'unused').exists()


def test_cpu_micro_pilot_save_reload_and_source_preservation(source,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('Pilot must remain CPU-only'))
    root,manifest=source
    before=(root/'model.safetensors').read_bytes()
    report=mlp_pilot.pilot(root,manifest,execute=True,output=tmp_path/'run')
    assert report['status']=='CPU_OPERATOR_PILOT_COMPLETED'
    assert report['gpu_execution']=='NOT_PERFORMED' and report['model_quality']=='NOT_MEASURED'
    assert report['pruning']['parameter_elements_after'] < report['pruning']['parameter_elements_before']
    assert (root/'model.safetensors').read_bytes()==before
    assert report['artifact']['size_bytes'] <= mlp_pilot.MAX_COMPONENT_BYTES
    assert read_json(tmp_path/'run'/'pilot.json')['artifact_scope']=='ONE_MLP_PAIR_NOT_A_DEPLOYABLE_BUNNY_CHECKPOINT'


def test_loaded_source_values_and_paired_slicing_match_checkpoint(source):
    from safetensors.torch import load_file
    from compression.pruning import prune, PruningConfig
    root,manifest=source
    plan=mlp_pilot.inspect_component(root,manifest)
    state=load_file(str(root/'model.safetensors'))
    model,activation=mlp_pilot._load_pair(root,plan)
    original={field:state[key].float().clone() for field,key in zip(mlp_pilot.FIELDS,plan['keys'])}
    for field,value in model.model.layers[0].mlp.state_dict().items():
        assert torch.equal(value,original[field])
    inputs=torch.randn(1,16,plan['hidden_size'],generator=torch.Generator().manual_seed(42))
    report=prune(model,PruningConfig(method='mlp_neurons',sparsity=.25))
    keep=report['layers'][0]['kept_neurons']
    assert len(keep)==plan['original_intermediate']-int(.25*plan['original_intermediate'])
    mlp=model.model.layers[0].mlp
    assert torch.equal(mlp.fc1.weight,original['fc1.weight'][keep])
    assert torch.equal(mlp.fc1.bias,original['fc1.bias'][keep])
    assert torch.equal(mlp.fc2.weight,original['fc2.weight'][:,keep])
    assert torch.equal(mlp.fc2.bias,original['fc2.bias'])
    hidden=activation(torch.nn.functional.linear(inputs,original['fc1.weight'],original['fc1.bias']))
    mask=torch.zeros(plan['original_intermediate']);mask[keep]=1
    reference=torch.nn.functional.linear(hidden*mask,original['fc2.weight'],original['fc2.bias'])
    torch.testing.assert_close(mlp.fc2(activation(mlp.fc1(inputs))),reference,rtol=1e-4,atol=1e-4)


@pytest.mark.parametrize('kind',['ram','disk','existing','source','source_value','cap'])
def test_pilot_rejects_before_loading_or_overwriting(source,tmp_path,monkeypatch,kind):
    root,manifest=source
    destination=tmp_path/'run'
    monkeypatch.setattr(mlp_pilot,'_load_pair',lambda *a:pytest.fail('Admission failure reached loading'))
    if kind=='ram':monkeypatch.setattr(mlp_pilot.psutil,'virtual_memory',lambda:SimpleNamespace(available=0))
    elif kind=='disk':
        import shutil
        monkeypatch.setattr(shutil,'disk_usage',lambda *a:SimpleNamespace(free=0))
    elif kind=='existing':
        destination.mkdir();(destination/'keep').write_text('preserve')
    elif kind=='source':
        with (root/'model.safetensors').open('ab') as stream:stream.write(b'altered')
    elif kind=='source_value':
        from safetensors.torch import load_file, save_file
        path=root/'model.safetensors'
        state=load_file(str(path));state['model.layers.0.mlp.fc1.weight'] += .25
        save_file(state,str(path),metadata={'format':'pt'})
    else:monkeypatch.setattr(mlp_pilot,'MAX_COMPONENT_BYTES',1)
    if kind in {'ram','disk','existing','cap'}:
        monkeypatch.setattr(mlp_pilot,'file_hash',lambda *a:pytest.fail('Admission failure reached payload hashing'))
    with pytest.raises((ValueError,SafetensorError)) as failure:
        mlp_pilot.pilot(root,manifest,execute=True,output=destination)
    if kind=='source_value':assert 'Selected source file differs' in str(failure.value)
    if kind=='existing':assert (destination/'keep').read_text()=='preserve'
    else:assert not destination.exists()
