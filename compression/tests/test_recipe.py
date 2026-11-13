from dataclasses import replace
from pathlib import Path
import json
import weakref

import pytest
import torch
from safetensors.torch import load_file

from evaluation.tests.test_baseline import tiny_bunny
from evaluation.adapters import LocalAdapter
from evaluation.data import load_examples
from evaluation.inspection import compatibility
from compression import recipe
from experiments.manifests import model_manifest
from experiments.store import write_json, read_json
from experiments.provenance import canonical_hash


@pytest.fixture
def source(tiny_bunny,tmp_path):
    from VLM.quantization.awq_model_loader import _enable_project_imports
    _enable_project_imports()
    from bunny.model.language_model.bunny_phi import BunnyPhiConfig,BunnyPhiForCausalLM
    config=BunnyPhiConfig.from_pretrained(tiny_bunny.model_path,local_files_only=True)
    config.num_hidden_layers=32
    torch.manual_seed(81)
    BunnyPhiForCausalLM(config).save_pretrained(tiny_bunny.model_path)
    manifest=tmp_path/'manifest.json'
    write_json(manifest,model_manifest(tiny_bunny.model_path,tiny_bunny.model_identifier,tiny_bunny.model_revision,hash_large=True))
    return tiny_bunny,manifest,tmp_path/'recipe'


def test_32_layer_recipe_matches_source_and_visual_generation(source):
    config,manifest,output=source
    initial=LocalAdapter();initial.load(config)
    protected={k:v.clone() for k,v in initial.model.state_dict().items() if '.mlp.' not in k}
    declaration=recipe.build(config.model_path,manifest,output,execute=True)
    assert declaration['complete_model']
    target=replace(config,treatment='structured_pruning',treatment_artifact_path=str(output))
    restored=LocalAdapter();restored.load(target)
    dense=load_file(str(Path(config.model_path)/'model.safetensors'))
    document,_=recipe.load(output,config.model_path)
    assert len(document['layers'])==32
    for record,block in zip(document['layers'],restored.model.model.layers):
        keep=record['kept_neurons'];prefix=f"model.layers.{record['layer']}.mlp."
        assert torch.equal(block.mlp.fc1.weight,dense[prefix+'fc1.weight'][keep])
        assert torch.equal(block.mlp.fc1.bias,dense[prefix+'fc1.bias'][keep])
        assert torch.equal(block.mlp.fc2.weight,dense[prefix+'fc2.weight'][:,keep])
        assert torch.equal(block.mlp.fc2.bias,dense[prefix+'fc2.bias'])
        assert block.mlp.fc1.out_features==block.mlp.fc2.in_features==24
    assert restored.model.config.intermediate_size==24
    assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in protected.items())
    record=load_examples(config)[0][0]
    with torch.inference_mode():assert restored.decode(restored.prepare(record),restored.generate(restored.prepare(record))).token_ids
    assert restored.treatment_identity['artifact_storage_bytes'] < 1024**2
    assert restored.treatment_identity['deployed_parameter_bytes'] > 0


def test_compiler_releases_layer_tensors_and_plan_does_not_execute(source,monkeypatch):
    config,manifest,output=source
    original=recipe.layer_result;previous=[]
    def inspect(*args):
        assert all(ref() is None for ref in previous)
        record,values=original(*args)
        previous[:]=[weakref.ref(t) for t in values]
        return record,values
    monkeypatch.setattr(recipe,'layer_result',inspect)
    checkpoint_type=recipe.TensorCheckpoint
    contexts=[]
    class TrackedCheckpoint(checkpoint_type):
        def __enter__(self):
            assert all(ref() is None for ref in contexts)
            contexts.append(weakref.ref(self))
            return super().__enter__()
    monkeypatch.setattr(recipe,'TensorCheckpoint',TrackedCheckpoint)
    recipe.build(config.model_path,manifest,output)
    assert not output.exists()
    recipe.build(config.model_path,manifest,output,execute=True)
    assert len(contexts)==34  # one dry plan, one build plan, 32 layer windows


@pytest.mark.parametrize('damage',['hash','missing_layer','keep','width','tensor_hash','config','partial'])
def test_invalid_recipe_rejected(source,damage):
    config,manifest,output=source
    recipe.build(config.model_path,manifest,output,execute=True)
    doc=read_json(output/'recipe.json')
    if damage=='partial':
        (output/'treatment.json').unlink()
    else:
        if damage=='missing_layer':doc['layers'].pop()
        elif damage in ('keep','width','tensor_hash'):
            record=read_json(output/'layers'/'0000.json')
            if damage=='keep':record['kept_neurons'][0]=True
            elif damage=='width':record['remaining_width']=999
            else:record['tensor_hashes']['fc1.weight']='0'*64
            (output/'layers'/'0000.json').write_text(json.dumps(record))
            doc['layers'][0]['receipt_sha256']=canonical_hash(record)
        elif damage=='config':doc['plan']['hidden_size']+=1
        else:doc['status']='CORRUPT'
        (output/'recipe.json').write_text(json.dumps(doc))
        if damage!='hash':
            wrapper=read_json(output/'treatment.json');wrapper['recipe_sha256']=canonical_hash(doc)
            (output/'treatment.json').write_text(json.dumps(wrapper))
    target=replace(config,treatment='structured_pruning',treatment_artifact_path=str(output))
    with pytest.raises((ValueError,OSError)):
        LocalAdapter().load(target)


def test_interrupted_build_resumes_and_partial_pilot_is_not_full_model(source,monkeypatch):
    config,manifest,output=source
    original=recipe.layer_result
    def interrupt(checkpoint,layout,*args):
        if layout['layer']==1:raise RuntimeError('simulated interruption')
        return original(checkpoint,layout,*args)
    monkeypatch.setattr(recipe,'layer_result',interrupt)
    with pytest.raises(RuntimeError):recipe.build(config.model_path,manifest,output,execute=True)
    saved=(output/'layers'/'0000.json').read_bytes()
    assert not (output/'treatment.json').exists()
    monkeypatch.setattr(recipe,'layer_result',original)
    recipe.build(config.model_path,manifest,output,execute=True,resume=True)
    assert (output/'layers'/'0000.json').read_bytes()==saved
    pilot=output.parent/'pilot'
    recipe.build(config.model_path,manifest,pilot,layers=[0,1],execute=True)
    target=replace(config,treatment='structured_pruning',treatment_artifact_path=str(pilot))
    assert compatibility(target)['errors']
    with pytest.raises(ValueError,match='Partial'):recipe.load(pilot,config.model_path)


def test_low_ram_fails_before_payload_hash_or_model_load(source,monkeypatch):
    from types import SimpleNamespace
    config,manifest,output=source
    monkeypatch.setattr(recipe.psutil,'virtual_memory',lambda:SimpleNamespace(available=0))
    monkeypatch.setattr(recipe,'verify_model_manifest',lambda *a,**k:pytest.fail('Payload read before admission'))
    with pytest.raises(ValueError,match='RAM'):recipe.build(config.model_path,manifest,output,execute=True)
    assert not output.exists()


def test_low_disk_fails_before_payload_hash_or_output(source,monkeypatch):
    from types import SimpleNamespace
    config,manifest,output=source
    monkeypatch.setattr(recipe.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    monkeypatch.setattr(recipe,'verify_model_manifest',lambda *a,**k:pytest.fail('Payload read before disk admission'))
    with pytest.raises(ValueError,match='disk'):recipe.build(config.model_path,manifest,output,execute=True)
    assert not output.exists()


def test_recipe_metadata_size_is_admitted_before_hashing(tmp_path,monkeypatch):
    from experiments import assets
    with (tmp_path/'oversized.json').open('wb') as stream:
        stream.truncate(recipe.DISK_RESERVE+1)
    monkeypatch.setattr(assets,'_hash_file',lambda *a:pytest.fail('Oversized metadata read before admission'))
    with pytest.raises(ValueError,match='byte budget'):recipe.storage_bytes(tmp_path)


def test_reconstruction_ram_gate_precedes_full_model_loading(source,monkeypatch):
    from types import SimpleNamespace
    config,manifest,output=source
    recipe.build(config.model_path,manifest,output,execute=True)
    monkeypatch.setattr(recipe.psutil,'virtual_memory',lambda:SimpleNamespace(available=0))
    from bunny.model.language_model.bunny_phi import BunnyPhiForCausalLM
    monkeypatch.setattr(BunnyPhiForCausalLM,'from_pretrained',lambda *a,**k:pytest.fail('Model allocated before RAM admission'))
    target=replace(config,treatment='structured_pruning',treatment_artifact_path=str(output))
    with pytest.raises(ValueError,match='RAM'):LocalAdapter().load(target)


@pytest.mark.parametrize('damage',['missing','dimensions'])
def test_source_layout_invalid_before_output(source,damage):
    from safetensors.torch import save_file
    config,manifest,output=source
    path=Path(config.model_path)/'model.safetensors'
    state=load_file(str(path));key='model.layers.31.mlp.fc2.weight'
    if damage=='missing':del state[key]
    else:state[key]=state[key][:,:-1].contiguous()
    save_file(state,str(path),metadata={'format':'pt'})
    with pytest.raises(ValueError,match='dimension/coverage'):recipe.build(config.model_path,manifest,output,execute=True)
    assert not output.exists()


def test_recipe_evaluation_records_residency_separately(source):
    from evaluation.runner import evaluate
    config,manifest,output=source
    recipe.build(config.model_path,manifest,output,execute=True)
    target=replace(config,treatment='structured_pruning',treatment_artifact_path=str(output))
    run=evaluate(target,output.parent/'evaluations')
    assert run.latest()['status']=='COMPLETED'
    storage=read_json(run.path/'deployment_storage.json')
    assert storage['recipe_bytes']==recipe.storage_bytes(output)
    assert storage['runtime_parameter_bytes']>0 and storage['base_dependency_weight_bytes']>0


def test_source_dtype_and_bias_are_preserved_before_runtime_cast(source):
    from safetensors.torch import save_file
    config,manifest,output=source
    path=Path(config.model_path)/'model.safetensors'
    state=load_file(str(path))
    for key in state:
        if '.mlp.' in key:state[key]=state[key].half()
    save_file(state,str(path),metadata={'format':'pt'})
    manifest.unlink();write_json(manifest,model_manifest(config.model_path,config.model_identifier,config.model_revision,hash_large=True))
    recipe.build(config.model_path,manifest,output,execute=True)
    adapter=LocalAdapter();adapter.load(config)
    adapter.config=replace(config,treatment='structured_pruning')
    recipe.apply(adapter,output)
    for index,block in enumerate(adapter.model.model.layers):
        assert block.mlp.fc1.weight.dtype==block.mlp.fc2.weight.dtype==torch.float16
        assert torch.equal(block.mlp.fc2.bias,state[f'model.layers.{index}.mlp.fc2.bias'])


def test_valid_looking_wrong_selection_and_changed_source_rejected(source):
    from safetensors.torch import save_file
    config,manifest,output=source
    recipe.build(config.model_path,manifest,output,execute=True)
    doc=read_json(output/'recipe.json')
    record=read_json(output/'layers'/'0000.json')
    original=record['kept_neurons']
    candidate=list(range(24))
    if candidate==original:candidate=list(range(8,32))
    record['kept_neurons']=candidate
    (output/'layers'/'0000.json').write_text(json.dumps(record))
    doc['layers'][0]['receipt_sha256']=canonical_hash(record)
    (output/'recipe.json').write_text(json.dumps(doc))
    declaration=read_json(output/'treatment.json');declaration['recipe_sha256']=canonical_hash(doc)
    (output/'treatment.json').write_text(json.dumps(declaration))
    target=replace(config,treatment='structured_pruning',treatment_artifact_path=str(output))
    with pytest.raises(ValueError,match='ranking'):LocalAdapter().load(target)
    path=Path(config.model_path)/'model.safetensors';state=load_file(str(path))
    state['model.layers.0.mlp.fc1.weight']+=.1
    save_file(state,str(path),metadata={'format':'pt'})
    with pytest.raises(ValueError,match='differs'):recipe.load(output,config.model_path)


def test_real_32_by_7680_selection_count_uses_bounded_receipts(source):
    config,manifest,output=source
    from bunny.model.language_model.bunny_phi import BunnyPhiConfig,BunnyPhiForCausalLM
    model_config=BunnyPhiConfig.from_pretrained(config.model_path,local_files_only=True)
    model_config.hidden_size=8;model_config.intermediate_size=10240
    BunnyPhiForCausalLM(model_config).save_pretrained(config.model_path)
    manifest.unlink();write_json(manifest,model_manifest(config.model_path,config.model_identifier,config.model_revision,hash_large=True))
    recipe.build(config.model_path,manifest,output,execute=True)
    loaded,_=recipe.load(output,config.model_path)
    assert len(loaded['layers'])==32
    assert all(len(layer['kept_neurons'])==7680 for layer in loaded['layers'])
    assert (output/'recipe.json').stat().st_size < 1024**2
    assert recipe.storage_bytes(output) < recipe.DISK_RESERVE
