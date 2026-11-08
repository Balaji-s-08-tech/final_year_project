"""Mission 8 negative cases: real local random fixtures, never downloaded weights."""
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from safetensors.torch import load_file, save_file
from evaluation.tests.test_baseline import tiny_bunny
from evaluation.adapters import LocalAdapter
from evaluation.data import load_examples
from evaluation.runner import evaluate
from experiments.store import ExperimentStore
from compression.pruning import prune, PruningConfig, reconstruct_state
from compression.synthetic import fixture
from VLM.quantization.awq_quantize import PackedAWQLinear, _quant_params, _dequantize


def test_missing_encoder_fails(tiny_bunny):
    path=Path(tiny_bunny.vision_path)/'model.safetensors'
    state=load_file(path); del state['vision_model.embeddings.patch_embedding.weight']
    save_file(state,path,metadata={'format':'pt'})
    with pytest.raises((ValueError,RuntimeError),match='[Cc]heckpoint|[Cc]overage'):
        LocalAdapter().load(tiny_bunny)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_dense_fails_and_tracks(tiny_bunny,tmp_path,value):
    path=Path(tiny_bunny.model_path)/'model.safetensors'; state=load_file(path)
    state['lm_head.weight'].fill_(value); save_file(state,path,metadata={'format':'pt'})
    with pytest.raises(ValueError,match='[Nn]onfinite'):
        evaluate(tiny_bunny,tmp_path/'runs')
    assert ExperimentStore(tmp_path/'runs').runs()[0].latest()['status']=='FAILED'


def test_internal_prompt_truncation_rejected(tiny_bunny):
    adapter=LocalAdapter(); adapter.load(tiny_bunny)
    adapter.model.config.tokenizer_model_max_length=1
    with pytest.raises(ValueError,match='truncat'):
        adapter.prepare(load_examples(tiny_bunny)[0][0])


def test_pruned_config_and_reload(tmp_path):
    model=fixture()[0]; prune(model,PruningConfig(method='mlp_neurons',sparsity=.5))
    assert model.config.intermediate_size==16
    model.save_pretrained(tmp_path)
    loaded=type(model).from_pretrained(tmp_path,local_files_only=True)
    torch.testing.assert_close(loaded(torch.tensor([[1,7]])).logits,model(torch.tensor([[1,7]])).logits)


def test_nonfinite_bias_reconstruction_rejected():
    model=fixture()[0]; state=model.state_dict(); state['model.layers.0.mlp.fc1.bias']=torch.full_like(state['model.layers.0.mlp.fc1.bias'],float('nan'))
    with pytest.raises(ValueError,match='[Nn]onfinite'):
        reconstruct_state(model,state)


def test_half_conversion_preserves_master_scales():
    linear=torch.nn.Linear(2,1,bias=False); linear.weight.data.fill_(1e-6)
    packed=PackedAWQLinear.from_linear(linear,8,2); expected=packed(torch.eye(2))
    packed.half()
    assert packed.scales.dtype==torch.float32
    torch.testing.assert_close(packed(torch.eye(2)),expected)


@pytest.mark.parametrize('bits',[4,8])
def test_extreme_quantization_rejects_overflow(bits):
    maximum=torch.finfo(torch.float32).max
    with pytest.raises(ValueError,match='represent|finite|overflow'):
        q,s,z=_quant_params(torch.tensor([[-maximum,maximum]]),bits,2)
        _dequantize(q,s,z,2)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_runtime_logits_rejected(tiny_bunny,value):
    adapter=LocalAdapter(); adapter.load(tiny_bunny)
    adapter.model.lm_head.weight.data.fill_(value)
    prepared=adapter.prepare(load_examples(tiny_bunny)[0][0])
    with pytest.raises(ValueError,match='Nonfinite logits'):
        adapter.generate(prepared)


@pytest.mark.parametrize('damage',['corrupt','shape'])
def test_bad_encoder_rejected(tiny_bunny,damage):
    path=Path(tiny_bunny.vision_path)/'model.safetensors'
    if damage=='corrupt': path.write_bytes(b'invalid checkpoint')
    else:
        state=load_file(path); state['vision_model.embeddings.patch_embedding.weight']=torch.ones(1)
        save_file(state,path,metadata={'format':'pt'})
    with pytest.raises(Exception): LocalAdapter().load(tiny_bunny)


def test_comparison_never_verifies_missing_or_changed_identity(tmp_path):
    from experiments.compare import compare
    model={'identifier':'fixture','revision':'r1','architecture':'phi','config_sha256':'config'}
    data={'identifier':'fixture','revision':'v1','fingerprint':'data','evaluation':{'split':'test'}}
    runs=[]
    for digest in ('1'*64,'2'*64):
        r=ExperimentStore(tmp_path).create('test',{},model=model|{'checkpoint_sha256':digest},dataset=data)
        r.transition('RUNNING'); r.transition('COMPLETED'); runs.append(r)
    report=compare(*[r.path for r in runs])
    assert not report['equivalent_protocol_verified']
    assert report['comparison_status']=='NON_COMPARABLE'
    assert report['missing_evidence']
    assert not compare(runs[0].path,runs[0].path)['equivalent_protocol_verified']


def test_invalid_packed_bias_export_rejected(tmp_path):
    from VLM.quantization.awq_quantize import _save_output
    from VLM.quantization.awq_config import AWQConfig
    model,tok,*_=fixture()
    model.model.layers[0].mlp.fc1=PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1,4,8)
    model.model.layers[0].mlp.fc1.bias=torch.zeros(3)
    with pytest.raises(ValueError,match='bias shape'):
        _save_output(tmp_path,model,tok,AWQConfig(bits=4,group_size=8),{}, {})


def test_content_overlap_rejected(tmp_path):
    from compression.pipeline import run_suite,PipelineConfig
    model,tok,_,records,protocol=fixture()
    samples=[tok(records[0]['question'],return_tensors='pt')['input_ids']]
    with pytest.raises(ValueError,match='overlap'):
        run_suite(model,tok,samples,records,calibration_ids=['renamed'],protocol=protocol,output_root=tmp_path,config=PipelineConfig(treatments=('dense',)))


def test_prompt_boundaries_and_lengths(tiny_bunny):
    adapter=LocalAdapter(); adapter.load(tiny_bunny); record=load_examples(tiny_bunny)[0][0]
    prepared=adapter.prepare(record)
    adapter.model.config.tokenizer_model_max_length=prepared.effective_tokens
    assert adapter.generate(adapter.prepare(record)).shape[1]>prepared.prompt_tokens
    adapter.model.config.tokenizer_model_max_length=prepared.effective_tokens-1
    with pytest.raises(ValueError,match='truncat'): adapter.prepare(record)
    with pytest.raises(ValueError,match='image-token'): adapter.prepare(record|{'question':'<image>'},formatted=True)


def test_unmarked_random_weights_cannot_be_pretrained_baseline(tiny_bunny):
    import json
    path=Path(tiny_bunny.model_path)/'config.json'; config=json.loads(path.read_text()); config.pop('weight_origin')
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='Unverified pretrained identity'): LocalAdapter().load(tiny_bunny)


def test_bad_historical_record_is_isolated(tmp_path,monkeypatch):
    import json
    from backend import main
    monkeypatch.setattr(main,'PROJECT_ROOT',tmp_path)
    path=tmp_path/'checkpoints'/'broken'/'benchmark_manifest.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'results':{'x':{'benchmark':{'schema_version':1,'metrics':{}}}}}))
    report=main.benchmark_results()
    assert report['invalid_records'] and report['status']=='not_measured'


def test_nested_suite_history_discovered(tmp_path):
    root=ExperimentStore(tmp_path); parent=root.create('suite',{})
    child=ExperimentStore(parent.path/'runs').create('dense',{})
    assert child.path in [r.path for r in root.runs()]


def test_actual_calibration_usage_is_recorded():
    from VLM.quantization.awq_calibration import collect_linear_inputs
    model=fixture()[0]; linear=model.model.layers[0].mlp.fc1
    collect_linear_inputs(model,{'fc1':linear},[torch.tensor([[1,4]])]*4,torch.device('cpu'),2)
    assert model._awq_calibration_observation['forwarded_sequences']==1
    assert model._awq_calibration_observation['requested_sequences']==4


def test_pruning_metadata_mismatch_rejected():
    from evaluation.integrity import validate_packed
    model=fixture()[0]; model.config.pruned_intermediate_sizes=[999]
    with pytest.raises(ValueError,match='metadata'): validate_packed(model)


def test_unused_duplicate_weight_export_not_counted(tiny_bunny):
    from evaluation.inspection import local_checkpoint
    root=Path(tiny_bunny.model_path); before=local_checkpoint(root)['artifact_bytes']
    (root/'unused.safetensors').write_bytes(b'not consumed by pretrained loader')
    assert local_checkpoint(root)['artifact_bytes']==before


def test_ratio_is_explicitly_filesystem_only(tmp_path):
    from VLM.quantization.benchmark import benchmark_manifest,summarize
    a,b=tmp_path/'a',tmp_path/'b'; a.write_bytes(b'1234'); b.write_bytes(b'12')
    metric=summarize(benchmark_manifest(str(a),None,None,str(b)))['metrics']['compression_ratio']
    assert metric['value']==2 and metric['metadata']['research_equivalence']=='UNVERIFIED'


def test_lora_recovery_explicitly_unsupported(tmp_path):
    from VLM.quantization.awq_model_loader import load_bunny_recovered_model
    directory=tmp_path/'lora'; directory.mkdir(); (directory/'adapter_config.json').write_text('{}')
    checkpoint=tmp_path/'pruned.pt'; torch.save({},checkpoint)
    with pytest.raises(ValueError,match='UNSUPPORTED recovery'):
        load_bunny_recovered_model(base_model='local-only',pruned_checkpoint=str(checkpoint),recovery_checkpoint=str(directory), trusted=True)


def test_legacy_sampler_fails_before_acquisition():
    from VLM.quantization.awq_model_loader import _enable_project_imports
    _enable_project_imports()
    from LLMPruner.datasets.example_samples import get_bunny
    with pytest.raises(ValueError,match='UNSUPPORTED legacy'):
        get_bunny(None,1,8,device='cpu')


def test_manifest_identity_mismatch_rejected(tiny_bunny,tmp_path):
    import json
    from experiments.manifests import model_manifest
    path=Path(tiny_bunny.model_path)/'config.json'; data=json.loads(path.read_text()); data.pop('weight_origin'); path.write_text(json.dumps(data))
    manifest=model_manifest(tiny_bunny.model_path,'WRONG_MODEL',tiny_bunny.model_revision,hash_large=True)
    target=tmp_path/'manifest.json'; target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity mismatch'):
        LocalAdapter().load(replace(tiny_bunny,model_manifest_path=str(target)))
