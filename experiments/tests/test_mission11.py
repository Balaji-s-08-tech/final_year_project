"""Adversarial comparison and loader tests; tiny CPU fixtures only."""
import copy
from pathlib import Path
import pytest
import torch
from experiments.store import ExperimentStore,write_json,read_json
from experiments.provenance import canonical_hash,capture
from experiments.compare import compare
from experiments.schema import benchmark,measurement
from compression.synthetic import fixture
from VLM.quantization.awq_quantize import PackedAWQLinear,_save_output
from VLM.quantization.awq_config import AWQConfig
from VLM.quantization.awq_model_loader import load_quantized_bunny_checkpoint


def pair(tmp_path,monkeypatch,status='COMPLETED'):
    import experiments.store as module
    real=capture
    def clean(*args):
        source,env=real(*args); source.update(commit='a'*40,dirty=False);return source,env
    monkeypatch.setattr(module,'capture',clean)
    asset_root=tmp_path/'source';asset_root.mkdir(parents=True,exist_ok=True);(asset_root/'weights').write_bytes(b'local synthetic weights')
    from experiments.provenance import file_hash
    write_json(asset_root/'config.json',{'model_type':'phi','hidden_size':16})
    digest=file_hash(asset_root/'weights',None)
    m={'identifier':'SYNTHETIC','architecture':'phi','revision':'v1','config_sha256':canonical_hash({'model_type':'phi','hidden_size':16}),'checkpoint_sha256':digest['sha256']}
    data_path=tmp_path/'heldout.jsonl';data_path.write_text('{"id":"heldout","question":"fixture question","answers":["fixture answer"]}\n')
    configuration={'seed':42,'adapter':'portable_awq_text','dataset_path':str(data_path),'max_samples':1,'max_input_tokens':32,'max_new_tokens':2,'prompt_template':'{question}'}
    from types import SimpleNamespace
    from evaluation.data import load_examples
    records,fingerprint=load_examples(SimpleNamespace(**configuration))
    d={'identifier':'SYNTHETIC_DATA','revision':'v1','fingerprint':fingerprint,'evaluation':{'sample_ids':[r['id'] for r in records],'split':'test','scorer':'exact_v1','prompt_template':'{question}','max_input_tokens':32,'max_new_tokens':2}}
    runs=[]
    for i in range(2):
        r=ExperimentStore(tmp_path).create('audit',configuration,model=m,dataset=d)
        if status=='COMPLETED':
            r.transition('RUNNING');r.transition(status,metrics=benchmark({'inference_latency':measurement('inference_latency',i+1,status='MEASURED',method='fixture')}))
        elif status in ('FAILED','CANCELLED'):r.transition(status,error='intentional fixture failure')
        region={'path':str(tmp_path/'source'),'status':'CONTENT_HASHED','files':{'weights':digest,'config.json':file_hash(asset_root/'config.json',None)}}
        assets={'model':region,'encoder':copy.deepcopy(region)};assets['sha256']=canonical_hash(assets)
        write_json(r.path/'assets.json',assets)
        write_json(r.path/'effective_inference.json',{'generation':{'max_new_tokens':2,'do_sample':False,'num_beams':1,'use_cache':True,'pad_token_id':0,'eos_token_id':2},'tokenizer':{'class':'fixture','name':'fixture'},'processor':{'size':4}})
        runs.append(r)
    return runs


def rewrite(path,value):path.write_text(__import__('json').dumps(value))


def test_empty_evidence_is_not_verified(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch)
    for r in runs:rewrite(r.path/'assets.json',{})
    assert not compare(*[r.path for r in runs])['equivalent_protocol_verified']


def test_failed_runs_excluded(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch,'FAILED'); report=compare(*[r.path for r in runs])
    assert report['comparison_status']!='COMPARABLE'
    assert not report['equivalent_protocol_verified']


def test_portable_metadata_conflict(tmp_path):
    model,tok,*_=fixture();model.model.layers[0].mlp.fc1=PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1,4,8)
    p=tmp_path/'model.pt';torch.save({'format':'bunny-awq-portable-v1','model':model,'tokenizer':tok,'metadata':{'quantization':{'bits':8}}},p)
    with pytest.raises(ValueError):load_quantized_bunny_checkpoint(str(p), trusted=True)


def test_invalid_attention_rejected(tmp_path):
    model,tok,*_=fixture();model.model.layers[0].self_attn.q_proj=torch.nn.Linear(16,15)
    model.model.layers[0].mlp.fc1=PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1,4,8)
    p=tmp_path/'model.pt';torch.save({'format':'bunny-awq-portable-v1','model':model,'tokenizer':tok,'metadata':{}},p)
    with pytest.raises(ValueError):load_quantized_bunny_checkpoint(str(p), trusted=True)


@pytest.mark.parametrize('damage',['empty','null','list','bad_hash','missing_files','null_generation','missing_generation','invalid_bool','bad_config_hash','conflicting_hash'])
def test_evidence_adversarial(tmp_path,monkeypatch,damage):
    runs=pair(tmp_path,monkeypatch)
    for r in runs:
        assets=read_json(r.path/'assets.json');effective=read_json(r.path/'effective_inference.json')
        if damage=='empty':assets={}
        if damage=='null':assets=None
        if damage=='list':assets=[]
        if damage=='bad_hash':assets['model']['files']['weights']['sha256']='not-a-hash'
        if damage=='missing_files':assets['encoder']['files']={}
        if damage=='null_generation':effective['generation']=None
        if damage=='missing_generation':effective['generation']={}
        if damage=='invalid_bool':effective['generation']['use_cache']='true'
        if damage=='bad_config_hash':rewrite(r.path/'config.json',{'changed':True})
        rewrite(r.path/'assets.json',assets);rewrite(r.path/'effective_inference.json',effective)
    if damage=='conflicting_hash':
        path=runs[1].path/'experiment.json';identity=read_json(path);identity['model']['checkpoint_sha256']='f'*64;rewrite(path,identity)
    report=compare(*[r.path for r in runs])
    assert not report['equivalent_protocol_verified'] and not report['eligible_for_measured_comparison']


def test_verified_and_conditional_positive(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch); paths=[r.path for r in runs]
    report=compare(*paths)
    assert report['verification_status']=='VERIFIED',report
    for r in runs:
        env=read_json(r.path/'environment.json');env.pop('cpu');rewrite(r.path/'environment.json',env)
    report=compare(*paths)
    assert report['verification_status']=='CONDITIONALLY_VERIFIED' and not report['equivalent_protocol_verified']


@pytest.mark.parametrize('status',['FAILED','CANCELLED','CREATED','RUNNING'])
def test_invalid_lifecycle_never_compares_metrics(tmp_path,monkeypatch,status):
    runs=pair(tmp_path,monkeypatch,status)
    if status=='RUNNING':
        for r in runs:r.transition('RUNNING')
    for r in runs:
        p=sorted(r.path.glob('state_*.json'))[-1];state=read_json(p)
        state['metrics']=benchmark({'inference_latency':measurement('inference_latency',1,status='MEASURED',method='fixture')});rewrite(p,state)
    report=compare(*[r.path for r in runs])
    assert report['verification_status']=='NON_COMPARABLE' and report['exclusions']
    assert all(d['difference'] is None for d in report['deltas'].values())


def test_collection_excludes_invalid_runs(tmp_path,monkeypatch):
    from experiments.compare import compare_collection
    good=pair(tmp_path/'good',monkeypatch);bad=pair(tmp_path/'bad',monkeypatch,'FAILED')
    malformed=tmp_path/'malformed';malformed.mkdir();(malformed/'experiment.json').write_text('{bad')
    report=compare_collection(good[0].path,[good[1].path,bad[0].path,malformed])
    assert report['included_count']==1 and report['excluded_count']==2
    assert all(x['report']['exclusions'] for x in report['exclusions'])


def exported(tmp_path,bits=4):
    model,tok,*_=fixture();model.model.layers[0].mlp.fc1=PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1,bits,8)
    return _save_output(tmp_path,model,tok,AWQConfig(bits=bits,group_size=8),{}, {})


@pytest.mark.parametrize('field,value',[('num_attention_heads',0),('num_attention_heads',-1),('num_attention_heads',2.0),('num_attention_heads',True),('hidden_size',15),('num_key_value_heads',1)])
def test_header_rejects_invalid_attention_before_pickle(tmp_path,monkeypatch,field,value):
    p=exported(tmp_path); side=Path(str(p)+'.metadata.json');data=read_json(side)
    data['architecture']['config'][field]=value;data['model_config_sha256']=canonical_hash(data['architecture']['config']);rewrite(side,data)
    import VLM.quantization.awq_model_loader as loader
    monkeypatch.setattr(loader,'_torch_load',lambda *a:pytest.fail('Constructed pickle before invalid header rejection'))
    with pytest.raises(ValueError):loader.load_quantized_bunny_checkpoint(str(p), trusted=True)


@pytest.mark.parametrize('precision,bits',[('INT4',4),('AWQ INT8',4),('AWQ INT3',3),('AWQ INT4',True),('AWQ INT4',None)])
def test_header_precision_rejected_before_pickle(tmp_path,monkeypatch,precision,bits):
    p=exported(tmp_path);side=Path(str(p)+'.metadata.json');data=read_json(side);data['precision']=precision;data['quantization']['bits']=bits;rewrite(side,data)
    import VLM.quantization.awq_model_loader as loader
    monkeypatch.setattr(loader,'_torch_load',lambda *a:pytest.fail('Constructed invalid precision payload'))
    with pytest.raises(ValueError):loader.load_quantized_bunny_checkpoint(str(p), trusted=True)


def test_inner_metadata_reconciled(tmp_path):
    from experiments.provenance import file_hash
    p=exported(tmp_path);payload=torch.load(p,weights_only=False);payload['metadata']['quantization']['bits']=8;torch.save(payload,p)
    side=Path(str(p)+'.metadata.json');data=read_json(side);digest=file_hash(p,None);data.update(sha256=digest['sha256'],size_bytes=digest['size_bytes']);rewrite(side,data)
    with pytest.raises(ValueError,match='Inner/outer precision'):load_quantized_bunny_checkpoint(str(p), trusted=True)


@pytest.mark.parametrize('bits',[4,8])
def test_valid_portable_forward_and_comparison_boundary(tmp_path,monkeypatch,bits):
    p=exported(tmp_path/'export',bits);_,model,metadata=load_quantized_bunny_checkpoint(str(p), trusted=True)
    assert torch.isfinite(model(torch.tensor([[1,4]])).logits).all()
    runs=pair(tmp_path/'runs',monkeypatch)
    for r in runs:
        path=r.path/'experiment.json';identity=read_json(path);identity['model']['config_sha256']=read_json(Path(str(p)+'.metadata.json'))['model_config_sha256'];identity['model']['checkpoint_sha256']=__import__('experiments.provenance',fromlist=['file_hash']).file_hash(p,None)['sha256'];rewrite(path,identity)
        assets=read_json(r.path/'assets.json');assets['model']={'path':str(p),'status':'CONTENT_HASHED','files':{p.name:__import__('experiments.provenance',fromlist=['file_hash']).file_hash(p,None)}}
        assets.pop('sha256');assets['sha256']=canonical_hash(assets);rewrite(r.path/'assets.json',assets)
    assert compare(*[r.path for r in runs])['verification_status']=='VERIFIED'


@pytest.mark.parametrize('damage',['missing_file','changed_file','bad_size','escape','null_record','duplicate_key'])
def test_unverifiable_assets_rejected(tmp_path,monkeypatch,damage):
    runs=pair(tmp_path,monkeypatch)
    if damage=='missing_file':(tmp_path/'source'/'weights').unlink()
    if damage=='changed_file':(tmp_path/'source'/'weights').write_bytes(b'different')
    for r in runs:
        p=r.path/'assets.json';data=read_json(p)
        if damage=='bad_size':data['model']['files']['weights']['size_bytes']=-1
        if damage=='escape':data['model']['files']['../outside']=data['model']['files'].pop('weights')
        if damage=='null_record':data['model']['files']['weights']=None
        data.pop('sha256');data['sha256']=canonical_hash(data);rewrite(p,data)
        if damage=='duplicate_key':p.write_text('{"model":{},"model":{}}')
    assert not compare(*[r.path for r in runs])['eligible_for_measured_comparison']


def test_asset_hash_budget_is_explicit(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch);paths=[r.path for r in runs]
    assert not compare(*paths,max_asset_hash_bytes=1)['equivalent_protocol_verified']
    assert compare(*paths,max_asset_hash_bytes=None)['equivalent_protocol_verified']


def test_invalid_attention_export_rejected_before_serialization(tmp_path,monkeypatch):
    model,tok,*_=fixture();model.model.layers[0].self_attn.q_proj=torch.nn.Linear(16,15)
    model.model.layers[0].mlp.fc1=PackedAWQLinear.from_linear(model.model.layers[0].mlp.fc1,4,8)
    monkeypatch.setattr(torch,'save',lambda *a,**k:pytest.fail('Serialized incompatible attention'))
    with pytest.raises(ValueError,match='projection dimensions'):_save_output(tmp_path,model,tok,AWQConfig(bits=4,group_size=8),{}, {})


def test_malformed_and_partial_records_are_excluded(tmp_path,monkeypatch):
    runs=pair(tmp_path,monkeypatch)
    path=sorted(runs[1].path.glob('state_*.json'))[-1];state=read_json(path);state['metrics']['metrics']['inference_latency']['value']=None;rewrite(path,state)
    report=compare(*[r.path for r in runs]);assert report['verification_status']=='NON_COMPARABLE' and report['exclusions']


def test_header_projection_conflict_rejected_before_pickle(tmp_path,monkeypatch):
    p=exported(tmp_path);side=Path(str(p)+'.metadata.json');data=read_json(side);data['architecture']['blocks'][0]['linears']['self_attn.q_proj']=[16,15];rewrite(side,data)
    import VLM.quantization.awq_model_loader as loader
    monkeypatch.setattr(loader,'_torch_load',lambda *a:pytest.fail('Unpickled invalid projection'))
    with pytest.raises(ValueError,match='projection dimensions'):loader.load_quantized_bunny_checkpoint(str(p), trusted=True)


def test_conflicting_calibration_group_size_rejected(tmp_path):
    from experiments.provenance import file_hash
    p=exported(tmp_path);payload=torch.load(p,weights_only=False);payload['metadata']['quantization']['group_size']=3;torch.save(payload,p)
    side=Path(str(p)+'.metadata.json');data=read_json(side);digest=file_hash(p,None);data.update(sha256=digest['sha256'],size_bytes=digest['size_bytes']);rewrite(side,data)
    with pytest.raises(ValueError,match='configuration conflict'):load_quantized_bunny_checkpoint(str(p), trusted=True)
