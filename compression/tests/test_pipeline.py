from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from compression.pipeline import run_suite,PipelineConfig,load_artifact,TREATMENTS
from compression.synthetic import fixture
from experiments.store import read_json,ExperimentStore
from evaluation.tests.test_baseline import tiny_bunny


def test_all_treatments_roundtrip_and_fixed_protocol(tmp_path):
    model,tok,samples,records,protocol=fixture()
    before={k:v.clone() for k,v in model.state_dict().items()}
    suite=run_suite(model,tok,samples,records,calibration_ids=['cal-one','cal-two'],protocol=protocol,output_root=tmp_path)
    assert suite.latest()['status']=='COMPLETED'
    report=read_json(suite.path/'comparison.json')
    assert [r['treatment'] for r in report['results']]==list(TREATMENTS)
    assert len({r['protocol_sha256'] for r in report['results']})==1
    for result in report['results']:
        loaded=load_artifact(result['path'],trusted=True)
        assert loaded['metadata']['treatment']==result['treatment']
        assert read_json(Path(result['path'])/'artifact_manifest.json')['size_bytes']>0
        assert result['metrics']['metrics']['peak_gpu_memory']['value'] is None
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_overlap_budget_and_missing_protocol_block_before_runs(tmp_path):
    model,tok,samples,records,protocol=fixture()
    with pytest.raises(ValueError,match='disjoint'):
        run_suite(model,tok,samples,records,calibration_ids=['eval-one','cal-two'],protocol=protocol,output_root=tmp_path)
    with pytest.raises(ValueError,match='budget'):
        run_suite(model,tok,samples,records,calibration_ids=['one','two'],protocol=protocol,output_root=tmp_path,config=PipelineConfig(max_parameter_elements=1))
    with pytest.raises(ValueError):
        run_suite(model,tok,samples,records,calibration_ids=['one','two'],protocol=protocol|{'split':''},output_root=tmp_path)
    assert not list(tmp_path.glob('**/experiment.json'))


def test_failures_are_preserved(tmp_path):
    model,tok,samples,records,protocol=fixture()
    class Bad:
        task='text'
        def __call__(self,*args): raise RuntimeError('intentional CPU fixture failure')
    with pytest.raises(RuntimeError):
        run_suite(model,tok,samples,records,calibration_ids=['one','two'],protocol=protocol,output_root=tmp_path,predictor=Bad())
    suite=ExperimentStore(tmp_path).runs()[0]
    assert suite.latest()['status']=='FAILED'
    assert ExperimentStore(suite.path/'runs').runs()[0].latest()['status']=='FAILED'


def test_real_tiny_multimodal_combined_path(tiny_bunny,tmp_path):
    from evaluation.adapters import LocalAdapter
    from evaluation.data import load_examples
    adapter=LocalAdapter(); adapter.load(tiny_bunny)
    records,_=load_examples(tiny_bunny)
    class Visual:
        task='vqa'
        def __call__(self,model,record):
            adapter.model=model
            prepared=adapter.prepare(record,formatted=True)
            with torch.inference_mode(): output=adapter.generate(prepared)
            generation=adapter.decode(prepared,output)
            self.effective_config=adapter.effective()
            return {'text':generation.text,'token_ids':generation.token_ids}
    protocol={'dataset_identifier':'TINY_RANDOM_VISUAL_FIXTURE','dataset_revision':'v1','split':'heldout_fixture','task':'vqa',
              'model_identifier':'RANDOM_BUNNY_SIGLIP','model_revision':'v1','prompt_template':tiny_bunny.prompt_template}
    run=run_suite(adapter.model,adapter.tokenizer,[torch.tensor([[1,7,8]])],records,calibration_ids=['cal-only'],protocol=protocol,
                  output_root=tmp_path/'runs',config=PipelineConfig(treatments=('dense','structured_awq'),scale_grid=2,clip_grid=2),predictor=Visual())
    assert run.latest()['status']=='COMPLETED'


def test_cli_is_synthetic_cpu_only(tmp_path):
    result=subprocess.run([sys.executable,'-m','compression','--output',str(tmp_path/'runs')],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert read_json(Path(result.stdout.strip())/'comparison.json')['evidence']=='SYNTHETIC_OR_SMALL_CPU_ONLY'
