import json
import pytest
from evaluation.tests.test_baseline import tiny_bunny
from compression.execute import execute
from experiments.store import read_json


def test_stage_checkpoints_resume_and_evaluation(tiny_bunny, tmp_path):
    recovery=tmp_path/'batches.json'
    recovery.write_text(json.dumps([{'input_ids':[[4,5,6]],'attention_mask':[[1,1,1]],'labels':[[4,5,6]]}]))
    calibration=tmp_path/'calibration.txt'; calibration.write_text('USER is red\nASSISTANT is no\n')
    plan={'stages':['dense','prune','recover','awq'],'calibration_data':str(calibration),
          'recovery_batches':str(recovery),'evaluate':True}
    root=execute(tiny_bunny,plan,tmp_path/'run')
    assert read_json(root/'03-complete.json')['treatment']=='pruning_awq'
    initial=(root/'03-complete.json').read_bytes()
    execute(tiny_bunny,plan,root,resume=True)
    assert (root/'03-complete.json').read_bytes()==initial
    assert read_json(root/'evaluation-complete.json')['run']
    calibration.write_text('changed input\n')
    with pytest.raises(ValueError,match='identical plan'):
        execute(tiny_bunny,plan,root,resume=True)


def test_existing_output_never_overwritten(tiny_bunny,tmp_path):
    root=tmp_path/'existing';root.mkdir();(root/'important').write_text('keep')
    with pytest.raises(ValueError):execute(tiny_bunny,{'stages':['dense']},root)
    assert (root/'important').read_text()=='keep'
