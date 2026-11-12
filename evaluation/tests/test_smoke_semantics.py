from dataclasses import replace
import json
from pathlib import Path
from evaluation.tests.test_baseline import tiny_bunny, ControlledAdapter
from evaluation.adapters import Generation
from evaluation.runner import evaluate
from experiments.store import read_json


class RedSquareAdapter(ControlledAdapter):
    def decode(self, prepared, output):
        return Generation(' The square is red.', [7, 2])


def test_successful_smoke_does_not_rewrite_exact_match(tiny_bunny, tmp_path):
    path = Path(tiny_bunny.dataset_path)
    row = json.loads(path.read_text()); row['answers'] = ['red']
    path.write_text(json.dumps(row)+'\n')
    run = evaluate(replace(tiny_bunny, evaluation_purpose='smoke_test'), tmp_path/'runs', adapter=RedSquareAdapter())
    assert run.latest()['metrics']['metrics']['baseline_accuracy']['value'] == 0
    assert read_json(run.path/'evaluation_summary.json')['score_interpretation'] == 'DIAGNOSTIC_ONLY'
    assert read_json(run.path/'smoke_test_summary.json')['status'] == 'PASSED_EXECUTION'
    assert read_json(run.path/'smoke_test_summary.json')['semantic_answer_correctness'] == 'NOT_AUTOMATICALLY_ASSESSED'
