"""Print a small offline snapshot from actual completed synthetic suite records."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.store import read_json


def snapshot(root):
    root=Path(root)
    comparison=read_json(root/'comparison.json')
    rows=[]
    for result in comparison['results']:
        run=Path(result['path'])
        state=read_json(sorted(run.glob('state_*.json'))[-1])
        if state['status']!='COMPLETED': raise ValueError('Only completed fixture runs may be exported')
        meta=read_json(run/'compression.json')
        metrics=result['metrics']['metrics']
        def measured(*keys):
            return next((metrics[key]['value'] for key in keys if metrics.get(key,{}).get('status')=='MEASURED'),None)
        identity=read_json(run/'experiment.json')
        rows.append({'treatment':result['treatment'],'parameters':meta['architecture']['logical_parameter_elements'],
            'artifact_bytes':measured('original_model_size','compressed_model_size'),
            'accuracy':measured('baseline_accuracy','compressed_accuracy'),
            'preparation_and_eval_seconds':measured('execution_duration'), 'vram_bytes':None,
            'source_commit':identity['source']['commit'], 'source_dirty':identity['source']['dirty']})
    return {'evidence':'MEASURED_RANDOM_CPU_TEXT_FIXTURE_ONLY',
        'warning':'Not Bunny-3B research results. Duration includes compression and fixture evaluation; it is not inference latency. Serialized container size is not deployment size.',
        'rows':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',required=True)
    print(json.dumps(snapshot(parser.parse_args().suite),indent=2,allow_nan=False))
