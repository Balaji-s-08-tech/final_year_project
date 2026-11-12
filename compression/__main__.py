from experiments.store import read_json
from experiments.jsonio import loads as strict_json_loads
import argparse
import json
from pathlib import Path
from compression.synthetic import fixture
from compression.pipeline import run_suite, PipelineConfig


def main():
    parser=argparse.ArgumentParser(description='Synthetic CPU-only suite; no remote/model loading options')
    parser.add_argument('--output',required=True)
    parser.add_argument('--config', help='Optional PipelineConfig JSON; inputs remain synthetic and CPU-only')
    parser.add_argument('--with-recovery', action='store_true', help='Include bounded synthetic MLP + LoRA and MLP + LoRA + INT4 treatments')
    args=parser.parse_args()
    config=PipelineConfig(**read_json(args.config)).validate() if args.config else PipelineConfig()
    model,tokenizer,samples,records,protocol=fixture()
    recovery = {}
    if args.with_recovery:
        from dataclasses import replace
        import torch
        from compression.pipeline import RECOVERY_TREATMENTS
        config = replace(config, treatments=tuple(dict.fromkeys((*config.treatments, *RECOVERY_TREATMENTS))))
        ids = tokenizer('answer beta', return_tensors='pt')['input_ids']
        recovery = {'recovery_batches': [{'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': ids.clone()}],
                    'recovery_ids': ['train-one']}
    run=run_suite(model,tokenizer,samples,records,calibration_ids=['cal-one','cal-two'],protocol=protocol,output_root=args.output,config=config,**recovery)
    print(run.path)


if __name__=='__main__': main()
