import argparse
import json
from pathlib import Path
from compression.synthetic import fixture
from compression.pipeline import run_suite, PipelineConfig


def main():
    parser=argparse.ArgumentParser(description='Synthetic CPU-only suite; no remote/model loading options')
    parser.add_argument('--output',required=True)
    parser.add_argument('--config', help='Optional PipelineConfig JSON; inputs remain synthetic and CPU-only')
    args=parser.parse_args()
    config=PipelineConfig(**json.loads(Path(args.config).read_text())).validate() if args.config else PipelineConfig()
    model,tokenizer,samples,records,protocol=fixture()
    run=run_suite(model,tokenizer,samples,records,calibration_ids=['cal-one','cal-two'],protocol=protocol,output_root=args.output,config=config)
    print(run.path)


if __name__=='__main__': main()
