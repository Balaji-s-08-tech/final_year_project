import argparse
import json
from evaluation.config import EvaluationConfig
from evaluation.preflight import preflight
from experiments.store import write_json


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation preflight; execution is opt-in")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="Exclusive report file, or experiment root for --execute")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-gpu-execution", action="store_true", help="Use only AFTER separate billing/GPU authorization")
    args = parser.parse_args()
    config = EvaluationConfig.load(args.config)
    if args.execute:
        from evaluation.runner import evaluate
        print(evaluate(config, args.output, authorize_gpu=args.authorize_gpu_execution).path)
    else:
        report = preflight(config)
        write_json(args.output, report)
        print(json.dumps({"status": report["status"], "errors": report["errors"]}, indent=2))
        return 2 if report["errors"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
