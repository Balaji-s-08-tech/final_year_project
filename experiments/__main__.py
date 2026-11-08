import argparse
import json
from pathlib import Path
from experiments.compare import compare, csv_report
from experiments.store import ExperimentStore, publish, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Create, inspect, and compare immutable experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--root", default="checkpoints/experiments")
    create.add_argument("--name", required=True)
    create.add_argument("--config", required=True)
    history = sub.add_parser("history")
    history.add_argument("--root", default="checkpoints/experiments")
    comparison = sub.add_parser("compare")
    comparison.add_argument("baseline")
    comparison.add_argument("candidate")
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--csv", required=True)
    args = parser.parse_args()
    if args.command == "create":
        config = read_json(args.config)
        print(ExperimentStore(args.root).create(args.name, config, model=config.get("model"), dataset=config.get("dataset"), source_paths=config.get("source_paths", ())).path)
    elif args.command == "history":
        records = []
        for run in ExperimentStore(args.root).runs():
            try:
                state = run.latest()
                if not isinstance(state, dict) or state.get('run_id') != run.path.name or state.get('status') not in {'CREATED', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'}:
                    raise ValueError('Invalid lifecycle object or run identity')
                records.append(state)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
                if hasattr(exc, 'as_dict'):
                    error.update(exc.as_dict())
                records.append({'status': 'INVALID', 'path': str(run.path), 'stage': 'lifecycle', 'error': error})
        print(json.dumps(records, indent=2, allow_nan=False))
    else:
        report = compare(args.baseline, args.candidate)
        write_json(args.output, report)
        publish(Path(args.csv), lambda stream: stream.write(csv_report(report).encode()))


if __name__ == "__main__":
    main()
