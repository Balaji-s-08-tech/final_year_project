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
        print(json.dumps([run.latest() for run in ExperimentStore(args.root).runs()], indent=2))
    else:
        report = compare(args.baseline, args.candidate)
        write_json(args.output, report)
        publish(Path(args.csv), lambda stream: stream.write(csv_report(report).encode()))


if __name__ == "__main__":
    main()
