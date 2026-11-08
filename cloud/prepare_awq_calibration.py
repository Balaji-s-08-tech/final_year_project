from __future__ import annotations
from experiments.jsonio import loads as strict_json_loads

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    records = strict_json_loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Bunny training JSON must be a list of conversation records")

    texts = []
    for record in records:
        if not isinstance(record, dict):
            continue
        turns = record.get("conversations")
        if not isinstance(turns, list):
            continue
        parts = [
            turn["value"].replace("<image>", " ").strip()
            for turn in turns
            if isinstance(turn, dict) and isinstance(turn.get("value"), str) and turn["value"].strip()
        ]
        text = "\n".join(part for part in parts if part)
        if text:
            texts.append(text)

    if len(texts) < args.samples:
        raise ValueError(f"Found {len(texts)} usable conversations; requested {args.samples}")
    selected = random.Random(args.seed).sample(texts, args.samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in selected),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "samples": len(selected), "seed": args.seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
