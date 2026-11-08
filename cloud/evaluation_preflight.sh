#!/usr/bin/env bash
# Read-only inspection only. Does not provision resources or execute inference.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'Usage: bash cloud/evaluation_preflight.sh <edited-config.json> <new-report.json>' >&2
  exit 2
fi
python -m evaluation --config "$1" --output "$2"
