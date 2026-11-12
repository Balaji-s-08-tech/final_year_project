# Reproduce engineering and future experiments

Base commit: `4b07bd7c3c593caffee345a5912327bff54fc832`.
Working branch: `codex/final-year-research-demo`.

## CPU verification

```sh
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$PWD/.cache/huggingface" OMP_NUM_THREADS=1
.venv-cpu/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider
.venv-cpu/bin/python -m pytest evaluation/tests/test_multimodal_treatments.py compression/tests/test_execute.py backend/tests/test_demo.py -q
cd Interface
npm ci
npm run lint
npm run build
```

Install CPU dependencies in a project environment using `requirements-cpu-dev.txt`.
API tests in managed Codex require sandbox network permission for loopback/asyncio.
No GPU or pretrained assets are needed by the generated tiny fixtures.

## Real experiment input contract

Evaluation JSONL records contain `id`, `question`, nonempty `answers` and a local
`image` path. Config paths resolve relative to their configuration file. The config
declares model/dataset identity and revisions, vision path, dataset split, prompt
with exactly one `<image>` and `{question}`, device/dtype, sample and token bounds,
manifest paths and memory budgets. Use true licensed revisions and complete local
assets; template identifiers are rejected.

Keep calibration, recovery and held-out data disjoint. Existing manifest helpers
check selected IDs/images/fingerprints, not semantic duplication or pretraining
contamination. Preparation plans hash their input files. Manually tokenized recovery
batches do not establish dataset split provenance; document and review that lineage
before publishing a research comparison. Normalized exact match is implemented;
an official dataset scorer is not supplied by this delivery.

Dense and compressed evaluations share prompt/scorer/split and token bounds.
Each creates immutable experiment records, actual prediction token IDs, sample
timings, source/environment identity and measured/unavailable metric statuses.
Protected base assets plus a separately verified treatment distinguish protocol
identity from the transformed weights. The existing generic comparison utility
remains conservative and may reject transformed runs with different asset/effective
records; inspect their recorded evidence rather than claiming automatic comparability.

The offline fixture snapshot was exported from actual prior local synthetic suite
records with `scripts/export_demo_fixture.py --suite SUITE_DIRECTORY`. It discloses
dirty source and contains no research-model claims. Its elapsed values include
compression and evaluation and must not be interpreted as inference latency.
