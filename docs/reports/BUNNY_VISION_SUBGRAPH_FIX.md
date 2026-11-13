# Bunny embedded vision subgraph correction

Base: `8d5754df7c8de5c3f99af94ddcd94260ba05e180`.
Fix branch: `codex/fix-bunny-vision-subgraph`.

## Cause and feature dependency

The previous validator incorrectly required every standalone SigLIP tensor to
appear in Bunny's embedded copy and iterated over every standalone tensor. The
user's real Kaggle reproduction exposed the mismatch: 421 embedded versus 448
standalone tensors. This report does not claim a new GPU run.

Inspected code: repository `SiglipVisionTower.feature_select`, ordinary/S2 forward
paths, and Transformers 4.37.2 `SiglipEncoder.forward` and
`SiglipVisionTransformer.forward`. Hidden states contain embeddings at index zero
and one state after each encoder block. For depth 27, `select_layer=-2` resolves
to index 26: embeddings followed by blocks 0 through 25. Encoder block 26,
post-layernorm and the pooling head execute in Hugging Face's full forward pass,
but their outputs do not feed the selected earlier hidden state. They remain
loaded and validated in the standalone checkpoint; nothing is removed at runtime.

The correction derives the required key set from configured depth, actual module
structure and the selector. Later encoder blocks and the pooling head may be
absent from the embedded copy. Post-layernorm is conservatively still required
and compared as part of the supported export contract, matching the supplied
layout. Thus this 27-block case has 421 required tensors and 27 optional tensors
(16 from block 26 and 11 from the head), with no fixed 27-key acceptance rule.

## Validation contract

- Check supported ordinary/S2 Bunny wrapper and feature-selection implementation,
  Transformers 4.37.2 semantics, actual encoder class/depth and selector bounds.
- Require all keys outside the derived optional downstream groups; reject every
  unknown key. Optional groups may be wholly omitted or wholly supplied, never
  partially supplied. Full embedded exports remain supported.
- Iterate exactly over embedded keys, comparing **every supplied tensor**, including
  optional groups if present. Preserve shape, floating dtype, finite-after-conversion
  and exact-after-FP16-value checks. Missing optional source tensors are never read.
- Keep strict language/projector coverage, loading diagnostics, standalone SigLIP
  coverage and trusted legacy behavior unchanged.
- Persist resolved hidden-state index, required key count and standalone-only
  count in the existing loading report.

Selecting the final hidden state now requires the last encoder block; the partial
421-key export is rejected for that selection. Unknown selector implementations,
invalid indices and config/runtime depth conflicts fail closed. Upgrading
Transformers requires a reviewed update of these dependency semantics.

## Regression evidence

`evaluation/tests/test_vision_subgraph.py` uses actual Transformers SigLIP modules
with 27 blocks and tiny widths to reproduce exactly 421/448 key counts. No real
checkpoint weights are downloaded. The test loads the partial embedded export
through LocalAdapter, thereby exercising both key validation and tensor iteration.

A numerical dependency test perturbs the final block, post-layernorm and pooling
head with finite values: final/pooled outputs change, while selected features
remain bit-for-bit equal. Negative cases cover required key loss (including the
retained post-layernorm), changed required weights, shapes/dtypes, unknown vision/
language/projector keys, partial optional groups, changed supplied optional weights,
final/out-of-range/invalid selectors, changed feature selection and incompatible
depth. Earlier full-export and standalone loader regressions remain active.

## Executed validation

Environment: Python 3.12.14 / torch 2.2.2+cpu. Tests used
`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
HF_HOME=/workspace/final_year_project/.cache/huggingface OMP_NUM_THREADS=1`.

```sh
.venv-cpu/bin/python -m pytest backend/tests VLM/quantization/tests evaluation/tests experiments/tests compression/tests cloud/tests LLM-Pruner/tests -q -p no:cacheprovider
```

Result: **611 passed, zero failures/skips, 2 warnings, 56.84 seconds**. Warnings
are the existing Starlette/AnyIO deprecation and PEFT fixture-vocabulary notice.
No existing test assertion was weakened or disabled. The focused initial subgraph
and previous checkpoint modules passed 40 tests before the three final negative
cases were added; all 18 new cases are included in the complete final suite.
All **222 Python sources** compiled with `py_compile.compile(..., doraise=True)`
into a temporary directory, and `git diff --check` passed.

Changed files: `evaluation/checkpoint_layout.py`,
`evaluation/tests/test_vision_subgraph.py`, `BUNNY_GPU_BASELINE_REPRODUCTION.md`
and this report. No historical experiment metrics or real model assets were changed.

Actual pinned checkpoint/GPU validation remains for the user's next Kaggle run;
this correction establishes CPU architecture/layout and dependency correctness only.
