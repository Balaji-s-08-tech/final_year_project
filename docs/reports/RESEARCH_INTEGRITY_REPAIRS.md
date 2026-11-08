# Integration audit resolution — Mission 3.5

The 2026-11-08 audit at a1b3272 found five gaps despite 236 passing CPU tests.
This increment adds regression evidence for each rather than declaring GPU readiness:

| Finding | Resolution | Evidence |
| --- | --- | --- |
| No supervised targets after truncation | bounded eligibility requires shifted targets; expanded targets checked; finite loss enforced | rejected ignored-only samples, retained partially supervised samples, NaN loss rejection |
| Divergent GPU gates | runtime_policy at supported evaluator, AWQ, loader, backend and probe boundaries; status polling avoids CUDA context | CPU-only admission tests; HTTP override deliberately unavailable |
| Null serialized units | reject null/wrong unit and null metadata before normalization | positive/negative document tests |
| Missing structured results | nested MEASURED values, including zero, detected; completed evaluator states discovered; UI displays versioned records | API import/history regressions, frontend type/build checks |
| Incomplete effective provenance | bounded asset manifest plus effective generation/model/tokenizer/processor snapshot; comparisons warn on differences/incomplete hashes | changed tokenizer fingerprint test; real tiny evaluator tests |

Large files, external producer trust, actual recovery training, GPU fit and official
quality evaluation remain separate evidence requirements. No full model/data download
or GPU execution occurred. These are CPU-validated repairs, not research gains.
