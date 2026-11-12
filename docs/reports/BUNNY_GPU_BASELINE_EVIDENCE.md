# Supplied Kaggle Bunny GPU baseline evidence

This is a review of user-supplied evidence, **not a GPU experiment rerun by this
engineering session**. The original archive and its historical metrics were not
modified or imported into the existing experiment store.

Source: `BUNNY_GPU_BASELINE_EVIDENCE.zip`, SHA256
`6e8cf754f76d9f26a9c293c470a140eec568574585c44f8c4a283863d50e8ca8`.
It contains patched loader source, configuration, manifests, environment,
COMPLETED lifecycle state, prediction, metrics and logs; it does not contain the
model weights. Tensor equality cannot be independently recomputed from this archive.

## Recorded setup

- Bunny: `BAAI/Bunny-v1_0-3B`, revision
  `0958a104bd39a448fc310b3d1b1fc3d398dc3fcb`.
- SigLIP: `google/siglip-so400m-patch14-384`, revision
  `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`.
- Python 3.12.3, torch 2.2.2+cu121, Transformers 4.37.2, Tesla T4; execution
  config selected `cuda:0`, FP16, batch 1, one sample, zero warmups, 16 new tokens.
- The environment lists two T4 devices. This is evidence of a single selected
  device run, not multi-GPU execution or combined VRAM.
- The user reports that 421 embedded vision tensors match standalone SigLIP after
  FP16 conversion. Local patches accepted those keys and 440 unused text/head
  keys. These numbers describe this checkpoint, not universal validation rules.

## Historical observations, retained exactly

| Observation | Recorded value |
| --- | --- |
| Prediction | ` The square is red.` |
| Reference | `red` |
| Generated tokens | 6, including EOS |
| Synchronized generation latency | 2.4565940510001383 seconds |
| Peak allocated VRAM | 7,141,774,848 bytes |
| Peak reserved VRAM | 7,482,638,336 bytes |
| Execution duration | 38.394406120999975 seconds |
| Normalized exact match | 0 correct / 1 sample |
| Model + standalone vision weight-file bytes | 9,876,575,208 |

The output visibly identifies red. Whole-answer exact match compares the complete
normalized sentence with `red`, so it returns zero. That score is neither a loader
failure nor evidence that the color was missed. It remains zero in the original
record; no relaxed scorer or replacement accuracy was applied.

This is a successful loading/image-generation **smoke test**, not an official VQA
benchmark or evidence of quality retention. One sample and zero warmups cannot
establish stable latency statistics, throughput gains or compression benefits.

## Provenance limits

`experiment/experiment.json` has no Git commit/branch/dirty identity. It records
source-file hashes, including the supplied adapter hash
`9a5f4e09f7c4faed941ed6cec28e84650e0c1f758ca591d975d5b8a5ae201272`.
Do not retroactively assign this run to the new stabilization commit. Its source
used checkpoint-specific counts; the reviewed implementation now uses configured
metadata and runtime tensor equality. Future runs will create new records and may
have different timing due to the added validation work.

The reproduction script generates the same 384×384 white image with a red square
at coordinates 110..274. In this CPU environment its PNG matched the supplied image
byte-for-byte: SHA256
`87346886812b4708332c8e46e5874921f4b6258764d29ad0762511a3d4dfdcc3`.
Future runs use a newly labeled generated smoke dataset revision and new output
directories. No original sample or score is overwritten.
