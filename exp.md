# Experiment Archive Notice

Detailed historical experiment logs, intermediate configs, and legacy research outputs have been moved to `archives/`.

Current official checkpoints and runnable configs are:

- `rd`
  - config: `configs/pretrain_rd_best.yaml`
  - latest checkpoint: `artifacts/pretrained_decoder_rd_best.pt`
  - best checkpoint: `artifacts/pretrained_decoder_rd_best_best.pt`
- `full`
  - config: `configs/pretrain_full_best.yaml`
  - latest checkpoint: `artifacts/pretrained_decoder_full_best.pt`
  - best checkpoint: `artifacts/pretrained_decoder_full_best_best.pt`

Current `rd` best branch:

- `1024` full-coverage sliding window
- `sliding_window_stride=512`
- from scratch
- full-validation CE: `1.8351`
- best checkpoint: `artifacts/pretrained_decoder_rd_best_best.pt`

Archived materials:

- historical experiment notes: `archives/exp.md`
- research configs: `archives/configs/research/`
- research checkpoints and summaries: `archives/artifacts/research/`
- historical logs: `archives/logs/`

## Active Sliding-Window Batch

Reference baseline:

- experiment id: `EXP-RD-SLIDING-BASELINE-300-001`
- setup: `max_length=1024`, `random_crop=true`, `sliding_window_stride=null`
- best validation loss: `2.764584154146198`

Full-coverage sliding-window comparison:

- `EXP-RD-SLIDING-300-002`
  - `data.random_crop=false`
  - `data.sliding_window_stride=1024`
  - best validation loss: `2.422784344175074`
  - classification: useful
- `EXP-RD-SLIDING-300-003`
  - `data.random_crop=false`
  - `data.sliding_window_stride=512`
  - best validation loss: `2.3037683321482296`
  - classification: useful

Recommendation:

- for the `rd` 1024 branch, using full-coverage sliding windows is clearly better than single-window random crop under the same 300-second budget
- `stride=512` is currently the strongest 300-second baseline on this branch
- next batch should tune around this new sliding-window baseline rather than return to the no-truncation branch

## Active No-Truncation Batch

Reference baseline:

- experiment id: `EXP-RD-NOTRUNC-BASELINE-300-001`
- setup: `max_length=null`, `length_bucketing=true`, `max_tokens_per_batch=8192`
- best validation loss: `5.336567202039044`

First token-budget sweep:

- `EXP-RD-NOTRUNC-BASELINE-300-002`
  - `max_tokens_per_batch=4096`
  - best validation loss: `5.681452613805263`
  - classification: worse
- `EXP-RD-NOTRUNC-BASELINE-300-003`
  - `max_tokens_per_batch=16384`
  - best validation loss: `6.115915381656409`
  - classification: worse
- `EXP-RD-NOTRUNC-BASELINE-300-004`
  - `max_tokens_per_batch=32768`
  - best validation loss: not available within the 300-second budget
  - final step: `205`
  - classification: worse

Recommendation:

- the first sweep suggests that simply increasing token budget is not helping on MPS
- next batch should probe a smaller neighborhood around the baseline, or change a different knob such as `learning_rate` or `batch_size` while keeping the no-truncation pipeline fixed

## Active No-Truncation Learning-Rate Batch

Reference baseline:

- experiment id: `EXP-RD-NOTRUNC-BASELINE-300-001`
- setup: `max_length=null`, `length_bucketing=true`, `max_tokens_per_batch=8192`
- best validation loss: `5.336567202039044`

Second learning-rate sweep:

- `EXP-RD-NOTRUNC-BASELINE-300-005`
  - `training.learning_rate=3e-4`
  - best validation loss: `7.493586902719949`
  - classification: worse
- `EXP-RD-NOTRUNC-BASELINE-300-006`
  - `training.learning_rate=8e-4`
  - best validation loss: `6.423977997053251`
  - classification: worse
- `EXP-RD-NOTRUNC-BASELINE-300-007`
  - `training.learning_rate=1e-3`
  - best validation loss: `6.200319615583576`
  - classification: worse

Recommendation:

- this sweep suggests the no-truncation branch is still under-optimized and does not like the current learning-rate range
- next batch should try a different training knob rather than pushing learning rate further, for example `batch_size` around the baseline or a smaller `max_tokens_per_batch` neighborhood around `8192`

## Active No-Truncation Token-Budget Neighborhood Batch

Reference baseline:

- experiment id: `EXP-RD-NOTRUNC-BASELINE-300-001`
- setup: `max_length=null`, `length_bucketing=true`, `max_tokens_per_batch=8192`
- best validation loss: `5.336567202039044`

Second token-budget neighborhood sweep:

- `EXP-RD-NOTRUNC-BASELINE-300-008`
  - `data.max_tokens_per_batch=6144`
  - best validation loss: `6.989866653195414`
  - classification: worse
- `EXP-RD-NOTRUNC-BASELINE-300-009`
  - `data.max_tokens_per_batch=10240`
  - best validation loss: `6.529763286742806`
  - classification: worse
- `EXP-RD-NOTRUNC-BASELINE-300-010`
  - `data.max_tokens_per_batch=12288`
  - best validation loss: `6.311214550416019`
  - classification: worse

Recommendation:

- the local token-budget search around `8192` is still clearly worse than the baseline
- this no-truncation branch does not currently look worth extending further on token budget alone
- if we continue, the next batch should change a different knob, such as `batch_size`, or revisit the validation/training schedule rather than pushing token budget again

## Active Sliding-Window Neighborhood Batch

Reference baseline:

- experiment id: `EXP-RD-SLIDING-300-003`
- setup: `max_length=1024`, `random_crop=false`, `sliding_window_stride=512`
- best validation loss: `2.3037683321482296`

Stride neighborhood sweep:

- `EXP-RD-SLIDING-300-004`
  - `data.sliding_window_stride=256`
  - best validation loss: `3.2007620330916446`
  - classification: worse
- `EXP-RD-SLIDING-300-005`
  - `data.sliding_window_stride=384`
  - best validation loss: `2.8975163639434687`
  - classification: worse
- `EXP-RD-SLIDING-300-006`
  - `data.sliding_window_stride=768`
  - best validation loss: `3.3747011853710123`
  - classification: worse

Recommendation:

- none of the local stride settings improved on the `stride=512` reference
- the `rd` sliding-window branch looks saturated under the current 300-second budget
- if we continue this branch, the next batch should change a different knob rather than continue narrowing the stride search

## Active Sliding-Window Learning-Rate Batch

Reference baseline:

- experiment id: `EXP-RD-SLIDING-300-003`
- setup: `max_length=1024`, `random_crop=false`, `sliding_window_stride=512`
- best validation loss: `2.3037683321482296`

Learning-rate sweep:

- `EXP-RD-SLIDING-300-007`
  - `training.learning_rate=5e-4`
  - best validation loss: `3.072189261190166`
  - classification: worse
- `EXP-RD-SLIDING-300-008`
  - `training.learning_rate=7e-4`
  - best validation loss: `2.882074626804121`
  - classification: worse
- `EXP-RD-SLIDING-300-009`
  - `training.learning_rate=8e-4`
  - best validation loss: `2.762234331283692`
  - classification: worse

Recommendation:

- none of the tested learning rates improved on the `stride=512` reference
- the `rd` sliding-window branch still looks strongest at the current reference configuration
- if we continue this branch, the next batch should change a different knob rather than keep narrowing learning rate around the same point

## Active Sliding-Window Batch-Size Batch

Reference baseline:

- experiment id: `EXP-RD-SLIDING-300-003`
- setup: `max_length=1024`, `random_crop=false`, `sliding_window_stride=512`
- best validation loss: `2.3037683321482296`

Batch-size sweep:

- `EXP-RD-SLIDING-300-010`
  - `training.batch_size=6`
  - best validation loss: `3.342595467472292`
  - classification: worse
- `EXP-RD-SLIDING-300-011`
  - `training.batch_size=12`
  - best validation loss: `3.4011293977525017`
  - classification: worse
- `EXP-RD-SLIDING-300-012`
  - `training.batch_size=16`
  - best validation loss: `2.998826434264549`
  - classification: worse

Recommendation:

- all tested batch sizes are clearly worse than the `stride=512` reference
- the `rd` sliding-window branch does not currently look promising under batch-size tuning alone
- if we continue, the next batch should change a different knob rather than expand the batch-size sweep

## Active Sliding-Window Long-Budget Promotion

Reference:

- promoted candidate: `EXP-RD-SLIDING-300-003`
- setup: `max_length=1024`, `random_crop=false`, `sliding_window_stride=512`
- reason for promotion: strongest clean `300s` baseline on the sliding-window branch

Long-budget result:

- `EXP-RD-SLIDING-LONG-001`
  - `training.max_duration_seconds=7200`
  - `training.num_eval_batches=null`
  - from-scratch run with full-validation model selection
  - best validation loss: `1.8351061700845759`
  - best step: `61500`
  - classification versus official `rd` best `1.9336`: useful

Recommendation:

- promote the `1024 + sliding_window_stride=512` branch to the new official `rd` best
- update `configs/pretrain_rd_best.yaml`, best-model exports, and `docs/decoder_pretrain.md`
