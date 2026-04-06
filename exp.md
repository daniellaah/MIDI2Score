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

Archived materials:

- historical experiment notes: `archives/exp.md`
- research configs: `archives/configs/research/`
- research checkpoints and summaries: `archives/artifacts/research/`
- historical logs: `archives/logs/`

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
