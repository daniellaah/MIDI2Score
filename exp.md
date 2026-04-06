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
