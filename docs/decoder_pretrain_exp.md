# Decoder Pretraining Experiments

Last updated: 2026-04-08

## Purpose

This file tracks the `rd` decoder pretraining baseline and any follow-up tuning results.
Use the current `600s` baseline as the comparison point for future experiments.

## Current Official Model

- method doc: [`./decoder_pretrain.md`](./decoder_pretrain.md)
- runnable config: [`../configs/pretrain_rd_best.yaml`](../configs/pretrain_rd_best.yaml)
- official best checkpoint: [`../artifacts/pretrained_decoder_rd_best_best.pt`](../artifacts/pretrained_decoder_rd_best_best.pt)
- official latest checkpoint: [`../artifacts/pretrained_decoder_rd_best.pt`](../artifacts/pretrained_decoder_rd_best.pt)

### Long-Run Official Best

- recipe:
  - `max_length = 1024`
  - `sliding_window_stride = 512`
  - `d_model = 256`
  - `nhead = 4`
  - `num_layers = 2`
  - `dim_feedforward = 1024`
  - positional encoding: `sinusoidal`
  - `batch_size = 16`
  - `learning_rate = 6e-4`
  - scheduler: `linear`
  - `warmup_steps = 500`
- full-validation metrics:
  - CE loss: `1.8092`
  - perplexity: `6.1053`
  - token accuracy: `0.5988`
  - top-5 accuracy: `0.7908`

## Baseline For Future Tuning

Use this `600s` run as the current short-budget baseline.

- run dir: [`../artifacts/runs/2026-04-08_01-05-45_750559`](../artifacts/runs/2026-04-08_01-05-45_750559)
- summary: [`../artifacts/runs/2026-04-08_01-05-45_750559/summary.json`](../artifacts/runs/2026-04-08_01-05-45_750559/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_01-05-45_750559/config.yaml`](../artifacts/runs/2026-04-08_01-05-45_750559/config.yaml)

### 600s Baseline Metrics

- best validation loss: `2.2262`
- final step: `3781`
- elapsed seconds: `600.06`
- device: `mps`
- average step time: `0.0115s`
- average tokens per second: `856341.3`

## Stable Conclusions

- The current stable baseline remains:
  - `1024` context
  - sliding-window training with stride `512`
  - `256 / 4 / 2 / 1024`
  - `sinusoidal` positional encoding
- `600s` short-budget confirmation is stable enough to use as the tuning reference.
- Long-run official best still uses the same recipe and remains the model to beat.

## Update Rules

When adding a new experiment here:

1. Record only changes relative to the current `600s` baseline.
2. Keep one short result block per experiment.
3. If a new recipe becomes the official best, update:
   - [`./decoder_pretrain.md`](./decoder_pretrain.md)
   - [`./decoder_pretrain_exp.md`](./decoder_pretrain_exp.md)
4. If a run is clearly worse or uninformative, omit it instead of keeping a full log dump.
