# Decoder Pretraining Experiments

Last updated: 2026-04-11

## Current Baseline

- status: accepted `rd` baseline on `main`
- config: [`../configs/pretrain_rd_best.yaml`](../configs/pretrain_rd_best.yaml)
- run dir: [`../artifacts/runs/2026-04-10_22-28-37_632453`](../artifacts/runs/2026-04-10_22-28-37_632453)
- summary: [`../artifacts/runs/2026-04-10_22-28-37_632453/summary.json`](../artifacts/runs/2026-04-10_22-28-37_632453/summary.json)

### Recipe

- `max_length = 1024`
- `sliding_window_stride = 256`
- `d_model = 512`
- `nhead = 8`
- `num_layers = 4`
- `dim_feedforward = 2048`
- `dropout = 0.05`
- residual layout: `pre_norm`
- norm type: `rmsnorm`
- activation: `swiglu`
- positional encoding: `sinusoidal`
- `tie_embeddings = true`
- optimizer: `adamw`
- `learning_rate = 6e-4`
- `weight_decay = 0.1`
- `beta1 = 0.9`
- `beta2 = 0.95`
- `grad_clip_norm = 2.0`
- scheduler: `cosine`
- `warmup_steps = 500`

### Metrics

- best validation loss: `1.6735`
- final step: `10500`
- elapsed seconds: `7301.52`
- average tokens per second: `20383.4`

## Promotion Path

- baseline 7200s rerun: `1.7708`
- `weight_decay = 1e-2`: `1.7670`
- `d_model = 512, nhead = 8, dim_feedforward = 2048`: `1.7520`
- `weight_decay = 0.1`: `1.7248`
- `AdamW beta2 = 0.95`: `1.7197`
- `dropout = 0.1`: `1.6922`
- `dropout = 0.05`: `1.6834`
- `grad_clip_norm = 1.0`: `1.6831`
- `grad_clip_norm = 2.0`: `1.6735`

## High-Signal Findings

### Confirmed Helpful

- Stronger regularization helped repeatedly under the 7200s budget:
  `weight_decay = 0.1`, `dropout = 0.05`, `grad_clip_norm = 2.0`.
- `AdamW beta2 = 0.95` improved over the default `0.999`.
- Wider than the older 384-d model helped, but only up to the current 512-d baseline.

### Confirmed Unhelpful

- Larger context under the same wall-clock budget was consistently bad:
  `2048` alone, `2048 + RoPE`, and `2048 + RoPE-NTK` all severely reduced step throughput and validation quality.
- Alternative positional encodings were all worse than sinusoidal on the current setup:
  `RoPE`, `RoPE-NTK`, `ALiBi`, and `learned`.
- `length_bucketing = true` was worse in this codepath under the 7200s comparison.
- Pushing width too far under fixed time also hurt:
  `d_model = 640` and `d_model = 768` were both worse than `512`.

## Discarded Directions

- `weight_decay = 1e-3`
- `d_model = 768, nhead = 12, dim_feedforward = 3072`
- `d_model = 640, nhead = 10, dim_feedforward = 2560`
- `RoPE`
- `RoPE-NTK`
- `ALiBi`
- `learned positional encoding`
- `dropout = 0.2`
- `length_bucketing = true`
- `learning_rate = 4e-4`
- `max_length = 2048`

## Notes

- The main comparison metric is always validation loss under the same `7200s` training budget.
- Longer-context variants were not compute-fair in practice even when wall-clock was fixed, because throughput collapsed too sharply.
- Tuning history should continue from the baseline above unless a new baseline is explicitly promoted.
