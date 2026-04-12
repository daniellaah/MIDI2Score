# Decoder Pretraining Experiments

Last updated: 2026-04-11

## Usage

- This file tracks the current accepted `rd` baseline and the latest high-signal experiment outcomes.
- Use [`../exp/apr9.tsv`](../exp/apr9.tsv) and [`../exp/apr9-7200.tsv`](../exp/apr9-7200.tsv) for the full Apr 9 result tables.
- Treat the top baseline section below as the current comparison point for future tuning.

## Baseline: 7200s

### Status

- This is the current accepted baseline on `main`.

### Recipe

- config: [`../configs/pretrain_rd_best.yaml`](../configs/pretrain_rd_best.yaml)
- `max_length = 1024`
- `sliding_window_stride = 256`
- `d_model = 512`
- `nhead = 8`
- `num_layers = 4`
- `dim_feedforward = 2048`
- `dropout = 0.05`
- positional encoding: `sinusoidal`
- residual layout: `pre_norm`
- norm type: `rmsnorm`
- activation: `swiglu`
- `tie_embeddings = true`
- `batch_size = 16`
- optimizer: `adamw`
- `learning_rate = 6e-4`
- `weight_decay = 0.1`
- `beta1 = 0.9`
- `beta2 = 0.95`
- `grad_clip_norm = 2.0`
- scheduler: `cosine`
- `warmup_steps = 500`

### Promotion Path

- baseline 7200s rerun: `1.7708`
- `weight_decay = 1e-2`: `1.7670`
- `d_model = 512, nhead = 8, dim_feedforward = 2048`: `1.7520`
- `weight_decay = 0.1`: `1.7248`
- `AdamW beta2 = 0.95`: `1.7197`
- `dropout = 0.1`: `1.6922`
- `dropout = 0.05`: `1.6834`
- `grad_clip_norm = 1.0`: `1.6831`
- `grad_clip_norm = 2.0`: `1.6735`

### Run

- run dir: [`../artifacts/runs/2026-04-10_22-28-37_632453`](../artifacts/runs/2026-04-10_22-28-37_632453)
- summary: [`../artifacts/runs/2026-04-10_22-28-37_632453/summary.json`](../artifacts/runs/2026-04-10_22-28-37_632453/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-10_22-28-37_632453/config.yaml`](../artifacts/runs/2026-04-10_22-28-37_632453/config.yaml)

### Metrics

- best validation loss: `1.6735`
- final step: `10500`
- elapsed seconds: `7301.52`
- device: `mps`
- average tokens per second: `20383.4`

## Apr 9 7200s Autoresearch Summary

### Kept

- `weight_decay = 1e-2`
  - best validation loss: `1.7670`

- `d_model = 512, nhead = 8, dim_feedforward = 2048`
  - best validation loss: `1.7520`

- `weight_decay = 0.1`
  - best validation loss: `1.7248`

- `AdamW beta2 = 0.95`
  - best validation loss: `1.7197`

- `dropout = 0.1`
  - best validation loss: `1.6922`

- `dropout = 0.05`
  - best validation loss: `1.6834`

- `grad_clip_norm = 1.0`
  - best validation loss: `1.6831`

- `grad_clip_norm = 2.0`
  - best validation loss: `1.6735`

### Discarded

- `weight_decay = 1e-3`
  - best validation loss: `1.7712`

- `d_model = 768, nhead = 12, dim_feedforward = 3072`
  - best validation loss: `1.7905`

- `RoPE`
  - best validation loss: `1.7909`

- `d_model = 640, nhead = 10, dim_feedforward = 2560`
  - best validation loss: `1.6983`

- `learning_rate = 4e-4`
  - best validation loss: `1.6769`

- `dropout = 0.2`
  - best validation loss: `1.7820`

- `length_bucketing = true`
  - best validation loss: `1.8360`

- `learned positional encoding`
  - best validation loss: `1.7803`

- `ALiBi`
  - best validation loss: `2.2922`

- `max_length = 2048`
  - best validation loss: `3.6509`

- `max_length = 2048` + `RoPE`
  - best validation loss: `2.8790`

- `max_length = 2048` + `RoPE-NTK` + `stride = 512` + `batch_size = 4`
  - best validation loss: `2.1639`

## Notes

- The full per-experiment commit/loss table for this round is in [`../exp/apr9-7200.tsv`](../exp/apr9-7200.tsv).
- Longer-context variants were not compute-fair in practice under the fixed `7200s` wall-clock budget because throughput collapsed too sharply.

## Baseline: 3600s

### Status

- This was the accepted baseline before the 7200s autoresearch round.
- It is preserved for historical reference only.

### Recipe

- config: [`../configs/pretrain_rd_best.yaml`](../configs/pretrain_rd_best.yaml)
- `max_length = 1024`
- `sliding_window_stride = 256`
- `d_model = 384`
- `nhead = 6`
- `num_layers = 4`
- `dim_feedforward = 1536`
- positional encoding: `sinusoidal`
- residual layout: `pre_norm`
- norm type: `rmsnorm`
- activation: `swiglu`
- `tie_embeddings = true`
- `batch_size = 16`
- optimizer: `adamw`
- `weight_decay = 1e-4`
- scheduler: `cosine`
- `warmup_steps = 500`

### Promotion Path

- `4 layers` became the initial 3600s baseline on 2026-04-08.
- `weight tying` improved the 3600s baseline.
- `weight_decay = 1e-4` improved again.
- `d_model = 384, nhead = 6, dim_feedforward = 1536` improved again.
- `cosine` scheduler improved again and became the 3600s baseline.

### Run

- run dir: [`../artifacts/runs/2026-04-09_09-24-23_512432`](../artifacts/runs/2026-04-09_09-24-23_512432)
- summary: [`../artifacts/runs/2026-04-09_09-24-23_512432/summary.json`](../artifacts/runs/2026-04-09_09-24-23_512432/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-09_09-24-23_512432/config.yaml`](../artifacts/runs/2026-04-09_09-24-23_512432/config.yaml)

### Metrics

- best validation loss: `1.7823`
- final step: `8500`
- elapsed seconds: `3678.08`
- device: `mps`
- average step time: `0.1796s`
- average tokens per second: `61667.3`

## Apr 9 3600s Autoresearch Summary

### Kept

- `weight tying`
  - run: [`../artifacts/runs/2026-04-09_02-15-02_062221`](../artifacts/runs/2026-04-09_02-15-02_062221)
  - best validation loss: `1.7913`
  - decision: promoted at the time

- `weight_decay = 1e-4`
  - run: [`../artifacts/runs/2026-04-09_03-15-56_163129`](../artifacts/runs/2026-04-09_03-15-56_163129)
  - best validation loss: `1.7838`
  - decision: promoted at the time

- `d_model = 384, nhead = 6, dim_feedforward = 1536`
  - run: [`../artifacts/runs/2026-04-09_05-19-36_477486`](../artifacts/runs/2026-04-09_05-19-36_477486)
  - best validation loss: `1.7831`
  - decision: promoted at the time

- `cosine` scheduler
  - run: [`../artifacts/runs/2026-04-09_09-24-23_512432`](../artifacts/runs/2026-04-09_09-24-23_512432)
  - best validation loss: `1.7823`
  - decision: current best at the time

### Discarded

- `label_smoothing = 0.05`
  - best validation loss: `1.8361`

- `label_smoothing = 0.10`
  - best validation loss: `1.9030`

- `weight_decay = 1e-3`
  - best validation loss: `1.7848`

- `d_model = 512, nhead = 8, dim_feedforward = 2048`
  - best validation loss: `1.7863`

- `d_model = 768, nhead = 12, dim_feedforward = 3072`
  - best validation loss: `1.9432`

- `RoPE`
  - best validation loss: `1.8156`

- `grad_clip_norm = 0.5`
  - best validation loss: `1.7858`

- `grad_clip_norm = 1.0`
  - best validation loss: `1.7927`

- `grad_clip_norm = 2.0`
  - best validation loss: `1.8009`

## Historical 600s Baseline

### Status

- This section preserves the earlier 600s tuning baseline and its surrounding experiments for reference only.
- It is not the current comparison point.

### Recipe

- config at the time: `256 / 4 / 2 layers / ff 1024`
- `max_length = 1024`
- `sliding_window_stride = 256`
- positional encoding: `sinusoidal`
- residual layout: `pre_norm`
- norm type: `rmsnorm`
- activation: `swiglu`
- `tie_embeddings = false`
- optimizer: `adamw`
- scheduler: `linear`

### Run

- run dir: [`../artifacts/runs/2026-04-08_14-39-35_064237`](../artifacts/runs/2026-04-08_14-39-35_064237)
- summary: [`../artifacts/runs/2026-04-08_14-39-35_064237/summary.json`](../artifacts/runs/2026-04-08_14-39-35_064237/summary.json)

### Metrics

- best validation loss: `2.1690`
- final step: `3500`
- elapsed seconds: `623.78`
- average step time: `0.0121s`
- average tokens per second: `917671.5`

## Historical 600s Experiments

### Kept Or Later Absorbed

- `Pre-Norm`
  - run: [`../artifacts/runs/2026-04-08_02-01-25_246490`](../artifacts/runs/2026-04-08_02-01-25_246490)
  - best validation loss: `2.1940`

- `RMSNorm`
  - run: [`../artifacts/runs/2026-04-08_02-11-59_535641`](../artifacts/runs/2026-04-08_02-11-59_535641)
  - best validation loss: `2.1894`

- `SwiGLU`
  - run: [`../artifacts/runs/2026-04-08_02-22-12_364605`](../artifacts/runs/2026-04-08_02-22-12_364605)
  - best validation loss: `2.1916`

### Rejected

- `GEGLU`
  - run: [`../artifacts/runs/2026-04-08_02-32-17_458295`](../artifacts/runs/2026-04-08_02-32-17_458295)
  - best validation loss: `2.2028`

- `AdamW` under the earlier 600s baseline
  - run: [`../artifacts/runs/2026-04-08_02-42-22_596085`](../artifacts/runs/2026-04-08_02-42-22_596085)
  - best validation loss: `2.2227`

- `4 Layers + Pre-Norm` under 600s
  - run: [`../artifacts/runs/2026-04-08_02-52-27_577379`](../artifacts/runs/2026-04-08_02-52-27_577379)
  - best validation loss: `2.5782`
