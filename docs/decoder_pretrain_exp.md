# Decoder Pretraining Experiments

Last updated: 2026-04-08

## Usage

- Treat this file as the running experiment log for `rd`.
- Ignore all tuning history before the baseline entry below.
- Add new experiments as new top-level sections with the same format.
- Compare every new run against the current baseline unless explicitly replaced.

## Baseline: 600s

### Status

- This is the only accepted baseline for future tuning.

### Recipe

- config: [`../configs/pretrain_rd_best.yaml`](../configs/pretrain_rd_best.yaml)
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

### Run

- run dir: [`../artifacts/runs/2026-04-08_01-05-45_750559`](../artifacts/runs/2026-04-08_01-05-45_750559)
- summary: [`../artifacts/runs/2026-04-08_01-05-45_750559/summary.json`](../artifacts/runs/2026-04-08_01-05-45_750559/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_01-05-45_750559/config.yaml`](../artifacts/runs/2026-04-08_01-05-45_750559/config.yaml)

### Metrics

- best validation loss: `2.2262`
- final step: `3781`
- elapsed seconds: `600.06`
- device: `mps`
- average step time: `0.0115s`
- average tokens per second: `856341.3`

### Decision

- Use this run as the comparison point for all future tuning work.

## Template For New Experiments

Copy this structure for each new run:

```md
## Experiment: <name>

### Change

- Describe only what changed relative to the current baseline.

### Run

- run dir: `...`
- summary: `...`
- config snapshot: `...`

### Metrics

- best validation loss: `...`
- final step: `...`
- elapsed seconds: `...`
- average step time: `...`
- average tokens per second: `...`

### Comparison To Baseline

- better / worse / inconclusive
- short explanation

### Decision

- keep exploring / reject / promote to new baseline
```
