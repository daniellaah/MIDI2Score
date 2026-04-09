# Decoder Pretraining Experiments

Last updated: 2026-04-08

## Usage

- Treat this file as the running experiment log for `rd`.
- Ignore all tuning history before the baseline entry below.
- Add new experiments as new top-level sections with the same format.
- Compare every new run against the current baseline unless explicitly replaced.

## Baseline: 3600s

### Status

- This is the only accepted baseline for future tuning.

### Recipe

- config: [`../configs/pretrain_rd_best.yaml`](../configs/pretrain_rd_best.yaml)
- `max_length = 1024`
- `sliding_window_stride = 256`
- `d_model = 256`
- `nhead = 4`
- `num_layers = 4`
- `dim_feedforward = 1024`
- positional encoding: `sinusoidal`
- residual layout: `pre_norm`
- norm type: `rmsnorm`
- activation: `swiglu`
- `tie_embeddings = false`
- `batch_size = 16`
- optimizer: `adamw`
- `learning_rate = 6e-4`
- scheduler: `linear`
- `warmup_steps = 500`

### Run

- run dir: [`../artifacts/runs/2026-04-08_20-49-04_250119`](../artifacts/runs/2026-04-08_20-49-04_250119)
- summary: [`../artifacts/runs/2026-04-08_20-49-04_250119/summary.json`](../artifacts/runs/2026-04-08_20-49-04_250119/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_20-49-04_250119/config.yaml`](../artifacts/runs/2026-04-08_20-49-04_250119/config.yaml)

### Metrics

- best validation loss: `1.8180`
- final step: `12000`
- elapsed seconds: `3616.72`
- device: `mps`
- average step time: `0.1099s`
- average tokens per second: `100749.7`

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

## Experiment: Pre-Norm

### Change

- Change only `residual_layout` from `post_norm` to `pre_norm`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_02-01-25_246490`](../artifacts/runs/2026-04-08_02-01-25_246490)
- summary: [`../artifacts/runs/2026-04-08_02-01-25_246490/summary.json`](../artifacts/runs/2026-04-08_02-01-25_246490/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_02-01-25_246490/config.yaml`](../artifacts/runs/2026-04-08_02-01-25_246490/config.yaml)

### Metrics

- best validation loss: `2.1940`
- final step: `4000`
- elapsed seconds: `629.80`
- average step time: `0.0105s`
- average tokens per second: `935699.5`

### Comparison To Baseline

- better
- Improves validation loss by about `0.0322` over the 600s baseline.

### Decision

- keep exploring
- Pre-Norm is a useful candidate and should be considered in combinations.

## Experiment: RMSNorm

### Change

- Change only `norm_type` from `layernorm` to `rmsnorm`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_02-11-59_535641`](../artifacts/runs/2026-04-08_02-11-59_535641)
- summary: [`../artifacts/runs/2026-04-08_02-11-59_535641/summary.json`](../artifacts/runs/2026-04-08_02-11-59_535641/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_02-11-59_535641/config.yaml`](../artifacts/runs/2026-04-08_02-11-59_535641/config.yaml)

### Metrics

- best validation loss: `2.1894`
- final step: `4000`
- elapsed seconds: `608.33`
- average step time: `0.0108s`
- average tokens per second: `913418.7`

### Comparison To Baseline

- better
- Improves validation loss by about `0.0368` over the 600s baseline.

### Decision

- keep exploring
- RMSNorm is one of the strongest single changes.

## Experiment: SwiGLU

### Change

- Change only `activation` from `relu` to `swiglu`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_02-22-12_364605`](../artifacts/runs/2026-04-08_02-22-12_364605)
- summary: [`../artifacts/runs/2026-04-08_02-22-12_364605/summary.json`](../artifacts/runs/2026-04-08_02-22-12_364605/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_02-22-12_364605/config.yaml`](../artifacts/runs/2026-04-08_02-22-12_364605/config.yaml)

### Metrics

- best validation loss: `2.1916`
- final step: `3729`
- elapsed seconds: `600.03`
- average step time: `0.0107s`
- average tokens per second: `915380.7`

### Comparison To Baseline

- better
- Improves validation loss by about `0.0346` over the 600s baseline.

### Decision

- keep exploring
- SwiGLU is a useful single change and worth combining with normalization changes.

## Experiment: GEGLU

### Change

- Change only `activation` from `relu` to `geglu`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_02-32-17_458295`](../artifacts/runs/2026-04-08_02-32-17_458295)
- summary: [`../artifacts/runs/2026-04-08_02-32-17_458295/summary.json`](../artifacts/runs/2026-04-08_02-32-17_458295/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_02-32-17_458295/config.yaml`](../artifacts/runs/2026-04-08_02-32-17_458295/config.yaml)

### Metrics

- best validation loss: `2.2028`
- final step: `3559`
- elapsed seconds: `600.06`
- average step time: `0.0112s`
- average tokens per second: `876133.0`

### Comparison To Baseline

- better
- Improves validation loss slightly, but less than Pre-Norm, RMSNorm, and SwiGLU.

### Decision

- reject
- GEGLU is dominated by SwiGLU in this setting.

## Experiment: AdamW

### Change

- Change only optimizer from `adam` to `adamw`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_02-42-22_596085`](../artifacts/runs/2026-04-08_02-42-22_596085)
- summary: [`../artifacts/runs/2026-04-08_02-42-22_596085/summary.json`](../artifacts/runs/2026-04-08_02-42-22_596085/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_02-42-22_596085/config.yaml`](../artifacts/runs/2026-04-08_02-42-22_596085/config.yaml)

### Metrics

- best validation loss: `2.2227`
- final step: `3673`
- elapsed seconds: `600.08`
- average step time: `0.0111s`
- average tokens per second: `884314.5`

### Comparison To Baseline

- inconclusive
- Very close to baseline, but not a meaningful improvement.

### Decision

- reject
- AdamW is not worth promoting over the earlier baseline recipe.

## Experiment: 4 Layers + Pre-Norm

### Change

- Change `num_layers` from `2` to `4`.
- Change `residual_layout` from `post_norm` to `pre_norm`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_02-52-27_577379`](../artifacts/runs/2026-04-08_02-52-27_577379)
- summary: [`../artifacts/runs/2026-04-08_02-52-27_577379/summary.json`](../artifacts/runs/2026-04-08_02-52-27_577379/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_02-52-27_577379/config.yaml`](../artifacts/runs/2026-04-08_02-52-27_577379/config.yaml)

### Metrics

- best validation loss: `2.5782`
- final step: `2355`
- elapsed seconds: `600.10`
- average step time: `0.0546s`
- average tokens per second: `180669.1`

### Comparison To Baseline

- worse
- Much slower and substantially worse under a fixed 600s budget.

### Decision

- reject
- 4 layers is not competitive under the current wall-clock budget.

## Experiment: Pre-Norm + RMSNorm

### Change

- Change `residual_layout` to `pre_norm`.
- Change `norm_type` to `rmsnorm`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_03-03-19_756125`](../artifacts/runs/2026-04-08_03-03-19_756125)
- summary: [`../artifacts/runs/2026-04-08_03-03-19_756125/summary.json`](../artifacts/runs/2026-04-08_03-03-19_756125/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_03-03-19_756125/config.yaml`](../artifacts/runs/2026-04-08_03-03-19_756125/config.yaml)

### Metrics

- best validation loss: `2.2322`
- final step: `3500`
- elapsed seconds: `610.87`
- average step time: `0.0115s`
- average tokens per second: `856555.8`

### Comparison To Baseline

- worse
- The two normalization changes do not combine well at 600s.

### Decision

- reject

## Experiment: Pre-Norm + SwiGLU

### Change

- Change `residual_layout` to `pre_norm`.
- Change `activation` to `swiglu`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_03-13-35_452041`](../artifacts/runs/2026-04-08_03-13-35_452041)
- summary: [`../artifacts/runs/2026-04-08_03-13-35_452041/summary.json`](../artifacts/runs/2026-04-08_03-13-35_452041/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_03-13-35_452041/config.yaml`](../artifacts/runs/2026-04-08_03-13-35_452041/config.yaml)

### Metrics

- best validation loss: `2.1890`
- final step: `3500`
- elapsed seconds: `631.19`
- average step time: `0.0096s`
- average tokens per second: `1019986.4`

### Comparison To Baseline

- better
- Improves validation loss by about `0.0372` and is the best pairwise combination among the tested pairs.

### Decision

- keep exploring
- This is a strong combination candidate.

## Experiment: RMSNorm + SwiGLU

### Change

- Change `norm_type` to `rmsnorm`.
- Change `activation` to `swiglu`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_03-24-11_644578`](../artifacts/runs/2026-04-08_03-24-11_644578)
- summary: [`../artifacts/runs/2026-04-08_03-24-11_644578/summary.json`](../artifacts/runs/2026-04-08_03-24-11_644578/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_03-24-11_644578/config.yaml`](../artifacts/runs/2026-04-08_03-24-11_644578/config.yaml)

### Metrics

- best validation loss: `2.2812`
- final step: `3363`
- elapsed seconds: `600.08`
- average step time: `0.0128s`
- average tokens per second: `767658.3`

### Comparison To Baseline

- worse
- This combination is clearly worse than both single changes.

### Decision

- reject

## Experiment: Pre-Norm + RMSNorm + SwiGLU

### Change

- Change `residual_layout` to `pre_norm`.
- Change `norm_type` to `rmsnorm`.
- Change `activation` to `swiglu`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_03-34-30_735649`](../artifacts/runs/2026-04-08_03-34-30_735649)
- summary: [`../artifacts/runs/2026-04-08_03-34-30_735649/summary.json`](../artifacts/runs/2026-04-08_03-34-30_735649/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_03-34-30_735649/config.yaml`](../artifacts/runs/2026-04-08_03-34-30_735649/config.yaml)

### Metrics

- best validation loss: `2.1756`
- final step: `3500`
- elapsed seconds: `625.45`
- average step time: `0.0129s`
- average tokens per second: `764033.6`

### Comparison To Baseline

- better
- Improves validation loss by about `0.0506`, which is the strongest result in this round of 600s experiments.

### Decision

- keep exploring
- This is the strongest candidate so far and should be the first one to confirm in a follow-up run.

## Experiment: AdamW On New Baseline

### Change

- Change only optimizer from `adam` to `adamw`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_04-12-42_579448`](../artifacts/runs/2026-04-08_04-12-42_579448)
- summary: [`../artifacts/runs/2026-04-08_04-12-42_579448/summary.json`](../artifacts/runs/2026-04-08_04-12-42_579448/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_04-12-42_579448/config.yaml`](../artifacts/runs/2026-04-08_04-12-42_579448/config.yaml)

### Metrics

- best validation loss: `2.1756`
- final step: `3539`
- elapsed seconds: `600.05`
- average step time: `0.0105s`
- average tokens per second: `935106.8`

### Comparison To Baseline

- inconclusive
- The result is effectively identical to the baseline.

### Decision

- reject
- Do not update the baseline.

### Note

- Even though the 600s result was effectively tied with the previous `adam` baseline, the project baseline has now been switched to `adamw` as the default optimizer choice.

## Experiment: Embedding / Output Weight Tying

### Change

- Change only `tie_embeddings` from `false` to `true`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_04-22-59_986013`](../artifacts/runs/2026-04-08_04-22-59_986013)
- summary: [`../artifacts/runs/2026-04-08_04-22-59_986013/summary.json`](../artifacts/runs/2026-04-08_04-22-59_986013/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_04-22-59_986013/config.yaml`](../artifacts/runs/2026-04-08_04-22-59_986013/config.yaml)

### Metrics

- best validation loss: `2.2159`
- final step: `3526`
- elapsed seconds: `600.01`
- average step time: `0.0100s`
- average tokens per second: `979298.3`

### Comparison To Baseline

- worse
- Weight tying improves throughput but hurts validation loss.

### Decision

- reject

## Experiment: Gradient Clipping Sweep

### Change

- Keep the new baseline fixed.
- Sweep `grad_clip_norm` over `0.5`, `1.0`, `2.0`.

### Runs

- `0.5`:
  - run dir: [`../artifacts/runs/2026-04-08_05-03-43_222505`](../artifacts/runs/2026-04-08_05-03-43_222505)
  - summary: [`../artifacts/runs/2026-04-08_05-03-43_222505/summary.json`](../artifacts/runs/2026-04-08_05-03-43_222505/summary.json)
  - best validation loss: `2.1730`
- `1.0`:
  - run dir: [`../artifacts/runs/2026-04-08_05-13-47_715927`](../artifacts/runs/2026-04-08_05-13-47_715927)
  - summary: [`../artifacts/runs/2026-04-08_05-13-47_715927/summary.json`](../artifacts/runs/2026-04-08_05-13-47_715927/summary.json)
  - best validation loss: `2.1736`
- `2.0`:
  - run dir: [`../artifacts/runs/2026-04-08_05-23-52_207348`](../artifacts/runs/2026-04-08_05-23-52_207348)
  - summary: [`../artifacts/runs/2026-04-08_05-23-52_207348/summary.json`](../artifacts/runs/2026-04-08_05-23-52_207348/summary.json)
  - best validation loss: `2.1757`

### Comparison To Baseline

- all three are effectively tied with the baseline
- `0.5` is numerically best, but the gain is very small

### Decision

- inconclusive
- Gradient clipping may help slightly, but the effect is too small to promote on a single 600s run.

## Experiment: Label Smoothing Sweep

### Change

- Keep the new baseline fixed.
- Sweep `label_smoothing` over `0.05`, `0.10`.

### Runs

- `0.05`:
  - run dir: [`../artifacts/runs/2026-04-08_05-33-56_654196`](../artifacts/runs/2026-04-08_05-33-56_654196)
  - summary: [`../artifacts/runs/2026-04-08_05-33-56_654196/summary.json`](../artifacts/runs/2026-04-08_05-33-56_654196/summary.json)
  - best validation loss: `2.2210`
- `0.10`:
  - run dir: [`../artifacts/runs/2026-04-08_05-44-11_677744`](../artifacts/runs/2026-04-08_05-44-11_677744)
  - summary: [`../artifacts/runs/2026-04-08_05-44-11_677744/summary.json`](../artifacts/runs/2026-04-08_05-44-11_677744/summary.json)
  - best validation loss: `2.2779`

### Comparison To Baseline

- worse
- Label smoothing hurts this decoder pretraining setup.

### Decision

- reject

## Experiment: Sliding-Window Stride Sweep

### Change

- Keep the new baseline fixed.
- Sweep `sliding_window_stride` over `256`, `768`, `1024`.

### Runs

- `256`:
  - run dir: [`../artifacts/runs/2026-04-08_05-54-16_367544`](../artifacts/runs/2026-04-08_05-54-16_367544)
  - summary: [`../artifacts/runs/2026-04-08_05-54-16_367544/summary.json`](../artifacts/runs/2026-04-08_05-54-16_367544/summary.json)
  - best validation loss: `2.1690`
- `768`:
  - run dir: [`../artifacts/runs/2026-04-08_06-04-31_475722`](../artifacts/runs/2026-04-08_06-04-31_475722)
  - summary: [`../artifacts/runs/2026-04-08_06-04-31_475722/summary.json`](../artifacts/runs/2026-04-08_06-04-31_475722/summary.json)
  - best validation loss: `2.1924`
- `1024`:
  - run dir: [`../artifacts/runs/2026-04-08_06-14-36_206196`](../artifacts/runs/2026-04-08_06-14-36_206196)
  - summary: [`../artifacts/runs/2026-04-08_06-14-36_206196/summary.json`](../artifacts/runs/2026-04-08_06-14-36_206196/summary.json)
  - best validation loss: `2.2292`

### Comparison To Baseline

- `256` is better
- `768` is slightly worse
- `1024` is clearly worse

### Decision

- promote to new baseline
- `sliding_window_stride = 256` is now the accepted tuning baseline.

### Follow-Up On AdamW Baseline

- confirm run: [`../artifacts/runs/2026-04-08_14-39-35_064237`](../artifacts/runs/2026-04-08_14-39-35_064237)
- summary: [`../artifacts/runs/2026-04-08_14-39-35_064237/summary.json`](../artifacts/runs/2026-04-08_14-39-35_064237/summary.json)
- best validation loss: `2.1690`

Updated conclusion:

- `sliding_window_stride = 256` still improves over the `adamw` baseline.
- this remains the accepted baseline change.

## Experiment: 4 Layers On Current Baseline

### Change

- Change only `num_layers` from `2` to `4`.

### Run

- run dir: [`../artifacts/runs/2026-04-08_15-42-05_557631`](../artifacts/runs/2026-04-08_15-42-05_557631)
- summary: [`../artifacts/runs/2026-04-08_15-42-05_557631/summary.json`](../artifacts/runs/2026-04-08_15-42-05_557631/summary.json)
- config snapshot: [`../artifacts/runs/2026-04-08_15-42-05_557631/config.yaml`](../artifacts/runs/2026-04-08_15-42-05_557631/config.yaml)

### Metrics

- best validation loss: `2.3808`
- final step: `2142`
- elapsed seconds: `600.13`
- average step time: `0.1011s`
- average tokens per second: `109930.0`

### Comparison To Baseline

- much worse
- The deeper model reduces throughput by about an order of magnitude under the same wall-clock budget.

### Decision

- reject
- 4 layers is not competitive at 600s on the current hardware and recipe.
