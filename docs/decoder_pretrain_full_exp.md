# Decoder Pretraining Experiments (`full`)

This file keeps only the high-value experiment history for the full-data decoder pretraining branch.

Detailed superseded trial outputs are archived under `archives/`.

## Current Official Best

- config: `configs/pretrain_full_best.yaml`
- dataset: `data/huggingface_full`
- checkpoint:
  - latest: `artifacts/pretrained_decoder_full_best.pt`
  - best: `artifacts/pretrained_decoder_full_best_best.pt`
- recipe:
  - `max_length = 256`
  - training: sliding-window chunks
  - `sliding_window_stride = 160`
  - validation/test: deterministic sliding-window coverage
  - `d_model = 256`
  - `nhead = 4`
  - `num_layers = 2`
  - `dim_feedforward = 1024`
  - `dropout = 0.0`
  - `batch_size = 8`
  - `learning_rate = 6e-4`
  - scheduler: `linear`
  - `warmup_steps = 500`
  - length bucketing: `false`
- full-validation metrics on the saved best checkpoint:
  - CE loss: `2.1232`
  - perplexity: `8.3580`
  - token accuracy: `0.5258`
  - top-5 accuracy: `0.7480`
- source experiment:
  - `EXP-FULL-RDREF-003_sliding160_dmodel256_ff1024_lr6e4_linearwarmup_bs8_long`

## Validation Policy

Use these rules for all reported final numbers:

- validation loss is **token-weighted**, not batch-averaged
- long validation sequences use **sliding-window evaluation**
- overlapping validation windows use a **loss mask**
- each target token is counted **exactly once**
- final reported metrics must use the **full validation split**

## Stable Lessons

### What Clearly Helps

- short-context sliding window (`256`, stride `160`) beats the tested `1024` long-context alternatives
- `d_model = 256`, `dim_feedforward = 1024`, and `dropout = 0.0` are strong defaults
- `learning_rate = 6e-4` with linear warmup/decay is the strongest stable optimizer recipe found so far

### What Clearly Does Not Help Right Now

- length bucketing on the current official branch
- `1024 + bucketing`
- optimizer-side tweaks on the tested `1024` backup branch:
  - weight decay
  - longer warmup
  - gradient clipping
  - label smoothing
- deeper 3-layer variants on the tested branches

## Strongest Backup Direction

### `1024` Long-Context Backup

This branch is the strongest tested long-context alternative on `full`, but it is still worse than the official `256 + sliding` recipe.

- branch:
  - `max_length = 1024`
  - no sliding-window training
  - no length bucketing
  - `d_model = 256`
  - `dim_feedforward = 1024`
  - `batch_size = 12`
  - `learning_rate = 6e-4`
  - scheduler: `linear`
  - `warmup_steps = 500`
- best validation loss:
  - `2.1344`
- source experiment:
  - `archives/artifacts/research/EXP-FULL-RDREF-010_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs12_linearwarmup_long/best.pt`

Interpretation:

- the `rd`-style `1024 + no bucketing` recipe transfers better than the earlier `1024 + bucketing` branch
- but it still does not beat the official `full` best

## Selected Milestones

1. Early short-context baseline:
   - `256 + sliding window` clearly beat early `1024` attempts

2. Transfer of the stronger `rd` recipe improved the official `full` run.
   - key outcome:
     - `d_model = 256`
     - `dim_feedforward = 1024`
     - `learning_rate = 6e-4`
     - linear warmup/decay
   - this reduced full-validation CE to `2.1232`

## Avoid Repeating These Dead Ends

- `1024 + bucketing`
- optimizer-side tweaks on the tested `1024` backup branch:
  - weight decay
  - longer warmup
  - gradient clipping
  - label smoothing
- deeper 3-layer variants

## Future Update Format

When adding a new block, use exactly this format:

### `<experiment family or decision>`

- setup:
  - only list the knobs that matter
- reference:
  - what current baseline this should be compared against
- result:
  - CE loss
  - optionally perplexity / accuracy if this is a promotion candidate
- interpretation:
  - one short statement explaining whether the result changes the current recommendation
- decision:
  - `promote`
  - `keep as backup`
  - `do not continue`

## Maintenance Rule

If a new experiment does **not** change:
- the official best
- the strongest backup branch
- or an important negative lesson

then do **not** add it here.
