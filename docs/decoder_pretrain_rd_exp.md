# Decoder Pretraining Experiments (`rd`)

This file keeps only the high-value experiment history for the reduced (`rd`) decoder pretraining branch.

Detailed superseded trial outputs are archived under `archives/`.

## Current Official Best

- config: `configs/pretrain_rd_best.yaml`
- dataset: `data/huggingface`
- checkpoint:
  - latest: `artifacts/pretrained_decoder_rd_best.pt`
  - best: `artifacts/pretrained_decoder_rd_best_best.pt`
- recipe:
  - `max_length = 1024`
  - training: full-coverage sliding windows
  - `sliding_window_stride = 512`
  - validation/test: full-coverage sliding-window evaluation
  - `d_model = 256`
  - `nhead = 4`
  - `num_layers = 2`
  - `dim_feedforward = 1024`
  - `dropout = 0.0`
  - `batch_size = 16`
  - `learning_rate = 6e-4`
  - scheduler: `linear`
  - `warmup_steps = 500`
  - length bucketing: `false`
- full-validation metrics on the saved best checkpoint:
  - CE loss: `1.8092`
  - perplexity: `6.1053`
  - token accuracy: `0.5988`
  - top-5 accuracy: `0.7908`
  - evaluated tokens: `4,416,152`
- source experiment:
  - `EXP-RD-SLIDING-LONG-003_bs16_confirm`

## Validation Policy

Use these rules for all reported final numbers:

- validation loss is **token-weighted**, not batch-averaged
- long validation sequences use **sliding-window evaluation**
- overlapping validation windows use a **loss mask**
- each target token is counted **exactly once**
- final reported metrics must use the **full validation split**

## Stable Lessons

### What Clearly Helps

- `1024` context is better than shorter random-crop baselines.
- Full-coverage sliding windows are better than using only a single random crop per long sample.
- `d_model = 256`, `dim_feedforward = 1024`, and `dropout = 0.0` are strong defaults.
- `learning_rate = 6e-4` with linear warmup/decay is the strongest stable optimizer recipe found so far.
- `batch_size = 16` is better than `batch_size = 8` on the official sliding-window branch.

### What Clearly Does Not Help Right Now

- length bucketing on the current official branch
- no-truncation + token-budget batching
- `2048` context under the current model scale
- deeper 3-layer variants
- dropout on the current best branch
- learned absolute positional encoding
- ALiBi
- flash attention backend on the tested setup
- RoPE has not yet beaten the official long-run sinusoidal best on the `1024` branch

## Selected Milestones

1. Width and feed-forward scaling established the first strong baseline.
   - key outcome: `d_model = 256` and `dim_feedforward = 1024` are worth keeping

2. Optimizer cleanup improved stability.
   - key outcome: `learning_rate = 6e-4` with linear warmup/decay beat the earlier stronger-noise settings

3. Validation was corrected from batch-average / prefix-only style toward full-coverage token-weighted evaluation.
   - key outcome: official metrics now reflect full validation coverage instead of truncated prefix-only reporting

4. Training moved from random-crop-only to full-coverage sliding-window training.
   - key experiment:
     - `EXP-RD-SLIDING-LONG-001`
   - key outcome:
     - sliding-window coverage is the right data pipeline for `rd`

5. `batch_size = 16` was promoted on the sliding-window branch.
   - reference `300s` baseline:
     - `EXP-RD-SLIDING-BASELINE-300-013`
     - CE `3.1823`
   - stronger `300s` candidate:
     - `EXP-RD-SLIDING-BASELINE-300-016`
     - CE `3.0432`
   - long-budget confirmation:
     - `EXP-RD-SLIDING-LONG-002_bs16`
     - CE `1.8102`
   - second confirmation:
     - `EXP-RD-SLIDING-LONG-003_bs16_confirm`
     - CE `1.8092`
   - key outcome:
     - `batch_size = 16` is now the official `rd` best setting

6. RoPE was re-checked on the current official `1024` baseline.
   - strict `600s` positional-only comparison:
     - `sinusoidal`
       - `EXP-RD-POSSTRICT600-SINUSOIDAL-001`
       - CE `2.2322`
     - `rope`
       - `EXP-RD-POSSTRICT600-ROPE-001`
       - CE `2.3511`
   - long-budget follow-up:
     - `EXP-RD-POSSTRICT7200-ROPE-001`
     - CE `1.8421`
   - interpretation:
     - RoPE is competitive and more memory-efficient than sinusoidal on the tested setup, but it still has not surpassed the current official long-run sinusoidal best (`1.8092`)
   - decision:
     - keep as a backup positional variant, do not promote

## Avoid Repeating These Dead Ends

- `2048` context under the current model scale
- no-truncation with `max_tokens_per_batch`
- length bucketing on the official branch
- learned positional encoding
- ALiBi
- flash attention backend on current hardware and implementation
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
- or an important negative lesson

then do **not** add it here.
