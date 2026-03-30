# Decoder Pretraining on `huggingface_full`

Last updated: 2026-03-27

## Recommended Full-Data Run

### Data

- dataset: `data/huggingface_full`
- tokenizer: `data/tokenizer_full.json`
- training split size: `279,931`
- validation split size: `69,573`
- test split size: `87,403`

### Input

- one dataset row produces one token sequence
- `max_length = 256`
- training: sliding-window chunks with `sliding_window_stride = 160`
- validation / test: deterministic sliding-window coverage with the same stride
- training batches do not use length bucketing in the recommended run
- batch padding: dynamic
- language-model targets:
  - `input_tokens = tokens[:, :-1]`
  - `output_tokens = tokens[:, 1:]`

### Model

- model type: decoder-only Transformer language model
- vocab size: `5000`
- `d_model = 256`
- `nhead = 4`
- `num_layers = 2`
- `dim_feedforward = 1024`
- `dropout = 0.0`
- `max_length = 256`
- positional encoding: sinusoidal

### Training

- initialization: from scratch
- objective: next-token prediction
- model-selection metric: validation cross-entropy loss
- `batch_size = 8`
- `learning_rate = 6e-4`
- scheduler: `linear`
- `warmup_steps = 500`
- `min_lr_ratio = 0.1`
- data loading: sliding window
- `sliding_window_stride = 160`
- `eval_every = 500`
- `early_stopping_patience = 20`
- `early_stopping_min_delta = 0.0`
- wall-clock cap: `3600` seconds
- actual stop condition: early stopping before the time cap

### Final Result

- best validation loss: `2.101936012506485`
- final step: `223000`
- elapsed seconds: `2293.68`
- best checkpoint: `artifacts/research/EXP-FULL-RDREF-003_sliding160_dmodel256_ff1024_lr6e4_linearwarmup_bs8_long/best.pt`
- latest checkpoint: `artifacts/research/EXP-FULL-RDREF-003_sliding160_dmodel256_ff1024_lr6e4_linearwarmup_bs8_long/latest.pt`

## Long-Context Backup Result

- alternative direction: `max_length = 1024` with random-crop training and no sliding window
- best tested long-context model:
  - `length_bucketing = false`
  - `d_model = 256`
  - `dim_feedforward = 1024`
  - `batch_size = 12`
  - `learning_rate = 6e-4`
  - `scheduler = linear`
  - `warmup_steps = 500`
  - `min_lr_ratio = 0.1`
  - best validation loss: `2.1344317700713873`
  - best checkpoint: `artifacts/research/EXP-FULL-RDREF-010_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs12_linearwarmup_long/best.pt`

- older long-context branch with `length_bucketing = true`:
  - best validation loss: `2.401298375800252`
  - best checkpoint: `artifacts/research/EXP-FULL-LONGCTX-007_crop1024_bucket_dmodel256_ff1024_bs8_long/best.pt`

Interpretation:

- the `rd`-style `1024 + no bucketing` recipe transfers better than the earlier `1024 + bucketing` branch
- but it still does not beat the current recommended `256 + sliding window` run
- a later optimizer sweep on this branch found:
  - `weight_decay = 1e-4` improved the `300s` smoke result (`3.3392` vs `3.4875`)
  - but the long-budget follow-up still finished worse than the branch best (`2.2827` vs `2.1344`)
  - longer warmup, gradient clipping, and label smoothing all degraded the branch

## Note

- this is the current recommended result on `huggingface_full`
- compared with the earlier recommended full run (`2.2531`), transferring the `rd`-validated training recipe and increasing `d_model` improved validation loss to `2.1019`
