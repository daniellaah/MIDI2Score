# Decoder Pretraining on `huggingface_full`

Last updated: 2026-03-25

## Recommended Full-Data Run

### Data

- dataset: `data/huggingface_full`
- tokenizer: `data/tokenizer_full.json`
- training split size: `279,931`
- validation split size: `69,573`
- test split size: `87,403`

### Model

- model type: decoder-only Transformer language model
- vocab size: `5000`
- `d_model = 128`
- `nhead = 4`
- `num_layers = 2`
- `dim_feedforward = 1024`
- `dropout = 0.0`
- `max_length = 256`

### Training

- initialization: from scratch
- objective: next-token prediction
- model-selection metric: validation cross-entropy loss
- `batch_size = 8`
- `learning_rate = 8e-4`
- data loading: sliding window
- `sliding_window_stride = 160`
- `eval_every = 500`
- `early_stopping_patience = 20`
- `early_stopping_min_delta = 0.0`
- wall-clock cap: `3600` seconds
- actual stop condition: early stopping before the time cap

### Final Result

- best validation loss: `2.253071304410696`
- final step: `211000`
- elapsed seconds: `2346.28`
- best checkpoint: `artifacts/research/EXP-FULL-FINAL-007_stride160_dmodel128_ff1024_dropout0_lr8e4_bs8_es20/best.pt`
- latest checkpoint: `artifacts/research/EXP-FULL-FINAL-007_stride160_dmodel128_ff1024_dropout0_lr8e4_bs8_es20/latest.pt`

## Long-Context Backup Result

- alternative direction: `max_length = 1024` with random-crop training and `length_bucketing = true`
- best tested long-context model:
  - `d_model = 256`
  - `dim_feedforward = 1024`
  - `batch_size = 8`
  - best validation loss: `2.401298375800252`
  - best checkpoint: `artifacts/research/EXP-FULL-LONGCTX-007_crop1024_bucket_dmodel256_ff1024_bs8_long/best.pt`

Interpretation:

- `1024 + bucketing` is trainable and improves substantially over the same setup without bucketing
- but it still does not beat the current recommended `256 + sliding window` run

## Note

- this is the current recommended result on `huggingface_full`
- compared with the random-crop full baseline (`2.4494`), sliding window plus targeted tuning improved validation loss to `2.2531`
