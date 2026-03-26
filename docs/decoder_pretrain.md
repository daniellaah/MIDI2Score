# Decoder Pretraining

Last updated: 2026-03-25

## Final Recommended Model

### Data

- dataset: `data/huggingface`
- tokenizer: `data/tokenizer_rd.json`
- training split size: `29,514`
- validation split size: `7,308`
- test split size: `9,086`

### Input

- one dataset row produces one token sequence
- `max_length = 1024`
- training: random contiguous crop for overlength samples
- validation / test: deterministic prefix crop for overlength samples
- training batches use length bucketing
- batch padding: dynamic
- language-model targets:
  - `input_tokens = tokens[:, :-1]`
  - `output_tokens = tokens[:, 1:]`

### Decoder

- model type: decoder-only Transformer language model
- vocab size: `5000`
- `d_model = 256`
- `nhead = 4`
- `num_layers = 2`
- `dim_feedforward = 1024`
- `dropout = 0.0`
- positional encoding: sinusoidal

### Training

- objective: next-token prediction
- model-selection metric: validation cross-entropy loss
- `batch_size = 8`
- `learning_rate = 8e-4`
- initialization: from scratch
- run cap: `3600` seconds
- validation cadence: `eval_every = 500`
- early stopping:
  - `early_stopping_patience = 20`
  - `early_stopping_min_delta = 0.0`
- safety cap: `num_steps = 1000000`

### Final Result

- best validation loss: `1.9615836385637522`
- best checkpoint: `artifacts/research/EXP-RD-LONGCTX-004_crop1024_bucket_dmodel256_ff1024_bs8_fullbudget/best.pt`
- latest checkpoint: `artifacts/research/EXP-RD-LONGCTX-004_crop1024_bucket_dmodel256_ff1024_bs8_fullbudget/latest.pt`
- actual stop condition: early stopping before the `3600`-second cap
- note: this replaced the previous short-context recommendation (`2.0914`)

## Loss Curves

![Decoder pretraining loss curves](decoder_pretrain_losses.svg)
