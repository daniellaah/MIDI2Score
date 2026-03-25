# Decoder Pretraining on `huggingface_full`

Last updated: 2026-03-24

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
- `dim_feedforward = 512`
- `dropout = 0.0`
- `max_length = 256`

### Training

- initialization: from scratch
- objective: next-token prediction
- model-selection metric: validation cross-entropy loss
- `batch_size = 8`
- `learning_rate = 8e-4`
- data loading: sliding window
- `sliding_window_stride = 128`
- `eval_every = 500`
- `early_stopping_patience = 20`
- `early_stopping_min_delta = 0.0`
- wall-clock cap: `3600` seconds
- actual stop condition: early stopping before the time cap

### Final Result

- best validation loss: `2.3425204092636704`
- final step: `104000`
- elapsed seconds: `956.39`
- best checkpoint: `artifacts/research/EXP-FULL-FINAL-002_sliding256_stride128_dmodel128_ff512_dropout0_lr8e4_bs8_es20/best.pt`
- latest checkpoint: `artifacts/research/EXP-FULL-FINAL-002_sliding256_stride128_dmodel128_ff512_dropout0_lr8e4_bs8_es20/latest.pt`

## Note

- this is the current recommended result on `huggingface_full`
- compared with the random-crop full baseline (`2.4494`), sliding window improved validation loss to `2.3425`
