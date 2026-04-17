# Decoder Pretraining

Last updated: 2026-04-14

## Current Best Baseline

Current best `7200s` baseline run:
- run: `artifacts/runs/2026-04-14_07-19-10_429605`
- summary: `artifacts/runs/2026-04-14_07-19-10_429605/summary.json`
- config: `artifacts/runs/2026-04-14_07-19-10_429605/config.yaml`
- `best validation loss = 1.636279`
- `average tokens/sec = 42033.8`

## Model

- model family: decoder-only Transformer LM
- objective: next-token prediction over linearized MusicXML token ids
- `vocab_size = 5000`
- `d_model = 512`
- `nhead = 8`
- `num_layers = 4`
- `dim_feedforward = 2048`
- `dropout = 0.05`
- activation: `swiglu`
- normalization: `rmsnorm`
- residual layout: `pre_norm`
- positional encoding: `sinusoidal`
- embedding/output weight tying: enabled
- `max_length = 1024`

## Training

- optimizer: `AdamW`
- `learning_rate = 6e-4`
- `weight_decay = 0.1`
- `beta1 = 0.9`
- `beta2 = 0.95`
- gradient clipping: `2.0`
- scheduler: cosine decay
- `warmup_steps = 500`
- `min_lr_ratio = 0.1`
- `batch_size = 64`
- wall-clock budget: `7200s`

## Data

- dataset: `data/huggingface`
- split source: Hugging Face `DatasetDict`
- training windows:
  - `max_length = 1024`
  - `sliding_window_stride = 256`
- training batching:
  - length-bucketed dynamic batching
  - `max_tokens_per_batch = 16384`
  - `pad_to_length_multiple = 64`
  - `bucket_padding_noise = 0.0`
- training worker count: `num_workers = 0`
- `PAD_TOKEN_ID = 0`
- `BOS_TOKEN_ID = 1`
- `EOS_TOKEN_ID = 2`

## Validation

Validation is fixed across experiments:

- `max_length = 1024`
- `sliding_window_stride = 256`
- fixed-order validation
- fixed-batch validation
- `eval_batch_size = 16`
- token-weighted average loss over scored tokens

## Files

- training entry: `run_pretrain.py`
- model: `midi2score/model.py`
- training loop: `midi2score/train.py`
- data pipeline: `midi2score/data.py`
- experiment history: `docs/decoder_pretrain_exp.md`
- batching notes: `docs/data_preprocessing.md`

## References

- Transformer decoder: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- RMSNorm: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- SwiGLU / GLU variants: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Weight tying: [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- AdamW: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Sliding-window evaluation background: [Hugging Face perplexity guide](https://huggingface.co/docs/transformers/en/perplexity)
