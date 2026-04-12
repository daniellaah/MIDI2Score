# Decoder Pretraining

Last updated: 2026-04-11

## Overview

This document records the stable decoder pretraining recipe for `rd`.
Use [`decoder_pretrain_exp.md`](decoder_pretrain_exp.md) for tuning history and keep this file focused on the accepted method.

## Current Baseline

- config: [`../configs/pretrain_rd_best.yaml`](../configs/pretrain_rd_best.yaml)
- model: decoder-only Transformer LM
- objective: next-token prediction over linearized MusicXML token ids
- accepted baseline run: [`../artifacts/runs/2026-04-10_22-28-37_632453`](../artifacts/runs/2026-04-10_22-28-37_632453)
- accepted baseline summary: [`../artifacts/runs/2026-04-10_22-28-37_632453/summary.json`](../artifacts/runs/2026-04-10_22-28-37_632453/summary.json)

### Model

- `d_model = 512`
- `nhead = 8`
- `num_layers = 4`
- `dim_feedforward = 2048`
- `dropout = 0.05`
- activation: `swiglu`
- norm: `rmsnorm`
- residual layout: `pre_norm`
- positional encoding: `sinusoidal`
- embedding/output weight tying: enabled
- context length: `1024`

### Optimization

- optimizer: `AdamW`
- `learning_rate = 6e-4`
- `weight_decay = 0.1`
- `beta1 = 0.9`
- `beta2 = 0.95`
- gradient clipping: `2.0`
- scheduler: cosine decay
- `warmup_steps = 500`
- `min_lr_ratio = 0.1`

### Data

- dataset: [`../data/huggingface`](../data/huggingface)
- split source: Hugging Face `DatasetDict`
- training windowing: sliding window over each tokenized sequence
- `max_length = 1024`
- `sliding_window_stride = 256`
- length bucketing: disabled in the accepted baseline
- padding: dynamic padding inside each batch with `PAD_TOKEN_ID = 0`

## Design Choices and References

| Component | Current Choice | Why We Use It | Reference |
| --- | --- | --- | --- |
| Data source | Tokenized `rd` Hugging Face dataset | Stable training/eval input and reproducible splits | Project dataset pipeline |
| Sequence handling | Sliding windows with dynamic padding | Covers long sequences while keeping batch compute bounded | [Hugging Face perplexity guide](https://huggingface.co/docs/transformers/en/perplexity) |
| Model family | Decoder-only Transformer LM | Matches autoregressive next-token pretraining | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| Norm and residual layout | RMSNorm + Pre-Norm | More stable and better than earlier LayerNorm/Post-Norm baselines in our experiments | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) |
| FFN activation | SwiGLU | Consistently stronger than simpler FFN activations in modern LMs | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) |
| Positional encoding | Sinusoidal | Best result among tested sinusoidal / learned / RoPE / ALiBi variants on the current setup | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| Weight tying | Enabled | Reduces parameters and improved our baseline | [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) |
| Optimizer | AdamW | Standard decoupled weight decay optimizer for Transformer pretraining | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) |
| LR schedule | Cosine with warmup | Better than earlier linear/no-schedule variants in our tuning | [SGDR](https://arxiv.org/abs/1608.03983) |
| Validation loss | Token-weighted average over scored tokens | Correctly averages by effective token count rather than by batch | [Hugging Face perplexity guide](https://huggingface.co/docs/transformers/en/perplexity) |

## End-to-End Pipeline

1. Load tokenized `DatasetDict` from disk.
2. Build sliding-window samples for training or evaluation.
3. Dynamically pad each batch and construct `input_tokens`, `output_tokens`, `padding_mask`, and `loss_mask`.
4. Train the decoder LM with next-token cross-entropy.
5. Evaluate validation loss with token-weighted averaging over all scored validation tokens.
6. Save `latest.pt`, `best.pt`, `config.yaml`, `summary.json`, and `train.csv` into the run directory.

## Data and Sequence Handling

### Dataset

- Loader entry: [`../midi2score/data.py`](../midi2score/data.py)
- Expected feature: `input_ids`
- Training split uses overlapping windows over each stored token sequence.
- Validation split uses sliding windows plus `loss_mask` so each target token is counted once.

### Special Tokens

- `PAD_TOKEN_ID = 0`
- `BOS_TOKEN_ID = 1`
- `EOS_TOKEN_ID = 2`

### Sequence Policy

- Training:
  use all windows produced by `max_length=1024` and `stride=256`
- Validation:
  use sliding windows and compute loss only on newly covered target tokens
- Loss aggregation:
  average over effective scored tokens, not over batches

## Model Architecture

### High-Level Structure

- token embedding
- sinusoidal positional encoding
- dropout
- Transformer decoder stack
- tied output projection

### Decoder Layer

- causal self-attention with `nn.MultiheadAttention`
- Pre-Norm residual order
- RMSNorm before attention and feed-forward
- feed-forward block with SwiGLU

## Training and Evaluation

### Training Objective

- autoregressive next-token cross-entropy
- ignore padded targets via `ignore_index = pad_token_id`

### Evaluation Metrics

- validation loss
- perplexity
- token accuracy
- top-5 accuracy

### Selection Rule

- compare runs only under the same wall-clock budget
- primary metric: lowest validation loss

## Files

- config: [`../configs/pretrain_rd_best.yaml`](../configs/pretrain_rd_best.yaml)
- experiment log: [`decoder_pretrain_exp.md`](decoder_pretrain_exp.md)
- training entry: [`../run_pretrain.py`](../run_pretrain.py)
- training loop: [`../midi2score/train.py`](../midi2score/train.py)
- data pipeline: [`../midi2score/data.py`](../midi2score/data.py)
- model: [`../midi2score/model.py`](../midi2score/model.py)

## Notes

- The accepted baseline is fixed-length `1024`; longer-context variants were worse under the same `7200s` budget because throughput collapsed too sharply.
- Keep future tuning history in [`decoder_pretrain_exp.md`](decoder_pretrain_exp.md).
