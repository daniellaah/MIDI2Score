# Pretrain

This document records the current pretraining recipe.

## Data Preprocess

- Use `bar-aware` chunking on raw LMX and cut at explicit `measure` boundaries, treating bars as meaningful symbolic-music structure units in line with bar-structured modeling work such as MuPT and SymPAC ([MuPT](https://arxiv.org/abs/2404.06393), [SymPAC](https://arxiv.org/abs/2409.03055)).
- Use `overlap_bars = 2` as the current project chunk-overlap setting.
- Add `BOS` only to the first chunk of a piece and `EOS` only to the last chunk of a piece as the current project chunk-boundary setting.
- Constrain chunk size by final encoded length rather than raw LMX token count, because training operates on final `input_ids`, not raw text tokens ([Hugging Face fixed-length perplexity guidance](https://huggingface.co/docs/transformers/perplexity)).
- Use the upstream tokenizer pipeline `LMX token -> chr(33 + index_in_ALL_TOKENS) -> BPE` so bar-aware generated training data stays in the same tokenization space as the existing HF dataset, following the symbolic-music BPE direction described in *Byte Pair Encoding for Symbolic Music* ([Byte Pair Encoding for Symbolic Music](https://arxiv.org/abs/2301.11975), [linearized-musicxml](https://pypi.org/project/linearized-musicxml/)).
- Use offline-generated bar-aware chunks as the training dataset so the training loader consumes final chunks directly and does not apply train-side chunking again.
- Use `max_length = 1024` for train chunks.
- Use `max_tokens_per_batch = 16384`.
- Use `pad_to_length_multiple = 64`, which follows the standard padded-shape batching pattern used by max-token samplers in libraries such as AllenNLP and fairseq ([AllenNLP MaxTokensBatchSampler](https://docs.allennlp.org/main/api/data/samplers/max_tokens_batch_sampler/), [fairseq batch_by_size](https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/data/fairseq_dataset.html)).
- Use `length_bucketing = true` so batches group similar sequence lengths before token-budget batching, matching the same high-level idea used by `group_by_length` / length-grouped samplers in common training pipelines ([Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)).
- Use `bucket_padding_noise = 0.0` so batching remains deterministic.
- Use sliding-window validation with `max_length = 1024` and `stride = 512` for the fixed-length causal model, following the standard strided-evaluation approach for fixed-length language models ([Hugging Face fixed-length perplexity guidance](https://huggingface.co/docs/transformers/perplexity)).
- Count overlapping validation targets only once through `loss_mask`, which follows the same fixed-length evaluation principle of scoring only the valid target region in overlapping windows ([Hugging Face fixed-length perplexity guidance](https://huggingface.co/docs/transformers/perplexity)).
- Use fixed-batch evaluation with `eval_batch_size = 16`.
- Use full validation with `num_eval_batches = null`.
- Use fast validation with `num_eval_batches = ceil(len(full_eval_loader) / 10)`.

Current training dataset:

- [data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full_dataset](../data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full_dataset)

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
- default wall-clock budget: `7200s`

Files:

- training entry: `run_pretrain.py`
- model: `pretrain/decoder.py`
- training loop: `pretrain/trainer.py`
- data pipeline: `pretrain/data.py`
- bar-aware chunk generation: `pretrain/bar_aware_chunk.py`
- experiment history: `exp/decoder_pretrain_exp.md`

## Reference

- Transformer decoder: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- RMSNorm: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- SwiGLU / GLU variants: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Weight tying: [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- AdamW: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Fixed-length evaluation background: [Hugging Face perplexity guide](https://huggingface.co/docs/transformers/en/perplexity)
