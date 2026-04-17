# Data Preprocessing

This document records only the current preprocessing recipe used before model optimization.

## Train Chunk

Current train chunk recipe:

- Use `bar-aware` chunking on raw LMX and cut at explicit `measure` boundaries, treating bars as meaningful symbolic-music structure units in line with bar-structured modeling work such as MuPT and SymPAC ([MuPT](https://arxiv.org/abs/2404.06393), [SymPAC](https://arxiv.org/abs/2409.03055)).
- Use `overlap_bars = 2` as the current project chunk-overlap setting.
- Add `BOS` only to the first chunk of a piece and `EOS` only to the last chunk of a piece as the current project chunk-boundary setting.
- Constrain chunk size by final encoded length rather than raw LMX token count, because training operates on final `input_ids`, not raw text tokens ([Hugging Face fixed-length perplexity guidance](https://huggingface.co/docs/transformers/perplexity)).
- Use the upstream tokenizer pipeline `LMX token -> chr(33 + index_in_ALL_TOKENS) -> BPE` so bar-aware generated training data stays in the same tokenization space as the existing HF dataset, following the symbolic-music BPE direction described in *Byte Pair Encoding for Symbolic Music* ([Byte Pair Encoding for Symbolic Music](https://arxiv.org/abs/2301.11975), [linearized-musicxml](https://pypi.org/project/linearized-musicxml/)).
- Use offline-generated bar-aware chunks as the training dataset so the training loader consumes final chunks directly and does not apply train-side sliding-window chunking again.
- Use `max_length = 1024` for train chunks.

Implementation:

- [midi2score/bar_aware_chunk.py](../midi2score/bar_aware_chunk.py)
- [midi2score/data.py](../midi2score/data.py)
- [tests/test_bar_aware_chunk.py](../tests/test_bar_aware_chunk.py)

Current training dataset:

- [data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full_dataset](../data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full_dataset)

## Train Batching

Current train batching recipe:

- Use `max_tokens_per_batch = 16384`.
- Use `pad_to_length_multiple = 64`, which follows the standard padded-shape batching pattern used by max-token samplers in libraries such as AllenNLP and fairseq ([AllenNLP MaxTokensBatchSampler](https://docs.allennlp.org/main/api/data/samplers/max_tokens_batch_sampler/), [fairseq batch_by_size](https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/data/fairseq_dataset.html)).
- Use `length_bucketing = true` so batches group similar sequence lengths before token-budget batching, matching the same high-level idea used by `group_by_length` / length-grouped samplers in common training pipelines ([Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)).
- Use `bucket_padding_noise = 0.0` so batching remains deterministic.

Implementation:

- [midi2score/data.py](../midi2score/data.py)

## Validation Chunk

Current full validation recipe:

- Use sliding-window validation with `max_length = 1024` and `stride = 512` for the fixed-length causal model, following the standard strided-evaluation approach for fixed-length language models ([Hugging Face fixed-length perplexity guidance](https://huggingface.co/docs/transformers/perplexity)).
- Count overlapping targets only once through `loss_mask`, which follows the same fixed-length evaluation principle of scoring only the valid target region in overlapping windows ([Hugging Face fixed-length perplexity guidance](https://huggingface.co/docs/transformers/perplexity)).
- Use fixed-batch evaluation with `eval_batch_size = 16`.
- Use full validation with `num_eval_batches = null`.

Current fast validation recipe:

- Keep the same chunking and fixed-batch evaluation as full validation.
- Use `num_eval_batches = ceil(len(full_eval_loader) / 10)`.

Implementation:

- [midi2score/data.py](../midi2score/data.py)
