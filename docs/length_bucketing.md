# Length Bucketing

## Purpose

This document records the batching method currently used for decoder pretraining experiments and the validated results that still matter.

The goal of length bucketing is to improve training efficiency without changing the model architecture or the validation protocol.

## Scope

Code location:
- `midi2score/data.py`

Training-side controls:
- `max_tokens_per_batch`
- `bucket_padding_noise`
- `pad_to_length_multiple`

Validation is intentionally fixed and does not depend on those controls.

## Canonical Validation Protocol

All results in this document use the same validation recipe:

- `max_length = 1024`
- `sliding_window_stride = 256`
- fixed-order validation
- fixed-batch validation
- `eval_batch_size = 16`
- token-weighted average loss over scored tokens

This is the only valid basis for cross-run comparison here.

## Current Method

Training windows are built from tokenized sequences with:

- `max_length = 1024`
- `sliding_window_stride = 256`

1. training windows are shuffled
2. indices are grouped into large pools
3. each pool is sorted by approximate length
4. batches are emitted under a token budget

The effective token-budget rule is:

`round_up(max(sequence_length in batch), pad_to_length_multiple) * batch_size <= max_tokens_per_batch`

This means:

- `batch_size` is an upper bound on example count
- `max_tokens_per_batch` is the real training budget
- `pad_to_length_multiple` affects both collation shape and sampler budget

## Why `pad_to_length_multiple` Matters

`pad_to_length_multiple`
- acts on sequence length
- rounds batch length up to a fixed multiple such as `32`, `64`, or `128`
- reduces shape diversity
- directly changes the padded shape used by attention and feed-forward compute

This was more useful than sample-count alignment in our experiments because the dominant constraint is `max_tokens_per_batch`. Under a tight token budget, regularizing sequence length helped more than trying to regularize the number of examples in each batch.

## Validated Results

### 600s Sampler-Speed Sweep

Selection rule:
- choose the configuration with the highest `tokens/sec`
- treat `valid_loss` here as a secondary monitoring field, not the selection metric

Validated sweep:

| max_tokens_per_batch | pad_to_length_multiple | valid_loss | toks/sec | status |
| ---: | ---: | ---: | ---: | --- |
| 14336 | 32 | 3.249211 | 44323.5 | keep |
| 14336 | 64 | 3.234957 | 44066.1 | discard |
| 14336 | 128 | 3.227507 | 43377.1 | discard |
| 16384 | 32 | 3.061776 | 42693.5 | discard |
| 16384 | 64 | 3.058532 | 42828.9 | discard |
| 16384 | 128 | 3.058387 | 41682.8 | discard |
| 18432 | 32 | 4.495638 | 37965.5 | discard |
| 18432 | 64 | 4.543844 | 40838.4 | discard |
| 18432 | 128 | 4.531093 | 41930.3 | discard |

Interpretation:

- `14336 + pad32` is the fastest short-budget configuration.
- `16384 + pad64` is slightly slower at `600s`, but its loss profile is much better.
- throughput alone is not sufficient for promotion to the long-budget baseline.

### 7200s Canonical Runs

Selection rule:
- choose the configuration with the lowest `validation loss`

| recipe | valid_loss | toks/sec | status |
| --- | ---: | ---: | --- |
| fixed batch baseline | 1.674948 | 28016.1 | discard |
| `length_bucketing + max_tokens 16384 + pad64` | 1.636279 | 42033.8 | keep |
| `length_bucketing + max_tokens 14336 + pad32` | 1.662629 | 40886.7 | discard |

Interpretation:

- `length_bucketing + max_tokens 16384 + pad64` is the current best long-budget batching recipe.
- `14336 + pad32` wins the `600s` throughput race but does not win the `7200s` objective.
- the best validated improvement over the fixed-batch baseline is therefore:
  - better `validation loss`
  - much higher `tokens/sec`

## Current Recommendation

Use the following batching recipe for current `7200s` decoder pretraining:

- `max_tokens_per_batch = 16384`
- `pad_to_length_multiple = 64`
- `bucket_padding_noise = 0.0`

Use the canonical fixed validation protocol unchanged.

## References

- Hugging Face Trainer: `group_by_length` / `LengthGroupedSampler`
  - [https://huggingface.co/docs/transformers/en/main_classes/trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- AllenNLP `MaxTokensBatchSampler`
  - [https://docs.allennlp.org/main/api/data/samplers/max_tokens_batch_sampler/](https://docs.allennlp.org/main/api/data/samplers/max_tokens_batch_sampler/)
- fairseq dataset batching / `batch_by_size`
  - [https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/data/fairseq_dataset.html](https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/data/fairseq_dataset.html)
