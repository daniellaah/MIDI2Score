# Length Bucketing

## Goal

The purpose of `length bucketing` is to reduce padding waste within each batch and improve training throughput without changing the model architecture.

For decoder pretraining, the main batching problems are:

1. Large length variance inside a batch causes excessive padding.
2. With a fixed `batch_size`, long-sequence batches are too heavy while short-sequence batches are too light.
3. Fully dynamic batch shapes can make allocator and kernel behavior unstable.

The current data pipeline addresses these issues with a small set of batching controls.

## Current Implementation

Code location:
- `midi2score/data.py`

Supported configuration fields:

| Field | Purpose |
| --- | --- |
| `length_bucketing` | Sort examples by approximate length inside large pools before batching |
| `bucket_padding_noise` | Add small random noise to bucket sorting to avoid rigid length ordering |
| `max_tokens_per_batch` | Build batches under a token budget instead of a fixed sample count |
| `required_batch_size_multiple` | Try to emit dynamic batches whose sample count matches a fixed multiple |
| `pad_to_length_multiple` | Pad batch length to a fixed multiple during collation to reduce shape diversity |

## Batching Pipeline

### 1. Sliding Windows

Original token sequences are expanded into sliding windows using:

- `max_length`
- `sliding_window_stride`

The sampler batches windows, not raw full-length sequences.

### 2. Length Bucketing

When `length_bucketing` is enabled:

1. Training examples are shuffled.
2. Indices are split into large pools.
3. Each pool is sorted by descending sequence length.
4. If `bucket_padding_noise` is enabled, small random perturbations are added before sorting.

This keeps similarly sized samples together while avoiding a completely deterministic ordering every epoch.

### 3. Max Tokens Per Batch

When `max_tokens_per_batch` is enabled, batches are built dynamically instead of being split directly by a fixed sample count.

The current budget rule is:

`max(sequence_length in batch) * batch_size <= max_tokens_per_batch`

This uses the padded batch length as a proxy for total token cost.

In this setup:

- `batch_size` is treated as `max_examples_per_batch`
- `max_tokens_per_batch` is treated as `token_budget_per_batch`

### 4. Required Batch Size Multiple

When `required_batch_size_multiple > 1`:

- dynamic batches try to flush at a sample count aligned to that multiple
- tail batches are still allowed to be smaller
- under a tight token budget, some non-tail batches may still be smaller than the requested multiple

This is a common engineering optimization for more regular kernel behavior.

### 5. Pad To Length Multiple

Instead of implementing explicit batch-shape tables, the current pipeline uses:

- `pad_to_length_multiple`

During collation, batch length is rounded up to a fixed multiple such as:

- `8`
- `16`
- `64`

This reduces shape diversity while keeping the implementation simple.

## Why We Did Not Add Explicit Batch Shape Tables

Some systems, especially fairseq-style pipelines, support a small predefined set of batch shapes. That can further reduce dynamic shape variation.

For this project, that approach would make the pipeline heavier:

- the sampler would need to emit shape metadata
- collation would need to obey external shapes instead of the batch's own maximum length
- the DataLoader interface would become more complex

For now, the lighter compromise is:

- `required_batch_size_multiple`
- `pad_to_length_multiple`

If shape instability remains a real bottleneck later, explicit batch shape tables can be reconsidered.

## Practical Recommendations

### Training

Recommended defaults for training:

- `length_bucketing = true`
- `max_tokens_per_batch` as the main batch budget
- `batch_size` only as an upper bound on example count
- `bucket_padding_noise > 0`

### Validation

Validation should prioritize stability and repeatability over aggressive dynamic batching.

Preferred properties:

- fixed ordering
- stable loss accounting
- stable batch shape behavior

### Moving to a Smaller GPU

The most important portability knob is:

- `max_tokens_per_batch`

If the target GPU has less memory, batch size will not shrink automatically. The token budget must be reduced manually.

## Experiment Results

The tables below summarize completed `600s` experiments. They are split into two rounds:

- the first round compares baseline batching variants
- the second round refines the `length bucketing + max_tokens_per_batch` setup

### Round 1: 600s Comparisons

| valid_loss | toks/sec | step time | peak mem | description |
| ---: | ---: | ---: | ---: | --- |
| 4.8384 | 24312.8 | 0.4569s | 35651.4 MiB | baseline |
| 5.0415 | 26516.9 | 0.4119s | 91096.5 MiB | baseline, length bucketing |
| 3.4203 | 27036.9 | 0.4096s | 40617.4 MiB | baseline, BF16, but this run actually reached `698.0s`, so it is not a strict `600s` comparison |
| 4.8276 | 22640.1 | 0.4815s | 40245.1 MiB | baseline, `max_tokens_per_batch=16384`, `batch_size` upper bound = 64 |
| 4.6188 | 28326.0 | 0.5516s | 110888.1 MiB | baseline, `length bucketing + max_tokens_per_batch=16384`, `batch_size` upper bound = 64 |
| 4.6183 | 29162.9 | 0.5358s | 107245.0 MiB | baseline, `length bucketing + max_tokens_per_batch=16384 + BF16`, `batch_size` upper bound = 64 |

Round 1 takeaways:

- `length bucketing` alone did not improve validation loss.
- `max_tokens_per_batch` alone also did not help.
- `length bucketing + max_tokens_per_batch` was the first clearly useful combination.
- In these runs, `BF16` mainly improved throughput rather than optimization quality at matched step count.

### Round 2: 600s Comparisons

This round starts from `length bucketing + max_tokens_per_batch=16384` and adds:

- `bucket_padding_noise`
- `pad_to_length_multiple`
- `BF16`
- `required_batch_size_multiple`

Experiments 4 and 5 ran longer than `600s` because higher throughput allowed them to reach additional steps before the time-budget check stopped the run. For a fair comparison, the most reliable shared checkpoint is the validation loss at `step=500`.

| exp | step500 val loss | toks/sec | step time | peak mem | description |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 4.6188 | 26347.2 | 0.5922s | 109965.4 MiB | `length bucketing + max_tokens_per_batch=16384` |
| 2 | 4.6159 | 27640.1 | 0.5623s | 126557.2 MiB | exp1 + `bucket_padding_noise=0.1` |
| 3 | 4.6186 | 31215.5 | 0.4953s | 107742.1 MiB | exp2 + `pad_to_length_multiple=64` |
| 4 | 4.6178 | 36504.7 | 0.4222s | 114190.6 MiB | exp3 + `BF16` |
| 5 | 4.6056 | 37594.7 | 0.4053s | 100410.6 MiB | exp4 + `required_batch_size_multiple=4` |

Run-level best validation losses:

| exp | best val loss | elapsed | final step | description |
| --- | ---: | ---: | ---: | --- |
| 1 | 4.6188 | 600.7s | 603 | `length bucketing + max_tokens_per_batch=16384` |
| 2 | 4.6159 | 600.3s | 611 | exp1 + `bucket_padding_noise=0.1` |
| 3 | 4.6186 | 600.9s | 821 | exp2 + `pad_to_length_multiple=64` |
| 4 | 3.1281 | 678.1s | 1000 | exp3 + `BF16` |
| 5 | 3.0582 | 647.3s | 1000 | exp4 + `required_batch_size_multiple=4` |

Round 2 takeaways:

- `bucket_padding_noise=0.1` gave a small positive gain.
- `pad_to_length_multiple=64` improved throughput substantially without hurting validation loss.
- `BF16` further improved throughput, but wall-clock-limited runs then became harder to compare directly by final best validation loss.
- The most promising direction so far is:
  - `length_bucketing = true`
  - `bucket_padding_noise = 0.1`
  - `max_tokens_per_batch = 16384`
  - `pad_to_length_multiple = 64`
  - `required_batch_size_multiple = 4`

## References

- Hugging Face Trainer: `group_by_length` / `LengthGroupedSampler`
  - [https://huggingface.co/docs/transformers/en/main_classes/trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- AllenNLP `MaxTokensBatchSampler`
  - [https://docs.allennlp.org/main/api/data/samplers/max_tokens_batch_sampler/](https://docs.allennlp.org/main/api/data/samplers/max_tokens_batch_sampler/)
- fairseq dataset batching / `batch_by_size`
  - [https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/data/fairseq_dataset.html](https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/data/fairseq_dataset.html)
- fairseq tasks docs: `required_batch_size_multiple`
  - [https://fairseq.readthedocs.io/en/v0.10.2/tasks.html](https://fairseq.readthedocs.io/en/v0.10.2/tasks.html)
- NVIDIA NeMo: dynamic batching and sequence packing
  - [https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html)
- NVIDIA Megatron Bridge: packed sequences
  - [https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/training/packed-sequences.html](https://docs.nvidia.com/nemo/megatron-bridge/0.1.0/training/packed-sequences.html)
