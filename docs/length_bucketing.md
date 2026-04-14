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
| `length_bucketing` | Enable length grouping on the training split |
| `bucket_padding_noise` | Add small random noise to bucket sorting to avoid rigid length ordering |
| `max_tokens_per_batch` | Build training batches under a token budget instead of a fixed sample count |
| `required_batch_size_multiple` | Try to emit dynamic batches whose sample count matches a fixed multiple |
| `pad_to_length_multiple` | Pad training batch length to a fixed multiple during collation to reduce shape diversity |

Public API:

- `build_dataloader()`
- `collate_fn()`
- `LengthBucketedDynamicBatchSampler`

## Batching Pipeline

### 1. Sliding Windows

Original token sequences are expanded into sliding windows using:

- `max_length`
- `sliding_window_stride`

The sampler batches windows, not raw full-length sequences.

### 2. Length Bucketing

When `length_bucketing` is enabled on the training split:

1. Training examples are shuffled.
2. Indices are split into large pools.
3. Each pool is sorted by descending sequence length.
4. If `bucket_padding_noise` is enabled, small random perturbations are added before sorting.

This keeps similarly sized samples together while avoiding a completely deterministic ordering every epoch.

### 3. Max Tokens Per Batch

When `max_tokens_per_batch` is enabled, training batches are built dynamically instead of being split directly by a fixed sample count.

The current budget rule is:

`round_up(max(sequence_length in batch), pad_to_length_multiple) * batch_size <= max_tokens_per_batch`

This uses the effective padded batch length so the sampler budget matches the real collated shape.

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

Instead of implementing explicit batch-shape tables, the current pipeline uses `pad_to_length_multiple`.

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
- no dependence on training-side length bucketing or token-budget batching

Current rule:

- `build_dataloader()` uses separate internal train/eval paths
- training may use `LengthBucketedDynamicBatchSampler`
- validation always uses fixed batching, `shuffle = false`, and `pad_to_length_multiple = 1`
- cross-run comparison should also keep `eval_batch_size` fixed

### Moving to a Smaller GPU

The most important portability knob is:

- `max_tokens_per_batch`

If the target GPU has less memory, batch size will not shrink automatically. The token budget must be reduced manually.

## Experiment Results

The table below summarizes the latest `600s` reruns under the canonical validation recipe:

- fixed-batch validation
- `shuffle = false`
- no validation length bucketing
- no validation token budget
- `eval_batch_size = 16`

Because some faster runs can still finish a validation pass after the `600s` boundary, the fairest comparison is the first shared validation point at `step=500`.

### Latest 600s Comparison

| exp | step500 val loss | toks/sec | step time | peak mem | description |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 4.8384 | 24212.0 | 0.4594s | 35943.4 MiB | baseline |
| 2 | 4.6188 | 31680.9 | 0.4927s | 107698.9 MiB | `length_bucketing + max_tokens_per_batch=16384` |
| 3 | 4.6159 | 29999.6 | 0.5173s | 119031.0 MiB | exp2 + `bucket_padding_noise=0.1` |
| 4 | 4.5809 | 35236.3 | 0.4376s | 73221.6 MiB | exp2 + `pad_to_length_multiple=64` |
| 5 | 4.6592 | 39823.9 | 0.3814s | 89175.1 MiB | exp2 + `bucket_padding_noise=0.1` + `pad_to_length_multiple=64` |
| 6 | 4.6553 | 29824.0 | 0.5012s | 72226.5 MiB | exp5 + `required_batch_size_multiple=4` |

### Run-Level Best Metrics

These values are still useful for operational monitoring, but they should not be used as the primary fairness metric for fixed-wall-clock comparisons when some runs evaluate past the budget boundary.

| exp | best val loss | elapsed | final step | description |
| --- | ---: | ---: | ---: | --- |
| 1 | 4.8384 | 600.09s | 823 | baseline |
| 2 | 4.6188 | 600.43s | 724 | `length_bucketing + max_tokens_per_batch=16384` |
| 3 | 4.6159 | 600.33s | 647 | exp2 + `bucket_padding_noise=0.1` |
| 4 | 4.5809 | 600.13s | 848 | exp2 + `pad_to_length_multiple=64` |
| 5 | 3.1303 | 726.55s | 1000 | exp2 + `bucket_padding_noise=0.1` + `pad_to_length_multiple=64` |
| 6 | 4.6553 | 600.17s | 779 | exp5 + `required_batch_size_multiple=4` |

### Takeaways

- Under the canonical fixed validation recipe, `length bucketing + max_tokens_per_batch` is still clearly better than the plain baseline on both validation loss and throughput.
- `bucket_padding_noise=0.1` gives only a weak loss gain and slightly hurts throughput.
- `pad_to_length_multiple=64` is the strongest single follow-up change on top of `length_bucketing + max_tokens_per_batch=16384`.
- combining `bucket_padding_noise=0.1` and `pad_to_length_multiple=64` is worse than using `pad_to_length_multiple=64` alone
- `required_batch_size_multiple=4` does not help under the current token-budget regime.

Current best direction for further batching experiments:

- `length_bucketing = true`
- `max_tokens_per_batch = 16384`
- `pad_to_length_multiple = 64`

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
