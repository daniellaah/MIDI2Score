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

The table below summarizes the latest `600s` reruns after:

- removing the temporary `RoPE` path
- tightening wall-clock budget checks

Because some faster runs can still finish a validation pass after the `600s` boundary, the fairest comparison is the first shared validation point at `step=500`.

### Latest 600s Comparison

| exp | step500 val loss | toks/sec | step time | peak mem | description |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 4.8384 | 26362.6 | 0.4204s | 35769.2 MiB | baseline |
| 2 | 4.8389 | 28188.5 | 0.3929s | 41051.4 MiB | baseline + `BF16` |
| 3 | 4.6188 | 31991.4 | 0.4889s | 117543.9 MiB | `length bucketing + max_tokens_per_batch=16384` |
| 4 | 4.6186 | 32903.6 | 0.4679s | 107644.5 MiB | exp3 + `bucket_padding_noise=0.1` + `pad_to_length_multiple=64` |
| 5 | 4.6181 | 38484.7 | 0.4005s | 114694.6 MiB | exp4 + `BF16` |
| 6 | 4.6071 | 41843.8 | 0.3635s | 99514.6 MiB | exp5 + `required_batch_size_multiple=4` |

### Run-Level Best Metrics

These values are still useful for operational monitoring, but they should not be used as the primary fairness metric for fixed-wall-clock comparisons when some runs evaluate past the budget boundary.

| exp | best val loss | elapsed | final step | description |
| --- | ---: | ---: | ---: | --- |
| 1 | 4.8384 | 600.0s | 951 | baseline |
| 2 | 3.4196 | 660.8s | 1000 | baseline + `BF16` |
| 3 | 4.6188 | 600.4s | 809 | `length bucketing + max_tokens_per_batch=16384` |
| 4 | 4.6186 | 601.4s | 908 | exp3 + `bucket_padding_noise=0.1` + `pad_to_length_multiple=64` |
| 5 | 3.1282 | 644.6s | 1000 | exp4 + `BF16` |
| 6 | 3.0577 | 600.3s | 1052 | exp5 + `required_batch_size_multiple=4` |

### Takeaways

- `length bucketing + max_tokens_per_batch` is clearly better than the plain baseline on both validation loss and throughput.
- `bucket_padding_noise=0.1` gives a small but consistent gain over rigid length sorting.
- `pad_to_length_multiple=64` improves throughput without hurting validation loss.
- `BF16` is mainly a throughput optimization in these comparisons. At matched validation steps, it does not change loss much by itself.
- `required_batch_size_multiple=4` is the strongest variant so far. In this rerun it improves throughput, lowers peak memory relative to exp5, and gives the best shared `step=500` validation loss.

Current best direction for further batching experiments:

- `length_bucketing = true`
- `bucket_padding_noise = 0.1`
- `max_tokens_per_batch = 16384`
- `pad_to_length_multiple = 64`
- `required_batch_size_multiple = 4`

### 7200s Follow-Up

The strongest `600s` variant was also rerun for a full `7200s` budget.

Configuration:

- `length_bucketing = true`
- `bucket_padding_noise = 0.1`
- `max_tokens_per_batch = 16384`
- `required_batch_size_multiple = 4`
- `pad_to_length_multiple = 64`
- `batch_size = 64`
- `precision = bf16`

Result:

| best val loss | best step | final step | elapsed | toks/sec | step time | peak mem |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.6485 | 9000 | 10279 | 7200.15s | 35488.1 | 0.4287s | 125480.8 MiB |

Compared with the fixed-batch `7200s` baseline:

- baseline best val loss: `1.6735`
- baseline average tokens per second: `20383.4`
- length-bucketing variant average tokens per second: `35488.1`
- throughput improvement: about `74%`

This run is therefore a meaningful improvement in both validation loss and training throughput, and it supports keeping the batching path active for future work.

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
