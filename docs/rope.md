# RoPE Research

Last updated: 2026-04-15

## Goal

Evaluate whether vanilla RoPE can outperform the current sinusoidal baseline under the current decoder pretraining setup, without changing:

- model size
- context length
- batching recipe
- canonical validation protocol

## Canonical Baseline

Reference baseline under the current decoder implementation with explicit `q/k/v` projections and optional cross-attention:

- positional encoding: `sinusoidal`
- `learning_rate = 6e-4`
- `weight_decay = 0.1`
- `grad_clip_norm = 2.0`
- `warmup_steps = 500`
- `dropout = 0.05`
- batching:
  - `max_tokens_per_batch = 16384`
  - `pad_to_length_multiple = 64`
  - `bucket_padding_noise = 0.0`

Canonical validation remains fixed:

- `max_length = 1024`
- `sliding_window_stride = 256`
- fixed-order validation
- fixed-batch validation
- `eval_batch_size = 16`
- token-weighted average loss over scored tokens

## 600s Reference

| positional_encoding | valid_loss | toks/sec | step time | peak mem | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| sinusoidal | 4.6129 | 32587.2 | 0.4733s | 69390.0 MiB | [`2026-04-14_19-20-03_961258`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-14_19-20-03_961258) |
| rope | 4.0283 | 30068.9 | 0.5123s | 73712.0 MiB | [`2026-04-14_19-30-12_388869`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-14_19-30-12_388869) |

Interpretation:

- Vanilla RoPE is clearly better than sinusoidal early.
- Vanilla RoPE is slower.
- The short-budget signal alone was not enough; long-budget behavior had to be retuned.

## 1800s Autoresearch

All rows below use the same `1024` context length and canonical validation.

| variant | valid_loss | toks/sec | step time | peak mem | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| rope + `lr=4e-4` | 1.9285 | 36871.3 | 0.4195s | 90195.0 MiB | [`2026-04-15_00-16-38_196582`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_00-16-38_196582) |
| rope + `warmup=1000` | 1.9086 | 35775.1 | 0.4323s | 88887.0 MiB | [`2026-04-15_00-46-44_219691`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_00-46-44_219691) |
| rope + `clip=1.0` | 1.8828 | 35670.9 | 0.4336s | 88276.0 MiB | [`2026-04-15_01-16-50_267035`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_01-16-50_267035) |
| rope + `warmup=1000` + `clip=1.0` | 1.9071 | 36853.1 | 0.4198s | 87782.0 MiB | [`2026-04-15_01-47-40_523404`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_01-47-40_523404) |
| rope + `lr=4e-4` + `clip=1.0` | 1.9258 | 36340.2 | 0.4256s | 88700.0 MiB | [`2026-04-15_02-18-17_503469`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_02-18-17_503469) |
| rope + `clip=1.0` + `weight_decay=0.05` | 1.8880 | 37322.7 | 0.4147s | 87782.0 MiB | [`2026-04-15_05-03-50_656533`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_05-03-50_656533) |
| rope + `clip=1.0` + `min_lr_ratio=0.0` | 1.8828 | 36462.0 | 0.4242s | 89211.0 MiB | [`2026-04-15_05-34-15_667910`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_05-34-15_667910) |
| rope + `clip=1.0` + `dropout=0.0` | 1.8696 | 44192.8 | 0.3501s | 62530.0 MiB | [`2026-04-15_06-04-36_792839`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_06-04-36_792839) |
| rope + `clip=1.0` + `dropout=0.02` | 1.8837 | 36806.1 | 0.4203s | 90161.0 MiB | [`2026-04-15_08-38-32_408429`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_08-38-32_408429) |

Interpretation:

- Lowering `grad_clip_norm` from `2.0` to `1.0` was the first consistent improvement.
- Lowering `learning_rate`, increasing `warmup_steps`, or changing `min_lr_ratio` did not help.
- `dropout=0.0` looked strong at `1800s`, but that signal did not hold at `7200s`.

## 7200s Canonical Comparison

| variant | valid_loss | toks/sec | step time | peak mem | status | notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| sinusoidal baseline | 1.6360 | 37466.7 | 0.4128s | 88794.2 MiB | baseline | [`2026-04-14_21-45-29_874161`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-14_21-45-29_874161) |
| pure rope | 1.6609 | 35171.3 | 0.4397s | 91110.2 MiB | discard | [`2026-04-14_19-41-34_990887`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-14_19-41-34_990887) |
| rope + `clip=1.0` | 1.6489 | 36652.0 | 0.4219s | 91823.0 MiB | discard | [`2026-04-15_02-48-38_259571`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_02-48-38_259571) |
| rope + `clip=1.0` + `dropout=0.0` | 1.7427 | 44452.4 | 0.3477s | 63028.0 MiB | discard | [`2026-04-15_06-35-02_165794`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/artifacts/runs/2026-04-15_06-35-02_165794) |

## Current Conclusion

Under the current decoder implementation:

- Vanilla RoPE improves early validation loss, but does not beat the sinusoidal baseline at `7200s`.
- The most likely pattern is:
  - faster early convergence
  - earlier plateau later in training
- Pure RoPE is therefore not the current best positional encoding on this branch.

## References

- RoFormer: [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
- Round and Round We Go! What makes Rotary Positional Encodings useful?: [https://arxiv.org/abs/2410.06205](https://arxiv.org/abs/2410.06205)
- LLaMA: [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)
- OLMo: [https://arxiv.org/abs/2402.00838](https://arxiv.org/abs/2402.00838)
- Music Transformer: [https://arxiv.org/abs/1809.04281](https://arxiv.org/abs/1809.04281)
- MusicBERT: [https://arxiv.org/abs/2106.05630](https://arxiv.org/abs/2106.05630)
