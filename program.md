# Research Program

This file defines how an agent should run automatic tuning for decoder pretraining.

The current priority is **`rd` decoder pretraining only**.

## Goal

Improve the decoder-only language model on the `rd` dataset without changing the task definition.

Primary metric:

- validation cross-entropy loss
- lower is better

Secondary metrics for promotion candidates:

- perplexity
- token accuracy
- top-5 accuracy

## Current Official `rd` Best

Always start from the current official `rd` recipe unless a new experiment explicitly changes one knob.

- config: [`configs/pretrain_rd_best.yaml`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/configs/pretrain_rd_best.yaml)
- experiment summary: [`docs/decoder_pretrain_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_exp.md)
- method doc: [`docs/decoder_pretrain.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain.md)

Current official recipe:

- dataset: `data/huggingface`
- tokenizer: `data/tokenizer_rd.json`
- `max_length = 1024`
- training: full-coverage sliding windows
- `sliding_window_stride = 512`
- validation/test: full-coverage sliding-window evaluation
- `d_model = 256`
- `nhead = 4`
- `num_layers = 2`
- `dim_feedforward = 1024`
- `dropout = 0.0`
- `batch_size = 16`
- `learning_rate = 6e-4`
- scheduler: `linear`
- `warmup_steps = 500`
- length bucketing: `false`

Current official full-validation metrics:

- CE loss: `1.8092`
- perplexity: `6.1053`
- token accuracy: `0.5988`
- top-5 accuracy: `0.7908`

## Comparison Budget

For new tuning on the current `rd` family:

- use **`600s`** as the default comparison budget
- compare challenger runs only against a baseline with the **same budget**
- do not mix `300s`, `600s`, `7200s`, and `10800s` results in one decision

Use these budget tiers:

- `600s`: baseline building and local hyperparameter search
- `1800s`: follow-up only for clearly useful `600s` winners
- `7200s` or more: confirmation / promotion runs only

## Important Files

- [`docs/decoder_pretrain_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_exp.md): `rd` experiment history and decisions
- [`docs/decoder_pretrain.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain.md): current `rd` method description
- [`configs/pretrain_rd_best.yaml`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/configs/pretrain_rd_best.yaml): official `rd` best config
- [`run_pretrain.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_pretrain.py): training entrypoint

## Allowed Changes

Only change decoder-pretraining experiment configs unless there is a clear blocker.

Preferred knobs for the current `rd` family:

- `training.batch_size`
- `training.learning_rate`
- `training.warmup_steps`
- `training.weight_decay`
- `training.grad_clip_norm`
- `training.label_smoothing`
- `data.sliding_window_stride`
- `training.max_duration_seconds`

Second-tier knobs:

- `model.d_model`
- `model.dim_feedforward`
- `model.num_layers`
- `data.max_length`
- matching `model.max_length`

## Current Negative Priors

Do not re-run these directions unless there is a new explicit hypothesis:

- `2048` context on `rd`
- no-truncation with `max_tokens_per_batch`
- length bucketing on the official `rd` branch
- learned positional encoding
- ALiBi
- RoPE
- flash attention backend
- deeper 3-layer variants

These are not theoretical bans; they are current low-priority directions with negative evidence.

## Constraints

- Change one meaningful variable at a time.
- Keep dataset fixed to `data/huggingface`.
- Keep tokenizer fixed to `data/tokenizer_rd.json`.
- Do not overwrite official checkpoints or official logs.
- Start managed experiments from a clean git worktree.
- Use [`run_pretrain.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_pretrain.py) with `--experiment-id` so outputs land under:
  - `configs/research/<experiment_id>.yaml`
  - `artifacts/research/<experiment_id>/`
  - `logs/research/<experiment_id>.csv`
  - `logs/tensorboard/research/<experiment_id>/`
- Read [`docs/decoder_pretrain_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_exp.md) before proposing new work.
- Each experiment summary must record the git commit and branch.

## Decision Rule

Compare against the strongest clean baseline that matches the same:

- dataset branch
- model family
- validation policy
- time budget

For the current `rd` family, use these thresholds:

- `useful`: improves validation CE by at least `0.03`
- `no clear effect`: within `0.03` of the reference
- `worse`: degrades validation CE by more than `0.03`

Promotion rule:

- do **not** promote a new official best from a single short-budget result
- a promotion candidate should either:
  - win again in a second run at the same budget, or
  - win at a longer confirmation budget

## Required Workflow

1. Read [`docs/decoder_pretrain_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_exp.md).
2. Identify the current matching baseline for the target budget.
3. Choose a small batch of experiments that each change only one meaningful variable relative to that baseline.
4. Commit code changes before running the batch if the worktree is dirty.
5. Run the batch with [`run_pretrain.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_pretrain.py) and `--experiment-id`.
6. Read each generated `summary.json`.
7. Update [`docs/decoder_pretrain_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_exp.md) only if the batch changed:
   - the official best
   - a strong backup direction
   - or an important negative lesson
8. Give a short recommendation for the next batch.

## Subagent Pattern

Subagents are useful for bounded, structured sweeps.

Recommended use:

- one subagent may own one sweep family
- each subagent should use one shared baseline and vary only one knob
- do not let multiple subagents write overlapping code or rewrite docs independently
- on a single local machine, **only one actual training run may execute at a time**
- if wall-clock budget is part of the comparison, parallel training is invalid because runs compete for the same hardware and become unfair
- subagents may still be used for:
  - planning the next sweep
  - preparing commands
  - summarizing finished runs
  - reading and updating docs after runs complete

Good sequential batches:

- `batch_size` sweep with `3` candidates
- `learning_rate` sweep with `3` candidates
- `warmup_steps` sweep with `3` candidates

Avoid parallelizing:

- any training runs whose results are compared by wall-clock budget
- long confirmation runs that may need result-dependent follow-up
- overlapping sweeps that compare against different baselines

## Recommended Search Order

For the current `rd` family, tune in this order:

1. baseline confirmation at `600s`
2. `batch_size`
3. `learning_rate`
4. `warmup_steps`
5. `weight_decay`
6. `grad_clip_norm`
7. `label_smoothing`
8. `sliding_window_stride`
9. only then reconsider structure changes

## Example Command

```bash
uv run python run_pretrain.py \
  --config configs/pretrain_rd_best.yaml \
  --experiment-id EXP-RD-TUNE600-BS12-001 \
  --set training.batch_size=12 \
  --set training.max_duration_seconds=600 \
  --note "600s rd tuning sweep: batch_size 12 on current 1024 sliding baseline"
```

## Notes

- `training.num_steps` remains a safety cap; timed experiments should stop because of `training.max_duration_seconds`.
- [`run_pretrain.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_pretrain.py) checks git cleanliness by default in managed experiment mode.
- Use `--allow-dirty-git` only for local smoke tests, not for managed research runs.
- Keep this file aligned with [`docs/decoder_pretrain_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_exp.md). If the official best changes, update both.
