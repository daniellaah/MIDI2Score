# Research Program

This file defines how an agent should run automatic tuning for decoder pretraining.

## Goal

Improve the decoder-only language model on the rd dataset.

Primary metric:

- validation cross-entropy loss
- lower is better

Current reference points from:
- [`docs/decoder_pretrain_rd_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_rd_exp.md)
- [`docs/decoder_pretrain_full_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_full_exp.md)

- historical best step-based run: `d_model=128`, `16000` steps, best validation loss `2.9924`
- current best 180-second timed run: `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, `learning_rate=7.5e-4`, best validation loss `2.7349`

Important:

- historical results before this document update were compared by fixed step budget
- new automatic experiments must be compared by wall-clock budget instead
- do not treat the old step-based numbers as a fair timed baseline
- first create and record a timed baseline before making strong claims about new timed runs

## Important Files

- [`docs/decoder_pretrain_rd_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_rd_exp.md): `rd` experiment history and decisions
- [`docs/decoder_pretrain_full_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_full_exp.md): `full` experiment history and decisions
- [`configs/pretrain_baseline.yaml`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/configs/pretrain_baseline.yaml): current baseline config
- [`run_pretrain.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_pretrain.py): single entrypoint for direct training and managed experiments

## Allowed Changes

Only change decoder-pretraining experiment configs unless there is a clear blocker.

Preferred knobs:

- `model.d_model`
- `model.num_layers`
- `model.dim_feedforward`
- `model.dropout`
- `training.learning_rate`
- `training.batch_size`
- `data.max_length`
- matching `model.max_length`

## Constraints

- Change one meaningful variable at a time.
- Keep dataset fixed to `data/huggingface`.
- Keep tokenizer fixed to `data/tokenizer_rd.json`.
- Do not overwrite baseline checkpoints or baseline logs.
- Start managed experiments from a clean git worktree.
- Use [`run_pretrain.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_pretrain.py) with `--experiment-id` so outputs land under:
  - `configs/research/<experiment_id>.yaml`
  - `artifacts/research/<experiment_id>/`
  - `logs/research/<experiment_id>.csv`
  - `logs/tensorboard/research/<experiment_id>/`
- Before choosing a new change, read both [`docs/decoder_pretrain_rd_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_rd_exp.md) and [`docs/decoder_pretrain_full_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_full_exp.md) and avoid repeating experiments.
- Each experiment summary must record the git commit and branch.

## Decision Rule

Compare against the strongest clean baseline that matches the same training budget.

For new managed runs:

- use a wall-clock budget, not a step budget
- default timed budget is now `300` seconds
- because the budget changed from `180` to `300` seconds, establish a fresh 300-second baseline before promoting new winners
- until then, old 180-second results are only rough context for choosing search directions

Classify outcomes this way:

- `useful`: improves validation loss by at least `0.03`
- `no clear effect`: within `0.03` of reference
- `worse`: degrades validation loss by more than `0.03`

## Required Workflow

1. Read [`docs/decoder_pretrain_rd_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_rd_exp.md) and [`docs/decoder_pretrain_full_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_full_exp.md).
2. Choose one or more new experiments for the same batch.
3. Commit code changes before running the batch if the worktree is dirty.
4. Run the batch with [`run_pretrain.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_pretrain.py) and `--experiment-id`.
5. Read each generated `summary.json`.
6. Update the appropriate experiment summary with:
   - [`docs/decoder_pretrain_rd_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_rd_exp.md) for `rd`
   - [`docs/decoder_pretrain_full_exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/docs/decoder_pretrain_full_exp.md) for `full`
   - a new `EXP-xxx` block
   - what changed
   - the result
   - the conclusion
7. Give a short recommendation for the next batch.

Batch rule:

- a single subagent may run multiple experiments in sequence
- each experiment inside the batch must still change only one meaningful variable relative to the chosen reference config

## Example Command

```bash
uv run python run_pretrain.py \
  --config configs/pretrain_baseline.yaml \
  --experiment-id EXP-TIMED-300-001_best180_baseline \
  --set model.d_model=128 \
  --set model.dim_feedforward=512 \
  --set model.dropout=0.0 \
  --set training.learning_rate=0.00075 \
  --set training.num_steps=1000000 \
  --set training.max_duration_seconds=300 \
  --note "300-second baseline with current best 180-second structure"
```

## Notes

- `training.num_steps` still exists as a safety cap, but timed experiments should normally stop because of `training.max_duration_seconds`.
- [`run_pretrain.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_pretrain.py) now checks git cleanliness by default in managed experiment mode.
- Use `--allow-dirty-git` only for local smoke tests, not for managed research runs.
