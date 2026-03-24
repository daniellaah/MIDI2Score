# Research Program

This file defines how an agent should run automatic tuning for decoder pretraining.

## Goal

Improve the decoder-only language model on the rd dataset.

Primary metric:

- validation cross-entropy loss
- lower is better

Current reference points from [`exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/exp.md):

- historical best step-based run: `d_model=128`, `16000` steps, best validation loss `2.9924`

Important:

- historical results before this document update were compared by fixed step budget
- new automatic experiments must be compared by wall-clock budget instead
- do not treat the old step-based numbers as a fair timed baseline
- first create and record a timed baseline before making strong claims about new timed runs

## Important Files

- [`exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/exp.md): experiment history and decisions
- [`configs/pretrain_baseline.yaml`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/configs/pretrain_baseline.yaml): current baseline config
- [`run_experiment.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_experiment.py): standardized experiment runner

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
- Use [`run_experiment.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_experiment.py) so outputs land under:
  - `configs/research/<experiment_id>.yaml`
  - `artifacts/research/<experiment_id>/`
  - `logs/research/<experiment_id>.csv`
  - `logs/tensorboard/research/<experiment_id>/`
- Before choosing a new change, read [`exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/exp.md) and avoid repeating experiments.
- Each experiment summary must record the git commit and branch.

## Decision Rule

Compare against the strongest clean baseline that matches the same training budget.

For new managed runs:

- use a wall-clock budget, not a step budget
- recommended first timed budget: `180` seconds
- after the first timed baseline is established, compare later timed runs against that baseline
- until then, old step-based runs are only rough context

Classify outcomes this way:

- `useful`: improves validation loss by at least `0.03`
- `no clear effect`: within `0.03` of reference
- `worse`: degrades validation loss by more than `0.03`

## Required Workflow

1. Read [`exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/exp.md).
2. Choose one new experiment.
3. Commit code changes before running the experiment if the worktree is dirty.
4. Run it with [`run_experiment.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_experiment.py).
4. Read the generated `summary.json`.
5. Update [`exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/exp.md) with:
   - a new `EXP-xxx` block
   - what changed
   - the result
   - the conclusion
6. Give a short recommendation for the next experiment.

## Example Command

```bash
uv run python run_experiment.py \
  --base-config configs/pretrain_baseline.yaml \
  --experiment-id EXP-TIMED-001_dmodel128_baseline \
  --set model.d_model=128 \
  --set training.num_steps=1000000 \
  --set training.max_duration_seconds=180 \
  --note "Timed baseline with current best known width"
```

## Notes

- `training.num_steps` still exists as a safety cap, but timed experiments should normally stop because of `training.max_duration_seconds`.
- [`run_experiment.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_experiment.py) now checks git cleanliness by default.
- Use `--allow-dirty-git` only for local smoke tests, not for managed research runs.
