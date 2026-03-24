# Research Program

This file defines how an agent should run automatic tuning for decoder pretraining.

## Goal

Improve the decoder-only language model on the rd dataset.

Primary metric:

- validation cross-entropy loss
- lower is better

Current reference points from [`exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/exp.md):

- baseline `max_length=256`, `16000` steps: best validation loss `3.6623`
- `max_length=512` at the same step budget: best validation loss `3.6453`, currently treated as `no clear effect`

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
- Use [`run_experiment.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_experiment.py) so outputs land under:
  - `configs/research/<experiment_id>.yaml`
  - `artifacts/research/<experiment_id>/`
  - `logs/research/<experiment_id>.csv`
  - `logs/tensorboard/research/<experiment_id>/`
- Before choosing a new change, read [`exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/exp.md) and avoid repeating experiments.

## Decision Rule

Compare against the strongest clean baseline that matches the same training budget.

For now, use:

- reference best validation loss `3.6623`
- reference budget `16000` steps on rd data

Classify outcomes this way:

- `useful`: improves validation loss by at least `0.03`
- `no clear effect`: within `0.03` of reference
- `worse`: degrades validation loss by more than `0.03`

## Required Workflow

1. Read [`exp.md`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/exp.md).
2. Choose one new experiment.
3. Run it with [`run_experiment.py`](/Users/daboluo/MyWorkSpace/GitHub/MIDI2Score/run_experiment.py).
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
  --experiment-id EXP-005_dmodel128 \
  --set model.d_model=128 \
  --set training.num_steps=16000 \
  --note "Single-variable width increase from baseline" \
  --reference-best-loss 3.6623
```

## Notes

- The current project compares runs by fixed step budget, not fixed wall-clock budget.
- That is an adaptation of the `autoresearch` idea, not a full copy.
- If you later switch to wall-clock budgeting, first establish a new timed baseline before comparing results.
