# Experiments

This file is a change log for model and training experiments.

Rules for future updates:

- add one new experiment block per meaningful change
- always state what changed relative to the previous experiment
- always record the main metric after the change
- always write a short conclusion: `useful`, `no clear effect`, or `worse`
- if multiple things changed at once, say that the result is harder to attribute

## Fixed Baseline

Task:

- decoder-only language model pretraining on rd LMX data

Model:

- vocab size: `5000`
- `d_model=64`
- `nhead=4`
- `num_layers=2`
- `dim_feedforward=128`
- `dropout=0.1`
- `activation=relu`
- `max_length=256`

Data:

- dataset: `data/huggingface`
- tokenizer: `data/tokenizer_rd.json`
- training uses random crop
- validation uses deterministic prefix crop

Training:

- `batch_size=4`
- `learning_rate=1e-3`
- `eval_every=500`
- `num_eval_batches=64`

Main metric:

- validation cross-entropy loss
- practical comparison reference for the timed batch-size sweep: `2.5787` from `EXP-TIMED-300-002_dmodel128_ff512_dropout0_lr6e4`
- that `2.5787` result was the current best observed 300-second candidate before the batch-size sweep, but its gain over the prior `2.6071` baseline was just under the promotion threshold

Artifacts:

- latest checkpoint: `artifacts/pretrained_decoder_baseline.pt`
- best checkpoint: `artifacts/pretrained_decoder_best.pt`
- CSV log: `logs/pretrain_rd.csv`
- TensorBoard log dir: `logs/tensorboard/pretrain_rd`

## Experiment Log

### EXP-000 Smoke Run

Date:

- 2026-03-23

Change vs previous:

- initial real-data smoke test
- validation loop and best-checkpoint saving enabled

Run:

- `num_steps=5`

Result:

- training ran successfully on real rd data
- loss started above random baseline and moved in the right direction

Conclusion:

- useful
- this confirmed the real-data pretraining pipeline was wired correctly

### EXP-001 First Full rd Run

Date:

- 2026-03-23

Change vs previous:

- increased training length from `5` to `8000` steps

Run:

- `num_steps=8000`

Result:

- best validation loss: `4.1044`
- validation loss fell steadily from about `7.10` at step `500` to `4.1044` at step `8000`
- no obvious overfitting signal

Conclusion:

- useful
- longer training clearly helped

### EXP-002 Resume + Logging + Longer Training

Date:

- 2026-03-23

Change vs previous:

- resumed from `artifacts/pretrained_decoder_baseline.pt`
- increased training length from `8000` to `12000` steps
- added CSV logging
- added TensorBoard logging

Run:

- `num_steps=12000`
- resumed from step `8000`

Result:

- best validation loss: `3.8551`
- validation loss improved from `4.1044` to `3.8551`
- validation crossed below `4.0` at step `10000` with `3.9808`
- latest visible validation values:
  - step `8500`: `4.0974`
  - step `9000`: `4.0556`
  - step `9500`: `4.0321`
  - step `10000`: `3.9808`
  - step `10500`: `3.9408`
  - step `11000`: `3.9103`
  - step `11500`: `3.8858`
  - step `12000`: `3.8551`

Conclusion:

- useful
- continuing the same training run beyond `8000` steps still helped materially
- CSV and TensorBoard logging are useful for inspection, but they are observability changes, not optimization changes

Notes:

- this resume used an old checkpoint that did not contain optimizer state
- model weights resumed correctly
- optimizer restarted fresh
- despite that limitation, validation loss still improved steadily

### EXP-003 Continue Training to 16000

Date:

- 2026-03-23

Change vs previous:

- increased training length from `12000` to `16000` steps
- resumed from a checkpoint that now included optimizer state

Run:

- `num_steps=16000`
- resumed from step `12000`

Result:

- best validation loss: `3.6623`
- validation loss improved from `3.8551` to `3.6623`
- latest visible validation values:
  - step `12500`: `3.8414`
  - step `13000`: `3.8207`
  - step `13500`: `3.7785`
  - step `14000`: `3.7589`
  - step `14500`: `3.7307`
  - step `15000`: `3.6924`
  - step `15500`: `3.6869`
  - step `16000`: `3.6623`

Conclusion:

- useful
- continuing the same configuration still gave a meaningful gain
- the curve is still improving, but more slowly than earlier in training

Notes:

- this resume restored both model weights and optimizer state
- no obvious instability appeared after resuming

### EXP-004 Increase max_length to 512

Date:

- 2026-03-23

Change vs previous:

- increased `max_length` from `256` to `512`
- kept model width, depth, learning rate, batch size, and training length the same
- ran from scratch instead of resuming the `256`-length model

Run:

- config: `configs/pretrain_rd_maxlen512.yaml`
- `num_steps=16000`

Result:

- best validation loss: `3.6453`
- baseline at `max_length=256` and `16000` steps: `3.6623`
- absolute improvement over baseline: `0.0170`
- validation crossed below `4.0` at step `10000` with `3.9584`
- latest visible validation values:
  - step `12500`: `3.8216`
  - step `13000`: `3.7995`
  - step `13500`: `3.7664`
  - step `14000`: `3.7575`
  - step `14500`: `3.7364`
  - step `15000`: `3.6799`
  - step `15500`: `3.6600`
  - step `16000`: `3.6453`

Conclusion:

- no clear effect
- the result is slightly better than the `256` baseline, but the gain is small enough that it may be noise
- this change should not yet be treated as a confirmed improvement

Notes:

- artifacts:
  - latest checkpoint: `artifacts/pretrained_decoder_rd_maxlen512.pt`
  - best checkpoint: `artifacts/pretrained_decoder_rd_maxlen512_best.pt`
  - CSV log: `logs/pretrain_rd_maxlen512.csv`
  - TensorBoard log dir: `logs/tensorboard/pretrain_rd_maxlen512`
- this run took longer per step than the `256` context runs, so the small metric gain came with higher training cost

### EXP-005 Increase d_model to 128

Date:

- 2026-03-23

Change vs previous:

- increased `model.d_model` from `64` to `128`
- kept all other settings the same
- trained from scratch for the same `16000` step budget

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-005_dmodel128`
- `num_steps=16000`

Result:

- best validation loss: `3.0250`
- reference baseline at `d_model=64` and `16000` steps: `3.6623`
- absolute improvement over reference: `0.6373`
- validation crossed below `4.0` at step `4000` with `3.9231`
- latest visible validation values:
  - step `12000`: `3.1341`
  - step `12500`: `3.1199`
  - step `13000`: `3.0944`
  - step `13500`: `3.0798`
  - step `14000`: `3.0948`
  - step `14500`: `3.0526`
  - step `15000`: `3.0373`
  - step `15500`: `3.0293`
  - step `16000`: `3.0250`

Conclusion:

- useful
- this is a clear improvement over the `d_model=64` reference at the same training budget
- the gain is large enough to be treated as real, not noise

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-005_dmodel128/latest.pt`
  - best checkpoint: `artifacts/research/EXP-005_dmodel128/best.pt`
  - CSV log: `logs/research/EXP-005_dmodel128.csv`
- TensorBoard log dir: `logs/tensorboard/research/EXP-005_dmodel128`
- optimizer state was not resumed because this run started from scratch

### EXP-006 Re-run best width after random-crop fix

Date:

- 2026-03-24

Change vs previous:

- kept the best known hyperparameters the same
- reran `d_model=128` after fixing training random crop so repeated sample visits can see different windows

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-006_dmodel128_rerun_after_cropfix`
- `num_steps=16000`

Result:

- best validation loss: `2.9924`
- previous `d_model=128` result before crop fix: `3.0250`
- absolute improvement over previous best-width run: `0.0326`
- latest visible validation values:
  - step `12000`: `3.1783`
  - step `12500`: `3.1166`
  - step `13000`: `3.1041`
  - step `13500`: `3.0604`
  - step `14000`: `3.0426`
  - step `14500`: `3.0419`
  - step `15000`: `3.0278`
  - step `15500`: `3.0337`
  - step `16000`: `2.9924`

Conclusion:

- useful
- the gain is smaller than the original width increase, but it is still an improvement over the old `d_model=128` run
- this suggests the crop fix improved training signal rather than just cleaning up code

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-006_dmodel128_rerun_after_cropfix/latest.pt`
  - best checkpoint: `artifacts/research/EXP-006_dmodel128_rerun_after_cropfix/best.pt`
  - CSV log: `logs/research/EXP-006_dmodel128_rerun_after_cropfix.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-006_dmodel128_rerun_after_cropfix`
- this rerun is the current best result on rd data

### EXP-007 First Timed Baseline

Date:

- 2026-03-24

Change vs previous:

- switched the comparison budget from fixed steps to a `180` second wall-clock limit
- kept the current best known width `d_model=128`
- used a large step cap as safety only

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-001_dmodel128_baseline`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `3.0486`
- elapsed time: `180.01` seconds
- final step reached: `13803`
- validation loss at the timed cutoff was still improving

Conclusion:

- useful
- this is the first clean wall-clock baseline for future managed autotuning
- future timed runs should compare against this baseline, not the older fixed-step results

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-001_dmodel128_baseline/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-001_dmodel128_baseline/best.pt`
  - CSV log: `logs/research/EXP-TIMED-001_dmodel128_baseline.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-001_dmodel128_baseline`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `7ba8c77fa9fc072194c7521fe222385b9eeec74f`
  - worktree clean: `true`

### EXP-008 Timed Feedforward Width Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `model.dim_feedforward` from `128` to `256`
- kept `d_model=128`, `num_layers=2`, and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-002_dmodel128_ff256`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `3.0224`
- reference timed baseline at `d_model=128`, `dim_feedforward=128`: `3.0486`
- absolute improvement over reference: `0.0262`
- final step reached: `15206`
- elapsed time: `180.01` seconds

Conclusion:

- no clear effect
- the result is slightly better than the timed baseline, but the gain is smaller than the `0.03` cutoff
- this is a promising direction, but not yet a confirmed improvement

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-002_dmodel128_ff256/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-002_dmodel128_ff256/best.pt`
  - CSV log: `logs/research/EXP-TIMED-002_dmodel128_ff256.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-002_dmodel128_ff256`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `b8a8db963ebea9c0174fc54b9922660f946eb9df`
  - worktree clean: `true`

### EXP-009 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `training.learning_rate` from `1e-3` to `5e-4`
- kept `d_model=128`, `num_layers=2`, `dim_feedforward=128`, and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-003_dmodel128_lr0005`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `3.0887`
- reference timed baseline at `d_model=128`, `learning_rate=1e-3`: `3.0486`
- absolute delta vs reference: `+0.0401`
- final step reached: `15424`
- elapsed time: `180.00` seconds

Conclusion:

- worse
- lowering the learning rate degraded the timed baseline
- this setting should not be promoted

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-003_dmodel128_lr0005/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-003_dmodel128_lr0005/best.pt`
  - CSV log: `logs/research/EXP-TIMED-003_dmodel128_lr0005.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-003_dmodel128_lr0005`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `b8a8db963ebea9c0174fc54b9922660f946eb9df`
  - worktree clean: `true`

### EXP-010 Timed Depth Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `model.num_layers` from `2` to `3`
- kept `d_model=128`, `dim_feedforward=128`, and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-004_dmodel128_layers3`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `3.2993`
- reference timed baseline at `num_layers=2`: `3.0486`
- absolute delta vs reference: `+0.2507`
- final step reached: `12093`
- elapsed time: `180.00` seconds

Conclusion:

- worse
- increasing depth to 3 layers was a clear regression under the timed budget
- this setting should be avoided for now

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-004_dmodel128_layers3/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-004_dmodel128_layers3/best.pt`
  - CSV log: `logs/research/EXP-TIMED-004_dmodel128_layers3.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-004_dmodel128_layers3`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `b8a8db963ebea9c0174fc54b9922660f946eb9df`
  - worktree clean: `true`

### EXP-011 Timed Feedforward Width Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `model.dim_feedforward` from `128` to `384`
- kept `d_model=128` and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-005_dmodel128_ff384`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.9873`
- reference timed baseline at `d_model=128`, `dim_feedforward=128`: `3.0486`
- absolute improvement over reference: `0.0613`
- final step reached: `16029`
- elapsed time: `180.01` seconds

Conclusion:

- useful
- this is a real improvement over the timed baseline
- the wider feedforward stack is worth keeping in the search space

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-005_dmodel128_ff384/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-005_dmodel128_ff384/best.pt`
  - CSV log: `logs/research/EXP-TIMED-005_dmodel128_ff384.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-005_dmodel128_ff384`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `0f2e9c70a6099e80726e3497062ff299c81c9391`
  - worktree clean: `true`

### EXP-012 Timed Feedforward Width Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `model.dim_feedforward` from `128` to `512`
- kept `d_model=128` and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-006_dmodel128_ff512`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.9853`
- reference timed baseline at `d_model=128`, `dim_feedforward=128`: `3.0486`
- absolute improvement over reference: `0.0633`
- final step reached: `16135`
- elapsed time: `180.01` seconds

Conclusion:

- useful
- this is the best result of the feedforward sweep so far
- the larger feedforward width is a strong candidate for the timed baseline

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-006_dmodel128_ff512/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-006_dmodel128_ff512/best.pt`
  - CSV log: `logs/research/EXP-TIMED-006_dmodel128_ff512.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-006_dmodel128_ff512`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `0f2e9c70a6099e80726e3497062ff299c81c9391`
  - worktree clean: `true`

### EXP-013 Timed Dropout Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `model.dropout` from `0.1` to `0.0`
- kept `d_model=128`, `dim_feedforward=128`, and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-007_dmodel128_dropout0`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.7713`
- reference timed baseline at `d_model=128`, `dropout=0.1`: `3.0486`
- absolute improvement over reference: `0.2773`
- final step reached: `19078`
- elapsed time: `180.01` seconds

Conclusion:

- useful
- this is the strongest result of the batch by a wide margin
- zero dropout should be the current default candidate in the timed search

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-007_dmodel128_dropout0/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-007_dmodel128_dropout0/best.pt`
  - CSV log: `logs/research/EXP-TIMED-007_dmodel128_dropout0.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-007_dmodel128_dropout0`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `0f2e9c70a6099e80726e3497062ff299c81c9391`
  - worktree clean: `true`

### EXP-014 Timed Feedforward Width Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `model.dim_feedforward` from `128` to `384`
- kept `d_model=128` and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-005_dmodel128_ff384`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.9873`
- reference timed baseline at `d_model=128`, `dim_feedforward=128`: `3.0486`
- absolute improvement over reference: `0.0613`
- final step reached: `16029`
- elapsed time: `180.01` seconds

Conclusion:

- worse
- against the current timed baseline `2.7713`, this run regressed
- this width increase should not be promoted over the current best search state

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-005_dmodel128_ff384/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-005_dmodel128_ff384/best.pt`
  - CSV log: `logs/research/EXP-TIMED-005_dmodel128_ff384.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-005_dmodel128_ff384`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `0f2e9c70a6099e80726e3497062ff299c81c9391`
  - worktree clean: `true`

### EXP-015 Timed Feedforward Width Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `model.dim_feedforward` from `128` to `512`
- kept `d_model=128` and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-006_dmodel128_ff512`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.9853`
- reference timed baseline at `d_model=128`, `dim_feedforward=128`: `3.0486`
- absolute improvement over reference: `0.0633`
- final step reached: `16135`
- elapsed time: `180.01` seconds

Conclusion:

- worse
- against the current timed baseline `2.7713`, this run regressed
- it is not better than the current best timed candidate

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-006_dmodel128_ff512/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-006_dmodel128_ff512/best.pt`
  - CSV log: `logs/research/EXP-TIMED-006_dmodel128_ff512.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-006_dmodel128_ff512`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `0f2e9c70a6099e80726e3497062ff299c81c9391`
  - worktree clean: `true`

### EXP-016 Timed Structure Baseline

Date:

- 2026-03-24

Change vs previous:

- kept the current best structural candidate `d_model=128`, `dim_feedforward=512`, `dropout=0.0`
- used `learning_rate=1e-3` as the comparison point

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-008_dmodel128_ff512_dropout0_lr1e3`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.8007`
- reference current timed baseline: `2.7713`
- absolute delta vs reference: `+0.0294`
- final step reached: `16259`
- elapsed time: `180.01` seconds

Conclusion:

- no clear effect
- this baseline on the current best structure is slightly worse than the best timed score, but it does not clear the `worse` threshold

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-008_dmodel128_ff512_dropout0_lr1e3/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-008_dmodel128_ff512_dropout0_lr1e3/best.pt`
  - CSV log: `logs/research/EXP-TIMED-008_dmodel128_ff512_dropout0_lr1e3.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-008_dmodel128_ff512_dropout0_lr1e3`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `2ea47c94b29342d1fbde196d2d60f49e3571ab77`
  - worktree clean: `true`

### EXP-017 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `training.learning_rate` from `1e-3` to `7.5e-4`
- kept `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-009_dmodel128_ff512_dropout0_lr7p5e4`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.7349`
- reference current timed baseline: `2.7713`
- absolute improvement over reference: `0.0364`
- final step reached: `19631`
- elapsed time: `180.00` seconds

Conclusion:

- useful
- this is the new best timed result in the current search
- the smaller learning rate improved the current best structure

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-009_dmodel128_ff512_dropout0_lr7p5e4/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-009_dmodel128_ff512_dropout0_lr7p5e4/best.pt`
  - CSV log: `logs/research/EXP-TIMED-009_dmodel128_ff512_dropout0_lr7p5e4.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-009_dmodel128_ff512_dropout0_lr7p5e4`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `2ea47c94b29342d1fbde196d2d60f49e3571ab77`
  - worktree clean: `true`

### EXP-018 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `training.learning_rate` from `1e-3` to `1.5e-3`
- kept `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, and the `180` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-010_dmodel128_ff512_dropout0_lr1p5e3`
- `training.max_duration_seconds=180`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.8886`
- reference current timed baseline: `2.7713`
- absolute delta vs reference: `+0.1173`
- final step reached: `20500`
- elapsed time: `180.03` seconds

Conclusion:

- worse
- increasing the learning rate hurt the current best structure
- this setting should not be promoted

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-010_dmodel128_ff512_dropout0_lr1p5e3/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-010_dmodel128_ff512_dropout0_lr1p5e3/best.pt`
  - CSV log: `logs/research/EXP-TIMED-010_dmodel128_ff512_dropout0_lr1p5e3.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-010_dmodel128_ff512_dropout0_lr1p5e3`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `2ea47c94b29342d1fbde196d2d60f49e3571ab77`
  - worktree clean: `true`

### EXP-019 First 300-Second Baseline

Date:

- 2026-03-24

Change vs previous:

- increased the wall-clock budget from `180` seconds to `300` seconds
- kept the current best 180-second structure:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `learning_rate=7.5e-4`

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-001_dmodel128_ff512_dropout0_lr7p5e4`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.6071`
- elapsed time: `300.00` seconds
- final step reached: `33968`

Conclusion:

- useful
- this establishes the first clean 300-second baseline for the current structure
- later 300-second runs should compare against this value, not the older 180-second scores

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-001_dmodel128_ff512_dropout0_lr7p5e4/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-001_dmodel128_ff512_dropout0_lr7p5e4/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-001_dmodel128_ff512_dropout0_lr7p5e4.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-001_dmodel128_ff512_dropout0_lr7p5e4`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `c95181cb4deeea983230067730e541630fe632bb`
  - worktree clean: `true`

### EXP-020 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `training.learning_rate` from `7.5e-4` to `6e-4`
- kept `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, and the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-002_dmodel128_ff512_dropout0_lr6e4`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.5787`
- reference 300-second baseline: `2.6071`
- absolute improvement over reference: `0.0284`
- elapsed time: `300.00` seconds
- final step reached: `34072`

Conclusion:

- no clear effect
- this is the best score in the 300-second batch so far, but the improvement stays just under the `0.03` threshold

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-002_dmodel128_ff512_dropout0_lr6e4/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-002_dmodel128_ff512_dropout0_lr6e4/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-002_dmodel128_ff512_dropout0_lr6e4.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-002_dmodel128_ff512_dropout0_lr6e4`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `c95181cb4deeea983230067730e541630fe632bb`
  - worktree clean: `true`

### EXP-021 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `training.learning_rate` from `7.5e-4` to `9e-4`
- kept `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, and the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-003_dmodel128_ff512_dropout0_lr9e4`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.6211`
- reference 300-second baseline: `2.6071`
- absolute delta vs reference: `+0.0140`
- elapsed time: `300.01` seconds
- final step reached: `34169`

Conclusion:

- no clear effect
- this setting is slightly worse than the 300-second baseline, but not by enough to classify as `worse`

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-003_dmodel128_ff512_dropout0_lr9e4/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-003_dmodel128_ff512_dropout0_lr9e4/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-003_dmodel128_ff512_dropout0_lr9e4.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-003_dmodel128_ff512_dropout0_lr9e4`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `c95181cb4deeea983230067730e541630fe632bb`
  - worktree clean: `true`

### EXP-022 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `training.learning_rate` from `6e-4` to `5.5e-4`
- kept `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, and the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-004_dmodel128_ff512_dropout0_lr55e4`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.6318`
- reference current best 300-second candidate: `2.5787`
- absolute delta vs reference: `+0.0531`
- elapsed time: `300.01` seconds
- final step reached: `29029`

Conclusion:

- worse
- lowering the learning rate below `6e-4` degraded the current best timed candidate

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-004_dmodel128_ff512_dropout0_lr55e4/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-004_dmodel128_ff512_dropout0_lr55e4/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-004_dmodel128_ff512_dropout0_lr55e4.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-004_dmodel128_ff512_dropout0_lr55e4`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `86cec6265d57d1218b311958d23aaa54ac5d9e73`
  - worktree clean: `true`

### EXP-023 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `training.learning_rate` from `5.5e-4` to `6.5e-4`
- kept `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, and the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-005_dmodel128_ff512_dropout0_lr65e4`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.6205`
- reference current best 300-second candidate: `2.5787`
- absolute delta vs reference: `+0.0418`
- elapsed time: `300.01` seconds
- final step reached: `29100`

Conclusion:

- worse
- increasing the learning rate from `5.5e-4` did not recover performance

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-005_dmodel128_ff512_dropout0_lr65e4/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-005_dmodel128_ff512_dropout0_lr65e4/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-005_dmodel128_ff512_dropout0_lr65e4.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-005_dmodel128_ff512_dropout0_lr65e4`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `86cec6265d57d1218b311958d23aaa54ac5d9e73`
  - worktree clean: `true`

### EXP-024 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `training.learning_rate` from `6.5e-4` to `5.75e-4`
- kept `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, and the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-006_dmodel128_ff512_dropout0_lr575e4`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.6391`
- reference current best 300-second candidate: `2.5787`
- absolute delta vs reference: `+0.0604`
- elapsed time: `300.01` seconds
- final step reached: `29336`

Conclusion:

- worse
- this was the weakest of the three new learning-rate settings

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-006_dmodel128_ff512_dropout0_lr575e4/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-006_dmodel128_ff512_dropout0_lr575e4/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-006_dmodel128_ff512_dropout0_lr575e4.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-006_dmodel128_ff512_dropout0_lr575e4`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `86cec6265d57d1218b311958d23aaa54ac5d9e73`
  - worktree clean: `true`

### EXP-025 Timed Batch Size Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `training.batch_size` from `4` to `2`
- kept the current best 300-second candidate structure the same:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `learning_rate=6e-4`
- kept the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-007_dmodel128_ff512_dropout0_lr6e4_bs2`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.8829024005681276`
- reference current best 300-second candidate: `2.5787`
- absolute delta vs reference: `+0.30420240056812764`
- elapsed time: `300.0038280839799` seconds
- final step reached: `25018`

Conclusion:

- worse
- batch size `2` was a clear regression relative to the current best 300-second candidate

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-007_dmodel128_ff512_dropout0_lr6e4_bs2/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-007_dmodel128_ff512_dropout0_lr6e4_bs2/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-007_dmodel128_ff512_dropout0_lr6e4_bs2.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-007_dmodel128_ff512_dropout0_lr6e4_bs2`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `f1352183e540826e1aadb4a7d39885f42c34d390`
  - worktree clean: `true`

### EXP-026 Timed Batch Size Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `training.batch_size` from `2` to `1`
- kept the current best 300-second candidate structure the same:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `learning_rate=6e-4`
- kept the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-008_dmodel128_ff512_dropout0_lr6e4_bs1`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `3.635765951126814`
- reference current best 300-second candidate: `2.5787`
- absolute delta vs reference: `+1.057065951126814`
- elapsed time: `300.0021635420271` seconds
- final step reached: `20473`

Conclusion:

- worse
- batch size `1` was much worse than the current best 300-second candidate

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-008_dmodel128_ff512_dropout0_lr6e4_bs1/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-008_dmodel128_ff512_dropout0_lr6e4_bs1/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-008_dmodel128_ff512_dropout0_lr6e4_bs1.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-008_dmodel128_ff512_dropout0_lr6e4_bs1`
- git metadata recorded for the run:
  - branch: `main`
  - head commit: `f1352183e540826e1aadb4a7d39885f42c34d390`
  - worktree clean: `true`

### EXP-027 Timed Batch Size Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `training.batch_size` from `1` to `8`
- kept the current best 300-second candidate structure the same:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `learning_rate=6e-4`
- kept the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-009_dmodel128_ff512_dropout0_lr6e4_bs8`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.2380989687517285`
- reference current best 300-second candidate: `2.5787`
- absolute delta vs reference: `-0.34060103124827146`
- elapsed time: `300.0047808330273` seconds
- final step reached: `28304`

Conclusion:

- useful
- batch size `8` is a clear improvement over the previous current best 300-second candidate
- this is now the strongest observed 300-second result in the search so far

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-009_dmodel128_ff512_dropout0_lr6e4_bs8/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-009_dmodel128_ff512_dropout0_lr6e4_bs8/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-009_dmodel128_ff512_dropout0_lr6e4_bs8.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-009_dmodel128_ff512_dropout0_lr6e4_bs8`
- git metadata recorded for the run:
  - branch: `main`
  - head_commit: `f1352183e540826e1aadb4a7d39885f42c34d390`
  - worktree clean: `true`

### EXP-028 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- reduced `training.learning_rate` from `6e-4` to `5e-4`
- kept the current best 300-second candidate structure the same:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `batch_size=8`
- kept the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-010_dmodel128_ff512_dropout0_lr5e4_bs8`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.2414858089759946`
- reference current best 300-second candidate: `2.2381`
- absolute delta vs reference: `+0.0033858089759943866`
- elapsed time: `300.0022771669901` seconds
- final step reached: `25293`

Conclusion:

- no clear effect
- this setting is slightly worse than the current best 300-second candidate, but only by a very small margin

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-010_dmodel128_ff512_dropout0_lr5e4_bs8/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-010_dmodel128_ff512_dropout0_lr5e4_bs8/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-010_dmodel128_ff512_dropout0_lr5e4_bs8.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-010_dmodel128_ff512_dropout0_lr5e4_bs8`
- git metadata recorded for the run:
  - branch: `main`
  - head_commit: `f1352183e540826e1aadb4a7d39885f42c34d390`
  - worktree clean: `true`

### EXP-029 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `training.learning_rate` from `6e-4` to `7e-4`
- kept the current best 300-second candidate structure the same:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `batch_size=8`
- kept the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-011_dmodel128_ff512_dropout0_lr7e4_bs8`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.2228779271245003`
- reference current best 300-second candidate: `2.2381`
- absolute delta vs reference: `-0.015222072875499926`
- elapsed time: `300.00551675003953` seconds
- final step reached: `26653`

Conclusion:

- no clear effect
- this is an improvement, but still below the `0.03` threshold

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-011_dmodel128_ff512_dropout0_lr7e4_bs8/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-011_dmodel128_ff512_dropout0_lr7e4_bs8/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-011_dmodel128_ff512_dropout0_lr7e4_bs8.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-011_dmodel128_ff512_dropout0_lr7e4_bs8`
- git metadata recorded for the run:
  - branch: `main`
  - head_commit: `f1352183e540826e1aadb4a7d39885f42c34d390`
  - worktree clean: `true`

### EXP-030 Timed Learning Rate Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `training.learning_rate` from `7e-4` to `8e-4`
- kept the current best 300-second candidate structure the same:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `batch_size=8`
- kept the `300` second wall-clock budget the same

Run:

- config generated by `run_experiment.py`
- `experiment_id=EXP-TIMED-300-012_dmodel128_ff512_dropout0_lr8e4_bs8`
- `training.max_duration_seconds=300`
- `training.num_steps=1000000`

Result:

- best validation loss: `2.214011838659644`
- reference current best 300-second candidate: `2.2381`
- absolute delta vs reference: `-0.024088161340356073`
- elapsed time: `300.00774512498174` seconds
- final step reached: `29027`

Conclusion:

- no clear effect
- this is the strongest result in the local learning-rate sweep, but the improvement is still under the `0.03` threshold

Notes:

- artifacts:
  - latest checkpoint: `artifacts/research/EXP-TIMED-300-012_dmodel128_ff512_dropout0_lr8e4_bs8/latest.pt`
  - best checkpoint: `artifacts/research/EXP-TIMED-300-012_dmodel128_ff512_dropout0_lr8e4_bs8/best.pt`
  - CSV log: `logs/research/EXP-TIMED-300-012_dmodel128_ff512_dropout0_lr8e4_bs8.csv`
  - TensorBoard log dir: `logs/tensorboard/research/EXP-TIMED-300-012_dmodel128_ff512_dropout0_lr8e4_bs8`
- git metadata recorded for the run:
  - branch: `main`
  - head_commit: `f1352183e540826e1aadb4a7d39885f42c34d390`
  - worktree clean: `true`

### EXP-031 Timed Batch-Size Extension Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `training.batch_size` above the current best value `8`
- kept:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `learning_rate=8e-4`
  - `300` second wall-clock budget

Run:

- batch sweep over `10`, `12`, and `16`

Result:

- `batch_size=10` -> `2.2408`
- `batch_size=12` -> `2.2561`
- `batch_size=16` -> `2.2688`
- all three were worse than the current best reference `2.2140`

Conclusion:

- worse
- increasing batch size above `8` did not help under the fixed 300-second budget
- `batch_size=8` should be kept

### EXP-032 Timed Dropout Sweep

Date:

- 2026-03-24

Change vs previous:

- reintroduced small positive dropout values above the current best `0.0`
- kept:
  - `d_model=128`
  - `dim_feedforward=512`
  - `batch_size=8`
  - `learning_rate=8e-4`
  - `300` second wall-clock budget

Run:

- dropout sweep over `0.02`, `0.05`, and `0.08`

Result:

- `dropout=0.02` -> `2.2505`
- `dropout=0.05` -> `2.2716`
- `dropout=0.08` -> `2.3201`
- all three were clear regressions vs `2.2140`

Conclusion:

- worse
- `dropout > 0.0` is not worth continuing in the current best branch

### EXP-033 Timed Feedforward Extension Sweep

Date:

- 2026-03-24

Change vs previous:

- increased `model.dim_feedforward` beyond the current best promoted value `512`
- kept:
  - `d_model=128`
  - `dropout=0.0`
  - `batch_size=8`
  - `learning_rate=8e-4`
  - `300` second wall-clock budget

Run:

- feedforward sweep over `640`, `768`, and `1024`

Result:

- `dim_feedforward=640` -> `2.2217`
- `dim_feedforward=768` -> `2.2205`
- `dim_feedforward=1024` -> `2.2087`
- only `1024` slightly beat the current promoted reference `2.2140`, but only by `0.0053`

Conclusion:

- no clear effect
- `1024` is the best observed value in this sweep, but the gain is too small to confidently promote it over `512`
- this suggests the search is near a plateau

## Change Summary

Useful changes so far:

- switching from fake data to real rd HuggingFace data
- adding periodic validation
- saving the best checkpoint
- training longer from `8000` to `12000` steps
- training longer from `12000` to `16000` steps
- adding CSV and TensorBoard logging for analysis
- increasing `model.d_model` from `64` to `128`
- fixing training random crop so repeated sample visits can expose different windows
- establishing the first wall-clock timed baseline
- establishing the first 300-second timed baseline
- increasing `model.dim_feedforward` from `128` to `384`
- increasing `model.dim_feedforward` from `128` to `512`
- reducing `model.dropout` from `0.1` to `0.0`
- reducing `training.learning_rate` from `1e-3` to `7.5e-4` on the current best structure
- reducing `training.learning_rate` from `7.5e-4` to `6e-4` on the current best 300-second baseline
- reducing `training.learning_rate` from `6e-4` to `5.5e-4` on the current best 300-second candidate
- increasing `training.learning_rate` from `5.5e-4` to `6.5e-4` on the current best 300-second candidate
- increasing `training.batch_size` from `4` to `8` on the current best 300-second candidate
- reducing `training.learning_rate` from `6e-4` to `5e-4` on the current best 300-second candidate
- increasing `training.learning_rate` from `6e-4` to `7e-4` on the current best 300-second candidate
- increasing `training.learning_rate` from `7e-4` to `8e-4` on the current best 300-second candidate

No clear effect yet:

- increasing `max_length` from `256` to `512`
- increasing `model.dim_feedforward` from `128` to `256`
- keeping `d_model=128`, `dim_feedforward=512`, `dropout=0.0`, `learning_rate=1e-3`
- reducing `training.learning_rate` from `7.5e-4` to `9e-4` on the current best 300-second baseline
- reducing `training.learning_rate` from `6.5e-4` to `5.75e-4` on the current best 300-second candidate
- reducing `training.learning_rate` from `6e-4` to `5e-4` on the current best 300-second candidate
- increasing `training.learning_rate` from `6e-4` to `7e-4` on the current best 300-second candidate
- increasing `training.learning_rate` from `7e-4` to `8e-4` on the current best 300-second candidate
- increasing `model.dim_feedforward` from `512` to `1024` on the current best 300-second candidate

Worse changes:

- reducing `training.learning_rate` from `1e-3` to `5e-4`
- increasing `model.num_layers` from `2` to `3`
- increasing `training.learning_rate` from `1e-3` to `1.5e-3` on the current best structure
- increasing `model.dim_feedforward` from `128` to `384` or `512` when `dropout=0.1`
- reducing `training.learning_rate` from `7.5e-4` to `5.5e-4` on the current best 300-second candidate
- increasing `training.learning_rate` from `5.5e-4` to `6.5e-4` on the current best 300-second candidate
- reducing `training.learning_rate` from `6.5e-4` to `5.75e-4` on the current best 300-second candidate
- reducing `training.batch_size` from `4` to `2` on the current best 300-second candidate
- reducing `training.batch_size` from `2` to `1` on the current best 300-second candidate
- increasing `training.batch_size` above `8` on the current best 300-second candidate
- reintroducing `dropout > 0.0` on the current best 300-second candidate

## Next Experiment Candidates

- promoted best stable candidate:
  - `d_model=128`
  - `dim_feedforward=512`
  - `dropout=0.0`
  - `batch_size=8`
  - `learning_rate=8e-4`
  - best validation loss `2.214011838659644`
- best observed but not yet promoted candidate:
  - `d_model=128`
  - `dim_feedforward=1024`
  - `dropout=0.0`
  - `batch_size=8`
  - `learning_rate=8e-4`
  - best validation loss `2.2087`
- search appears close to plateau; stop unless there is a strong reason to spend more budget on sub-0.01 improvements

## Confirmation Pass: ff=512 vs ff=1024

Goal:

- verify whether `dim_feedforward=1024` is stably better than `dim_feedforward=512`
- keep all other settings fixed at the current best stable timed configuration:
  - `d_model=128`
  - `dropout=0.0`
  - `batch_size=8`
  - `learning_rate=8e-4`
  - `num_steps=1000000`
  - `max_duration_seconds=300`

Run summary:

- `EXP-034_ff512_confirm_A`
  - `ff=512`
  - best validation loss: `2.3238879162818193`
  - elapsed: `300.0077770409989`
  - final step: `15491`
  - conclusion: worse
- `EXP-035_ff512_confirm_B`
  - `ff=512`
  - best validation loss: `2.3320015957579017`
  - elapsed: `300.0084552500048`
  - final step: `15475`
  - conclusion: worse
- `EXP-036_ff1024_confirm_A`
  - `ff=1024`
  - best validation loss: `2.30784167163074`
  - elapsed: `300.0092004580074`
  - final step: `15465`
  - conclusion: worse
- `EXP-037_ff1024_confirm_B`
  - `ff=1024`
  - best validation loss: `2.2976198866963387`
  - elapsed: `300.0094164159964`
  - final step: `15026`
  - conclusion: worse

Conclusion:

- `ff=1024` did not show a stable advantage over `ff=512` in this confirmatory pass
- both `ff=512` reruns and both `ff=1024` reruns were worse than the current recommended stable model from the earlier tuned search
- `ff=512` should remain the recommended stable model
- `ff=1024` should not be promoted based on these reruns

## 2026-03-25 Full-Data Update

Goal:

- move from the earlier `huggingface_full` baseline to a stronger full-data recommendation
- test whether sliding-window tuning, length bucketing, and long-context runs can beat the short-context setup

Key results:

- `EXP-FULL-FINAL-005_stride192_dmodel256_ff512_dropout0_lr8e4_bs8_es20`
  - change: `d_model 128 -> 256`
  - best validation loss: `2.3426`
  - conclusion: worse
- `EXP-FULL-FINAL-006_stride192_dmodel128_ff1024_dropout0_lr8e4_bs8_es20`
  - change: `dim_feedforward 512 -> 1024`
  - best validation loss: `2.2797`
  - conclusion: useful
- `EXP-FULL-FINAL-007_stride160_dmodel128_ff1024_dropout0_lr8e4_bs8_es20`
  - change: `sliding_window_stride 192 -> 160`
  - best validation loss: `2.2531`
  - conclusion: useful
- `EXP-FULL-FINAL-008_stride224_dmodel128_ff1024_dropout0_lr8e4_bs8_es20`
  - change: `sliding_window_stride 192 -> 224`
  - best validation loss: `2.2735`
  - conclusion: no clear effect
- `EXP-FULL-FINAL-009_stride160_ff1024_bucketed_dmodel128_lr8e4_bs8_es20`
  - change: enable `length_bucketing`
  - best validation loss: `2.2954`
  - conclusion: worse
- `EXP-FULL-FINAL-010_stride160_dmodel128_ff1024_dropout0_lr8e4_bs8_es20_long`
  - change: longer budget on the current full best
  - best validation loss: `2.2726`
  - conclusion: no clear effect

Current recommended full-data model:

- dataset: `data/huggingface_full`
- tokenizer: `data/tokenizer_full.json`
- `max_length=256`
- sliding window enabled
- `sliding_window_stride=160`
- `d_model=128`
- `dim_feedforward=1024`
- `dropout=0.0`
- `batch_size=8`
- `learning_rate=8e-4`
- best validation loss: `2.2531`

Long-context full-data branch:

- `1024` and `2048` without bucketing were too slow and clearly worse under fixed budget
- bucketing was necessary for long-context throughput:
  - `EXP-FULL-LONGCTX-001_crop1024_ff1024_bs8_nobucket` -> `4.1969`
  - `EXP-FULL-LONGCTX-002_crop1024_ff1024_bs8_bucket` -> `3.2807`
  - `EXP-FULL-LONGCTX-003_crop2048_ff1024_bs4_nobucket` -> `5.2889`
  - `EXP-FULL-LONGCTX-004_crop2048_ff1024_bs4_bucket` -> `4.1662`
- stronger long-context model:
  - `EXP-FULL-LONGCTX-006_crop1024_bucket_dmodel256_ff1024_bs8_smoke` -> `3.0660`
  - `EXP-FULL-LONGCTX-007_crop1024_bucket_dmodel256_ff1024_bs8_long` -> `2.4013`
- conclusion:
  - `1024 + bucketing` is viable
  - but it still does not beat the recommended `256 + sliding` full-data model

## 2026-03-25 rd Update

Goal:

- revisit the rd recommendation with long-context training after full-data experiments showed that long-context only worked well when paired with bucketing and larger capacity

Key results:

- `EXP-RD-LONGCTX-001_crop1024_bucket_dmodel128_ff512_bs8_smoke`
  - change: first `rd` long-context smoke run
  - best validation loss: `3.4546`
  - conclusion: worse
- `EXP-RD-LONGCTX-002_crop1024_bucket_dmodel256_ff1024_bs8_smoke`
  - change: increase long-context model capacity
  - best validation loss: `2.5439`
  - conclusion: useful
- `EXP-RD-LONGCTX-003_crop1024_bucket_dmodel256_ff1024_bs8_long`
  - change: longer budget, but still capped by the old `num_steps=16000`
  - best validation loss: `2.0765`
  - conclusion: useful
- `EXP-RD-LONGCTX-004_crop1024_bucket_dmodel256_ff1024_bs8_fullbudget`
  - change: true full-budget rerun with `num_steps=1000000`
  - best validation loss: `1.9616`
  - conclusion: useful

Current recommended rd model:

- dataset: `data/huggingface`
- tokenizer: `data/tokenizer_rd.json`
- `max_length=1024`
- random crop training with `length_bucketing=true`
- `d_model=256`
- `dim_feedforward=1024`
- `dropout=0.0`
- `batch_size=8`
- `learning_rate=8e-4`
- best validation loss: `1.9616`

Main takeaways:

- on `rd`, long-context training only became competitive after all three changes were combined:
  - longer context
  - length bucketing
  - larger model capacity
- after those changes, the long-context branch beat the earlier short-context rd best (`2.0914`)

## 2026-03-26 rd Follow-up

Goal:

- optimize the new `rd` long-context branch more rigorously
- compare `length_bucketing` vs no bucketing on the same recipe
- test `batch_size = 16`
- add and validate warmup / scheduler support
- check whether dropout is broken or simply unhelpful

Key results:

- `EXP-RD-LONGCTX-027_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_smoke`
  - change: disable `length_bucketing` on the then-best `rd` branch
  - result: `300s`, `5136` steps, best validation loss `2.3004`
  - reference bucketing smoke: `EXP-RD-LONGCTX-013...` at `300s`, `3406` steps, best validation loss `2.7204`
  - conclusion: useful
- `EXP-RD-LONGCTX-032_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_long`
  - change: long-budget follow-up for the no-bucketing branch
  - result: best validation loss `1.8107`
  - conclusion: useful
- `EXP-RD-LONGCTX-028_crop1024_bucket_dmodel256_ff1024_lr6e4_bs16_smoke`
  - change: `batch_size 8 -> 16` on the older bucketing branch
  - result: best validation loss `2.5373`
  - conclusion: useful as a smoke signal
- `EXP-RD-LONGCTX-029_crop1024_bucket_dmodel256_ff1024_lr6e4_bs16_long`
  - change: long-budget follow-up for `batch_size = 16`
  - result: best validation loss `2.0639`
  - conclusion: worse
- `EXP-RD-LONGCTX-030_crop1024_bucket_dmodel256_ff1024_lr6e4_bs8_linearwarmup_smoke`
  - change: add `linear` warmup/decay on the bucketing branch
  - result: best validation loss `2.6237`
  - conclusion: useful
- `EXP-RD-LONGCTX-031_crop1024_bucket_dmodel256_ff1024_lr6e4_bs8_cosinewarmup_smoke`
  - change: add `cosine` warmup/decay on the bucketing branch
  - result: best validation loss `2.6578`
  - conclusion: useful, but worse than linear
- `EXP-RD-LONGCTX-033_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_linearwarmup_smoke`
  - change: move `linear` warmup/decay onto the strongest no-bucketing branch
  - result: best validation loss `2.2893` vs no-scheduler smoke `2.3004`
  - conclusion: no clear effect at smoke budget, but directionally positive
- `EXP-RD-LONGCTX-034_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_linearwarmup_long`
  - change: long-budget validation for `linear` warmup/decay on the strongest branch
  - result: best validation loss `1.8039`
  - conclusion: useful
- `EXP-RD-POSENC-001_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_linearwarmup_learned_smoke`
  - change: replace sinusoidal positional encoding with learned absolute position embeddings
  - result: best validation loss `3.5420`
  - reference smoke baseline on the same branch: `2.2893` with sinusoidal
  - conclusion: worse
- `EXP-RD-POSENC-002_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_linearwarmup_alibi_smoke`
  - change: replace sinusoidal positional encoding with ALiBi
  - result: best validation loss `3.4845`
  - reference smoke baseline on the same branch: `2.2893` with sinusoidal
  - conclusion: worse
- `EXP-RD-POSENC-003_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_linearwarmup_learned_long`
  - change: long-budget follow-up for learned absolute position embeddings
  - result: best validation loss `2.4982`
  - conclusion: worse
- `EXP-RD-POSENC-004_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_linearwarmup_alibi_long`
  - change: long-budget follow-up for ALiBi
  - result: best validation loss `2.6788`
  - conclusion: worse
- `EXP-RD-LONGCTX-021_crop1024_bucket_dmodel256_ff1024_lr6e4_dropout005_bs8_smoke`
  - change: set `dropout = 0.05`
  - result: best validation loss `3.7221`
  - conclusion: worse

Efficiency notes for `length_bucketing`:

- batch-shape benchmark on the same `rd` recipe over the first `512` train batches:
  - no bucketing:
    - average padded input length `950.26`
    - average non-pad tokens per batch `3654.66`
    - average padding fraction `51.87%`
  - bucketing:
    - average padded input length `464.38`
    - average non-pad tokens per batch `3651.72`
    - average padding fraction `2.40%`
- dataloader-only benchmark over `1024` batches:
  - no bucketing:
    - `1483.85` batches/s
    - `5.38M` non-pad tokens/s
  - bucketing:
    - `1543.99` batches/s
    - `5.43M` non-pad tokens/s
- interpretation:
  - bucketing is working as intended at the data-loader level
  - the regression appears in end-to-end MPS training, not in Python-side loading

Dropout investigation:

- code inspection found no obvious dropout bug
- dropout is applied at:
  - embedding output in `TransformerDecoderLM.decode`
  - attention modules
  - FFN hidden activations
  - residual branches in `TransformerDecoderLayer`
- training loop uses `model.train()` for train steps and `model.eval()` for validation
- `tests/test_decoder_pretraining.py` now includes a direct test that dropout changes outputs in train mode and becomes deterministic in eval mode

Current recommended rd model:

- dataset: `data/huggingface`
- tokenizer: `data/tokenizer_rd.json`
- `max_length=1024`
- `length_bucketing=false`
- `d_model=256`
- `num_layers=2`
- `dim_feedforward=1024`
- `dropout=0.0`
- `position_encoding_type=sinusoidal`
- `batch_size=8`
- `learning_rate=6e-4`
- `scheduler=linear`
- `warmup_steps=500`
- `min_lr_ratio=0.1`
- best validation loss: `1.8039`

## 2026-03-27 full Update Based on rd Best

Goal:

- transfer the `rd`-validated training recipe to the current `full` best branch
- test whether `linear` warmup, lower learning rate, and larger width also help on `full`

Key results:

- `EXP-FULL-RDREF-001_sliding160_dmodel128_ff1024_lr6e4_linearwarmup_bs8_smoke`
  - change: keep the old full short-context structure, but switch to `learning_rate=6e-4` with `linear` warmup
  - result: best validation loss `2.4881` vs old `300s` baseline `2.5364`
  - conclusion: useful
- `EXP-FULL-RDREF-002_sliding160_dmodel256_ff1024_lr6e4_linearwarmup_bs8_smoke`
  - change: add `d_model=256` on top of the rd-style optimizer schedule
  - result: best validation loss `2.3922`
  - conclusion: useful
- `EXP-FULL-RDREF-003_sliding160_dmodel256_ff1024_lr6e4_linearwarmup_bs8_long`
  - change: long-budget follow-up for the strongest smoke candidate
  - result: best validation loss `2.1019`
  - conclusion: useful

Current recommended full model:

- dataset: `data/huggingface_full`
- tokenizer: `data/tokenizer_full.json`
- `max_length=256`
- `sliding_window_stride=160`
- `d_model=256`
- `num_layers=2`
- `dim_feedforward=1024`
- `dropout=0.0`
- `batch_size=8`
- `learning_rate=6e-4`
- `scheduler=linear`
- `warmup_steps=500`
- `min_lr_ratio=0.1`
- best validation loss: `2.1019`

Main takeaway:

- the `rd`-validated optimizer schedule transferred cleanly to `full`
- on `full`, the biggest win came from combining that schedule with `d_model=256` on the existing short-context sliding-window branch
- this new short-context full model beats the previous best `2.2531` and also remains clearly better than the best tested `1024 + bucketing` backup branch (`2.4013`)

## 2026-03-27 full Long-Context Check Without Bucketing

Goal:

- test whether the final `rd` long-context recipe also works on `full`
- compare `1024 + no bucketing` against the earlier `1024 + bucketing` full run

Key result:

- `EXP-FULL-RDREF-004_crop1024_nobucket_dmodel256_ff1024_lr6e4_bs8_linearwarmup_long`
  - change: run `full` with `max_length=1024`, `length_bucketing=false`, `d_model=256`, `dim_feedforward=1024`, `learning_rate=6e-4`, `linear` warmup
  - result: best validation loss `2.1592`
  - comparison:
    - better than the earlier `1024 + bucketing` full long-context branch at `2.4013`
    - still worse than the current recommended short-context sliding-window full model at `2.1019`
  - conclusion: useful as the new long-context backup, but not promotable

Main takeaway:

- on `full`, the final `rd` long-context recipe transfers better than the earlier bucketing-based `1024` branch
- however, the recommended full model still remains the `256 + sliding_window_stride=160` branch
