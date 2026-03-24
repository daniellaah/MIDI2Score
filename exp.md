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

No clear effect yet:

- increasing `max_length` from `256` to `512`

Worse changes:

- none recorded yet

## Next Experiment Candidates

- increase `dim_feedforward` while keeping `d_model=128`
- test a smaller learning rate at `d_model=128`
- keep changing one variable at a time and compare against `3.0486` for the current timed baseline
