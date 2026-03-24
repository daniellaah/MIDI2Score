# MIDI2Score Dev Notes

Last updated: 2026-03-23

## 1. Current Scope

The project is currently focused on **decoder pretraining only**.

Active goal:

- pretrain a MusicXML / LMX decoder language model on the real tokenized dataset in `data/`

Not in active scope right now:

- seq2seq training
- MIDI tokenization integration
- real paired MIDI -> LMX fine-tuning
- evaluation metrics

The seq2seq model code is still kept in the repository because it will be needed later for fine-tuning, but the active training pipeline is decoder pretraining only.

## 2. Real Data That Exists Now

There are three important data layers in the repository.

### 2.1 Preprocessed Triplets

Path:

- `data/PDMX_preprocessed_rd/`

This directory contains per-part data triples:

- `lmx/`
- `mxl/`
- `midi/`

Metadata files:

- `dataset_info.csv`
- `dataset_info_with_partitions.csv`
- `error_dataset_info.csv`

Key point:

- each row in `dataset_info_with_partitions.csv` corresponds to a **part-level** sample
- each row stores paths to the aligned `lmx`, `musicxml`, and `midi` files
- the `partition` column contains `training / validation / test`

### 2.2 HuggingFace Reduced Dataset

Path:

- `data/huggingface/`

This is the reduced on-disk `DatasetDict` version for decoder pretraining.

Splits:

- `training`
- `validation`
- `test`

Feature schema:

- `input_ids: List[int32]`

### 2.3 HuggingFace Full Dataset

Path:

- `data/huggingface_full/`

This is the larger version of the same decoder-pretraining dataset.

### 2.4 Tokenizers

Paths:

- `data/tokenizer_rd.json`
- `data/tokenizer_full.json`

Important facts verified locally:

- vocab size: `5000`
- `[PAD] -> 0`
- `[BOS] -> 1`
- `[EOS] -> 2`

## 3. What The Current Training Pipeline Uses

The active pretraining pipeline uses the already tokenized HuggingFace dataset.

It does **not** re-tokenize MusicXML at runtime.

Data flow:

1. load a split from `data/huggingface` or `data/huggingface_full`
2. read `input_ids`
3. optionally crop / truncate to `max_length`
4. collate into:
   - `input_tokens`
   - `output_tokens`
   - `padding_mask`
5. train a decoder-only language model with next-token prediction

So the current setup is:

- source of truth for pretraining: `input_ids` in HuggingFace dataset
- tokenizer is used for validation / metadata consistency
- target model predicts the next LMX token

## 4. Relevant Code Layout

### 4.1 Data

- `midi2score/data/config.py`
  - `LanguageModelDataConfig`
  - points to a real on-disk HuggingFace dataset

- `midi2score/data/language_model_dataset.py`
  - `HuggingFaceLanguageModelDataset`
  - `LanguageModelBatch`
  - collate logic
  - dataloader builder

### 4.2 Models

- `midi2score/models/decoder_config.py`
  - `DecoderLanguageModelConfig`

- `midi2score/models/decoder_lm.py`
  - `TransformerDecoderLM`

- `midi2score/models/modules.py`
  - shared decoder blocks
  - positional encoding
  - causal mask

- `midi2score/models/transformer.py`
  - future seq2seq model
  - still contains decoder weight loading for later fine-tuning

### 4.3 Training

- `midi2score/trainers/config.py`
  - `TrainingConfig`
  - includes validation scheduling and best-checkpoint options

- `midi2score/trainers/device.py`
  - device selection

- `midi2score/trainers/pretrain_loop.py`
  - real decoder pretraining loop
  - periodic validation
  - best-checkpoint saving

- `midi2score/trainers/checkpoint.py`
  - checkpoint saving

- `midi2score/trainers/logging.py`
  - CSV logging
  - TensorBoard logging

- `midi2score/trainers/resume.py`
  - checkpoint resume helpers

### 4.5 Research / Autotuning

- `midi2score/research/experiment_runner.py`
  - standardized experiment config generation
  - standardized output paths for research runs
  - experiment summary JSON writing
  - git metadata capture for experiments

- `run_experiment.py`
  - command-line entrypoint for one controlled tuning run

- `program.md`
  - instructions for an agent doing autotuning

- `exp.md`
  - experiment history and conclusions

### 4.4 Config + Entrypoint

- `midi2score/config.py`
  - YAML loading for decoder pretraining

- `configs/pretrain_baseline.yaml`
  - default real-data pretraining config

- `pretrain.py`
  - command-line entrypoint

## 5. Important Config Behavior

`LanguageModelDataConfig` controls:

- which dataset to read
- which split to use
- max sequence length
- whether long sequences are randomly cropped in training
- tokenizer path for vocab checks

`DecoderLanguageModelConfig` controls:

- decoder architecture
- vocab size
- special token ids
- max model length

`TrainingConfig` controls:

- batch size
- learning rate
- number of steps
- optional wall-clock budget in seconds
- validation frequency
- optional number of validation batches
- device
- optional last-checkpoint output path
- optional best-checkpoint output path
- optional resume checkpoint path
- optional CSV log path
- optional TensorBoard log directory

There is an explicit setup validation step before training starts.

It checks:

- model and data special-token ids match
- model max length is at least data max length
- tokenizer vocab size matches model vocab size
- dataset path exists

## 6. Current Default Baseline

Default config:

- file: `configs/pretrain_baseline.yaml`
- dataset: `data/huggingface`
- vocab size: `5000`
- model length: `256`
- batch size: `4`
- steps: `16000`
- validate every: `500` steps
- validation batches per eval: `64`

Best verified baseline result so far:

- best validation loss `3.6623`

There is also a tracked comparison run with `max_length=512`, and it is currently recorded as `no clear effect` in `exp.md`.

## 7. How To Run

Install environment:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest -q
```

Run decoder pretraining:

```bash
uv run python pretrain.py --config configs/pretrain_baseline.yaml
```

Run one standardized research experiment:

```bash
uv run python run_experiment.py --base-config configs/pretrain_baseline.yaml --experiment-id my-exp --set training.num_steps=1000000 --set training.max_duration_seconds=180 --set model.dropout=0.0
```

Default checkpoint output:

```text
artifacts/pretrained_decoder_baseline.pt
```

Default best-checkpoint output:

```text
artifacts/pretrained_decoder_best.pt
```

Research experiment outputs:

```text
configs/research/<experiment_id>.yaml
artifacts/research/<experiment_id>/
logs/research/<experiment_id>.csv
logs/tensorboard/research/<experiment_id>/
```

## 8. What Was Removed

The old fake-data development path has been removed.

Removed categories:

- fake seq2seq datasets
- fake decoder-LM dataset
- fake-data training entrypoints
- fake-data tests

Reason:

- real data is now available
- keeping both fake and real paths would create confusion
- the current project scope is decoder pretraining on real data

## 9. What Still Exists But Is Not Active

These are still present, but not the current focus:

- `TransformerSeq2Seq`
- decoder-to-seq2seq weight transfer logic

These remain because the longer-term project plan is still:

1. decoder pretraining
2. later seq2seq fine-tuning with the pretrained decoder

But step 2 is intentionally deferred.

## 10. Recommended Reading Order For Another Agent

If another agent needs to get up to speed quickly, read in this order:

1. `docs/dev.md`
2. `configs/pretrain_baseline.yaml`
3. `midi2score/config.py`
4. `midi2score/data/config.py`
5. `midi2score/data/language_model_dataset.py`
6. `midi2score/models/decoder_config.py`
7. `midi2score/models/modules.py`
8. `midi2score/models/decoder_lm.py`
9. `midi2score/trainers/pretrain_loop.py`
10. `pretrain.py`

Then inspect tests:

- `tests/test_config.py`
- `tests/test_project_config.py`
- `tests/test_decoder_pretraining.py`
- `tests/test_pretrain_script.py`

## 11. Next Likely Development Steps

If work continues on decoder pretraining, the most natural next steps are:

1. keep running single-variable experiments and record them in `exp.md`
2. tune width / depth / dropout / learning rate with `run_experiment.py`
3. consider a separate full-dataset baseline once rd tuning stabilizes
4. establish and refresh timed baselines when the training code changes materially

If work later returns to seq2seq, the decoder-pretraining artifacts are already being saved in a form that can be reused.
