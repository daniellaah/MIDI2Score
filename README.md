# MIDI2Score Pretrain

This repository contains the current decoder pretraining pipeline for linearized MusicXML.

## Set up

This project uses `uv` for dependency management.

To set up the environment, run:

```bash
uv sync
```

## Training set

### Bar-aware chunk

The training set is generated offline from raw LMX using bar-aware chunking at explicit `measure` boundaries, following the same bar-structured symbolic-music modeling direction used in MuPT and SymPAC ([MuPT](https://arxiv.org/abs/2404.06393), [SymPAC](https://arxiv.org/abs/2409.03055)).

- chunk at bar boundaries
- `max_length = 1024`
- `overlap_bars = 2`
- `BOS` only on the first chunk of a piece
- `EOS` only on the last chunk of a piece
- chunk size constrained by final encoded length rather than raw LMX token count ([Hugging Face fixed-length perplexity guidance](https://huggingface.co/docs/transformers/perplexity))
- tokenizer pipeline `LMX token -> chr(33 + index_in_ALL_TOKENS) -> BPE`, aligned with the symbolic-music BPE direction described in *Byte Pair Encoding for Symbolic Music* ([Byte Pair Encoding for Symbolic Music](https://arxiv.org/abs/2301.11975), [linearized-musicxml](https://pypi.org/project/linearized-musicxml/))
- offline-generated chunks consumed directly by the training loader

Run to generate bar-aware training dataset:

```bash
uv run python -m pretrain.bar_aware_chunk \
  --dataset-root data/PDMX_preprocessed_rd \
  --partition training \
  --tokenizer-path data/tokenizer_rd.json \
  --base-dataset-path data/huggingface \
  --output-dataset-path data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full_dataset \
  --output-jsonl-path data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full.jsonl \
  --max-length 1024 \
  --overlap-bars 2
```

### Length bucketing and dynamic batching

Length bucketing means grouping training examples with similar sequence lengths before batching. Dynamic batching means limiting each batch by a token budget instead of a fixed example count. In practice, the two are used together: first reorder examples by similar lengths, then keep adding examples until `max_tokens_per_batch` is reached. This is the same general idea used by common max-token and length-grouped samplers in AllenNLP, fairseq, and Hugging Face Trainer.

Why it accelerates training:

- bar-aware chunking produces a wide length distribution rather than mostly fixed-length samples
- without bucketing, short and long samples land in the same batch and force excessive padding
- with bucketing, padding waste drops, so more of each step is spent on valid tokens instead of padded tokens
- with dynamic batching, the training loop keeps batch token count near a stable budget even when sample lengths vary

In this project, this is part of the main recipe rather than an optional micro-optimization, because bar-aware chunking creates a broad length distribution with many short and medium-length chunks plus a smaller set of near-max-length chunks. Without bucketing, that uneven distribution turns directly into padding waste.

Current settings:

- `length_bucketing = true`
- `max_tokens_per_batch = 16384`
- `pad_to_length_multiple = 64`

[`analysis/train_data_analysis.ipynb`](analysis/train_data_analysis.ipynb) shows the underlying length distribution and writes the figure to `analysis/img/current_train_chunk_length_distribution.png`.

Use [`analysis/length_bucketing_analysis.ipynb`](analysis/length_bucketing_analysis.ipynb) to benchmark the speed difference between `length_bucketing = true` and `false`.

## Validation set

The validation recipe is fixed:

- sliding-window validation
- `max_length = 1024`
- `stride = 512`
- overlapping targets counted once through `loss_mask` ([Hugging Face fixed-length perplexity guidance](https://huggingface.co/docs/transformers/perplexity))
- `eval_batch_size = 16`
- full validation uses `num_eval_batches = null`

Evaluation code:

- metrics implementation: [`pretrain/evaluate.py`](pretrain/evaluate.py)
- evaluation entry: [`run_pretrain.py`](run_pretrain.py)

Evaluate a checkpoint on the validation split:

```bash
uv run python run_pretrain.py \
  --eval-mode \
  --config configs/pretrain.yaml \
  --checkpoint artifacts/runs/<run_dir>/best.pt \
  --split validation
```

Evaluate a checkpoint on the test split:

```bash
uv run python run_pretrain.py \
  --eval-mode \
  --config configs/pretrain.yaml \
  --checkpoint artifacts/runs/<run_dir>/best.pt \
  --split test
```

## Model

The current model is a decoder-only Transformer language model for next-token prediction over linearized MusicXML token ids.

Core settings:

- `vocab_size = 5000`
- `d_model = 512`
- `nhead = 8`
- `num_layers = 4`
- `dim_feedforward = 2048`
- `dropout = 0.05`
- activation: `swiglu`
- normalization: `rmsnorm`
- residual layout: `pre_norm`
- positional encoding: `sinusoidal`
- embedding/output weight tying: enabled
- `max_length = 1024`

Code:

- model: [`pretrain/decoder.py`](pretrain/decoder.py)

## Training

The main training config is:

- [`configs/pretrain.yaml`](configs/pretrain.yaml)

Core settings:

- optimizer: `AdamW`
- `learning_rate = 6e-4`
- `weight_decay = 0.1`
- `beta1 = 0.9`
- `beta2 = 0.95`
- gradient clipping: `2.0`
- scheduler: cosine decay
- `warmup_steps = 500`
- `min_lr_ratio = 0.1`
- `batch_size = 64`
- default wall-clock budget: `7200s`

Run training:

```bash
uv run python run_pretrain.py --config configs/pretrain.yaml
```

Run a short verification job:

```bash
uv run python run_pretrain.py --config configs/tmp/pretrain_600s_verify.yaml
```

Code:

- entrypoint: [`run_pretrain.py`](run_pretrain.py)
- config loading: [`pretrain/config.py`](pretrain/config.py)
- training loop: [`pretrain/trainer.py`](pretrain/trainer.py)
- experiment history: [`exp/decoder_pretrain_exp.md`](exp/decoder_pretrain_exp.md)

## File structure

```text
.
├── .gitignore
├── README.md
├── analysis/
│   ├── img/
│   ├── length_bucketing_analysis.ipynb
│   └── train_data_analysis.ipynb
├── configs/
│   ├── lb_off.yaml
│   ├── lb_on.yaml
│   └── pretrain.yaml
├── exp/
│   └── decoder_pretrain_exp.md
├── pretrain/
│   ├── __init__.py
│   ├── bar_aware_chunk.py
│   ├── config.py
│   ├── data.py
│   ├── decoder.py
│   ├── evaluate.py
│   └── trainer.py
├── pyproject.toml
├── run_pretrain.py
├── tests/
│   ├── test_bar_aware_chunk.py
│   ├── test_config.py
│   ├── test_decoder_pretraining.py
│   ├── test_pretrain_script.py
│   └── test_project_config.py
└── uv.lock
```

## Reference

- Transformer decoder: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- RMSNorm: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- SwiGLU: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Weight tying: [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- AdamW: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Fixed-length evaluation: [Hugging Face perplexity guide](https://huggingface.co/docs/transformers/perplexity)
- Symbolic music BPE: [Byte Pair Encoding for Symbolic Music](https://arxiv.org/abs/2301.11975)
- Linearized MusicXML package: [linearized-musicxml](https://pypi.org/project/linearized-musicxml/)
- Bar-structured symbolic music modeling: [MuPT](https://arxiv.org/abs/2404.06393)
- Bar-structured symbolic music modeling: [SymPAC](https://arxiv.org/abs/2409.03055)
- Hugging Face. *Transformers Trainer documentation*.  
  https://huggingface.co/docs/transformers/main_classes/trainer

- NVIDIA. *NVIDIA NeMo Framework User Guide: Datasets and bucketing*.  
  https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html

- SpeechBrain. *Dynamic batching tutorial*.  
  https://speechbrain.readthedocs.io/en/stable/tutorials/advanced/dynamic-batching.html

- PyTorch. *Performance tuning guide*.  
  https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

- [AllenNLP MaxTokensBatchSampler](https://docs.allennlp.org/main/api/data/samplers/max_tokens_batch_sampler/)
- [fairseq batch_by_size](https://fairseq.readthedocs.io/en/v0.10.2/_modules/fairseq/data/fairseq_dataset.html)
