# Decoder Pretraining

Last updated: 2026-04-08

## Overview

This document summarizes the decoder pretraining pipeline for `rd`.

## Baseline

- Current baseline is still under active tuning.
- Fill in the accepted baseline only after it is finalized.

## Design Choices and References

| Component | Current Choice | Why We Use It | Reference |
| --- | --- | --- | --- |
| Data source | TBD | TBD | TBD |
| Tokenization | TBD | TBD | TBD |
| Model family | TBD | TBD | TBD |
| Objective | TBD | TBD | TBD |
| Dynamic padding | TBD | TBD | TBD |
| Training windowing | TBD | TBD | TBD |
| Validation windowing | TBD | TBD | TBD |
| Optimizer | TBD | TBD | TBD |
| LR schedule | TBD | TBD | TBD |

## End-to-End Pipeline

- Dataset preparation
- Window construction
- Batch collation
- Decoder pretraining
- Validation and reporting

## Data and Sequence Handling

### Dataset

- TBD

### Special Tokens

- TBD

### Sequence Policy

- TBD

### Tensor Shapes

- TBD

## Model Architecture

### High-Level Structure

- TBD

### Decoder Specification

- TBD

### Decoder Layer

- TBD

## Training and Evaluation

### Training Objective

- TBD

### Optimization

- TBD

### Evaluation Metrics

- TBD

## Files

- config:
- experiment log:
- training entry:
- training loop:
- data pipeline:
- model:

## Notes

- Keep this document focused on stable methodology only.
- Put tuning history in `docs/decoder_pretrain_exp.md`.
