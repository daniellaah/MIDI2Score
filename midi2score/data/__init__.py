"""Data utilities for MIDI2Score."""

from midi2score.data.config import FakeDataConfig, FakeLanguageModelDataConfig
from midi2score.data.fake_dataset import (
    FakeSeq2SeqDataset,
    Seq2SeqBatch,
    build_fake_dataloader,
    collate_seq2seq_batch,
)
from midi2score.data.language_model_dataset import (
    FakeLanguageModelDataset,
    LanguageModelBatch,
    build_fake_language_model_dataloader,
    collate_language_model_batch,
)

__all__ = [
    "FakeDataConfig",
    "FakeLanguageModelDataConfig",
    "FakeLanguageModelDataset",
    "FakeSeq2SeqDataset",
    "LanguageModelBatch",
    "Seq2SeqBatch",
    "build_fake_dataloader",
    "build_fake_language_model_dataloader",
    "collate_language_model_batch",
    "collate_seq2seq_batch",
]
