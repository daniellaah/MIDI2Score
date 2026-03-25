import pytest

from midi2score.data.config import LanguageModelDataConfig
from midi2score.models.config import ModelConfig
from midi2score.trainers import TrainingConfig


def test_model_config_validate() -> None:
    cfg = ModelConfig(
        src_vocab_size=128,
        tgt_vocab_size=256,
        d_model=256,
        nhead=8,
    )

    cfg.validate()
    assert cfg.head_dim == 32


def test_model_config_rejects_invalid_attention_shape() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            src_vocab_size=128,
            tgt_vocab_size=256,
            d_model=250,
            nhead=8,
        )


def test_language_model_data_config_rejects_invalid_split() -> None:
    with pytest.raises(ValueError, match="split"):
        LanguageModelDataConfig(dataset_path="data/huggingface", split="train")


def test_training_config_rejects_non_positive_time_budget() -> None:
    with pytest.raises(ValueError, match="max_duration_seconds"):
        TrainingConfig(max_duration_seconds=0.0)


def test_training_config_rejects_non_positive_early_stopping_patience() -> None:
    with pytest.raises(ValueError, match="early_stopping_patience"):
        TrainingConfig(early_stopping_patience=0, eval_every=1)


def test_training_config_requires_eval_for_early_stopping() -> None:
    with pytest.raises(ValueError, match="eval_every > 0"):
        TrainingConfig(early_stopping_patience=3, eval_every=0)
