import pytest

from midi2score.data.config import FakeDataConfig
from midi2score.models.config import ModelConfig


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


def test_fake_data_config_rejects_invalid_sequence_range() -> None:
    with pytest.raises(ValueError, match="min_source_length"):
        FakeDataConfig(min_source_length=16, max_source_length=8)
