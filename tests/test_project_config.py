from pathlib import Path

import pytest

from midi2score.config import DecoderPretrainProjectConfig, load_decoder_pretrain_config


def test_load_decoder_pretrain_config_reads_baseline_yaml() -> None:
    config = load_decoder_pretrain_config(Path("configs/pretrain_baseline.yaml"))

    assert isinstance(config, DecoderPretrainProjectConfig)
    assert config.model.vocab_size == 5000
    assert config.data.dataset_path == "data/huggingface"
    assert config.training.num_steps == 16000
    assert config.training.resume_checkpoint_path is None


def test_load_decoder_pretrain_config_requires_mapping_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("model: []\ndata: {}\ntraining: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model"):
        load_decoder_pretrain_config(config_path)
