from pathlib import Path

import pytest

from midi2score.config import PretrainConfig, load_pretrain_config


def test_load_pretrain_config_reads_pretrain_yaml() -> None:
    config = load_pretrain_config(Path("configs/pretrain.yaml"))

    assert isinstance(config, PretrainConfig)
    assert config.model.vocab_size == 5000
    assert config.data.dataset_path == "data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full_dataset"
    assert config.training.num_steps == 1000000
    assert config.training.resume_checkpoint_path is None


def test_load_pretrain_config_requires_mapping_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("model: []\ndata: {}\ntraining: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model"):
        load_pretrain_config(config_path)
