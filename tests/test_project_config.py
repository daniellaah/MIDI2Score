from pathlib import Path

import pytest

from midi2score.config import ProjectConfig, load_project_config


def test_load_project_config_reads_baseline_yaml() -> None:
    config = load_project_config(Path("configs/baseline.yaml"))

    assert isinstance(config, ProjectConfig)
    assert config.model.src_vocab_size == 128
    assert config.data.tgt_vocab_size == 160
    assert config.training.num_steps == 5


def test_load_project_config_requires_mapping_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("model: []\ndata: {}\ntraining: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model"):
        load_project_config(config_path)
