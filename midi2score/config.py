from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from midi2score.data import FakeDataConfig
from midi2score.data import FakeLanguageModelDataConfig
from midi2score.models import DecoderLanguageModelConfig, ModelConfig
from midi2score.trainers import TrainingConfig


@dataclass(slots=True)
class ProjectConfig:
    model: ModelConfig
    data: FakeDataConfig
    training: TrainingConfig


@dataclass(slots=True)
class DecoderPretrainProjectConfig:
    model: DecoderLanguageModelConfig
    data: FakeLanguageModelDataConfig
    training: TrainingConfig


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("Top-level config must be a mapping with model/data/training sections.")

    # Keep the YAML-to-dataclass mapping in one place so training code can stay
    # focused on model execution rather than parsing details.
    model_section = _get_section(raw_config, "model")
    data_section = _get_section(raw_config, "data")
    training_section = _get_section(raw_config, "training")

    return ProjectConfig(
        model=ModelConfig(**model_section),
        data=FakeDataConfig(**data_section),
        training=TrainingConfig(**training_section),
    )


def load_decoder_pretrain_config(path: str | Path) -> DecoderPretrainProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("Top-level config must be a mapping with model/data/training sections.")

    model_section = _get_section(raw_config, "model")
    data_section = _get_section(raw_config, "data")
    training_section = _get_section(raw_config, "training")

    return DecoderPretrainProjectConfig(
        model=DecoderLanguageModelConfig(**model_section),
        data=FakeLanguageModelDataConfig(**data_section),
        training=TrainingConfig(**training_section),
    )


def _get_section(raw_config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = raw_config.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"Config section {section_name!r} must be a mapping.")
    return section
