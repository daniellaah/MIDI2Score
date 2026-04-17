from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from pretrain.data import LmxDataConfig
from pretrain.decoder import DecoderLanguageModelConfig
from pretrain.trainer import TrainingConfig


@dataclass(slots=True)
class PretrainConfig:
    model: DecoderLanguageModelConfig
    data: LmxDataConfig
    training: TrainingConfig


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("Top-level config must be a mapping with model/data/training sections.")

    sections = {}
    for name in ("model", "data", "training"):
        section = raw_config.get(name)
        if not isinstance(section, dict):
            raise ValueError(f"Config section {name!r} must be a mapping.")
        sections[name] = section

    return PretrainConfig(
        model=DecoderLanguageModelConfig(**sections["model"]),
        data=LmxDataConfig(**sections["data"]),
        training=TrainingConfig(**sections["training"]),
    )
