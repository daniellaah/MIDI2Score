from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, UTC
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from midi2score.config import load_decoder_pretrain_config
from midi2score.data import LanguageModelDataConfig
from midi2score.model import DecoderLanguageModelConfig
from midi2score.research.git_utils import collect_git_metadata, require_clean_git_worktree
from midi2score.train import TrainingConfig, run_decoder_pretraining_loop

_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_ALLOWED_OVERRIDE_FIELDS = {
    "model": {field.name for field in fields(DecoderLanguageModelConfig)},
    "data": {field.name for field in fields(LanguageModelDataConfig)},
    "training": {field.name for field in fields(TrainingConfig)},
}


@dataclass(slots=True)
class ExperimentPaths:
    config_path: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    csv_log_path: Path
    tensorboard_log_dir: Path
    summary_path: Path


def parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def build_experiment_config(
    *,
    base_config_path: str | Path,
    experiment_id: str,
    overrides: dict[str, Any],
    output_root: str | Path = ".",
) -> tuple[Path, ExperimentPaths, dict[str, Any]]:
    if not _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(
            "experiment_id may only contain letters, numbers, dot, underscore, and hyphen."
        )

    output_root_path = Path(output_root)
    paths = _build_experiment_paths(output_root_path, experiment_id)
    raw_config = _load_raw_config(base_config_path)
    resolved_config = deepcopy(raw_config)
    _apply_overrides(resolved_config, overrides)
    _inject_standardized_output_paths(
        resolved_config,
        paths=paths,
        clear_resume="training.resume_checkpoint_path" not in overrides,
    )

    paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    return paths.config_path, paths, resolved_config


def run_research_experiment(
    *,
    base_config_path: str | Path,
    experiment_id: str,
    overrides: dict[str, Any],
    output_root: str | Path = ".",
    repo_root: str | Path = ".",
    note: str | None = None,
    reference_best_validation_loss: float | None = None,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    config_path, paths, resolved_config = build_experiment_config(
        base_config_path=base_config_path,
        experiment_id=experiment_id,
        overrides=overrides,
        output_root=output_root,
    )
    git_metadata = (
        require_clean_git_worktree(repo_root)
        if require_clean_git
        else collect_git_metadata(repo_root)
    )
    started_at = datetime.now(UTC)
    project_config = load_decoder_pretrain_config(config_path)
    result = run_decoder_pretraining_loop(
        project_config.model,
        project_config.data,
        project_config.training,
    )
    finished_at = datetime.now(UTC)

    summary = {
        "experiment_id": experiment_id,
        "base_config_path": str(Path(base_config_path).resolve()),
        "resolved_config_path": str(config_path.resolve()),
        "note": note,
        "overrides": overrides,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "best_validation_loss": result.best_validation_loss,
        "final_step": result.final_step,
        "elapsed_seconds": result.elapsed_seconds,
        "stopped_due_to_time_budget": result.stopped_due_to_time_budget,
        "stopped_due_to_early_stopping": result.stopped_due_to_early_stopping,
        "device": result.device,
        "resumed_from_checkpoint": result.resumed_from_checkpoint,
        "optimizer_state_loaded": result.optimizer_state_loaded,
        "checkpoint_path": str(paths.checkpoint_path.resolve()),
        "best_checkpoint_path": str(paths.best_checkpoint_path.resolve()),
        "csv_log_path": str(paths.csv_log_path.resolve()),
        "tensorboard_log_dir": str(paths.tensorboard_log_dir.resolve()),
        "summary_path": str(paths.summary_path.resolve()),
        "resolved_config": resolved_config,
        "git": git_metadata,
    }
    if reference_best_validation_loss is not None and result.best_validation_loss is not None:
        summary["reference_best_validation_loss"] = reference_best_validation_loss
        summary["delta_to_reference"] = (
            result.best_validation_loss - reference_best_validation_loss
        )

    paths.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _build_experiment_paths(output_root: Path, experiment_id: str) -> ExperimentPaths:
    artifacts_dir = output_root / "artifacts" / "research" / experiment_id
    logs_dir = output_root / "logs" / "research"
    tensorboard_dir = output_root / "logs" / "tensorboard" / "research" / experiment_id
    config_dir = output_root / "configs" / "research"
    return ExperimentPaths(
        config_path=config_dir / f"{experiment_id}.yaml",
        checkpoint_path=artifacts_dir / "latest.pt",
        best_checkpoint_path=artifacts_dir / "best.pt",
        csv_log_path=logs_dir / f"{experiment_id}.csv",
        tensorboard_log_dir=tensorboard_dir,
        summary_path=artifacts_dir / "summary.json",
    )


def _load_raw_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    if not isinstance(raw_config, dict):
        raise ValueError("Top-level config must be a mapping with model/data/training sections.")
    return raw_config


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Override key {dotted_key!r} must target a nested section such as model.d_model."
            )

        cursor: dict[str, Any] = config
        for part in parts[:-1]:
            next_value = cursor.get(part)
            if not isinstance(next_value, dict):
                raise ValueError(f"Unknown config section for override {dotted_key!r}.")
            cursor = next_value

        leaf_key = parts[-1]
        root_section = parts[0]
        if leaf_key not in cursor and leaf_key not in _ALLOWED_OVERRIDE_FIELDS.get(root_section, set()):
            raise ValueError(f"Unknown config field for override {dotted_key!r}.")
        cursor[leaf_key] = value


def _inject_standardized_output_paths(
    config: dict[str, Any],
    *,
    paths: ExperimentPaths,
    clear_resume: bool,
) -> None:
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("Config section 'training' must be a mapping.")

    training["save_checkpoint_path"] = str(paths.checkpoint_path.resolve())
    training["save_best_checkpoint_path"] = str(paths.best_checkpoint_path.resolve())
    training["csv_log_path"] = str(paths.csv_log_path.resolve())
    training["tensorboard_log_dir"] = str(paths.tensorboard_log_dir.resolve())
    if clear_resume:
        training["resume_checkpoint_path"] = None
