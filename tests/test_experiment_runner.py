from __future__ import annotations

import json
from pathlib import Path

import yaml

from midi2score.research import (
    build_experiment_config,
    parse_override_value,
    run_research_experiment,
)


def test_parse_override_value_handles_common_types() -> None:
    assert parse_override_value("128") == 128
    assert parse_override_value("0.001") == 0.001
    assert parse_override_value("true") is True
    assert parse_override_value("false") is False
    assert parse_override_value("none") is None
    assert parse_override_value("data/huggingface") == "data/huggingface"


def test_build_experiment_config_writes_standardized_paths(tmp_path: Path) -> None:
    config_path, paths, resolved_config = build_experiment_config(
        base_config_path=Path("configs/pretrain_baseline.yaml"),
        experiment_id="unit-test-exp",
        overrides={
            "training.num_steps": 1,
            "training.eval_every": 1,
            "training.num_eval_batches": 1,
        },
        output_root=tmp_path,
    )

    assert config_path.exists()
    assert resolved_config["training"]["resume_checkpoint_path"] is None
    assert resolved_config["training"]["save_checkpoint_path"] == str(
        paths.checkpoint_path.resolve()
    )
    assert resolved_config["training"]["csv_log_path"] == str(paths.csv_log_path.resolve())

    with config_path.open("r", encoding="utf-8") as handle:
        persisted = yaml.safe_load(handle)
    assert persisted["training"]["save_best_checkpoint_path"] == str(
        paths.best_checkpoint_path.resolve()
    )


def test_run_research_experiment_writes_summary_and_outputs(tmp_path: Path) -> None:
    summary = run_research_experiment(
        base_config_path=Path("configs/pretrain_baseline.yaml"),
        experiment_id="integration-test-exp",
        overrides={
            "training.num_steps": 1,
            "training.eval_every": 1,
            "training.num_eval_batches": 1,
            "training.batch_size": 2,
            "training.device": "cpu",
        },
        output_root=tmp_path,
        note="runner smoke test",
        reference_best_validation_loss=3.6623,
    )

    summary_path = Path(summary["summary_path"])
    assert summary_path.exists()
    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["best_checkpoint_path"]).exists()
    assert Path(summary["csv_log_path"]).exists()
    assert Path(summary["tensorboard_log_dir"]).exists()
    assert summary["experiment_id"] == "integration-test-exp"
    assert summary["note"] == "runner smoke test"
    assert summary["best_validation_loss"] is not None
    assert "delta_to_reference" in summary

    with summary_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["resolved_config"]["training"]["device"] == "cpu"
