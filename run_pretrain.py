from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from midi2score.config import load_pretrain_config
from midi2score.train import run_decoder_pretraining_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run decoder pretraining.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain.yaml"),
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory where timestamped run folders are created.",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional short note stored in summary.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = _create_run_dir(args.runs_root)
    run_dir_resolved = run_dir.resolve()

    resolved_config = _load_raw_config(args.config)
    _inject_run_output_paths(resolved_config, run_dir)

    config_path = run_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    started_at = datetime.now().astimezone()
    project_config = load_pretrain_config(config_path)
    result = run_decoder_pretraining_loop(
        project_config.model,
        project_config.data,
        project_config.training,
    )
    finished_at = datetime.now().astimezone()

    summary_path = run_dir / "summary.json"
    summary = {
        "run_dir": str(run_dir_resolved),
        "config_path": str(config_path.resolve()),
        "note": args.note,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "config": {
            "model": project_config.model.to_dict(),
            "data": project_config.data.to_dict(),
            "training": project_config.training.to_dict(),
        },
        "result": asdict(result),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"finished {len(result.losses)} pretraining steps on {result.device}")
    print(f"run_dir={run_dir_resolved}")
    print(
        f"final_step={result.final_step} elapsed_seconds={result.elapsed_seconds:.2f} "
        f"stopped_due_to_time_budget={result.stopped_due_to_time_budget}"
    )
    print(
        f"average_step_time_seconds={result.average_step_time_seconds:.4f} "
        f"average_tokens_per_second={result.average_tokens_per_second:.1f}"
    )
    if result.mps_peak_memory_bytes is not None:
        print(f"mps_peak_memory_mib={result.mps_peak_memory_bytes / (1024 * 1024):.1f}")
    if result.best_validation_loss is not None:
        print(f"best validation loss {result.best_validation_loss:.4f}")
    if result.target_validation_loss is not None:
        print(f"target_validation_loss={result.target_validation_loss:.4f}")
        if result.time_to_target_validation_loss_seconds is None:
            print("time_to_target_validation_loss_seconds=not_reached")
        else:
            print(
                "time_to_target_validation_loss_seconds="
                f"{result.time_to_target_validation_loss_seconds:.2f}"
            )
    print(f"saved checkpoint to {(run_dir / 'latest.pt').resolve()}")
    print(f"saved best checkpoint to {(run_dir / 'best.pt').resolve()}")
    print(f"saved summary to {summary_path.resolve()}")


def _create_run_dir(runs_root: Path) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _load_raw_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    if not isinstance(raw_config, dict):
        raise ValueError("Top-level config must be a mapping with model/data/training sections.")
    return raw_config


def _inject_run_output_paths(config: dict[str, Any], run_dir: Path) -> None:
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("Config section 'training' must be a mapping.")

    training["save_checkpoint_path"] = str((run_dir / "latest.pt").resolve())
    training["save_best_checkpoint_path"] = str((run_dir / "best.pt").resolve())
    training["csv_log_path"] = str((run_dir / "train.csv").resolve())
    training["tensorboard_log_dir"] = str((run_dir / "tensorboard").resolve())
    training["resume_checkpoint_path"] = None


if __name__ == "__main__":
    main()
