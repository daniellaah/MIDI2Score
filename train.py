from __future__ import annotations

import argparse
from pathlib import Path

from midi2score.config import load_project_config
from midi2score.trainers import run_training_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MIDI2Score baseline from YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML config file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_config = load_project_config(args.config)
    result = run_training_loop(
        project_config.model,
        project_config.data,
        project_config.training,
    )
    print(f"finished {len(result.losses)} steps on {result.device}")


if __name__ == "__main__":
    main()
