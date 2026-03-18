from __future__ import annotations

import argparse
from pathlib import Path

from midi2score.config import load_decoder_pretrain_config
from midi2score.trainers import run_decoder_pretraining_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain the MusicXML decoder language model from YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain_baseline.yaml"),
        help="Path to the YAML config file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_config = load_decoder_pretrain_config(args.config)
    result = run_decoder_pretraining_loop(
        project_config.model,
        project_config.data,
        project_config.training,
    )
    print(f"finished {len(result.losses)} pretraining steps on {result.device}")
    if result.checkpoint_path is not None:
        print(f"saved checkpoint to {result.checkpoint_path}")


if __name__ == "__main__":
    main()
