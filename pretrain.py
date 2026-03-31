from __future__ import annotations

import argparse
from pathlib import Path

from midi2score.config import load_decoder_pretrain_config
from midi2score.train import run_decoder_pretraining_loop


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
    print(
        f"final_step={result.final_step} elapsed_seconds={result.elapsed_seconds:.2f} "
        f"stopped_due_to_time_budget={result.stopped_due_to_time_budget}"
    )
    if result.best_validation_loss is not None:
        print(f"best validation loss {result.best_validation_loss:.4f}")
    if result.checkpoint_path is not None:
        print(f"saved checkpoint to {result.checkpoint_path}")
    if result.best_checkpoint_path is not None and result.best_validation_loss is not None:
        print(f"saved best checkpoint to {result.best_checkpoint_path}")


if __name__ == "__main__":
    main()
