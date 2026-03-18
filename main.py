from __future__ import annotations

import argparse

from midi2score.data import FakeDataConfig
from midi2score.models import ModelConfig
from midi2score.trainers import TrainingConfig, run_training_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal MIDI2Score training smoke test.")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps to run.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for fake training.")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_config = ModelConfig(
        src_vocab_size=128,
        tgt_vocab_size=160,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        max_source_length=64,
        max_target_length=64,
    )
    data_config = FakeDataConfig(
        num_samples=32,
        src_vocab_size=model_config.src_vocab_size,
        tgt_vocab_size=model_config.tgt_vocab_size,
        min_source_length=8,
        max_source_length=16,
        min_target_length=8,
        max_target_length=16,
        seed=17,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_steps=args.steps,
        device=args.device,
    )

    result = run_training_loop(model_config, data_config, training_config)
    print(f"finished {len(result.losses)} steps on {result.device}")


if __name__ == "__main__":
    main()
