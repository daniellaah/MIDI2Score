from __future__ import annotations

import argparse
import json
from pathlib import Path

from midi2score.research import parse_override_value, run_research_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a standardized decoder-pretraining tuning experiment."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/pretrain_baseline.yaml"),
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Short experiment identifier, used in output paths.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override in dotted form, e.g. model.d_model=128.",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional short note stored in the experiment summary.",
    )
    parser.add_argument(
        "--reference-best-loss",
        type=float,
        default=None,
        help="Optional reference validation loss for automatic delta reporting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = _parse_overrides(args.overrides)
    summary = run_research_experiment(
        base_config_path=args.base_config,
        experiment_id=args.experiment_id,
        overrides=overrides,
        note=args.note,
        reference_best_validation_loss=args.reference_best_loss,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_overrides(pairs: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be in KEY=VALUE form, got {pair!r}.")
        key, raw_value = pair.split("=", maxsplit=1)
        if not key:
            raise ValueError(f"Override key must be non-empty, got {pair!r}.")
        overrides[key] = parse_override_value(raw_value)
    return overrides


if __name__ == "__main__":
    main()
