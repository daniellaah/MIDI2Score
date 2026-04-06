from __future__ import annotations

import argparse
import json
from pathlib import Path

from midi2score.config import load_decoder_pretrain_config
from midi2score.train import run_decoder_pretraining_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run decoder pretraining directly or as a managed experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain_baseline.yaml"),
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional experiment id. When set, run in managed experiment mode.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override in dotted form, e.g. model.d_model=256.",
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
        help="Optional reference validation loss for delta reporting.",
    )
    parser.add_argument(
        "--allow-dirty-git",
        action="store_true",
        help="Allow running a managed experiment with a dirty worktree.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = _parse_overrides(args.overrides)

    if args.experiment_id is None:
        if overrides or args.note is not None or args.reference_best_loss is not None:
            raise ValueError("--set/--note/--reference-best-loss require --experiment-id.")
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
        return

    from midi2score.research import run_research_experiment

    summary = run_research_experiment(
        base_config_path=args.config,
        experiment_id=args.experiment_id,
        overrides=overrides,
        note=args.note,
        reference_best_validation_loss=args.reference_best_loss,
        require_clean_git=not args.allow_dirty_git,
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
        overrides[key] = _parse_override_value(raw_value)
    return overrides


def _parse_override_value(raw_value: str) -> object:
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


if __name__ == "__main__":
    main()
