from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_PATTERN = re.compile(r"^step=(\d+) pretrain_loss=([0-9.]+)")
VAL_PATTERN = re.compile(
    r"^step=(\d+) validation_loss=([0-9.]+) perplexity=([0-9.]+) token_acc=([0-9.]+) top5_acc=([0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot decoder pretraining losses from a log file.")
    parser.add_argument("--log", type=Path, required=True, help="Path to the training log.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving-average window for train loss.",
    )
    return parser.parse_args()


def parse_log(log_path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train_losses: list[tuple[int, float]] = []
    val_losses: list[tuple[int, float]] = []

    for line in log_path.read_text(encoding="utf-8").splitlines():
        train_match = TRAIN_PATTERN.match(line)
        if train_match is not None:
            train_losses.append((int(train_match.group(1)), float(train_match.group(2))))
            continue

        val_match = VAL_PATTERN.match(line)
        if val_match is not None:
            val_losses.append((int(val_match.group(1)), float(val_match.group(2))))

    if not train_losses:
        raise ValueError(f"No train losses found in {log_path}.")
    if not val_losses:
        raise ValueError(f"No validation losses found in {log_path}.")

    return train_losses, val_losses


def parse_csv(csv_path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train_losses: list[tuple[int, float]] = []
    val_losses: list[tuple[int, float]] = []

    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            metric = row["metric"]
            value = float(row["value"])
            split = row["split"]
            if split == "train" and metric == "loss":
                train_losses.append((step, value))
            elif split == "validation" and metric == "loss":
                val_losses.append((step, value))

    if not train_losses:
        raise ValueError(f"No train losses found in {csv_path}.")
    if not val_losses:
        raise ValueError(f"No validation losses found in {csv_path}.")

    return train_losses, val_losses


def moving_average(points: list[tuple[int, float]], window: int) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for index, (step, _) in enumerate(points):
        start = max(0, index - window + 1)
        window_values = [value for _, value in points[start : index + 1]]
        steps.append(step)
        values.append(sum(window_values) / len(window_values))
    return steps, values


def main() -> None:
    args = parse_args()
    if args.log.suffix == ".csv":
        train_losses, val_losses = parse_csv(args.log)
    else:
        train_losses, val_losses = parse_log(args.log)

    train_steps, train_values = moving_average(train_losses, window=args.window)
    val_steps = [step for step, _ in val_losses]
    val_values = [value for _, value in val_losses]
    best_step, best_value = min(val_losses, key=lambda item: item[1])

    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 7.2), dpi=150)
    axis.plot(train_steps, train_values, label=f"Train loss ({args.window}-step mean)", linewidth=2.2)
    axis.plot(val_steps, val_values, label="Validation loss", linewidth=2.2)
    axis.scatter([best_step], [best_value], color="tab:red", zorder=3)
    axis.annotate(
        f"best val: {best_value:.4f}",
        xy=(best_step, best_value),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=10,
    )

    axis.set_title("Decoder Pretraining Loss Curves (rd best from scratch)")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
