from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


@dataclass(slots=True)
class TrainingLogger:
    csv_path: str | None = None
    tensorboard_log_dir: str | None = None
    _writer: SummaryWriter | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.csv_path is not None:
            csv_file = Path(self.csv_path)
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            if not csv_file.exists():
                with csv_file.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["step", "split", "metric", "value"])

        if self.tensorboard_log_dir is not None:
            log_dir = Path(self.tensorboard_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, *, step: int, split: str, value: float, metric: str = "loss") -> None:
        if self.csv_path is not None:
            with Path(self.csv_path).open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([step, split, metric, f"{value:.8f}"])

        if self._writer is not None:
            self._writer.add_scalar(f"{metric}/{split}", value, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
