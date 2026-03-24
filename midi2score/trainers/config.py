from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_steps: int = 10
    max_duration_seconds: float | None = None
    log_every: int = 1
    eval_every: int = 0
    num_eval_batches: int | None = None
    device: str = "auto"
    save_checkpoint_path: str | None = None
    save_best_checkpoint_path: str | None = None
    resume_checkpoint_path: str | None = None
    csv_log_path: str | None = None
    tensorboard_log_dir: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}.")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0.0:
            raise ValueError(
                "max_duration_seconds must be positive when provided, "
                f"got {self.max_duration_seconds}."
            )
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}.")
        if self.eval_every < 0:
            raise ValueError(f"eval_every must be non-negative, got {self.eval_every}.")
        if self.num_eval_batches is not None and self.num_eval_batches <= 0:
            raise ValueError(
                f"num_eval_batches must be positive when provided, got {self.num_eval_batches}."
            )
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device {self.device!r}.")
        if self.save_checkpoint_path is not None and not isinstance(self.save_checkpoint_path, str):
            raise ValueError("save_checkpoint_path must be a string path or None.")
        if self.save_best_checkpoint_path is not None and not isinstance(
            self.save_best_checkpoint_path, str
        ):
            raise ValueError("save_best_checkpoint_path must be a string path or None.")
        for field_name in ("resume_checkpoint_path", "csv_log_path", "tensorboard_log_dir"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string path or None.")
        if self.resume_checkpoint_path is not None and not Path(self.resume_checkpoint_path).exists():
            raise ValueError(
                f"resume_checkpoint_path does not exist: {self.resume_checkpoint_path}"
            )

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)
