from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    label_smoothing: float = 0.0
    scheduler: str = "none"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    num_steps: int = 10
    max_duration_seconds: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
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
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError(
                f"grad_clip_norm must be positive when provided, got {self.grad_clip_norm}."
            )
        if self.label_smoothing < 0.0 or self.label_smoothing >= 1.0:
            raise ValueError(
                "label_smoothing must be in [0.0, 1.0), "
                f"got {self.label_smoothing}."
            )
        if self.scheduler not in {"none", "linear", "cosine"}:
            raise ValueError(f"Unsupported scheduler {self.scheduler!r}.")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}.")
        if self.min_lr_ratio < 0.0 or self.min_lr_ratio > 1.0:
            raise ValueError(
                "min_lr_ratio must be between 0.0 and 1.0, "
                f"got {self.min_lr_ratio}."
            )
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}.")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0.0:
            raise ValueError(
                "max_duration_seconds must be positive when provided, "
                f"got {self.max_duration_seconds}."
            )
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be positive when provided, "
                f"got {self.early_stopping_patience}."
            )
        if self.early_stopping_min_delta < 0.0:
            raise ValueError(
                "early_stopping_min_delta must be non-negative, "
                f"got {self.early_stopping_min_delta}."
            )
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}.")
        if self.eval_every < 0:
            raise ValueError(f"eval_every must be non-negative, got {self.eval_every}.")
        if self.early_stopping_patience is not None and self.eval_every == 0:
            raise ValueError("early_stopping_patience requires eval_every > 0.")
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
