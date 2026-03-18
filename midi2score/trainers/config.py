from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_steps: int = 10
    log_every: int = 1
    device: str = "auto"
    save_checkpoint_path: str | None = None
    pretrained_decoder_checkpoint: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}.")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device {self.device!r}.")
        for field_name in ("save_checkpoint_path", "pretrained_decoder_checkpoint"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string path or None.")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)
