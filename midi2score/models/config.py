from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ModelConfig:
    # These vocab sizes already include special tokens such as PAD/BOS/EOS.
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "relu"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_source_length: int = 512
    max_target_length: int = 512

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.nhead

    @property
    def first_regular_token_id(self) -> int:
        return max(self.pad_token_id, self.bos_token_id, self.eos_token_id) + 1

    def validate(self) -> None:
        # Fail fast on shape/configuration errors so we do not discover them only
        # after building a model or starting training.
        positive_int_fields = {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
        }
        for field_name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")

        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")

        if self.activation not in {"relu", "gelu"}:
            raise ValueError(
                f"activation must be one of {{'relu', 'gelu'}}, got {self.activation!r}."
            )

        if min(self.pad_token_id, self.bos_token_id, self.eos_token_id) < 0:
            raise ValueError("Special token ids must be non-negative.")

        min_vocab_size = self.first_regular_token_id + 1
        if self.src_vocab_size < min_vocab_size:
            raise ValueError(
                f"src_vocab_size must be at least {min_vocab_size} to fit special tokens."
            )
        if self.tgt_vocab_size < min_vocab_size:
            raise ValueError(
                f"tgt_vocab_size must be at least {min_vocab_size} to fit special tokens."
            )

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)
