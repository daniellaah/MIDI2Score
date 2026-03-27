from __future__ import annotations

from dataclasses import asdict, dataclass

from midi2score.models.config import ModelConfig


@dataclass(slots=True)
class DecoderLanguageModelConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "relu"
    position_encoding_type: str = "sinusoidal"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_length: int = 512

    def __post_init__(self) -> None:
        self.validate()

    @property
    def first_regular_token_id(self) -> int:
        return max(self.pad_token_id, self.bos_token_id, self.eos_token_id) + 1

    def validate(self) -> None:
        positive_int_fields = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "max_length": self.max_length,
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
        if self.position_encoding_type not in {"sinusoidal", "learned", "alibi"}:
            raise ValueError(
                "position_encoding_type must be one of "
                f"{{'sinusoidal', 'learned', 'alibi'}}, got {self.position_encoding_type!r}."
            )
        if min(self.pad_token_id, self.bos_token_id, self.eos_token_id) < 0:
            raise ValueError("Special token ids must be non-negative.")

        min_vocab_size = self.first_regular_token_id + 1
        if self.vocab_size < min_vocab_size:
            raise ValueError(
                f"vocab_size must be at least {min_vocab_size} to fit special tokens."
            )

    def assert_compatible_with_seq2seq(self, seq2seq_config: ModelConfig) -> None:
        expected_pairs = {
            "vocab_size": (self.vocab_size, seq2seq_config.tgt_vocab_size),
            "d_model": (self.d_model, seq2seq_config.d_model),
            "nhead": (self.nhead, seq2seq_config.nhead),
            "num_layers": (self.num_layers, seq2seq_config.num_decoder_layers),
            "dim_feedforward": (self.dim_feedforward, seq2seq_config.dim_feedforward),
            "pad_token_id": (self.pad_token_id, seq2seq_config.pad_token_id),
            "bos_token_id": (self.bos_token_id, seq2seq_config.bos_token_id),
            "eos_token_id": (self.eos_token_id, seq2seq_config.eos_token_id),
        }
        for field_name, (decoder_value, seq2seq_value) in expected_pairs.items():
            if decoder_value != seq2seq_value:
                raise ValueError(
                    f"Incompatible decoder config for {field_name}: "
                    f"{decoder_value} != {seq2seq_value}."
                )

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)
