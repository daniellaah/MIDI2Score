from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class FakeDataConfig:
    # This config describes a synthetic seq2seq task that mimics the real
    # training interface without depending on raw MIDI/MusicXML files.
    num_samples: int = 128
    src_vocab_size: int = 512
    tgt_vocab_size: int = 512
    min_source_length: int = 8
    max_source_length: int = 32
    min_target_length: int = 8
    max_target_length: int = 32
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    seed: int = 7
    semantic_vocab_size: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def first_regular_token_id(self) -> int:
        return max(self.pad_token_id, self.bos_token_id, self.eos_token_id) + 1

    @property
    def resolved_semantic_vocab_size(self) -> int:
        if self.semantic_vocab_size is not None:
            return self.semantic_vocab_size
        # The fake task maps both source and target tokens from a shared latent
        # token space so that the model sees a learnable relationship instead of
        # completely independent random sequences.
        return min(
            self.src_vocab_size - self.first_regular_token_id,
            self.tgt_vocab_size - self.first_regular_token_id,
            128,
        )

    def validate(self) -> None:
        positive_int_fields = {
            "num_samples": self.num_samples,
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "min_source_length": self.min_source_length,
            "max_source_length": self.max_source_length,
            "min_target_length": self.min_target_length,
            "max_target_length": self.max_target_length,
        }
        for field_name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")

        if min(self.pad_token_id, self.bos_token_id, self.eos_token_id) < 0:
            raise ValueError("Special token ids must be non-negative.")

        if self.min_source_length > self.max_source_length:
            raise ValueError("min_source_length cannot be greater than max_source_length.")
        if self.min_target_length > self.max_target_length:
            raise ValueError("min_target_length cannot be greater than max_target_length.")

        min_vocab_size = self.first_regular_token_id + 1
        if self.src_vocab_size < min_vocab_size:
            raise ValueError(
                f"src_vocab_size must be at least {min_vocab_size} to fit special tokens."
            )
        if self.tgt_vocab_size < min_vocab_size:
            raise ValueError(
                f"tgt_vocab_size must be at least {min_vocab_size} to fit special tokens."
            )

        if self.semantic_vocab_size is not None and self.semantic_vocab_size <= 0:
            raise ValueError("semantic_vocab_size must be positive when provided.")

        if self.resolved_semantic_vocab_size <= 0:
            raise ValueError("semantic_vocab_size resolves to zero; enlarge the vocab sizes.")

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)
