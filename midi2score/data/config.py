from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class LanguageModelDataConfig:
    # This points at an on-disk HuggingFace DatasetDict that already contains
    # tokenized LMX sequences under the `input_ids` field.
    dataset_path: str
    split: str = "training"
    max_length: int = 512
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tokenizer_path: str | None = None
    random_crop: bool = True
    crop_seed: int = 0
    sliding_window_stride: int | None = None
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        if self.split not in {"training", "validation", "test"}:
            raise ValueError(f"split must be training/validation/test, got {self.split!r}.")

        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}.")

        if min(self.pad_token_id, self.bos_token_id, self.eos_token_id) < 0:
            raise ValueError("Special token ids must be non-negative.")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2 for next-token prediction.")
        if self.crop_seed < 0:
            raise ValueError("crop_seed must be non-negative.")
        if self.sliding_window_stride is not None and self.sliding_window_stride <= 0:
            raise ValueError(
                "sliding_window_stride must be positive when provided, "
                f"got {self.sliding_window_stride}."
            )
        if self.tokenizer_path is not None and not self.tokenizer_path:
            raise ValueError("tokenizer_path must be a non-empty string or None.")

    def tokenizer_vocab_size(self) -> int | None:
        if self.tokenizer_path is None:
            return None

        tokenizer = json.loads(Path(self.tokenizer_path).read_text(encoding="utf-8"))
        vocab = tokenizer["model"]["vocab"]
        expected_specials = {
            "[PAD]": self.pad_token_id,
            "[BOS]": self.bos_token_id,
            "[EOS]": self.eos_token_id,
        }
        for token, expected_id in expected_specials.items():
            if vocab.get(token) != expected_id:
                raise ValueError(
                    f"Tokenizer special token {token} has id {vocab.get(token)}, "
                    f"expected {expected_id}."
                )
        return len(vocab)

    def to_dict(self) -> dict[str, int | str | bool | None]:
        return asdict(self)
