from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypedDict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from midi2score.data.config import FakeLanguageModelDataConfig


class LanguageModelExample(TypedDict):
    tokens: Tensor


@dataclass(slots=True)
class LanguageModelBatch:
    input_tokens: Tensor
    output_tokens: Tensor
    padding_mask: Tensor

    def to(self, device: torch.device | str) -> "LanguageModelBatch":
        return LanguageModelBatch(
            input_tokens=self.input_tokens.to(device),
            output_tokens=self.output_tokens.to(device),
            padding_mask=self.padding_mask.to(device),
        )


class FakeLanguageModelDataset(Dataset[LanguageModelExample]):
    def __init__(self, config: FakeLanguageModelDataConfig) -> None:
        self.config = config

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> LanguageModelExample:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self)}.")

        generator = torch.Generator().manual_seed(self.config.seed + index)
        body_length = int(
            torch.randint(
                self.config.min_length,
                self.config.max_length + 1,
                size=(1,),
                generator=generator,
            ).item()
        )
        semantic_tokens = torch.randint(
            low=0,
            high=self.config.resolved_semantic_vocab_size,
            size=(body_length,),
            generator=generator,
            dtype=torch.long,
        )

        # The fake score language has consistent local patterns, giving the
        # decoder a sequence modeling problem that is not pure noise.
        body_tokens = (
            (semantic_tokens * 11 + torch.arange(body_length)) % self.config.resolved_semantic_vocab_size
        ) + self.config.first_regular_token_id
        tokens = torch.cat(
            [
                torch.tensor([self.config.bos_token_id], dtype=torch.long),
                body_tokens.to(dtype=torch.long),
                torch.tensor([self.config.eos_token_id], dtype=torch.long),
            ]
        )
        return {"tokens": tokens}


def collate_language_model_batch(
    examples: list[LanguageModelExample],
    *,
    pad_token_id: int,
) -> LanguageModelBatch:
    if not examples:
        raise ValueError("collate_language_model_batch requires at least one example.")

    token_sequences = [example["tokens"] for example in examples]
    tokens = pad_sequence(token_sequences, batch_first=True, padding_value=pad_token_id)
    if tokens.size(1) < 2:
        raise ValueError("Language model sequences must contain at least BOS and EOS.")

    input_tokens = tokens[:, :-1]
    output_tokens = tokens[:, 1:]
    return LanguageModelBatch(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        padding_mask=input_tokens.eq(pad_token_id),
    )


def build_fake_language_model_dataloader(
    config: FakeLanguageModelDataConfig,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader[LanguageModelBatch]:
    dataset = FakeLanguageModelDataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_language_model_batch, pad_token_id=config.pad_token_id),
    )
