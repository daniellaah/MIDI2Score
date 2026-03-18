from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypedDict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from midi2score.data.config import FakeDataConfig


class Seq2SeqExample(TypedDict):
    src_tokens: Tensor
    tgt_tokens: Tensor


@dataclass(slots=True)
class Seq2SeqBatch:
    src_tokens: Tensor
    tgt_input_tokens: Tensor
    tgt_output_tokens: Tensor
    src_padding_mask: Tensor
    tgt_padding_mask: Tensor

    def to(self, device: torch.device | str) -> "Seq2SeqBatch":
        # Mirror the Tensor.to(...) ergonomics so the whole batch can be moved
        # to CPU/MPS/CUDA in one call inside the trainer.
        return Seq2SeqBatch(
            src_tokens=self.src_tokens.to(device),
            tgt_input_tokens=self.tgt_input_tokens.to(device),
            tgt_output_tokens=self.tgt_output_tokens.to(device),
            src_padding_mask=self.src_padding_mask.to(device),
            tgt_padding_mask=self.tgt_padding_mask.to(device),
        )


class FakeSeq2SeqDataset(Dataset[Seq2SeqExample]):
    def __init__(self, config: FakeDataConfig) -> None:
        self.config = config

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> Seq2SeqExample:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self)}.")

        generator = torch.Generator().manual_seed(self.config.seed + index)

        # Source and target lengths are sampled independently to exercise the
        # variable-length batching code that a real seq2seq task needs.
        src_body_length = int(
            torch.randint(
                self.config.min_source_length,
                self.config.max_source_length + 1,
                size=(1,),
                generator=generator,
            ).item()
        )
        tgt_body_length = int(
            torch.randint(
                self.config.min_target_length,
                self.config.max_target_length + 1,
                size=(1,),
                generator=generator,
            ).item()
        )
        semantic_length = max(src_body_length, tgt_body_length)

        semantic_tokens = torch.randint(
            low=0,
            high=self.config.resolved_semantic_vocab_size,
            size=(semantic_length,),
            generator=generator,
            dtype=torch.long,
        )

        src_body = self._project_semantic_tokens(
            semantic_tokens[:src_body_length],
            vocab_size=self.config.src_vocab_size,
            multiplier=5,
            bias=1,
        )
        tgt_body = self._project_semantic_tokens(
            torch.flip(semantic_tokens[:tgt_body_length], dims=(0,)),
            vocab_size=self.config.tgt_vocab_size,
            multiplier=7,
            bias=3,
        )

        # The source side ends with EOS, while the target side uses the usual
        # decoder convention: BOS + body + EOS.
        src_tokens = torch.cat(
            [
                src_body,
                torch.tensor([self.config.eos_token_id], dtype=torch.long),
            ]
        )
        tgt_tokens = torch.cat(
            [
                torch.tensor([self.config.bos_token_id], dtype=torch.long),
                tgt_body,
                torch.tensor([self.config.eos_token_id], dtype=torch.long),
            ]
        )

        return {
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
        }

    def _project_semantic_tokens(
        self,
        semantic_tokens: Tensor,
        *,
        vocab_size: int,
        multiplier: int,
        bias: int,
    ) -> Tensor:
        regular_vocab_size = vocab_size - self.config.first_regular_token_id
        # A simple deterministic projection keeps samples reproducible while
        # still making source and target token spaces look different.
        return (
            (semantic_tokens * multiplier + bias) % regular_vocab_size
        ) + self.config.first_regular_token_id


def collate_seq2seq_batch(
    examples: list[Seq2SeqExample],
    *,
    pad_token_id: int,
) -> Seq2SeqBatch:
    if not examples:
        raise ValueError("collate_seq2seq_batch requires at least one example.")

    src_sequences = [example["src_tokens"] for example in examples]
    tgt_sequences = [example["tgt_tokens"] for example in examples]

    src_tokens = pad_sequence(src_sequences, batch_first=True, padding_value=pad_token_id)
    tgt_tokens = pad_sequence(tgt_sequences, batch_first=True, padding_value=pad_token_id)

    if tgt_tokens.size(1) < 2:
        raise ValueError("Target sequences must contain at least BOS and EOS tokens.")

    # Teacher forcing uses the target sequence twice:
    # - decoder input:  BOS w1 w2 ... w(n-1)
    # - prediction goal: w1  w2 ... wn EOS
    tgt_input_tokens = tgt_tokens[:, :-1]
    tgt_output_tokens = tgt_tokens[:, 1:]

    return Seq2SeqBatch(
        src_tokens=src_tokens,
        tgt_input_tokens=tgt_input_tokens,
        tgt_output_tokens=tgt_output_tokens,
        src_padding_mask=src_tokens.eq(pad_token_id),
        tgt_padding_mask=tgt_input_tokens.eq(pad_token_id),
    )


def build_fake_dataloader(
    config: FakeDataConfig,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader[Seq2SeqBatch]:
    dataset = FakeSeq2SeqDataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_seq2seq_batch, pad_token_id=config.pad_token_id),
    )
