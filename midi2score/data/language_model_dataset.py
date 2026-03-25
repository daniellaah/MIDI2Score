from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TypedDict

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, get_worker_info

from midi2score.data.config import LanguageModelDataConfig


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


class HuggingFaceLanguageModelDataset(Dataset[LanguageModelExample]):
    def __init__(self, config: LanguageModelDataConfig) -> None:
        self.config = config
        dataset_dict = load_from_disk(config.dataset_path)
        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError(
                f"Expected a DatasetDict at {config.dataset_path!r}, got {type(dataset_dict)!r}."
            )
        self.dataset: HFDataset = dataset_dict[config.split]
        self._crop_generators: dict[int, torch.Generator] = {}
        self._window_index: list[tuple[int, int]] | None = None
        if config.sliding_window_stride is not None:
            self._window_index = self._build_window_index()

    def __len__(self) -> int:
        if self._window_index is not None:
            return len(self._window_index)
        return len(self.dataset)

    def __getitem__(self, index: int) -> LanguageModelExample:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self)}.")

        raw_index, window_start = self._resolve_index(index)
        tokens = self.dataset[raw_index]["input_ids"]
        tokens = self._trim_tokens(tokens, index=raw_index, window_start=window_start)
        if len(tokens) < 2:
            raise ValueError(f"Sample {index} is shorter than 2 tokens after trimming.")

        # The on-disk dataset is already tokenized, so the adapter only converts
        # lists of ids into tensors and applies optional length control.
        tokens = torch.tensor(tokens, dtype=torch.long)
        return {"tokens": tokens}

    def _trim_tokens(
        self,
        token_ids: list[int],
        *,
        index: int,
        window_start: int | None,
    ) -> list[int]:
        if len(token_ids) <= self.config.max_length:
            return token_ids

        if window_start is not None:
            return token_ids[window_start : window_start + self.config.max_length]

        if self.config.random_crop and self.config.split == "training":
            generator = self._get_crop_generator()
            max_start = len(token_ids) - self.config.max_length
            start = int(torch.randint(0, max_start + 1, size=(1,), generator=generator).item())
            return token_ids[start : start + self.config.max_length]

        # Validation and test should be deterministic.
        return token_ids[: self.config.max_length]

    def _resolve_index(self, index: int) -> tuple[int, int | None]:
        if self._window_index is None:
            return index, None
        return self._window_index[index]

    def _build_window_index(self) -> list[tuple[int, int]]:
        stride = self.config.sliding_window_stride
        if stride is None:
            return []

        window_index: list[tuple[int, int]] = []
        for raw_index in range(len(self.dataset)):
            token_ids = self.dataset[raw_index]["input_ids"]
            if len(token_ids) <= self.config.max_length:
                window_index.append((raw_index, 0))
                continue

            max_start = len(token_ids) - self.config.max_length
            starts = list(range(0, max_start + 1, stride))
            if starts[-1] != max_start:
                starts.append(max_start)
            window_index.extend((raw_index, start) for start in starts)
        return window_index

    def _get_crop_generator(self) -> torch.Generator:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        generator = self._crop_generators.get(worker_id)
        if generator is None:
            generator = torch.Generator()
            # Each worker gets its own reproducible random stream so repeated
            # visits to the same sample can see different windows over time.
            generator.manual_seed(self.config.crop_seed + 1_000_003 * worker_id)
            self._crop_generators[worker_id] = generator
        return generator


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


def build_language_model_dataloader(
    config: LanguageModelDataConfig,
    *,
    batch_size: int,
    shuffle: bool | None = None,
) -> DataLoader[LanguageModelBatch]:
    dataset = HuggingFaceLanguageModelDataset(config)
    if shuffle is None:
        shuffle = config.split == "training"
    generator = None
    if config.split == "training":
        generator = torch.Generator().manual_seed(config.crop_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=partial(collate_language_model_batch, pad_token_id=config.pad_token_id),
    )
