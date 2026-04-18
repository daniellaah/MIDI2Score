from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset

PAD_TOKEN_ID = 0
BUCKET_SIZE_MULTIPLIER = 50
VALIDATION_MAX_LENGTH = 1024
VALIDATION_SLIDING_WINDOW_STRIDE = 512
VALIDATION_PAD_TO_LENGTH_MULTIPLE = 1


@dataclass(slots=True)
class LmxDataConfig:
    dataset_path: str
    split: str = "training"
    max_length: int = 1024
    length_bucketing: bool = True
    bucket_padding_noise: float = 0.0
    max_tokens_per_batch: int | None = None
    pad_to_length_multiple: int = 1
    num_workers: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LmxWindow:
    raw_index: int
    start: int
    length: int
    loss_start: int = 0


def _load_split_dataset(dataset_path: str, split: str) -> HFDataset:
    dataset_dict = load_from_disk(dataset_path)
    if not isinstance(dataset_dict, DatasetDict):
        raise ValueError("dataset_path must point to a HuggingFace DatasetDict.")
    return dataset_dict[split]


class LmxTrainDataset(Dataset):
    def __init__(self, config: LmxDataConfig) -> None:
        self.config = config
        self.dataset = _load_split_dataset(config.dataset_path, "training")
        self.windows: list[LmxWindow] = []

        for raw_index in range(len(self.dataset)):
            token_ids = self.dataset[raw_index]["input_ids"]
            total_length = len(token_ids)
            if total_length > config.max_length:
                raise ValueError(
                    "Training samples must not exceed max_length. "
                    "Increase max_length or regenerate the training dataset."
                )
            self.windows.append(LmxWindow(raw_index=raw_index, start=0, length=total_length))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        window = self.windows[index]
        token_ids = self.dataset[window.raw_index]["input_ids"]
        tokens = token_ids[window.start : window.start + window.length]
        if len(tokens) < 2:
            raise ValueError("Samples must contain at least 2 tokens.")
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "loss_mask": torch.ones(len(tokens) - 1, dtype=torch.bool),
        }

    def sequence_length(self, index: int) -> int:
        return self.windows[index].length


class LmxEvalDataset(Dataset):
    def __init__(self, config: LmxDataConfig) -> None:
        self.config = config
        self.dataset = _load_split_dataset(config.dataset_path, "validation")
        self.windows: list[LmxWindow] = []
        step = min(VALIDATION_SLIDING_WINDOW_STRIDE, VALIDATION_MAX_LENGTH - 1)
        for raw_index in range(len(self.dataset)):
            token_ids = self.dataset[raw_index]["input_ids"]
            total_length = len(token_ids)
            if total_length <= VALIDATION_MAX_LENGTH:
                self.windows.append(LmxWindow(raw_index=raw_index, start=0, length=total_length, loss_start=0))
                continue
            max_start = total_length - VALIDATION_MAX_LENGTH
            starts = list(range(0, max_start + 1, step))
            if starts[-1] != max_start:
                starts.append(max_start)

            last_loss_position = 0
            for start in starts:
                length = min(total_length - start, VALIDATION_MAX_LENGTH)
                end = start + length
                self.windows.append(
                    LmxWindow(
                        raw_index=raw_index,
                        start=start,
                        length=length,
                        loss_start=max(0, last_loss_position - start),
                    )
                )
                last_loss_position = end - 1

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        window = self.windows[index]
        token_ids = self.dataset[window.raw_index]["input_ids"]
        tokens = token_ids[window.start : window.start + window.length]
        if len(tokens) < 2:
            raise ValueError("Samples must contain at least 2 tokens.")

        loss_mask = torch.ones(len(tokens) - 1, dtype=torch.bool)
        if window.loss_start > 0:
            loss_mask[: window.loss_start] = False
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "loss_mask": loss_mask,
        }


class LengthBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: LmxTrainDataset,
        batch_size: int,
        seed: int,
        shuffle: bool,
        length_bucketing: bool = True,
        max_tokens_per_batch: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_tokens_per_batch is not None and max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive when provided.")
        if dataset.config.pad_to_length_multiple <= 0:
            raise ValueError("pad_to_length_multiple must be positive.")
        if dataset.config.bucket_padding_noise < 0.0:
            raise ValueError("bucket_padding_noise must be non-negative.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.length_bucketing = length_bucketing
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pad_to_length_multiple = dataset.config.pad_to_length_multiple
        self.bucket_padding_noise = (
            dataset.config.bucket_padding_noise if shuffle and length_bucketing else 0.0
        )
        self.lengths = [dataset.sequence_length(index) for index in range(len(dataset))]
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for batch in self._planned_batches(generator):
            yield batch

    def __len__(self) -> int:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return len(self._planned_batches(generator))

    def _planned_batches(self, generator: torch.Generator) -> list[list[int]]:
        indices = self._sample_indices(generator)
        if not self.length_bucketing:
            return self._build_batches_from_sorted_pool(indices)
        batches: list[list[int]] = []
        bucket_size = self.batch_size * BUCKET_SIZE_MULTIPLIER
        for start in range(0, len(indices), bucket_size):
            pool = indices[start : start + bucket_size]
            sorted_pool = self._sort_pool(pool, generator)
            batches.extend(self._build_batches_from_sorted_pool(sorted_pool))

        if self.shuffle:
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            return [batches[index] for index in batch_order]
        return batches

    def _sample_indices(self, generator: torch.Generator) -> list[int]:
        if self.shuffle:
            return torch.randperm(len(self.dataset), generator=generator).tolist()
        return list(range(len(self.dataset)))

    def _sort_pool(self, pool: list[int], generator: torch.Generator) -> list[int]:
        ordered_pool = list(pool)
        if self.bucket_padding_noise == 0.0:
            ordered_pool.sort(key=self.lengths.__getitem__, reverse=True)
            return ordered_pool

        noisy_lengths: dict[int, float] = {}
        for index in ordered_pool:
            scale = 1.0 + ((torch.rand(1, generator=generator).item() * 2.0) - 1.0) * self.bucket_padding_noise
            noisy_lengths[index] = max(float(self.lengths[index]) * scale, 0.0)
        ordered_pool.sort(key=noisy_lengths.__getitem__, reverse=True)
        return ordered_pool

    def _build_batches_from_sorted_pool(self, pool: list[int]) -> list[list[int]]:
        if self.max_tokens_per_batch is None:
            return [
                pool[start : start + self.batch_size]
                for start in range(0, len(pool), self.batch_size)
            ]

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max_length = 0

        for index in pool:
            sequence_length = self.lengths[index]
            if not current_batch:
                effective_length = _effective_padded_length(sequence_length, self.pad_to_length_multiple)
                if effective_length > self.max_tokens_per_batch:
                    raise ValueError(
                        "A single sequence exceeds max_tokens_per_batch after padding. "
                        "Increase max_tokens_per_batch or reduce max_length."
                    )

            prospective_max_length = max(current_max_length, sequence_length)
            prospective_batch_size = len(current_batch) + 1
            effective_length = _effective_padded_length(prospective_max_length, self.pad_to_length_multiple)
            exceeds_examples = prospective_batch_size > self.batch_size
            exceeds_tokens = effective_length * prospective_batch_size > self.max_tokens_per_batch
            if current_batch and (exceeds_examples or exceeds_tokens):
                batches.append(current_batch)
                current_batch = []
                current_max_length = 0

            current_batch.append(index)
            current_max_length = max(current_max_length, sequence_length)

        if current_batch:
            batches.append(current_batch)
        return batches

def _effective_padded_length(length: int, pad_to_length_multiple: int) -> int:
    if pad_to_length_multiple <= 1:
        return length
    return ((length + pad_to_length_multiple - 1) // pad_to_length_multiple) * pad_to_length_multiple

@dataclass(slots=True)
class LmxBatch:
    input_tokens: Tensor
    output_tokens: Tensor
    padding_mask: Tensor
    loss_mask: Tensor | None = None

    def to(self, device: torch.device | str) -> "LmxBatch":
        return LmxBatch(
            input_tokens=self.input_tokens.to(device),
            output_tokens=self.output_tokens.to(device),
            padding_mask=self.padding_mask.to(device),
            loss_mask=None if self.loss_mask is None else self.loss_mask.to(device),
        )


def collate_fn(
    examples: list[dict[str, Tensor]],
    *,
    pad_to_length_multiple: int = 1,
) -> LmxBatch:
    tokens = pad_sequence(
        [example["tokens"] for example in examples],
        batch_first=True,
        padding_value=PAD_TOKEN_ID,
    )
    loss_mask = pad_sequence(
        [example["loss_mask"] for example in examples],
        batch_first=True,
        padding_value=False,
    )
    padded_length = _effective_padded_length(tokens.size(1), pad_to_length_multiple)
    if padded_length > tokens.size(1):
        tokens = F.pad(tokens, (0, padded_length - tokens.size(1)), value=PAD_TOKEN_ID)
    if padded_length - 1 > loss_mask.size(1):
        loss_mask = F.pad(loss_mask, (0, padded_length - 1 - loss_mask.size(1)), value=False)
    return LmxBatch(
        input_tokens=tokens[:, :-1],
        output_tokens=tokens[:, 1:],
        padding_mask=tokens[:, :-1].eq(PAD_TOKEN_ID),
        loss_mask=loss_mask,
    )


def build_train_dataloader(
    config: LmxDataConfig,
    batch_size: int,
    seed: int,
) -> DataLoader[LmxBatch]:
    dataset = LmxTrainDataset(config)
    return DataLoader(
        dataset=dataset,
        batch_sampler=LengthBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            length_bucketing=config.length_bucketing,
            max_tokens_per_batch=config.max_tokens_per_batch,
        ),
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn, pad_to_length_multiple=config.pad_to_length_multiple),
    )


def build_eval_dataloader(
    config: LmxDataConfig,
    batch_size: int,
) -> DataLoader[LmxBatch]:
    dataset = LmxEvalDataset(config)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn, pad_to_length_multiple=VALIDATION_PAD_TO_LENGTH_MULTIPLE),
    )
