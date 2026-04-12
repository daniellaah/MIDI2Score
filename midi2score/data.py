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


@dataclass(slots=True)
class LmxDataConfig:
    dataset_path: str
    split: str = "training"
    max_length: int = 512
    sliding_window_stride: int = 256
    length_bucketing: bool = False
    bucket_padding_noise: float = 0.0
    max_tokens_per_batch: int | None = None
    required_batch_size_multiple: int = 1
    pad_to_length_multiple: int = 1
    num_workers: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LmxSlidingWindow:
    raw_index: int
    start: int
    loss_start: int = 0


class LmxSlidingWindowDataset(Dataset):
    def __init__(self, config: LmxDataConfig) -> None:
        self.config = config
        dataset_dict = load_from_disk(config.dataset_path)
        if not isinstance(dataset_dict, DatasetDict):
            raise ValueError("dataset_path must point to a HuggingFace DatasetDict.")
        self.dataset: HFDataset = dataset_dict[config.split]
        self._window_index = (
            self._build_training_window_index()
            if config.split == "training"
            else self._build_evaluation_window_index()
        )

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        raw_index, window_start, loss_start = self._resolve_sliding_window_index(index)

        # get tokens from the dataset and apply the sliding window
        token_ids = self.dataset[raw_index]["input_ids"]
        if len(token_ids) <= self.config.max_length:
            tokens = token_ids
        else:
            tokens = token_ids[window_start : window_start + self.config.max_length]
        if len(tokens) < 2:
            raise ValueError("Samples must contain at least 2 tokens.")

        loss_mask = torch.ones(len(tokens) - 1, dtype=torch.bool)
        if loss_start > 0:
            loss_mask[:loss_start] = False
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "loss_mask": loss_mask,
        }

    def sequence_length(self, index: int) -> int:
        raw_index, window_start, _ = self._resolve_sliding_window_index(index)
        length = len(self.dataset[raw_index]["input_ids"])
        return min(length - window_start, self.config.max_length)

    def _resolve_sliding_window_index(self, index: int) -> tuple[int, int, int]:
        window = self._window_index[index]
        return window.raw_index, window.start, window.loss_start

    def _build_training_window_index(self) -> list[LmxSlidingWindow]:
        stride = self.config.sliding_window_stride
        max_length = self.config.max_length
        windows: list[LmxSlidingWindow] = []
        for raw_index in range(len(self.dataset)):
            token_ids = self.dataset[raw_index]["input_ids"]
            if len(token_ids) <= max_length:
                windows.append(LmxSlidingWindow(raw_index=raw_index, start=0))
                continue
            max_start = len(token_ids) - max_length
            starts = list(range(0, max_start + 1, stride))
            if starts[-1] != max_start:
                starts.append(max_start)
            windows.extend(LmxSlidingWindow(raw_index=raw_index, start=start) for start in starts)
        return windows

    def _build_evaluation_window_index(self) -> list[LmxSlidingWindow]:
        stride = self.config.sliding_window_stride
        max_length = self.config.max_length
        step = min(stride, max_length - 1)
        windows: list[LmxSlidingWindow] = []
        for raw_index in range(len(self.dataset)):
            token_ids = self.dataset[raw_index]["input_ids"]
            if len(token_ids) <= max_length:
                windows.append(LmxSlidingWindow(raw_index=raw_index, start=0, loss_start=0))
                continue
            max_start = len(token_ids) - max_length
            starts = list(range(0, max_start + 1, step))
            if starts[-1] != max_start:
                starts.append(max_start)
            last_loss_position = 0
            for start in starts:
                end = min(start + max_length, len(token_ids))
                loss_start = max(0, last_loss_position - start)
                windows.append(
                    LmxSlidingWindow(
                        raw_index=raw_index,
                        start=start,
                        loss_start=loss_start,
                    )
                )
                last_loss_position = end - 1
        return windows


class LengthBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: LmxSlidingWindowDataset,
        batch_size: int,
        seed: int,
        shuffle: bool,
        max_tokens_per_batch: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_tokens_per_batch is not None and max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive when provided.")
        if dataset.config.required_batch_size_multiple <= 0:
            raise ValueError("required_batch_size_multiple must be positive.")
        if dataset.config.pad_to_length_multiple <= 0:
            raise ValueError("pad_to_length_multiple must be positive.")
        if dataset.config.bucket_padding_noise < 0.0:
            raise ValueError("bucket_padding_noise must be non-negative.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.max_tokens_per_batch = max_tokens_per_batch
        self.required_batch_size_multiple = dataset.config.required_batch_size_multiple
        self.bucket_padding_noise = dataset.config.bucket_padding_noise if shuffle else 0.0
        self._epoch = 0
        self._lengths = [dataset.sequence_length(index) for index in range(len(dataset))]
        self._cached_length: int | None = None

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        batches = self._build_batches(indices)

        if self.shuffle:
            for batch_index in torch.randperm(len(batches), generator=generator).tolist():
                yield batches[batch_index]
            return

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self._uses_order_dependent_batching():
            return len(self._batches_for_next_epoch())
        if self._cached_length is None:
            self._cached_length = self._estimate_num_batches()
        return self._cached_length

    def _build_batches(self, indices: list[int]) -> list[list[int]]:
        if (
            not self.shuffle
            and self.max_tokens_per_batch is None
            and not self.dataset.config.length_bucketing
            and self.required_batch_size_multiple == 1
        ):
            return self._build_fixed_size_batches(indices)

        if not self.dataset.config.length_bucketing:
            return self._build_batches_from_pool(indices)

        bucket_size = self._bucket_size()
        batches: list[list[int]] = []
        generator = torch.Generator().manual_seed(self.seed + self._epoch - 1)
        for start in range(0, len(indices), bucket_size):
            pool = indices[start : start + bucket_size]
            self._sort_pool_by_length(pool, generator=generator)
            batches.extend(self._build_batches_from_pool(pool))
        return batches

    def _uses_order_dependent_batching(self) -> bool:
        return self.shuffle and (
            self.max_tokens_per_batch is not None
            or self.dataset.config.length_bucketing
            or self.required_batch_size_multiple > 1
        )

    def _batches_for_next_epoch(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        return self._build_batches(indices)

    def _estimate_num_batches(self) -> int:
        if self.max_tokens_per_batch is None and self.required_batch_size_multiple == 1:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

        if self.dataset.config.length_bucketing:
            ordered_lengths = sorted(self._lengths, reverse=True)
        else:
            ordered_lengths = self._lengths

        count = 0
        current_batch_size = 0
        current_max_length = 0
        for sequence_length in ordered_lengths:
            while current_batch_size > 0:
                prospective_max_length = max(current_max_length, sequence_length)
                prospective_batch_size = current_batch_size + 1
                if not self._would_exceed_batch_constraints(
                    prospective_max_length=prospective_max_length,
                    prospective_batch_size=prospective_batch_size,
                ):
                    break
                emittable_size = self._aligned_prefix_size(current_batch_size)
                if emittable_size == 0:
                    count += 1
                    current_batch_size = 0
                    current_max_length = 0
                    break
                count += 1
                current_batch_size -= emittable_size
                if current_batch_size == 0:
                    current_max_length = 0
            current_batch_size += 1
            current_max_length = max(current_max_length, sequence_length)

        if current_batch_size:
            count += self._tail_batch_count(current_batch_size)
        return count

    def _bucket_size(self) -> int:
        return self.batch_size * BUCKET_SIZE_MULTIPLIER

    def _build_fixed_size_batches(
        self,
        indices: list[int],
    ) -> list[list[int]]:
        return [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]

    def _build_batches_from_pool(
        self,
        pool: list[int],
    ) -> list[list[int]]:
        if self.max_tokens_per_batch is None and self.required_batch_size_multiple == 1:
            return self._build_fixed_size_batches(pool)

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max_length = 0

        for index in pool:
            sequence_length = self._lengths[index]
            while current_batch:
                prospective_max_length = max(current_max_length, sequence_length)
                prospective_batch_size = len(current_batch) + 1
                if not self._would_exceed_batch_constraints(
                    prospective_max_length=prospective_max_length,
                    prospective_batch_size=prospective_batch_size,
                ):
                    break
                emitted_batch, current_batch = self._split_batch_on_multiple(current_batch)
                if emitted_batch:
                    batches.append(emitted_batch)
                    current_max_length = self._max_batch_length(current_batch)
                    continue
                batches.append(current_batch)
                current_batch = []
                current_max_length = 0
            current_batch.append(index)
            current_max_length = max(current_max_length, sequence_length)

        if current_batch:
            batches.extend(self._finalize_tail_batch(current_batch))
        return batches

    def _sort_pool_by_length(self, pool: list[int], *, generator: torch.Generator) -> None:
        if self.bucket_padding_noise == 0.0:
            pool.sort(key=self._lengths.__getitem__, reverse=True)
            return

        noisy_lengths = {}
        for index in pool:
            length = float(self._lengths[index])
            noise_scale = 1.0 + ((torch.rand(1, generator=generator).item() * 2.0) - 1.0) * self.bucket_padding_noise
            noisy_lengths[index] = max(length * noise_scale, 0.0)
        pool.sort(key=noisy_lengths.__getitem__, reverse=True)

    def _would_exceed_batch_constraints(
        self,
        *,
        prospective_max_length: int,
        prospective_batch_size: int,
    ) -> bool:
        if prospective_batch_size > self.batch_size:
            return True
        if self.max_tokens_per_batch is not None:
            return prospective_max_length * prospective_batch_size > self.max_tokens_per_batch
        return False

    def _aligned_prefix_size(self, batch_size: int) -> int:
        if self.required_batch_size_multiple == 1:
            return batch_size
        return (batch_size // self.required_batch_size_multiple) * self.required_batch_size_multiple

    def _split_batch_on_multiple(self, batch: list[int]) -> tuple[list[int], list[int]]:
        aligned_size = self._aligned_prefix_size(len(batch))
        if aligned_size == 0:
            return [], batch
        if aligned_size == len(batch):
            return batch, []
        return batch[:aligned_size], batch[aligned_size:]

    def _finalize_tail_batch(self, batch: list[int]) -> list[list[int]]:
        emitted_batch, remainder = self._split_batch_on_multiple(batch)
        if not emitted_batch or not remainder:
            return [batch]
        return [emitted_batch, remainder]

    def _tail_batch_count(self, batch_size: int) -> int:
        aligned_size = self._aligned_prefix_size(batch_size)
        if aligned_size == 0 or aligned_size == batch_size:
            return 1
        return 2

    def _max_batch_length(self, batch: list[int]) -> int:
        if not batch:
            return 0
        return max(self._lengths[index] for index in batch)

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

def collate_language_model_batch(
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
    if pad_to_length_multiple > 1:
        padded_length = _round_up_to_multiple(tokens.size(1), pad_to_length_multiple)
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


def build_language_model_dataloader(
    config: LmxDataConfig,
    batch_size: int,
    seed: int,
    shuffle: bool | None = None,
) -> DataLoader[LmxBatch]:
    dataset = LmxSlidingWindowDataset(config)
    if shuffle is None:
        shuffle = config.split == "training"

    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": config.num_workers,
        "collate_fn": partial(
            collate_language_model_batch,
            pad_to_length_multiple=config.pad_to_length_multiple,
        ),
    }
    if (
        config.length_bucketing
        or config.max_tokens_per_batch is not None
        or config.required_batch_size_multiple > 1
    ):
        dataloader_kwargs["batch_sampler"] = LengthBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            max_tokens_per_batch=config.max_tokens_per_batch,
        )
    else:
        generator = None
        if config.split == "training":
            generator = torch.Generator().manual_seed(seed)
        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["shuffle"] = shuffle
        dataloader_kwargs["generator"] = generator
    return DataLoader(**dataloader_kwargs)


def _round_up_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return length
    return ((length + multiple - 1) // multiple) * multiple
