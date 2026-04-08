from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
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
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0
        self._lengths = [dataset.sequence_length(index) for index in range(len(dataset))]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        bucket_size = self._bucket_size()
        batches: list[list[int]] = []

        for start in range(0, len(indices), bucket_size):
            pool = indices[start : start + bucket_size]
            pool.sort(key=self._lengths.__getitem__, reverse=True)
            batches.extend(self._build_batches_from_sorted_pool(pool))

        if self.shuffle:
            for batch_index in torch.randperm(len(batches), generator=generator).tolist():
                yield batches[batch_index]
            return

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _bucket_size(self) -> int:
        return self.batch_size * BUCKET_SIZE_MULTIPLIER

    def _build_batches_from_sorted_pool(
        self,
        pool: list[int],
    ) -> list[list[int]]:
        batches = [
            pool[start : start + self.batch_size]
            for start in range(0, len(pool), self.batch_size)
        ]
        return batches

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

def collate_language_model_batch(examples: list[dict[str, Tensor]]) -> LmxBatch:
    tokens = pad_sequence(
        [example["tokens"] for example in examples],
        batch_first=True,
        padding_value=PAD_TOKEN_ID,
    )
    return LmxBatch(
        input_tokens=tokens[:, :-1],
        output_tokens=tokens[:, 1:],
        padding_mask=tokens[:, :-1].eq(PAD_TOKEN_ID),
        loss_mask=pad_sequence(
            [example["loss_mask"] for example in examples],
            batch_first=True,
            padding_value=False,
        ),
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
        "collate_fn": collate_language_model_batch,
    }
    if config.length_bucketing:
        dataloader_kwargs["batch_sampler"] = LengthBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
        )
    else:
        generator = None
        if config.split == "training":
            generator = torch.Generator().manual_seed(seed)
        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["shuffle"] = shuffle
        dataloader_kwargs["generator"] = generator
    return DataLoader(**dataloader_kwargs)
