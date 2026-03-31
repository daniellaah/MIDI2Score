from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TypedDict

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset, get_worker_info


@dataclass(slots=True)
class LanguageModelDataConfig:
    dataset_path: str
    split: str = "training"
    max_length: int = 512
    tokenizer_path: str | None = None
    random_crop: bool = True
    crop_seed: int = 0
    sliding_window_stride: int | None = None
    length_bucketing: bool = False
    bucket_size_multiplier: int = 50
    num_workers: int = 0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if self.split not in {"training", "validation", "test"}:
            raise ValueError("split must be one of training/validation/test.")
        if self.sliding_window_stride is not None and self.sliding_window_stride <= 0:
            raise ValueError("sliding_window_stride must be positive.")
        if self.bucket_size_multiplier <= 0:
            raise ValueError("bucket_size_multiplier must be positive.")

    def tokenizer_vocab_size(self) -> int | None:
        if self.tokenizer_path is None:
            return None
        tokenizer = json.loads(Path(self.tokenizer_path).read_text(encoding="utf-8"))
        return len(tokenizer["model"]["vocab"])

    def to_dict(self) -> dict[str, int | str | bool | None]:
        return asdict(self)


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
            raise ValueError("dataset_path must point to a HuggingFace DatasetDict.")
        self.dataset: HFDataset = dataset_dict[config.split]
        self._crop_generators: dict[int, torch.Generator] = {}
        self._window_index: list[tuple[int, int]] | None = None
        if config.sliding_window_stride is not None:
            self._window_index = self._build_window_index()

    def __len__(self) -> int:
        return len(self._window_index) if self._window_index is not None else len(self.dataset)

    def __getitem__(self, index: int) -> LanguageModelExample:
        raw_index, window_start = self._resolve_index(index)
        token_ids = self.dataset[raw_index]["input_ids"]
        tokens = self._trim_tokens(token_ids, window_start=window_start)
        if len(tokens) < 2:
            raise ValueError("Samples must contain at least 2 tokens.")
        return {"tokens": torch.tensor(tokens, dtype=torch.long)}

    def sequence_length(self, index: int) -> int:
        raw_index, window_start = self._resolve_index(index)
        length = len(self.dataset[raw_index]["input_ids"])
        if length <= self.config.max_length:
            return length
        if window_start is not None or self.config.random_crop:
            return self.config.max_length
        return self.config.max_length

    def _resolve_index(self, index: int) -> tuple[int, int | None]:
        if self._window_index is None:
            return index, None
        return self._window_index[index]

    def _trim_tokens(self, token_ids: list[int], *, window_start: int | None) -> list[int]:
        if len(token_ids) <= self.config.max_length:
            return token_ids
        if window_start is not None:
            return token_ids[window_start : window_start + self.config.max_length]
        if self.config.random_crop and self.config.split == "training":
            max_start = len(token_ids) - self.config.max_length
            start = int(
                torch.randint(
                    0,
                    max_start + 1,
                    size=(1,),
                    generator=self._get_crop_generator(),
                ).item()
            )
            return token_ids[start : start + self.config.max_length]
        return token_ids[: self.config.max_length]

    def _build_window_index(self) -> list[tuple[int, int]]:
        stride = self.config.sliding_window_stride
        assert stride is not None
        windows: list[tuple[int, int]] = []
        for raw_index in range(len(self.dataset)):
            token_ids = self.dataset[raw_index]["input_ids"]
            if len(token_ids) <= self.config.max_length:
                windows.append((raw_index, 0))
                continue
            max_start = len(token_ids) - self.config.max_length
            starts = list(range(0, max_start + 1, stride))
            if starts[-1] != max_start:
                starts.append(max_start)
            windows.extend((raw_index, start) for start in starts)
        return windows

    def _get_crop_generator(self) -> torch.Generator:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        generator = self._crop_generators.get(worker_id)
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(self.config.crop_seed + 1_000_003 * worker_id)
            self._crop_generators[worker_id] = generator
        return generator


class LengthBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        *,
        dataset: HuggingFaceLanguageModelDataset,
        batch_size: int,
        drop_last: bool,
        seed: int,
        bucket_size_multiplier: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_size_multiplier = bucket_size_multiplier
        self._epoch = 0
        self._lengths = [dataset.sequence_length(index) for index in range(len(dataset))]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        bucket_size = self.batch_size * self.bucket_size_multiplier
        batches: list[list[int]] = []

        for start in range(0, len(indices), bucket_size):
            pool = indices[start : start + bucket_size]
            pool.sort(key=self._lengths.__getitem__, reverse=True)
            for batch_start in range(0, len(pool), self.batch_size):
                batch = pool[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        for batch_index in torch.randperm(len(batches), generator=generator).tolist():
            yield batches[batch_index]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_language_model_batch(
    examples: list[LanguageModelExample],
    *,
    pad_token_id: int,
) -> LanguageModelBatch:
    tokens = pad_sequence(
        [example["tokens"] for example in examples],
        batch_first=True,
        padding_value=pad_token_id,
    )
    return LanguageModelBatch(
        input_tokens=tokens[:, :-1],
        output_tokens=tokens[:, 1:],
        padding_mask=tokens[:, :-1].eq(pad_token_id),
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

    dataloader_kwargs = {
        "dataset": dataset,
        "num_workers": config.num_workers,
        "collate_fn": partial(collate_language_model_batch, pad_token_id=config.pad_token_id),
    }
    if config.split == "training" and config.length_bucketing:
        dataloader_kwargs["batch_sampler"] = LengthBucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=False,
            seed=config.crop_seed,
            bucket_size_multiplier=config.bucket_size_multiplier,
        )
    else:
        generator = None
        if config.split == "training":
            generator = torch.Generator().manual_seed(config.crop_seed)
        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["shuffle"] = shuffle
        dataloader_kwargs["generator"] = generator
    return DataLoader(**dataloader_kwargs)
