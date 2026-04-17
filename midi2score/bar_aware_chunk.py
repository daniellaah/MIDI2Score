from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, TypeVar

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from lmx.linearization.vocabulary import ALL_TOKENS
from tokenizers import Tokenizer


T = TypeVar("T")
UNICODE_BYTE_BEGIN = 33


@dataclass(frozen=True, slots=True)
class BarChunkPlan:
    piece_path: Path
    token_start: int
    token_end: int
    start_bar: int
    end_bar: int
    is_first_chunk: bool
    is_last_chunk: bool
    is_bar_aligned: bool

    @property
    def token_length(self) -> int:
        return self.token_end - self.token_start

    @property
    def bar_count(self) -> int:
        return self.end_bar - self.start_bar + 1


@dataclass(frozen=True, slots=True)
class BarChunkSummary:
    partition: str
    max_length: int
    overlap_bars: int
    piece_count: int
    chunk_count: int
    avg_chunks_per_piece: float
    avg_chunk_length: float
    median_chunk_length: float
    p90_chunk_length: int
    max_chunk_length: int
    avg_bars_per_chunk: float
    median_bars_per_chunk: float
    bar_aligned_chunk_count: int
    non_bar_aligned_chunk_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EncodedBarChunk:
    piece_path: Path
    input_ids: list[int]
    start_bar: int
    end_bar: int
    is_first_chunk: bool
    is_last_chunk: bool
    is_bar_aligned: bool

    @property
    def token_length(self) -> int:
        return len(self.input_ids)


def list_partition_lmx_paths(
    dataset_root: str | Path,
    partition: str,
    dataset_info_path: str | Path | None = None,
) -> list[Path]:
    root = Path(dataset_root)
    info_path = Path(dataset_info_path) if dataset_info_path is not None else root / "dataset_info_with_partitions.csv"
    frame = pd.read_csv(info_path, usecols=["lmx", "partition"])
    rows = frame.loc[frame["partition"] == partition, "lmx"].dropna().tolist()
    return [root / relative_path for relative_path in rows]


def build_partition_bar_chunk_plans(
    dataset_root: str | Path,
    partition: str,
    max_length: int,
    overlap_bars: int = 1,
    dataset_info_path: str | Path | None = None,
) -> list[BarChunkPlan]:
    plans: list[BarChunkPlan] = []
    for piece_path in list_partition_lmx_paths(dataset_root, partition, dataset_info_path):
        plans.extend(
            plan_bar_aware_chunks(
                piece_path=piece_path,
                tokens=read_lmx_tokens(piece_path),
                max_length=max_length,
                overlap_bars=overlap_bars,
            )
        )
    return plans


def build_partition_encoded_bar_chunks(
    dataset_root: str | Path,
    partition: str,
    tokenizer_path: str | Path,
    max_length: int,
    overlap_bars: int = 1,
    bpe_dropout: float | None = None,
    dataset_info_path: str | Path | None = None,
    piece_limit: int | None = None,
    token_byte_map: dict[str, str] | None = None,
) -> list[EncodedBarChunk]:
    if bpe_dropout is None:
        bpe_dropout = _default_bpe_dropout_for_partition(partition)
    tokenizer = _load_tokenizer(tokenizer_path, bpe_dropout)
    if token_byte_map is None:
        token_byte_map = build_explicit_lmx_token_byte_map()
    encoded_chunks: list[EncodedBarChunk] = []

    piece_paths = list_partition_lmx_paths(dataset_root, partition, dataset_info_path)
    if piece_limit is not None:
        piece_paths = piece_paths[:piece_limit]

    for piece_path in piece_paths:
        tokens = read_lmx_tokens(piece_path)
        encoded_chunks.extend(
            _plan_and_encode_piece_chunks(
                piece_path=piece_path,
                piece_tokens=tokens,
                tokenizer=tokenizer,
                token_byte_map=token_byte_map,
                max_length=max_length,
                overlap_bars=overlap_bars,
            )
        )

    return encoded_chunks


def read_lmx_tokens(piece_path: str | Path) -> list[str]:
    return Path(piece_path).read_text(encoding="utf-8").split()


def build_explicit_lmx_token_byte_map(
    *,
    byte_offset: int = UNICODE_BYTE_BEGIN,
) -> dict[str, str]:
    return {token: chr(byte_offset + index) for index, token in enumerate(ALL_TOKENS)}


def find_measure_start_indices(tokens: Sequence[str]) -> list[int]:
    starts = [index for index, token in enumerate(tokens) if token == "measure"]
    if not starts:
        return [0] if tokens else []
    if starts[0] != 0:
        starts.insert(0, 0)
    return starts


def plan_bar_aware_chunks(
    piece_path: str | Path,
    tokens: Sequence[str],
    max_length: int,
    overlap_bars: int = 1,
) -> list[BarChunkPlan]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if overlap_bars < 0:
        raise ValueError("overlap_bars must be non-negative.")
    if not tokens:
        return []

    piece_path = Path(piece_path)
    bar_starts = find_measure_start_indices(tokens)
    if not bar_starts:
        return []
    bar_ends = [*bar_starts[1:], len(tokens)]

    chunks: list[BarChunkPlan] = []
    start_bar = 0

    while start_bar < len(bar_starts):
        start_index = bar_starts[start_bar]
        end_bar = start_bar
        end_index = bar_ends[end_bar]

        while end_bar + 1 < len(bar_starts):
            candidate_end = bar_ends[end_bar + 1]
            if candidate_end - start_index > max_length:
                break
            end_bar += 1
            end_index = candidate_end

        if end_index - start_index > max_length:
            chunks.extend(
                _plan_single_long_bar(
                    piece_path=piece_path,
                    bar_index=start_bar,
                    bar_start=start_index,
                    bar_end=bar_ends[start_bar],
                    max_length=max_length,
                    is_first_chunk=start_bar == 0 and not chunks,
                )
            )
            start_bar += 1
            continue

        is_first_chunk = not chunks
        is_last_chunk = end_index == len(tokens)
        chunks.append(
            BarChunkPlan(
                piece_path=piece_path,
                token_start=start_index,
                token_end=end_index,
                start_bar=start_bar,
                end_bar=end_bar,
                is_first_chunk=is_first_chunk,
                is_last_chunk=is_last_chunk,
                is_bar_aligned=True,
            )
        )
        if is_last_chunk:
            break

        next_start_bar = max(end_bar + 1 - overlap_bars, start_bar + 1)
        start_bar = min(next_start_bar, len(bar_starts) - 1)

    if chunks:
        chunks[-1] = BarChunkPlan(
            piece_path=chunks[-1].piece_path,
            token_start=chunks[-1].token_start,
            token_end=chunks[-1].token_end,
            start_bar=chunks[-1].start_bar,
            end_bar=chunks[-1].end_bar,
            is_first_chunk=chunks[-1].is_first_chunk,
            is_last_chunk=True,
            is_bar_aligned=chunks[-1].is_bar_aligned,
        )

    return chunks


def apply_piece_boundary_tokens(
    tokens: Sequence[T],
    *,
    bos_token: T | None,
    eos_token: T | None,
    is_first_chunk: bool,
    is_last_chunk: bool,
) -> list[T]:
    output: list[T] = []
    if bos_token is not None and is_first_chunk:
        output.append(bos_token)
    output.extend(tokens)
    if eos_token is not None and is_last_chunk:
        output.append(eos_token)
    return output


def write_bar_chunk_plans_jsonl(plans: Sequence[BarChunkPlan], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for plan in plans:
            record = {
                "piece_path": str(plan.piece_path),
                "token_start": plan.token_start,
                "token_end": plan.token_end,
                "token_length": plan.token_length,
                "start_bar": plan.start_bar,
                "end_bar": plan.end_bar,
                "bar_count": plan.bar_count,
                "is_first_chunk": plan.is_first_chunk,
                "is_last_chunk": plan.is_last_chunk,
                "is_bar_aligned": plan.is_bar_aligned,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def summarize_bar_chunk_plans(
    plans: Sequence[BarChunkPlan],
    *,
    partition: str,
    max_length: int,
    overlap_bars: int,
) -> BarChunkSummary:
    if not plans:
        raise ValueError("plans must not be empty.")

    token_lengths = sorted(plan.token_length for plan in plans)
    bar_counts = sorted(plan.bar_count for plan in plans)
    piece_ids = {plan.piece_path for plan in plans}
    bar_aligned_chunk_count = sum(1 for plan in plans if plan.is_bar_aligned)
    non_bar_aligned_chunk_count = len(plans) - bar_aligned_chunk_count

    return BarChunkSummary(
        partition=partition,
        max_length=max_length,
        overlap_bars=overlap_bars,
        piece_count=len(piece_ids),
        chunk_count=len(plans),
        avg_chunks_per_piece=len(plans) / len(piece_ids),
        avg_chunk_length=sum(token_lengths) / len(token_lengths),
        median_chunk_length=_median(token_lengths),
        p90_chunk_length=token_lengths[_quantile_index(len(token_lengths), 0.9)],
        max_chunk_length=token_lengths[-1],
        avg_bars_per_chunk=sum(bar_counts) / len(bar_counts),
        median_bars_per_chunk=_median(bar_counts),
        bar_aligned_chunk_count=bar_aligned_chunk_count,
        non_bar_aligned_chunk_count=non_bar_aligned_chunk_count,
    )


def write_bar_chunk_summary_json(summary: BarChunkSummary, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_encoded_bar_chunks_jsonl(chunks: Sequence[EncodedBarChunk], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "piece_path": str(chunk.piece_path),
                "input_ids": chunk.input_ids,
                "token_length": chunk.token_length,
                "start_bar": chunk.start_bar,
                "end_bar": chunk.end_bar,
                "is_first_chunk": chunk.is_first_chunk,
                "is_last_chunk": chunk.is_last_chunk,
                "is_bar_aligned": chunk.is_bar_aligned,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_bar_chunk_prototype_dataset_dict(
    encoded_chunks: Sequence[EncodedBarChunk],
    base_dataset_path: str | Path,
) -> DatasetDict:
    base_dataset = load_from_disk(str(base_dataset_path))
    if not isinstance(base_dataset, DatasetDict):
        raise ValueError("base_dataset_path must point to a HuggingFace DatasetDict.")

    training_rows = [{"input_ids": chunk.input_ids} for chunk in encoded_chunks]
    return DatasetDict(
        {
            "training": Dataset.from_list(training_rows),
            "validation": base_dataset["validation"],
            "test": base_dataset["test"],
        }
    )


def save_bar_chunk_prototype_dataset_dict(
    encoded_chunks: Sequence[EncodedBarChunk],
    *,
    base_dataset_path: str | Path,
    output_path: str | Path,
) -> None:
    dataset_dict = build_bar_chunk_prototype_dataset_dict(encoded_chunks, base_dataset_path)
    dataset_dict.save_to_disk(str(output_path))


def _plan_single_long_bar(
    *,
    piece_path: Path,
    bar_index: int,
    bar_start: int,
    bar_end: int,
    max_length: int,
    is_first_chunk: bool,
) -> list[BarChunkPlan]:
    chunks: list[BarChunkPlan] = []
    chunk_start = bar_start
    while chunk_start < bar_end:
        chunk_end = min(chunk_start + max_length, bar_end)
        chunks.append(
            BarChunkPlan(
                piece_path=piece_path,
                token_start=chunk_start,
                token_end=chunk_end,
                start_bar=bar_index,
                end_bar=bar_index,
                is_first_chunk=is_first_chunk and not chunks,
                is_last_chunk=False,
                is_bar_aligned=False,
            )
        )
        chunk_start = chunk_end
    return chunks


def _plan_and_encode_piece_chunks(
    *,
    piece_path: Path,
    piece_tokens: Sequence[str],
    tokenizer: Tokenizer,
    token_byte_map: dict[str, str],
    max_length: int,
    overlap_bars: int,
) -> list[EncodedBarChunk]:
    bar_starts = find_measure_start_indices(piece_tokens)
    if not bar_starts:
        return []
    bar_ends = [*bar_starts[1:], len(piece_tokens)]

    encoded_chunks: list[EncodedBarChunk] = []
    start_bar = 0

    while start_bar < len(bar_starts):
        start_index = bar_starts[start_bar]
        best_end_bar: int | None = None
        best_ids: list[int] | None = None

        for end_bar in range(start_bar, len(bar_starts)):
            end_index = bar_ends[end_bar]
            encoded_ids = _encode_chunk_ids(
                tokenizer=tokenizer,
                token_byte_map=token_byte_map,
                chunk_tokens=piece_tokens[start_index:end_index],
                include_bos=not encoded_chunks,
                include_eos=end_index == len(piece_tokens),
            )
            if len(encoded_ids) > max_length:
                break
            best_end_bar = end_bar
            best_ids = encoded_ids

        if best_end_bar is None or best_ids is None:
            encoded_chunks.extend(
                _split_single_bar_by_encoded_length(
                    piece_path=piece_path,
                    piece_tokens=piece_tokens,
                    tokenizer=tokenizer,
                    token_byte_map=token_byte_map,
                    bar_index=start_bar,
                    bar_start=bar_starts[start_bar],
                    bar_end=bar_ends[start_bar],
                    max_length=max_length,
                    is_first_chunk=not encoded_chunks,
                )
            )
            start_bar += 1
            continue

        end_index = bar_ends[best_end_bar]
        encoded_chunks.append(
            EncodedBarChunk(
                piece_path=piece_path,
                input_ids=best_ids,
                start_bar=start_bar,
                end_bar=best_end_bar,
                is_first_chunk=not encoded_chunks,
                is_last_chunk=end_index == len(piece_tokens),
                is_bar_aligned=True,
            )
        )
        if end_index == len(piece_tokens):
            break

        next_start_bar = max(best_end_bar + 1 - overlap_bars, start_bar + 1)
        start_bar = min(next_start_bar, len(bar_starts) - 1)

    if encoded_chunks:
        last = encoded_chunks[-1]
        encoded_chunks[-1] = EncodedBarChunk(
            piece_path=last.piece_path,
            input_ids=last.input_ids,
            start_bar=last.start_bar,
            end_bar=last.end_bar,
            is_first_chunk=last.is_first_chunk,
            is_last_chunk=True,
            is_bar_aligned=last.is_bar_aligned,
        )

    return encoded_chunks


def _encode_chunk_ids(
    *,
    tokenizer: Tokenizer,
    token_byte_map: dict[str, str],
    chunk_tokens: Sequence[str],
    include_bos: bool,
    include_eos: bool,
) -> list[int]:
    try:
        byte_str = "".join(token_byte_map[token] for token in chunk_tokens)
    except KeyError as exc:
        raise KeyError(f"Missing byte mapping for LMX token: {exc.args[0]!r}") from exc
    encoded_ids = tokenizer.encode(byte_str).ids
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    if not include_bos and bos_token_id is not None and encoded_ids and encoded_ids[0] == bos_token_id:
        encoded_ids = encoded_ids[1:]
    if not include_eos and eos_token_id is not None and encoded_ids and encoded_ids[-1] == eos_token_id:
        encoded_ids = encoded_ids[:-1]
    return encoded_ids


def _split_single_bar_by_encoded_length(
    *,
    piece_path: Path,
    piece_tokens: Sequence[str],
    tokenizer: Tokenizer,
    token_byte_map: dict[str, str],
    bar_index: int,
    bar_start: int,
    bar_end: int,
    max_length: int,
    is_first_chunk: bool,
) -> list[EncodedBarChunk]:
    encoded_ids = _encode_chunk_ids(
        tokenizer=tokenizer,
        token_byte_map=token_byte_map,
        chunk_tokens=piece_tokens[bar_start:bar_end],
        include_bos=is_first_chunk,
        include_eos=bar_end == len(piece_tokens),
    )
    if len(encoded_ids) <= max_length:
        return [
            EncodedBarChunk(
                piece_path=piece_path,
                input_ids=encoded_ids,
                start_bar=bar_index,
                end_bar=bar_index,
                is_first_chunk=is_first_chunk,
                is_last_chunk=bar_end == len(piece_tokens),
                is_bar_aligned=False,
            )
        ]

    chunks: list[EncodedBarChunk] = []
    start = 0
    while start < len(encoded_ids):
        end = min(start + max_length, len(encoded_ids))
        chunks.append(
            EncodedBarChunk(
                piece_path=piece_path,
                input_ids=encoded_ids[start:end],
                start_bar=bar_index,
                end_bar=bar_index,
                is_first_chunk=is_first_chunk and not chunks,
                is_last_chunk=end == len(encoded_ids) and bar_end == len(piece_tokens),
                is_bar_aligned=False,
            )
        )
        start = end
    return chunks


def _median(values: Sequence[int]) -> float:
    count = len(values)
    midpoint = count // 2
    if count % 2 == 1:
        return float(values[midpoint])
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _quantile_index(count: int, quantile: float) -> int:
    if count <= 0:
        raise ValueError("count must be positive.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    return min(count - 1, max(0, math.ceil(count * quantile) - 1))


def _load_tokenizer(tokenizer_path: str | Path, bpe_dropout: float) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.model.dropout = bpe_dropout
    return tokenizer


def _default_bpe_dropout_for_partition(partition: str) -> float:
    return 0.1 if partition == "training" else 0.0
