from pathlib import Path
import json

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict, load_from_disk
from tokenizers import Tokenizer

from pretrain.bar_aware_chunk import (
    apply_piece_boundary_tokens,
    build_bar_chunk_prototype_dataset_dict,
    build_explicit_lmx_token_byte_map,
    build_partition_encoded_bar_chunks,
    build_partition_bar_chunk_plans,
    find_measure_start_indices,
    list_partition_lmx_paths,
    plan_bar_aware_chunks,
    read_lmx_tokens,
    summarize_bar_chunk_plans,
    write_encoded_bar_chunks_jsonl,
    write_bar_chunk_plans_jsonl,
    write_bar_chunk_summary_json,
)


def test_list_partition_lmx_paths_uses_dataset_info_partition(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "lmx").mkdir()
    csv_path = root / "dataset_info_with_partitions.csv"
    pd.DataFrame(
        [
            {"lmx": "lmx/a.lmx", "partition": "training"},
            {"lmx": "lmx/b.lmx", "partition": "validation"},
            {"lmx": "lmx/c.lmx", "partition": "training"},
        ]
    ).to_csv(csv_path, index=False)

    paths = list_partition_lmx_paths(root, "training")

    assert paths == [root / "lmx/a.lmx", root / "lmx/c.lmx"]


def test_read_lmx_tokens_and_measure_starts() -> None:
    piece_path = Path("data/PDMX_preprocessed_rd/lmx/1/5/Qmb6B5xMaPXBgdDg4XCvL3UcoBdDsBWwqQy2ZHvkq7Criv-P1.lmx")

    tokens = read_lmx_tokens(piece_path)
    starts = find_measure_start_indices(tokens)

    assert tokens
    assert starts
    assert starts[0] == 0
    assert all(tokens[index] == "measure" or index == 0 for index in starts)


def test_plan_bar_aware_chunks_respects_overlap_and_max_length() -> None:
    tokens = [
        "measure", "a1", "a2",
        "measure", "b1", "b2",
        "measure", "c1", "c2",
        "measure", "d1", "d2",
    ]

    chunks = plan_bar_aware_chunks("piece.lmx", tokens, max_length=7, overlap_bars=1)

    assert [(chunk.start_bar, chunk.end_bar) for chunk in chunks] == [(0, 1), (1, 2), (2, 3)]
    assert [(chunk.token_start, chunk.token_end) for chunk in chunks] == [(0, 6), (3, 9), (6, 12)]
    assert chunks[0].is_first_chunk is True
    assert chunks[-1].is_last_chunk is True
    assert all(chunk.is_bar_aligned for chunk in chunks)


def test_plan_bar_aware_chunks_falls_back_when_single_bar_exceeds_max_length() -> None:
    tokens = ["measure", "a1", "a2", "a3", "a4", "a5", "a6"]

    chunks = plan_bar_aware_chunks("piece.lmx", tokens, max_length=3, overlap_bars=1)

    assert [(chunk.token_start, chunk.token_end) for chunk in chunks] == [(0, 3), (3, 6), (6, 7)]
    assert all(not chunk.is_bar_aligned for chunk in chunks)
    assert chunks[0].is_first_chunk is True
    assert chunks[-1].is_last_chunk is True


def test_apply_piece_boundary_tokens_only_on_piece_edges() -> None:
    assert apply_piece_boundary_tokens(
        [10, 11],
        bos_token=1,
        eos_token=2,
        is_first_chunk=True,
        is_last_chunk=False,
    ) == [1, 10, 11]
    assert apply_piece_boundary_tokens(
        [10, 11],
        bos_token=1,
        eos_token=2,
        is_first_chunk=False,
        is_last_chunk=True,
    ) == [10, 11, 2]
    assert apply_piece_boundary_tokens(
        [10, 11],
        bos_token=1,
        eos_token=2,
        is_first_chunk=False,
        is_last_chunk=False,
    ) == [10, 11]


def test_build_partition_bar_chunk_plans_reads_all_piece_paths(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    lmx_dir = root / "lmx"
    lmx_dir.mkdir(parents=True)
    (lmx_dir / "a.lmx").write_text("measure a1 a2 measure b1", encoding="utf-8")
    (lmx_dir / "b.lmx").write_text("measure c1 c2", encoding="utf-8")
    pd.DataFrame(
        [
            {"lmx": "lmx/a.lmx", "partition": "training"},
            {"lmx": "lmx/b.lmx", "partition": "training"},
        ]
    ).to_csv(root / "dataset_info_with_partitions.csv", index=False)

    plans = build_partition_bar_chunk_plans(root, "training", max_length=16, overlap_bars=1)

    assert len(plans) == 2
    assert {plan.piece_path.name for plan in plans} == {"a.lmx", "b.lmx"}


def test_summarize_and_write_bar_chunk_outputs(tmp_path: Path) -> None:
    plans = plan_bar_aware_chunks(
        "piece.lmx",
        [
            "measure", "a1", "a2",
            "measure", "b1", "b2",
            "measure", "c1", "c2",
        ],
        max_length=6,
        overlap_bars=1,
    )

    summary = summarize_bar_chunk_plans(plans, partition="training", max_length=6, overlap_bars=1)
    plans_path = tmp_path / "plans.jsonl"
    summary_path = tmp_path / "summary.json"
    write_bar_chunk_plans_jsonl(plans, plans_path)
    write_bar_chunk_summary_json(summary, summary_path)

    plan_rows = [json.loads(line) for line in plans_path.read_text(encoding="utf-8").splitlines()]
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(plan_rows) == len(plans)
    assert plan_rows[0]["piece_path"].endswith("piece.lmx")
    assert plan_rows[0]["token_length"] == 6
    assert summary_payload["partition"] == "training"
    assert summary_payload["chunk_count"] == len(plans)
    assert summary_payload["bar_aligned_chunk_count"] == len(plans)


def test_build_partition_encoded_bar_chunks_applies_boundary_bos_eos(tmp_path: Path) -> None:
    tokenizer = Tokenizer.from_file("data/tokenizer_rd.json")
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")

    chunks = build_partition_encoded_bar_chunks(
        dataset_root="data/PDMX_preprocessed_rd",
        partition="training",
        tokenizer_path="data/tokenizer_rd.json",
        max_length=32,
        overlap_bars=0,
        piece_limit=1,
    )

    assert len(chunks) >= 2
    assert chunks[0].input_ids[0] == bos_token_id
    assert chunks[0].input_ids[-1] != eos_token_id
    assert chunks[1].input_ids[0] != bos_token_id
    assert all(len(chunk.input_ids) <= 32 for chunk in chunks)
    assert chunks[-1].input_ids[-1] == eos_token_id


def test_build_explicit_lmx_token_byte_map_matches_known_tokens() -> None:
    token_byte_map = build_explicit_lmx_token_byte_map()

    assert token_byte_map["measure"] == "!"
    assert token_byte_map["voice:1"] == "\u00a2"
    assert token_byte_map["quarter"] == "\u00b6"


def test_build_partition_encoded_bar_chunks_uses_partition_default_bpe_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_dropouts: list[float] = []

    def fake_load_tokenizer(_tokenizer_path: str | Path, bpe_dropout: float) -> Tokenizer:
        captured_dropouts.append(bpe_dropout)
        return Tokenizer.from_file("data/tokenizer_rd.json")

    monkeypatch.setattr("pretrain.bar_aware_chunk._load_tokenizer", fake_load_tokenizer)

    build_partition_encoded_bar_chunks(
        dataset_root="data/PDMX_preprocessed_rd",
        partition="training",
        tokenizer_path="data/tokenizer_rd.json",
        max_length=32,
        overlap_bars=0,
        piece_limit=1,
    )
    build_partition_encoded_bar_chunks(
        dataset_root="data/PDMX_preprocessed_rd",
        partition="validation",
        tokenizer_path="data/tokenizer_rd.json",
        max_length=32,
        overlap_bars=0,
        piece_limit=1,
    )

    assert captured_dropouts == [0.1, 0.0]


def test_explicit_byte_map_reencodes_validation_split_exactly() -> None:
    token_byte_map = build_explicit_lmx_token_byte_map()
    tokenizer = Tokenizer.from_file("data/tokenizer_rd.json")
    validation_dataset = load_from_disk("data/huggingface")["validation"]
    piece_paths = list_partition_lmx_paths("data/PDMX_preprocessed_rd", "validation")[:3]

    for index, piece_path in enumerate(piece_paths):
        raw_tokens = read_lmx_tokens(piece_path)
        byte_str = "".join(token_byte_map[token] for token in raw_tokens)
        assert tokenizer.encode(byte_str).ids == validation_dataset[index]["input_ids"]


def test_write_encoded_bar_chunks_and_build_prototype_dataset(tmp_path: Path) -> None:
    root = tmp_path / "base_dataset"
    dataset_dict = DatasetDict(
        {
            "training": Dataset.from_list([{"input_ids": [1, 2, 3]}]),
            "validation": Dataset.from_list([{"input_ids": [4, 5, 6]}]),
            "test": Dataset.from_list([{"input_ids": [7, 8, 9]}]),
        }
    )
    dataset_dict.save_to_disk(str(root))

    chunks = build_partition_encoded_bar_chunks(
        dataset_root="data/PDMX_preprocessed_rd",
        partition="training",
        tokenizer_path="data/tokenizer_rd.json",
        max_length=1024,
        overlap_bars=0,
        piece_limit=1,
    )

    jsonl_path = tmp_path / "encoded_chunks.jsonl"
    write_encoded_bar_chunks_jsonl(chunks, jsonl_path)
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[0]["input_ids"]
    assert all(len(chunk.input_ids) <= 1024 for chunk in chunks)

    prototype = build_bar_chunk_prototype_dataset_dict(chunks, root)
    assert prototype["training"].num_rows == len(chunks)
    assert prototype["validation"].num_rows == 1
    assert prototype["test"].num_rows == 1
