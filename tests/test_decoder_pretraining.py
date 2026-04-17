from pathlib import Path
import csv

import pytest
import torch
from datasets import Dataset, DatasetDict

from pretrain.data import (
    LmxBatch,
    LmxDataConfig,
    LmxEvalDataset,
    LmxTrainDataset,
    LengthBucketBatchSampler,
    VALIDATION_MAX_LENGTH,
    build_eval_dataloader,
    build_train_dataloader,
    collate_fn,
)
from pretrain.decoder import DecoderLanguageModelConfig, TransformerDecoderLayer, TransformerDecoderLM
from pretrain.trainer import (
    DecoderEvaluationMetrics,
    MpsMemoryTracker,
    TrainingConfig,
    build_lr_scheduler,
    evaluate_decoder_language_model,
    evaluate_decoder_language_model_metrics,
    run_decoder_pretraining_loop,
)


class DummySlidingWindowDataset:
    def __init__(self, lengths: list[int], config: LmxDataConfig) -> None:
        self._lengths = lengths
        self.config = config

    def __len__(self) -> int:
        return len(self._lengths)

    def sequence_length(self, index: int) -> int:
        return self._lengths[index]


class DummyBatchingDataset(DummySlidingWindowDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        length = self.sequence_length(index)
        tokens = torch.arange(1, length + 1, dtype=torch.long)
        return {
            "tokens": tokens,
            "loss_mask": torch.ones(length - 1, dtype=torch.bool),
        }


def build_dummy_sampler_dataset(
    lengths: list[int],
    *,
    max_tokens_per_batch: int | None = None,
    bucket_padding_noise: float = 0.0,
    pad_to_length_multiple: int = 1,
) -> DummySlidingWindowDataset:
    return DummySlidingWindowDataset(
        lengths,
        LmxDataConfig(
            dataset_path="data/huggingface",
            split="training",
            max_length=64,
            bucket_padding_noise=bucket_padding_noise,
            max_tokens_per_batch=max_tokens_per_batch,
            pad_to_length_multiple=pad_to_length_multiple,
        ),
    )


def build_dummy_batching_dataset(
    lengths: list[int],
    *,
    max_tokens_per_batch: int | None = None,
    pad_to_length_multiple: int = 1,
    bucket_padding_noise: float = 0.0,
) -> DummyBatchingDataset:
    return DummyBatchingDataset(
        lengths,
        LmxDataConfig(
            dataset_path="data/huggingface",
            split="training",
            max_length=max(lengths),
            bucket_padding_noise=bucket_padding_noise,
            max_tokens_per_batch=max_tokens_per_batch,
            pad_to_length_multiple=pad_to_length_multiple,
        ),
    )


def build_dummy_batch(*, batch_size: int = 2, sequence_length: int = 8) -> LmxBatch:
    input_tokens = torch.full((batch_size, sequence_length), 1, dtype=torch.long)
    output_tokens = torch.full((batch_size, sequence_length), 2, dtype=torch.long)
    return LmxBatch(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        padding_mask=torch.zeros((batch_size, sequence_length), dtype=torch.bool),
        loss_mask=torch.ones((batch_size, sequence_length), dtype=torch.bool),
    )


def build_small_synthetic_batch() -> tuple[DecoderLanguageModelConfig, LmxBatch]:
    model_config = DecoderLanguageModelConfig(
        vocab_size=256,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=64,
    )
    return model_config, build_dummy_batch(batch_size=2, sequence_length=8)


def build_temporary_dataset_dict(tmp_path: Path, training_rows: list[list[int]]) -> Path:
    dataset_path = tmp_path / "dataset"
    DatasetDict(
        {
            "training": Dataset.from_list([{"input_ids": row} for row in training_rows]),
            "validation": Dataset.from_list([{"input_ids": [1, 2, 3, 4]}]),
            "test": Dataset.from_list([{"input_ids": [1, 2, 3, 4]}]),
        }
    ).save_to_disk(str(dataset_path))
    return dataset_path


def test_language_model_dataloader_reads_real_dataset() -> None:
    data_config = LmxDataConfig(
        dataset_path="data/bar_aware_chunk/training_bar_chunk_encoded_overlap2_full_dataset",
        split="training",
        max_length=1024,
    )
    batch = next(iter(build_train_dataloader(data_config, batch_size=2, seed=0)))

    assert batch.input_tokens.shape[0] == 2
    assert batch.input_tokens.shape == batch.output_tokens.shape
    assert batch.padding_mask.dtype == torch.bool
    assert batch.input_tokens[:, 0].ne(0).all()


def test_training_dataset_consumes_pregenerated_chunks_directly(tmp_path: Path) -> None:
    dataset_path = build_temporary_dataset_dict(
        tmp_path,
        training_rows=[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12],
        ],
    )
    config = LmxDataConfig(
        dataset_path=str(dataset_path),
        split="training",
        max_length=8,
    )

    dataset = LmxTrainDataset(config)

    assert len(dataset.windows) == 2
    assert [(window.start, window.length) for window in dataset.windows] == [(0, 5), (0, 7)]
    assert dataset[0]["tokens"].tolist() == [1, 2, 3, 4, 5]
    assert dataset[1]["tokens"].tolist() == [6, 7, 8, 9, 10, 11, 12]


def test_training_dataset_rejects_oversized_samples(tmp_path: Path) -> None:
    dataset_path = build_temporary_dataset_dict(
        tmp_path,
        training_rows=[
            [1, 2, 3, 4],
            [5, 6, 7, 8, 9],
        ],
    )
    config = LmxDataConfig(
        dataset_path=str(dataset_path),
        split="training",
        max_length=4,
    )

    with pytest.raises(ValueError, match="Training samples must not exceed max_length"):
        LmxTrainDataset(config)


def test_validation_sliding_window_scores_each_target_token_once() -> None:
    config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=32,
    )
    dataset = LmxEvalDataset(config)
    long_index = next(
        index
        for index in range(len(dataset.dataset))
        if len(dataset.dataset[index]["input_ids"]) > VALIDATION_MAX_LENGTH * 3
    )
    raw_tokens = dataset.dataset[long_index]["input_ids"]
    covered_targets: list[int] = []

    for spec in dataset.windows:
        if spec.raw_index != long_index:
            continue
        window = dataset[next(i for i, candidate in enumerate(dataset.windows) if candidate == spec)]
        scored_positions = window["loss_mask"].nonzero(as_tuple=False).flatten().tolist()
        covered_targets.extend(spec.start + 1 + position for position in scored_positions)

    assert covered_targets == list(range(1, len(raw_tokens)))


def test_validation_dataset_ignores_passed_window_arguments() -> None:
    narrow_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=32,
    )
    wide_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=2048,
    )

    narrow_dataset = LmxEvalDataset(narrow_config)
    wide_dataset = LmxEvalDataset(wide_config)

    assert len(narrow_dataset) == len(wide_dataset)
    assert [
        (window.raw_index, window.start, window.loss_start)
        for window in narrow_dataset.windows[:100]
    ] == [
        (window.raw_index, window.start, window.loss_start)
        for window in wide_dataset.windows[:100]
    ]
    assert max(window.length for window in narrow_dataset.windows[:100]) <= VALIDATION_MAX_LENGTH


def test_length_bucketed_dynamic_batch_sampler_groups_examples_by_length() -> None:
    dataset = build_dummy_sampler_dataset(
        [48, 12, 36, 64, 24, 8, 56, 16],
    )
    sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=4,
        seed=23,
        shuffle=True,
    )

    batch_indices = next(iter(sampler))
    lengths = [dataset.sequence_length(index) for index in batch_indices]

    assert lengths == sorted(lengths, reverse=True)


def test_dynamic_batch_sampler_without_length_bucketing_preserves_sample_order() -> None:
    dataset = build_dummy_sampler_dataset([48, 12, 36, 64, 24, 8, 56, 16])
    sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=4,
        seed=23,
        shuffle=False,
        length_bucketing=False,
    )

    batches = list(iter(sampler))

    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_length_bucketed_dynamic_batch_sampler_respects_max_tokens_per_batch() -> None:
    dataset = build_dummy_sampler_dataset(
        [64, 56, 48, 40, 32, 24, 16, 8],
        max_tokens_per_batch=128,
    )
    sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=8,
        seed=23,
        shuffle=False,
        max_tokens_per_batch=dataset.config.max_tokens_per_batch,
    )

    for batch_indices in sampler:
        lengths = [dataset.sequence_length(index) for index in batch_indices]
        padded_length = max(lengths)
        assert padded_length * len(batch_indices) <= 128


def test_build_dataloader_supports_dynamic_batch_size_with_token_budget() -> None:
    dataset = build_dummy_batching_dataset(
        [64, 60, 52, 48, 40, 36, 32, 28, 24, 20, 16],
        max_tokens_per_batch=192,
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("pretrain.data.LmxTrainDataset", lambda config: dataset)
        loader = build_train_dataloader(dataset.config, batch_size=8, seed=0)

        assert isinstance(loader.batch_sampler, LengthBucketBatchSampler)
        batch_sizes: list[int] = []
        for batch in loader:
            batch_sizes.append(batch.input_tokens.size(0))
            padded_token_length = batch.input_tokens.size(1) + 1
            assert padded_token_length * batch.input_tokens.size(0) <= 192

    assert len(batch_sizes) > 1
    assert all(batch_size <= 8 for batch_size in batch_sizes)


def test_build_dataloader_can_disable_length_bucketing() -> None:
    dataset = build_dummy_batching_dataset(
        [64, 60, 52, 48, 40, 36, 32, 28],
        max_tokens_per_batch=192,
    )
    dataset.config.length_bucketing = False

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("pretrain.data.LmxTrainDataset", lambda config: dataset)
        loader = build_train_dataloader(dataset.config, batch_size=8, seed=0)

        assert isinstance(loader.batch_sampler, LengthBucketBatchSampler)
        assert loader.batch_sampler.length_bucketing is False


def test_build_dataloader_keeps_validation_fixed_even_if_dynamic_fields_are_set() -> None:
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=256,
        bucket_padding_noise=0.1,
        max_tokens_per_batch=512,
        pad_to_length_multiple=64,
    )
    loader = build_eval_dataloader(data_config, batch_size=4)
    assert not isinstance(loader.batch_sampler, LengthBucketBatchSampler)

    batch_sizes: list[int] = []
    for batch_index, batch in enumerate(loader):
        batch_sizes.append(batch.input_tokens.size(0))
        if batch_index >= 24:
            break

    assert batch_sizes
    assert all(batch_size == 4 for batch_size in batch_sizes[:-1])


def test_collate_fn_supports_pad_to_length_multiple() -> None:
    examples = [
        {
            "tokens": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "loss_mask": torch.tensor([True, True, True, True], dtype=torch.bool),
        },
        {
            "tokens": torch.tensor([1, 2, 3], dtype=torch.long),
            "loss_mask": torch.tensor([True, True], dtype=torch.bool),
        },
    ]

    batch = collate_fn(examples, pad_to_length_multiple=8)

    assert batch.input_tokens.shape[1] + 1 == 8
    assert batch.output_tokens.shape[1] + 1 == 8
    assert batch.loss_mask is not None
    assert batch.loss_mask.shape[1] == 7


def test_length_bucketed_dynamic_batch_sampler_supports_padding_noise() -> None:
    baseline_dataset = build_dummy_sampler_dataset(
        [256, 248, 240, 232, 224, 216, 208, 200],
    )
    noisy_dataset = build_dummy_sampler_dataset(
        [256, 248, 240, 232, 224, 216, 208, 200],
        bucket_padding_noise=0.5,
    )
    baseline_sampler = LengthBucketBatchSampler(
        dataset=baseline_dataset,
        batch_size=8,
        seed=23,
        shuffle=True,
    )
    noisy_sampler = LengthBucketBatchSampler(
        dataset=noisy_dataset,
        batch_size=8,
        seed=23,
        shuffle=True,
    )

    baseline_batch = next(iter(baseline_sampler))
    noisy_batch = next(iter(noisy_sampler))

    assert len(noisy_batch) == 8
    assert baseline_batch != noisy_batch


def test_length_bucketed_dynamic_batch_sampler_len_matches_iteration_with_shuffle() -> None:
    dataset = build_dummy_sampler_dataset(
        [256, 224, 192, 160, 128, 96, 64, 48, 32, 16],
        max_tokens_per_batch=512,
    )
    sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=8,
        seed=23,
        shuffle=True,
        max_tokens_per_batch=dataset.config.max_tokens_per_batch,
    )

    assert len(sampler) == sum(1 for _ in sampler)


def test_decoder_language_model_forward_produces_vocab_logits() -> None:
    model_config, batch = build_small_synthetic_batch()
    model = TransformerDecoderLM(model_config)

    logits = model(batch.input_tokens, padding_mask=batch.padding_mask)

    assert logits.shape == (
        batch.input_tokens.size(0),
        batch.input_tokens.size(1),
        model_config.vocab_size,
    )


def test_decoder_language_model_supports_rope() -> None:
    model_config, batch = build_small_synthetic_batch()
    model_config = DecoderLanguageModelConfig(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_layers=model_config.num_layers,
        dim_feedforward=model_config.dim_feedforward,
        max_length=model_config.max_length,
        positional_encoding="rope",
    )
    model = TransformerDecoderLM(model_config)

    logits = model(batch.input_tokens, padding_mask=batch.padding_mask)

    assert logits.shape == (
        batch.input_tokens.size(0),
        batch.input_tokens.size(1),
        model_config.vocab_size,
    )


def test_decoder_language_model_supports_optional_cross_attention_memory() -> None:
    model_config, batch = build_small_synthetic_batch()
    model = TransformerDecoderLM(model_config)
    memory = torch.randn(
        batch.input_tokens.size(0),
        5,
        model_config.d_model,
    )
    memory_padding_mask = torch.zeros(
        (batch.input_tokens.size(0), 5),
        dtype=torch.bool,
    )

    logits = model(
        batch.input_tokens,
        padding_mask=batch.padding_mask,
        memory=memory,
        memory_padding_mask=memory_padding_mask,
    )

    assert logits.shape == (
        batch.input_tokens.size(0),
        batch.input_tokens.size(1),
        model_config.vocab_size,
    )


def test_transformer_decoder_layer_keeps_ffn_norm_consistent_with_and_without_memory() -> None:
    class TrackingNorm(torch.nn.Module):
        def __init__(self, name: str, calls: list[str]) -> None:
            super().__init__()
            self.name = name
            self.calls = calls

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls.append(self.name)
            return x

    class ZeroSelfAttention(torch.nn.Module):
        def forward(self, x: torch.Tensor, *, tgt_padding_mask=None) -> torch.Tensor:
            return torch.zeros_like(x)

    class ZeroCrossAttention(torch.nn.Module):
        def forward(self, x: torch.Tensor, *, memory: torch.Tensor, memory_padding_mask=None) -> torch.Tensor:
            return torch.zeros_like(x)

    class ZeroFeedForward(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    layer = TransformerDecoderLayer(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        positional_encoding="rope",
        max_length=64,
    )
    calls: list[str] = []
    layer.self_attn_norm = TrackingNorm("self", calls)
    layer.cross_attn_norm = TrackingNorm("cross", calls)
    layer.ffn_norm = TrackingNorm("ffn", calls)
    layer.self_attn = ZeroSelfAttention()
    layer.cross_attn = ZeroCrossAttention()
    layer.feedforward = ZeroFeedForward()

    tgt = torch.randn(2, 8, 32)
    layer(tgt)
    assert calls == ["self", "ffn"]

    calls.clear()
    memory = torch.randn(2, 5, 32)
    layer(tgt, memory=memory)
    assert calls == ["self", "cross", "ffn"]


@pytest.mark.parametrize(("activation",), [("relu",), ("gelu",), ("swiglu",), ("geglu",)])
def test_decoder_language_model_supports_activation_variants(activation: str) -> None:
    model_config, batch = build_small_synthetic_batch()
    model_config = DecoderLanguageModelConfig(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_layers=model_config.num_layers,
        dim_feedforward=model_config.dim_feedforward,
        max_length=model_config.max_length,
        activation=activation,
    )
    model = TransformerDecoderLM(model_config)

    logits = model(batch.input_tokens, padding_mask=batch.padding_mask)

    assert logits.shape[-1] == model_config.vocab_size


def test_decoder_language_model_always_ties_embeddings() -> None:
    model_config, batch = build_small_synthetic_batch()
    model = TransformerDecoderLM(model_config)

    logits = model(batch.input_tokens, padding_mask=batch.padding_mask)

    assert logits.shape[-1] == model_config.vocab_size
    assert model.output_projection.weight is model.tgt_embedding.weight


@pytest.mark.parametrize(("positional_encoding",), [("sinusoidal",), ("rope",)])
def test_decoder_attention_exposes_linear_qkv_for_lora(positional_encoding: str) -> None:
    model_config = DecoderLanguageModelConfig(
        vocab_size=128,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=64,
        positional_encoding=positional_encoding,
    )
    model = TransformerDecoderLM(model_config)
    attention = model.decoder.layers[0].self_attn

    assert isinstance(attention.q_proj, torch.nn.Linear)
    assert isinstance(attention.k_proj, torch.nn.Linear)
    assert isinstance(attention.v_proj, torch.nn.Linear)
    assert isinstance(attention.out_proj, torch.nn.Linear)

    cross_attention = model.decoder.layers[0].cross_attn
    assert isinstance(cross_attention.q_proj, torch.nn.Linear)
    assert isinstance(cross_attention.k_proj, torch.nn.Linear)
    assert isinstance(cross_attention.v_proj, torch.nn.Linear)
    assert isinstance(cross_attention.out_proj, torch.nn.Linear)


def test_length_bucketed_dynamic_batch_sampler_is_deterministic_without_shuffle() -> None:
    dataset = build_dummy_sampler_dataset(
        [64, 56, 48, 40, 32, 24, 16, 8],
    )
    sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=4,
        seed=23,
        shuffle=False,
    )

    first_pass = list(iter(sampler))
    second_pass = list(iter(sampler))

    assert first_pass == second_pass


def test_decoder_pretraining_loop_saves_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "decoder.pt"
    best_checkpoint_path = tmp_path / "decoder-best.pt"
    csv_log_path = tmp_path / "train.csv"
    model_config = DecoderLanguageModelConfig(
        vocab_size=5000,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=VALIDATION_MAX_LENGTH,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
    )
    training_config = TrainingConfig(
        seed=0,
        batch_size=4,
        num_steps=2,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        log_every=10,
        eval_every=1,
        num_eval_batches=1,
        target_validation_loss=100.0,
        device="cpu",
        save_checkpoint_path=str(checkpoint_path),
        save_best_checkpoint_path=str(best_checkpoint_path),
        csv_log_path=str(csv_log_path),
    )

    dummy_batch = build_dummy_batch()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("pretrain.trainer.build_train_dataloader", lambda *args, **kwargs: [dummy_batch])
        monkeypatch.setattr("pretrain.trainer.build_eval_dataloader", lambda *args, **kwargs: [dummy_batch])
        monkeypatch.setattr(
            "pretrain.trainer.evaluate_decoder_language_model_metrics",
            lambda *args, **kwargs: DecoderEvaluationMetrics(
                loss=1.0,
                perplexity=2.0,
                token_accuracy=0.5,
                top5_accuracy=0.75,
                evaluated_tokens=int(dummy_batch.loss_mask.sum().item()),
            ),
        )
        result = run_decoder_pretraining_loop(model_config, data_config, training_config)

    assert len(result.losses) == 2
    assert len(result.validation_losses) == 2
    assert checkpoint_path.exists()
    assert best_checkpoint_path.exists()
    assert result.best_validation_loss is not None
    with csv_log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["step", "split", "metric", "value"]
    assert any(row[1] == "train" and row[2] == "loss" for row in rows[1:])
    assert any(row[1] == "train" and row[2] == "step_time_seconds" for row in rows[1:])
    assert any(row[1] == "train" and row[2] == "tokens_per_second" for row in rows[1:])
    assert any(row[1] == "validation" and row[2] == "loss" for row in rows[1:])
    assert any(row[1] == "validation" and row[2] == "perplexity" for row in rows[1:])
    assert any(row[1] == "validation" and row[2] == "token_accuracy" for row in rows[1:])
    assert any(row[1] == "validation" and row[2] == "top5_accuracy" for row in rows[1:])
    assert any(row[1] == "validation" and row[2] == "time_to_target_validation_loss_seconds" for row in rows[1:])
    assert result.average_step_time_seconds > 0.0
    assert result.average_tokens_per_second > 0.0
    assert result.target_validation_loss == pytest.approx(100.0)
    assert result.time_to_target_validation_loss_seconds is not None
    assert result.time_to_target_validation_loss_seconds <= result.elapsed_seconds
    assert result.mps_peak_memory_bytes is None

def test_decoder_pretraining_loop_uses_fixed_validation_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_calls: list[tuple[LmxDataConfig, int]] = []
    dummy_batch = LmxBatch(
        input_tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
        output_tokens=torch.tensor([[2, 3, 4]], dtype=torch.long),
        padding_mask=torch.zeros((1, 3), dtype=torch.bool),
        loss_mask=torch.ones((1, 3), dtype=torch.bool),
    )

    def fake_build_train_loader(config: LmxDataConfig, *, batch_size: int, seed: int):
        captured_calls.append((config, batch_size))
        return [dummy_batch]

    def fake_build_eval_loader(config: LmxDataConfig, *, batch_size: int):
        captured_calls.append((config, batch_size))
        return [dummy_batch]

    monkeypatch.setattr("pretrain.trainer.build_train_dataloader", fake_build_train_loader)
    monkeypatch.setattr("pretrain.trainer.build_eval_dataloader", fake_build_eval_loader)
    monkeypatch.setattr(
        "pretrain.trainer.evaluate_decoder_language_model_metrics",
        lambda *args, **kwargs: DecoderEvaluationMetrics(
            loss=1.0,
            perplexity=2.0,
            token_accuracy=0.5,
            top5_accuracy=0.75,
            evaluated_tokens=3,
        ),
    )

    model_config = DecoderLanguageModelConfig(
        vocab_size=5000,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=VALIDATION_MAX_LENGTH,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        bucket_padding_noise=0.1,
        max_tokens_per_batch=128,
        pad_to_length_multiple=8,
    )
    training_config = TrainingConfig(
        seed=0,
        batch_size=8,
        eval_batch_size=16,
        num_steps=1,
        log_every=10,
        eval_every=1,
        device="cpu",
    )

    run_decoder_pretraining_loop(model_config, data_config, training_config)

    assert len(captured_calls) == 2
    train_config, train_batch_size = captured_calls[0]
    eval_config, eval_batch_size = captured_calls[1]

    assert train_config == data_config
    assert train_batch_size == 8

    assert eval_config.split == "validation"
    assert eval_batch_size == 16


def test_evaluate_decoder_language_model_returns_finite_loss() -> None:
    model_config, batch = build_small_synthetic_batch()
    model = TransformerDecoderLM(model_config)
    loader = [batch]

    loss = evaluate_decoder_language_model(
        model,
        loader,
        pad_token_id=model_config.pad_token_id,
        device="cpu",
        num_batches=1,
    )

    assert loss > 0.0


def test_evaluate_decoder_language_model_metrics_returns_expected_fields() -> None:
    model_config, batch = build_small_synthetic_batch()
    model = TransformerDecoderLM(model_config)
    loader = [batch]

    metrics = evaluate_decoder_language_model_metrics(
        model,
        loader,
        pad_token_id=model_config.pad_token_id,
        device="cpu",
        num_batches=1,
    )

    assert metrics.loss > 0.0
    assert metrics.perplexity >= 1.0
    assert 0.0 <= metrics.token_accuracy <= 1.0
    assert 0.0 <= metrics.top5_accuracy <= 1.0
    assert metrics.top5_accuracy >= metrics.token_accuracy
    assert metrics.evaluated_tokens > 0


def test_evaluate_decoder_language_model_metrics_uses_token_weighted_loss() -> None:
    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_logits = [
                torch.tensor(
                    [[[4.0, 0.0], [0.0, 4.0]]],
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [[[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [0.0, 4.0]]],
                    dtype=torch.float32,
                ),
            ]
            self.index = 0

        def eval(self):
            return self

        def forward(self, input_tokens, *, padding_mask=None):
            logits = self.batch_logits[self.index]
            self.index += 1
            return logits

    batches = [
        LmxBatch(
            input_tokens=torch.tensor([[1, 2]], dtype=torch.long),
            output_tokens=torch.tensor([[0, 1]], dtype=torch.long),
            padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        ),
        LmxBatch(
            input_tokens=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            output_tokens=torch.tensor([[0, 0, 0, 1]], dtype=torch.long),
            padding_mask=torch.zeros((1, 4), dtype=torch.bool),
        ),
    ]

    metrics = evaluate_decoder_language_model_metrics(
        FixedModel(),
        batches,
        pad_token_id=99,
        device="cpu",
    )

    loss_a = torch.nn.functional.cross_entropy(
        torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        torch.tensor([0, 1]),
        reduction="sum",
    )
    loss_b = torch.nn.functional.cross_entropy(
        torch.tensor([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [0.0, 4.0]]),
        torch.tensor([0, 0, 0, 1]),
        reduction="sum",
    )
    expected_loss = float((loss_a + loss_b) / 6)

    assert metrics.loss == pytest.approx(expected_loss)
    assert metrics.evaluated_tokens == 6


def test_evaluate_decoder_language_model_metrics_respects_loss_mask() -> None:
    class FixedModel(torch.nn.Module):
        def eval(self):
            return self

        def forward(self, input_tokens, *, padding_mask=None):
            return torch.tensor(
                [[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]],
                dtype=torch.float32,
            )

    batch = LmxBatch(
        input_tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
        output_tokens=torch.tensor([[0, 1, 0]], dtype=torch.long),
        padding_mask=torch.zeros((1, 3), dtype=torch.bool),
        loss_mask=torch.tensor([[False, True, True]], dtype=torch.bool),
    )

    metrics = evaluate_decoder_language_model_metrics(
        FixedModel(),
        [batch],
        pad_token_id=99,
        device="cpu",
    )

    expected_loss = float(
        torch.nn.functional.cross_entropy(
            torch.tensor([[0.0, 4.0], [4.0, 0.0]]),
            torch.tensor([1, 0]),
        )
    )

    assert metrics.loss == pytest.approx(expected_loss)
    assert metrics.evaluated_tokens == 2


def test_mps_memory_tracker_records_peak_driver_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(torch, "mps") or not hasattr(torch.mps, "driver_allocated_memory"):
        pytest.skip("torch.mps driver memory tracking is unavailable")

    tracker = MpsMemoryTracker(enabled=True)
    observed_memory = iter([128, 64, 512, 256])
    monkeypatch.setattr(
        "torch.mps.driver_allocated_memory",
        lambda: next(observed_memory),
    )

    tracker.record()
    tracker.record()
    tracker.record()
    tracker.record()

    assert tracker.peak_memory_bytes == 512


def test_decoder_dropout_is_active_only_in_train_mode() -> None:
    torch.manual_seed(0)
    model_config = DecoderLanguageModelConfig(
        vocab_size=128,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.5,
        max_length=16,
    )
    model = TransformerDecoderLM(model_config)
    input_tokens = torch.randint(0, model_config.vocab_size, (2, 8))

    model.train()
    train_first = model(input_tokens)
    train_second = model(input_tokens)
    assert not torch.allclose(train_first, train_second)

    model.eval()
    eval_first = model(input_tokens)
    eval_second = model(input_tokens)
    assert torch.allclose(eval_first, eval_second)


def test_resume_continues_from_saved_step(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "decoder.pt"
    csv_log_path = tmp_path / "resume.csv"
    model_config = DecoderLanguageModelConfig(
        vocab_size=5000,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=VALIDATION_MAX_LENGTH,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
    )

    first_stage = TrainingConfig(
        seed=0,
        batch_size=4,
        num_steps=2,
        log_every=10,
        eval_every=1,
        num_eval_batches=1,
        device="cpu",
        save_checkpoint_path=str(checkpoint_path),
    )
    dummy_batch = build_dummy_batch()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("pretrain.trainer.build_train_dataloader", lambda *args, **kwargs: [dummy_batch])
        monkeypatch.setattr("pretrain.trainer.build_eval_dataloader", lambda *args, **kwargs: [dummy_batch])
        monkeypatch.setattr(
            "pretrain.trainer.evaluate_decoder_language_model_metrics",
            lambda *args, **kwargs: DecoderEvaluationMetrics(
                loss=1.0,
                perplexity=2.0,
                token_accuracy=0.5,
                top5_accuracy=0.75,
                evaluated_tokens=int(dummy_batch.loss_mask.sum().item()),
            ),
        )
        run_decoder_pretraining_loop(model_config, data_config, first_stage)

        second_stage = TrainingConfig(
            seed=0,
            batch_size=4,
            num_steps=4,
            log_every=10,
            eval_every=2,
            num_eval_batches=1,
            device="cpu",
            save_checkpoint_path=str(checkpoint_path),
            resume_checkpoint_path=str(checkpoint_path),
            csv_log_path=str(csv_log_path),
        )
        result = run_decoder_pretraining_loop(model_config, data_config, second_stage)

    assert result.final_step == 4
    assert result.resumed_from_checkpoint == str(checkpoint_path)
    with csv_log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    logged_steps = [int(row[0]) for row in rows[1:] if row[1] == "train" and row[2] == "loss"]
    assert logged_steps == [3, 4]


def test_build_lr_scheduler_creates_warmup_cosine_schedule() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    training_config = TrainingConfig(
        num_steps=10,
        warmup_steps=2,
        min_lr_ratio=0.2,
    )

    scheduler = build_lr_scheduler(optimizer, training_config)

    assert scheduler is not None
    observed_lrs = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        observed_lrs.append(optimizer.param_groups[0]["lr"])

    assert observed_lrs[0] == pytest.approx(0.5)
    assert observed_lrs[1] == pytest.approx(1.0)
    assert observed_lrs[2] < observed_lrs[1]
    assert observed_lrs[-1] > 0.2


def test_resume_restores_scheduler_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "scheduler-resume.pt"
    model_config = DecoderLanguageModelConfig(
        vocab_size=5000,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=VALIDATION_MAX_LENGTH,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
    )

    first_stage = TrainingConfig(
        seed=0,
        batch_size=4,
        num_steps=2,
        log_every=10,
        eval_every=0,
        warmup_steps=1,
        min_lr_ratio=0.5,
        device="cpu",
        save_checkpoint_path=str(checkpoint_path),
    )
    dummy_batch = build_dummy_batch()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("pretrain.trainer.build_train_dataloader", lambda *args, **kwargs: [dummy_batch])
        run_decoder_pretraining_loop(model_config, data_config, first_stage)

        second_stage = TrainingConfig(
            seed=0,
            batch_size=4,
            num_steps=3,
            log_every=10,
            eval_every=0,
            warmup_steps=1,
            min_lr_ratio=0.5,
            device="cpu",
            save_checkpoint_path=str(checkpoint_path),
            resume_checkpoint_path=str(checkpoint_path),
        )
        result = run_decoder_pretraining_loop(model_config, data_config, second_stage)

    assert result.final_step == 3


def test_time_budget_stops_pretraining_early(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "timed.pt"
    model_config = DecoderLanguageModelConfig(
        vocab_size=5000,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=VALIDATION_MAX_LENGTH,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
    )
    training_config = TrainingConfig(
        seed=0,
        batch_size=4,
        num_steps=10,
        max_duration_seconds=1.0,
        log_every=10,
        eval_every=0,
        device="cpu",
        save_checkpoint_path=str(checkpoint_path),
    )

    clock_values = iter([0.0, 0.1, 0.6, 1.2, 1.8])
    monkeypatch.setattr(
        "pretrain.trainer.time.monotonic",
        lambda: next(clock_values),
    )

    monkeypatch.setattr("pretrain.trainer.build_train_dataloader", lambda *args, **kwargs: [build_dummy_batch()])
    result = run_decoder_pretraining_loop(model_config, data_config, training_config)

    assert result.stopped_due_to_time_budget is True
    assert result.final_step == 1
    assert len(result.losses) == 1
    assert result.elapsed_seconds == pytest.approx(1.8)


def test_early_stopping_stops_after_patience(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "early-stop.pt"
    model_config = DecoderLanguageModelConfig(
        vocab_size=5000,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=VALIDATION_MAX_LENGTH,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
    )
    training_config = TrainingConfig(
        seed=0,
        batch_size=4,
        num_steps=20,
        log_every=10,
        eval_every=2,
        num_eval_batches=1,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
        device="cpu",
        save_checkpoint_path=str(checkpoint_path),
    )

    validation_losses = iter([5.0, 5.5, 5.6])

    monkeypatch.setattr(
        "pretrain.trainer.evaluate_decoder_language_model_metrics",
        lambda *args, **kwargs: DecoderEvaluationMetrics(
            loss=next(validation_losses),
            perplexity=1.0,
            token_accuracy=0.0,
            top5_accuracy=0.0,
            evaluated_tokens=1,
        ),
    )
    monkeypatch.setattr("pretrain.trainer.build_train_dataloader", lambda *args, **kwargs: [build_dummy_batch()])
    monkeypatch.setattr("pretrain.trainer.build_eval_dataloader", lambda *args, **kwargs: [build_dummy_batch()])

    result = run_decoder_pretraining_loop(model_config, data_config, training_config)

    assert result.stopped_due_to_early_stopping is True
    assert result.stopped_due_to_time_budget is False
    assert result.final_step == 6
    assert result.best_validation_loss == pytest.approx(5.0)
