from pathlib import Path
import csv

import pytest
import torch

from midi2score.data import (
    LmxSlidingWindowDataset,
    LmxBatch,
    LmxDataConfig,
    LengthBucketBatchSampler,
    build_language_model_dataloader,
)
from midi2score.model import DecoderLanguageModelConfig, TransformerDecoderLM
from midi2score.train import (
    DecoderEvaluationMetrics,
    TrainingConfig,
    build_lr_scheduler,
    evaluate_decoder_language_model,
    evaluate_decoder_language_model_metrics,
    run_decoder_pretraining_loop,
)


def build_small_real_batch() -> tuple[DecoderLanguageModelConfig, object]:
    model_config = DecoderLanguageModelConfig(
        vocab_size=5000,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=64,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        sliding_window_stride=32,
    )
    batch = next(iter(build_language_model_dataloader(data_config, batch_size=4, seed=0, shuffle=False)))
    return model_config, batch


def test_language_model_dataloader_reads_real_dataset() -> None:
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        sliding_window_stride=32,
    )
    batch = next(iter(build_language_model_dataloader(data_config, batch_size=2, seed=0, shuffle=False)))

    assert batch.input_tokens.shape[0] == 2
    assert batch.input_tokens.shape == batch.output_tokens.shape
    assert batch.padding_mask.dtype == torch.bool
    assert batch.input_tokens[0, 0].item() == 1


def test_sliding_window_expands_long_examples_and_covers_tail() -> None:
    config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=32,
        sliding_window_stride=16,
    )
    dataset = LmxSlidingWindowDataset(config)
    long_index = next(
        index
        for index in range(len(dataset.dataset))
        if len(dataset.dataset[index]["input_ids"]) > config.max_length * 3
    )
    raw_tokens = dataset.dataset[long_index]["input_ids"]
    starts = [
        spec.start
        for spec in dataset._window_index or []
        if spec.raw_index == long_index
    ]

    assert len(starts) > 1
    assert starts[0] == 0
    assert starts[-1] == len(raw_tokens) - config.max_length

    last_window_dataset_index = next(
        i
        for i, spec in enumerate(dataset._window_index or [])
        if spec.raw_index == long_index and spec.start == starts[-1]
    )
    example = dataset[last_window_dataset_index]
    assert example["tokens"].tolist() == raw_tokens[starts[-1] : starts[-1] + config.max_length]
    assert example["loss_mask"].all()


def test_validation_sliding_window_scores_each_target_token_once() -> None:
    config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=32,
        sliding_window_stride=16,
    )
    dataset = LmxSlidingWindowDataset(config)
    long_index = next(
        index
        for index in range(len(dataset.dataset))
        if len(dataset.dataset[index]["input_ids"]) > config.max_length * 3
    )
    raw_tokens = dataset.dataset[long_index]["input_ids"]
    covered_targets: list[int] = []

    for spec in dataset._window_index or []:
        if spec.raw_index != long_index:
            continue
        window = dataset[next(i for i, candidate in enumerate(dataset._window_index or []) if candidate == spec)]
        scored_positions = window["loss_mask"].nonzero(as_tuple=False).flatten().tolist()
        covered_targets.extend(spec.start + 1 + position for position in scored_positions)

    assert covered_targets == list(range(1, len(raw_tokens)))


def test_length_bucket_batch_sampler_groups_examples_by_length() -> None:
    config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        sliding_window_stride=32,
        length_bucketing=True,
    )
    dataset = LmxSlidingWindowDataset(config)
    sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=4,
        seed=23,
        shuffle=True,
    )

    batch_indices = next(iter(sampler))
    lengths = [dataset.sequence_length(index) for index in batch_indices]

    assert lengths == sorted(lengths, reverse=True)


def test_decoder_language_model_forward_produces_vocab_logits() -> None:
    model_config, batch = build_small_real_batch()
    model = TransformerDecoderLM(model_config)

    logits = model(batch.input_tokens, padding_mask=batch.padding_mask)

    assert logits.shape == (
        batch.input_tokens.size(0),
        batch.input_tokens.size(1),
        model_config.vocab_size,
    )


@pytest.mark.parametrize(("activation",), [("relu",), ("gelu",), ("swiglu",), ("geglu",)])
def test_decoder_language_model_supports_activation_variants(activation: str) -> None:
    model_config, batch = build_small_real_batch()
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
    model_config, batch = build_small_real_batch()
    model = TransformerDecoderLM(model_config)

    logits = model(batch.input_tokens, padding_mask=batch.padding_mask)

    assert logits.shape[-1] == model_config.vocab_size
    assert model.output_projection.weight is model.tgt_embedding.weight


def test_validation_bucketing_is_deterministic_without_shuffle() -> None:
    config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=64,
        sliding_window_stride=32,
        length_bucketing=True,
    )
    dataset = LmxSlidingWindowDataset(config)
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
        max_length=64,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        sliding_window_stride=32,
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
        device="cpu",
        save_checkpoint_path=str(checkpoint_path),
        save_best_checkpoint_path=str(best_checkpoint_path),
        csv_log_path=str(csv_log_path),
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
    assert result.average_step_time_seconds > 0.0
    assert result.average_tokens_per_second > 0.0


def test_evaluate_decoder_language_model_returns_finite_loss() -> None:
    model_config, batch = build_small_real_batch()
    model = TransformerDecoderLM(model_config)
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=64,
        sliding_window_stride=32,
    )
    loader = build_language_model_dataloader(data_config, batch_size=2, seed=0, shuffle=False)

    loss = evaluate_decoder_language_model(
        model,
        loader,
        pad_token_id=model_config.pad_token_id,
        device="cpu",
        num_batches=1,
    )

    assert loss > 0.0


def test_evaluate_decoder_language_model_metrics_returns_expected_fields() -> None:
    model_config, _ = build_small_real_batch()
    model = TransformerDecoderLM(model_config)
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=64,
        sliding_window_stride=32,
    )
    loader = build_language_model_dataloader(data_config, batch_size=2, seed=0, shuffle=False)

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
        max_length=64,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        sliding_window_stride=32,
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
        max_length=64,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        sliding_window_stride=32,
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
        max_length=64,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        sliding_window_stride=32,
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
        "midi2score.train.time.monotonic",
        lambda: next(clock_values),
    )

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
        max_length=64,
    )
    data_config = LmxDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        sliding_window_stride=32,
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
        "midi2score.train.evaluate_decoder_language_model_metrics",
        lambda *args, **kwargs: DecoderEvaluationMetrics(
            loss=next(validation_losses),
            perplexity=1.0,
            token_accuracy=0.0,
            top5_accuracy=0.0,
            evaluated_tokens=1,
        ),
    )

    result = run_decoder_pretraining_loop(model_config, data_config, training_config)

    assert result.stopped_due_to_early_stopping is True
    assert result.stopped_due_to_time_budget is False
    assert result.final_step == 6
    assert result.best_validation_loss == pytest.approx(5.0)
