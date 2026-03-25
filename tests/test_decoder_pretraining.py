from pathlib import Path
import csv

import pytest
import torch

from midi2score.data import (
    HuggingFaceLanguageModelDataset,
    LanguageModelDataConfig,
    build_language_model_dataloader,
)
from midi2score.models import (
    DecoderLanguageModelConfig,
    ModelConfig,
    TransformerDecoderLM,
    TransformerSeq2Seq,
)
from midi2score.trainers import (
    TrainingConfig,
    evaluate_decoder_language_model,
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
    data_config = LanguageModelDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        tokenizer_path="data/tokenizer_rd.json",
        random_crop=False,
    )
    batch = next(iter(build_language_model_dataloader(data_config, batch_size=4, shuffle=False)))
    return model_config, batch


def test_language_model_dataloader_reads_real_dataset() -> None:
    data_config = LanguageModelDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        tokenizer_path="data/tokenizer_rd.json",
        random_crop=False,
    )
    batch = next(iter(build_language_model_dataloader(data_config, batch_size=2, shuffle=False)))

    assert batch.input_tokens.shape[0] == 2
    assert batch.input_tokens.shape == batch.output_tokens.shape
    assert batch.padding_mask.dtype == torch.bool
    assert batch.input_tokens[0, 0].item() == data_config.bos_token_id


def test_training_random_crop_changes_across_repeated_accesses() -> None:
    config = LanguageModelDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=32,
        tokenizer_path="data/tokenizer_rd.json",
        random_crop=True,
        crop_seed=23,
    )
    dataset = HuggingFaceLanguageModelDataset(config)
    long_index = next(
        index
        for index in range(len(dataset))
        if len(dataset.dataset[index]["input_ids"]) > config.max_length * 3
    )

    first_draws = [dataset[long_index]["tokens"].tolist() for _ in range(8)]
    second_dataset = HuggingFaceLanguageModelDataset(config)
    second_draws = [second_dataset[long_index]["tokens"].tolist() for _ in range(8)]

    assert len({tuple(draw) for draw in first_draws}) > 1
    assert first_draws == second_draws


def test_decoder_language_model_forward_produces_vocab_logits() -> None:
    model_config, batch = build_small_real_batch()
    model = TransformerDecoderLM(model_config)

    logits = model(batch.input_tokens, padding_mask=batch.padding_mask)

    assert logits.shape == (
        batch.input_tokens.size(0),
        batch.input_tokens.size(1),
        model_config.vocab_size,
    )


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
    data_config = LanguageModelDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        tokenizer_path="data/tokenizer_rd.json",
        random_crop=False,
    )
    training_config = TrainingConfig(
        batch_size=4,
        num_steps=2,
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
    assert rows[0] == ["step", "split", "loss"]
    assert any(row[1] == "train" for row in rows[1:])
    assert any(row[1] == "validation" for row in rows[1:])


def test_evaluate_decoder_language_model_returns_finite_loss() -> None:
    model_config, batch = build_small_real_batch()
    model = TransformerDecoderLM(model_config)
    data_config = LanguageModelDataConfig(
        dataset_path="data/huggingface",
        split="validation",
        max_length=64,
        tokenizer_path="data/tokenizer_rd.json",
        random_crop=False,
    )
    loader = build_language_model_dataloader(data_config, batch_size=2, shuffle=False)

    loss = evaluate_decoder_language_model(
        model,
        loader,
        pad_token_id=model_config.pad_token_id,
        device="cpu",
        num_batches=1,
    )

    assert loss > 0.0


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
    data_config = LanguageModelDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        tokenizer_path="data/tokenizer_rd.json",
        random_crop=False,
    )

    first_stage = TrainingConfig(
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
    logged_steps = [int(row[0]) for row in rows[1:] if row[1] == "train"]
    assert logged_steps == [3, 4]


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
    data_config = LanguageModelDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        tokenizer_path="data/tokenizer_rd.json",
        random_crop=False,
    )
    training_config = TrainingConfig(
        batch_size=4,
        num_steps=10,
        max_duration_seconds=1.0,
        log_every=10,
        eval_every=0,
        device="cpu",
        save_checkpoint_path=str(checkpoint_path),
    )

    clock_values = iter([0.0, 0.6, 1.2, 1.8])
    monkeypatch.setattr(
        "midi2score.trainers.pretrain_loop.time.monotonic",
        lambda: next(clock_values),
    )

    result = run_decoder_pretraining_loop(model_config, data_config, training_config)

    assert result.stopped_due_to_time_budget is True
    assert result.final_step == 2
    assert len(result.losses) == 2
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
    data_config = LanguageModelDataConfig(
        dataset_path="data/huggingface",
        split="training",
        max_length=64,
        tokenizer_path="data/tokenizer_rd.json",
        random_crop=False,
    )
    training_config = TrainingConfig(
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
        "midi2score.trainers.pretrain_loop.evaluate_decoder_language_model",
        lambda *args, **kwargs: next(validation_losses),
    )

    result = run_decoder_pretraining_loop(model_config, data_config, training_config)

    assert result.stopped_due_to_early_stopping is True
    assert result.stopped_due_to_time_budget is False
    assert result.final_step == 6
    assert result.best_validation_loss == pytest.approx(5.0)


def test_seq2seq_can_load_pretrained_decoder_weights() -> None:
    torch.manual_seed(0)
    decoder_config = DecoderLanguageModelConfig(
        vocab_size=72,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=32,
    )
    decoder_model = TransformerDecoderLM(decoder_config)
    seq2seq_config = ModelConfig(
        src_vocab_size=64,
        tgt_vocab_size=72,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        max_target_length=32,
    )
    seq2seq_model = TransformerSeq2Seq(seq2seq_config)

    before = seq2seq_model.tgt_embedding.weight.detach().clone()
    transferred_keys = seq2seq_model.load_pretrained_decoder_state(decoder_model.state_dict())
    after = seq2seq_model.tgt_embedding.weight.detach()

    assert "tgt_embedding.weight" in transferred_keys
    assert torch.equal(after, decoder_model.tgt_embedding.weight.detach())
    assert not torch.equal(before, after)
