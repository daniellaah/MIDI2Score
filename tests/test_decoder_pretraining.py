from pathlib import Path

import torch

from midi2score.data import FakeLanguageModelDataConfig, build_fake_language_model_dataloader
from midi2score.models import (
    DecoderLanguageModelConfig,
    ModelConfig,
    TransformerDecoderLM,
    TransformerSeq2Seq,
)
from midi2score.trainers import TrainingConfig, run_decoder_pretraining_loop


def test_decoder_language_model_forward_produces_vocab_logits() -> None:
    model_config = DecoderLanguageModelConfig(
        vocab_size=64,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=32,
    )
    data_config = FakeLanguageModelDataConfig(
        num_samples=8,
        vocab_size=64,
        min_length=5,
        max_length=9,
        seed=31,
    )
    batch = next(iter(build_fake_language_model_dataloader(data_config, batch_size=4)))
    model = TransformerDecoderLM(model_config)

    logits = model(batch.input_tokens, padding_mask=batch.padding_mask)

    assert logits.shape == (
        batch.input_tokens.size(0),
        batch.input_tokens.size(1),
        model_config.vocab_size,
    )


def test_decoder_pretraining_loop_saves_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "decoder.pt"
    model_config = DecoderLanguageModelConfig(
        vocab_size=64,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=32,
    )
    data_config = FakeLanguageModelDataConfig(
        num_samples=8,
        vocab_size=64,
        min_length=5,
        max_length=9,
        seed=37,
    )
    training_config = TrainingConfig(
        batch_size=4,
        num_steps=2,
        log_every=10,
        device="cpu",
        save_checkpoint_path=str(checkpoint_path),
    )

    result = run_decoder_pretraining_loop(model_config, data_config, training_config)

    assert len(result.losses) == 2
    assert checkpoint_path.exists()


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
