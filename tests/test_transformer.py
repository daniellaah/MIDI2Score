import torch

from midi2score.data import FakeDataConfig, build_fake_dataloader
from midi2score.models import ModelConfig, TransformerSeq2Seq
from midi2score.trainers import run_train_step


def build_small_batch() -> tuple[ModelConfig, object]:
    model_config = ModelConfig(
        src_vocab_size=64,
        tgt_vocab_size=72,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        max_source_length=32,
        max_target_length=32,
    )
    data_config = FakeDataConfig(
        num_samples=8,
        src_vocab_size=model_config.src_vocab_size,
        tgt_vocab_size=model_config.tgt_vocab_size,
        min_source_length=5,
        max_source_length=10,
        min_target_length=5,
        max_target_length=10,
        seed=13,
    )
    batch = next(iter(build_fake_dataloader(data_config, batch_size=4, shuffle=False)))
    return model_config, batch


def test_transformer_forward_produces_vocab_logits() -> None:
    model_config, batch = build_small_batch()
    model = TransformerSeq2Seq(model_config)

    logits = model(
        batch.src_tokens,
        batch.tgt_input_tokens,
        src_padding_mask=batch.src_padding_mask,
        tgt_padding_mask=batch.tgt_padding_mask,
    )

    assert logits.shape == (
        batch.tgt_input_tokens.size(0),
        batch.tgt_input_tokens.size(1),
        model_config.tgt_vocab_size,
    )


def test_run_train_step_updates_model_parameters() -> None:
    torch.manual_seed(0)
    model_config, batch = build_small_batch()
    model = TransformerSeq2Seq(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = model.output_projection.weight.detach().clone()
    output = run_train_step(
        model,
        batch,
        optimizer,
        pad_token_id=model_config.pad_token_id,
    )
    after = model.output_projection.weight.detach()

    assert torch.isfinite(torch.tensor(output.loss))
    assert output.logits.shape[-1] == model_config.tgt_vocab_size
    assert not torch.equal(before, after)
