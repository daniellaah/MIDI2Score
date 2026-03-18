from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import Adam

from midi2score.data import FakeLanguageModelDataConfig, build_fake_language_model_dataloader
from midi2score.models import DecoderLanguageModelConfig, TransformerDecoderLM
from midi2score.trainers.checkpoint import save_checkpoint
from midi2score.trainers.config import TrainingConfig
from midi2score.trainers.train_loop import resolve_device


@dataclass(slots=True)
class DecoderPretrainingResult:
    losses: list[float]
    device: str
    checkpoint_path: str | None


def run_decoder_pretraining_loop(
    model_config: DecoderLanguageModelConfig,
    data_config: FakeLanguageModelDataConfig,
    training_config: TrainingConfig,
) -> DecoderPretrainingResult:
    device = resolve_device(training_config.device)
    loader = build_fake_language_model_dataloader(
        data_config,
        batch_size=training_config.batch_size,
        shuffle=False,
    )
    model = TransformerDecoderLM(model_config).to(device)
    optimizer = Adam(model.parameters(), lr=training_config.learning_rate)

    losses: list[float] = []
    data_iterator = iter(loader)

    for step in range(1, training_config.num_steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch = batch.to(device)
        logits = model(batch.input_tokens, padding_mask=batch.padding_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.output_tokens.reshape(-1),
            ignore_index=model_config.pad_token_id,
        )
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())
        losses.append(loss_value)

        if step % training_config.log_every == 0:
            print(f"step={step} pretrain_loss={loss_value:.4f} device={device}")

    if training_config.save_checkpoint_path is not None:
        save_checkpoint(
            training_config.save_checkpoint_path,
            {
                "model_type": "decoder_lm",
                "model_config": model_config.to_dict(),
                "model_state": model.state_dict(),
            },
        )

    return DecoderPretrainingResult(
        losses=losses,
        device=device,
        checkpoint_path=training_config.save_checkpoint_path,
    )
