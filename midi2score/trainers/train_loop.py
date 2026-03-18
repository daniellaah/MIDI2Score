from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Adam

from midi2score.data import FakeDataConfig, build_fake_dataloader
from midi2score.models import ModelConfig, TransformerSeq2Seq
from midi2score.trainers.config import TrainingConfig
from midi2score.trainers.train_step import run_train_step


@dataclass(slots=True)
class TrainingRunResult:
    losses: list[float]
    device: str


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_training_loop(
    model_config: ModelConfig,
    data_config: FakeDataConfig,
    training_config: TrainingConfig,
) -> TrainingRunResult:
    device = resolve_device(training_config.device)

    loader = build_fake_dataloader(
        data_config,
        batch_size=training_config.batch_size,
        shuffle=False,
    )
    model = TransformerSeq2Seq(model_config).to(device)
    if training_config.pretrained_decoder_checkpoint is not None:
        model.load_pretrained_decoder_checkpoint(training_config.pretrained_decoder_checkpoint)
    optimizer = Adam(model.parameters(), lr=training_config.learning_rate)

    losses: list[float] = []
    data_iterator = iter(loader)

    for step in range(1, training_config.num_steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            # Reuse the fake dataset when we exhaust one pass. This keeps the
            # minimal training loop simple while still behaving like a real loop.
            data_iterator = iter(loader)
            batch = next(data_iterator)

        output = run_train_step(
            model,
            batch,
            optimizer,
            pad_token_id=model_config.pad_token_id,
            device=device,
        )
        losses.append(output.loss)

        if step % training_config.log_every == 0:
            print(f"step={step} loss={output.loss:.4f} device={device}")

    return TrainingRunResult(losses=losses, device=device)
