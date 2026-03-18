from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from midi2score.data.fake_dataset import Seq2SeqBatch


@dataclass(slots=True)
class TrainStepOutput:
    loss: float
    logits: Tensor


def run_train_step(
    model: nn.Module,
    batch: Seq2SeqBatch,
    optimizer: Optimizer,
    *,
    pad_token_id: int,
    device: torch.device | str = "cpu",
) -> TrainStepOutput:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch = batch.to(device)
    logits = model(
        batch.src_tokens,
        batch.tgt_input_tokens,
        src_padding_mask=batch.src_padding_mask,
        tgt_padding_mask=batch.tgt_padding_mask,
    )

    # Cross-entropy expects flattened class logits and target ids.
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.tgt_output_tokens.reshape(-1),
        ignore_index=pad_token_id,
    )
    loss.backward()
    optimizer.step()

    return TrainStepOutput(loss=float(loss.detach().item()), logits=logits.detach())
