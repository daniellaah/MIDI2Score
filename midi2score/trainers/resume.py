from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(slots=True)
class ResumeState:
    start_step: int
    best_validation_loss: float | None
    optimizer_loaded: bool
    scheduler_loaded: bool


def load_checkpoint_for_resume(
    checkpoint_path: str,
    *,
    model,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> ResumeState:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    optimizer_loaded = False
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_loaded = True

    scheduler_loaded = False
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scheduler_loaded = True

    return ResumeState(
        start_step=int(checkpoint.get("step", 0)),
        best_validation_loss=checkpoint.get("best_validation_loss", checkpoint.get("validation_loss")),
        optimizer_loaded=optimizer_loaded,
        scheduler_loaded=scheduler_loaded,
    )
