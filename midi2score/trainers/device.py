from __future__ import annotations

import torch


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
