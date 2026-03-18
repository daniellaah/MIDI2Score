from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str, payload: dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
