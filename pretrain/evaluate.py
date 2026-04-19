from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pretrain.data import LmxDataConfig, build_eval_dataloader
from pretrain.decoder import DecoderLanguageModelConfig, TransformerDecoderLM


@dataclass(slots=True)
class DecoderEvaluationMetrics:
    loss: float
    perplexity: float
    token_accuracy: float
    top5_accuracy: float
    evaluated_tokens: int


@dataclass(slots=True)
class MpsMemoryTracker:
    enabled: bool
    peak_memory_bytes: int = 0

    @classmethod
    def for_device(cls, device: torch.device | str) -> "MpsMemoryTracker":
        return cls(enabled=_supports_mps_memory_tracking(device))

    def record(self) -> None:
        if not self.enabled:
            return
        self.peak_memory_bytes = max(
            self.peak_memory_bytes,
            int(torch.mps.driver_allocated_memory()),
        )


def _supports_mps_memory_tracking(device: torch.device | str) -> bool:
    if str(device) != "mps":
        return False
    if not hasattr(torch, "mps"):
        return False
    return hasattr(torch.mps, "driver_allocated_memory")


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_decoder_language_model(
    model: TransformerDecoderLM,
    loader,
    *,
    pad_token_id: int,
    device: torch.device | str,
    num_batches: int | None = None,
    mps_memory_tracker: MpsMemoryTracker | None = None,
) -> float:
    return evaluate_decoder_language_model_metrics(
        model,
        loader,
        pad_token_id=pad_token_id,
        device=device,
        num_batches=num_batches,
        mps_memory_tracker=mps_memory_tracker,
    ).loss


def evaluate_decoder_language_model_metrics(
    model: TransformerDecoderLM,
    loader,
    *,
    pad_token_id: int,
    device: torch.device | str,
    num_batches: int | None = None,
    mps_memory_tracker: MpsMemoryTracker | None = None,
) -> DecoderEvaluationMetrics:
    model.eval()
    total_loss = 0.0
    total_valid_tokens = 0
    total_correct_tokens = 0
    total_top5_correct_tokens = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            if mps_memory_tracker is not None:
                mps_memory_tracker.record()
            logits = model(batch.input_tokens, padding_mask=batch.padding_mask)
            if mps_memory_tracker is not None:
                mps_memory_tracker.record()
            flat_targets = batch.output_tokens.reshape(-1)
            valid_mask = flat_targets.ne(pad_token_id)
            if batch.loss_mask is not None:
                valid_mask &= batch.loss_mask.reshape(-1)
            valid_targets = flat_targets[valid_mask]
            if valid_targets.numel() > 0:
                flat_logits = logits.reshape(-1, logits.size(-1))[valid_mask]
                total_loss += float(
                    F.cross_entropy(
                        flat_logits.float(),
                        valid_targets,
                        reduction="sum",
                    ).item()
                )
                predictions = flat_logits.argmax(dim=-1)
                topk = flat_logits.topk(k=min(5, flat_logits.size(-1)), dim=-1).indices
                total_valid_tokens += int(valid_targets.numel())
                total_correct_tokens += int(predictions.eq(valid_targets).sum().item())
                total_top5_correct_tokens += int(
                    topk.eq(valid_targets.unsqueeze(-1)).any(dim=-1).sum().item()
                )
            if num_batches is not None and batch_index >= num_batches:
                break

    average_loss = total_loss / total_valid_tokens
    return DecoderEvaluationMetrics(
        loss=average_loss,
        perplexity=float(torch.exp(torch.tensor(average_loss)).item()),
        token_accuracy=total_correct_tokens / total_valid_tokens,
        top5_accuracy=total_top5_correct_tokens / total_valid_tokens,
        evaluated_tokens=total_valid_tokens,
    )


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    model_config: DecoderLanguageModelConfig | None = None,
) -> tuple[TransformerDecoderLM, DecoderLanguageModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved_model_config = model_config
    if resolved_model_config is None:
        raw_model_config = checkpoint.get("model_config")
        if not isinstance(raw_model_config, dict):
            raise ValueError("Checkpoint is missing model_config.")
        resolved_model_config = DecoderLanguageModelConfig(**raw_model_config)

    model = TransformerDecoderLM(resolved_model_config)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    return model, resolved_model_config


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    data_config: LmxDataConfig,
    batch_size: int,
    device: str = "auto",
    num_batches: int | None = None,
    model_config: DecoderLanguageModelConfig | None = None,
) -> tuple[DecoderEvaluationMetrics, str]:
    resolved_device = resolve_device(device)
    model, resolved_model_config = load_model_from_checkpoint(
        checkpoint_path,
        device=resolved_device,
        model_config=model_config,
    )
    loader = build_eval_dataloader(
        data_config,
        batch_size=batch_size,
        pad_token_id=resolved_model_config.pad_token_id,
    )
    tracker = MpsMemoryTracker.for_device(resolved_device)
    metrics = evaluate_decoder_language_model_metrics(
        model,
        loader,
        pad_token_id=resolved_model_config.pad_token_id,
        device=resolved_device,
        num_batches=num_batches,
        mps_memory_tracker=tracker,
    )
    return metrics, resolved_device


def load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu")
