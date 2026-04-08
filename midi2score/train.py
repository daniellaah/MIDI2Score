from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.tensorboard import SummaryWriter

from midi2score.data import LmxDataConfig, build_language_model_dataloader
from midi2score.model import DecoderLanguageModelConfig, TransformerDecoderLM


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 0
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    label_smoothing: float = 0.0
    scheduler: str = "none"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    num_steps: int = 10
    max_duration_seconds: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    log_every: int = 1
    eval_every: int = 0
    num_eval_batches: int | None = None
    device: str = "auto"
    save_checkpoint_path: str | None = None
    save_best_checkpoint_path: str | None = None
    resume_checkpoint_path: str | None = None
    csv_log_path: str | None = None
    tensorboard_log_dir: str | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive.")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0.0, 1.0).")
        if self.scheduler not in {"none", "linear", "cosine"}:
            raise ValueError(f"Unsupported scheduler {self.scheduler!r}.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0.0:
            raise ValueError("max_duration_seconds must be positive.")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.early_stopping_patience is not None and self.eval_every == 0:
            raise ValueError("early_stopping_patience requires eval_every > 0.")
        if self.num_eval_batches is not None and self.num_eval_batches <= 0:
            raise ValueError("num_eval_batches must be positive.")
        if self.resume_checkpoint_path is not None and not Path(self.resume_checkpoint_path).exists():
            raise ValueError(f"resume_checkpoint_path does not exist: {self.resume_checkpoint_path}")

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


@dataclass(slots=True)
class DecoderEvaluationMetrics:
    loss: float
    perplexity: float
    token_accuracy: float
    top5_accuracy: float
    evaluated_tokens: int


@dataclass(slots=True)
class DecoderPretrainingResult:
    losses: list[float]
    validation_losses: list[tuple[int, float]]
    device: str
    checkpoint_path: str | None
    best_checkpoint_path: str | None
    best_validation_loss: float | None
    final_step: int
    resumed_from_checkpoint: str | None
    optimizer_state_loaded: bool
    elapsed_seconds: float
    stopped_due_to_time_budget: bool
    stopped_due_to_early_stopping: bool
    average_step_time_seconds: float
    average_tokens_per_second: float


@dataclass(slots=True)
class ResumeState:
    start_step: int
    best_validation_loss: float | None
    optimizer_loaded: bool
    scheduler_loaded: bool


@dataclass(slots=True)
class TrainingLogger:
    csv_path: str | None = None
    tensorboard_log_dir: str | None = None
    _writer: SummaryWriter | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.csv_path is not None:
            csv_file = Path(self.csv_path)
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            if not csv_file.exists():
                with csv_file.open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow(["step", "split", "metric", "value"])
        if self.tensorboard_log_dir is not None:
            log_dir = Path(self.tensorboard_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, *, step: int, split: str, value: float, metric: str = "loss") -> None:
        if self.csv_path is not None:
            with Path(self.csv_path).open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([step, split, metric, f"{value:.8f}"])
        if self._writer is not None:
            self._writer.add_scalar(f"{metric}/{split}", value, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(path: str, payload: dict) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


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


def build_lr_scheduler(optimizer: Adam, training_config: TrainingConfig) -> LambdaLR | None:
    if training_config.scheduler == "none" and training_config.warmup_steps == 0:
        return None

    total_steps = training_config.num_steps
    warmup_steps = training_config.warmup_steps
    min_lr_ratio = training_config.min_lr_ratio
    scheduler_name = training_config.scheduler

    def lr_lambda(step: int) -> float:
        current_step = max(step, 1)
        if warmup_steps > 0 and current_step <= warmup_steps:
            return current_step / warmup_steps
        if scheduler_name == "none":
            return 1.0
        decay_start = max(warmup_steps, 1)
        decay_steps = max(total_steps - decay_start, 1)
        progress = min(max((current_step - decay_start) / decay_steps, 0.0), 1.0)
        if scheduler_name == "linear":
            return 1.0 - progress * (1.0 - min_lr_ratio)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate_decoder_language_model(
    model: TransformerDecoderLM,
    loader,
    *,
    pad_token_id: int,
    device: torch.device | str,
    num_batches: int | None = None,
) -> float:
    return evaluate_decoder_language_model_metrics(
        model,
        loader,
        pad_token_id=pad_token_id,
        device=device,
        num_batches=num_batches,
    ).loss


def evaluate_decoder_language_model_metrics(
    model: TransformerDecoderLM,
    loader,
    *,
    pad_token_id: int,
    device: torch.device | str,
    num_batches: int | None = None,
) -> DecoderEvaluationMetrics:
    model.eval()
    total_loss = 0.0
    total_valid_tokens = 0
    total_correct_tokens = 0
    total_top5_correct_tokens = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            logits = model(batch.input_tokens, padding_mask=batch.padding_mask)
            flat_targets = batch.output_tokens.reshape(-1)
            valid_mask = flat_targets.ne(pad_token_id)
            if batch.loss_mask is not None:
                valid_mask &= batch.loss_mask.reshape(-1)
            valid_targets = flat_targets[valid_mask]
            if valid_targets.numel() > 0:
                flat_logits = logits.reshape(-1, logits.size(-1))[valid_mask]
                total_loss += float(
                    F.cross_entropy(
                        flat_logits,
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


def run_decoder_pretraining_loop(
    model_config: DecoderLanguageModelConfig,
    data_config: LmxDataConfig,
    training_config: TrainingConfig,
) -> DecoderPretrainingResult:
    _validate_setup(model_config, data_config)
    torch.manual_seed(training_config.seed)
    device = resolve_device(training_config.device)
    train_loader = build_language_model_dataloader(
        data_config,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )
    validation_loader = None
    if training_config.eval_every > 0:
        validation_loader = build_language_model_dataloader(
            replace(data_config, split="validation"),
            batch_size=training_config.batch_size,
            seed=training_config.seed,
            shuffle=False,
        )

    model = TransformerDecoderLM(model_config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, training_config)
    logger = TrainingLogger(
        csv_path=training_config.csv_log_path,
        tensorboard_log_dir=training_config.tensorboard_log_dir,
    )

    losses: list[float] = []
    validation_losses: list[tuple[int, float]] = []
    start_step = 0
    best_validation_loss: float | None = None
    best_step = 0
    consecutive_non_improving_evals = 0
    optimizer_state_loaded = False
    if training_config.resume_checkpoint_path is not None:
        resume_state = load_checkpoint_for_resume(
            training_config.resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_step = resume_state.start_step
        best_validation_loss = resume_state.best_validation_loss
        best_step = start_step
        optimizer_state_loaded = resume_state.optimizer_loaded

    iterator = iter(train_loader)
    started_at = time.monotonic()
    cumulative_step_seconds = 0.0
    cumulative_tokens = 0
    final_step = start_step
    stopped_due_to_time_budget = False
    stopped_due_to_early_stopping = False

    try:
        for step in range(start_step + 1, training_config.num_steps + 1):
            if (
                training_config.max_duration_seconds is not None
                and step > start_step + 1
                and time.monotonic() - started_at >= training_config.max_duration_seconds
            ):
                stopped_due_to_time_budget = True
                break
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device)
            step_started_at = time.monotonic()
            logits = model(batch.input_tokens, padding_mask=batch.padding_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.output_tokens.reshape(-1),
                ignore_index=model_config.pad_token_id,
                label_smoothing=training_config.label_smoothing,
            )
            loss.backward()
            if training_config.grad_clip_norm is not None:
                clip_grad_norm_(model.parameters(), max_norm=training_config.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            step_elapsed_seconds = time.monotonic() - step_started_at
            step_tokens = int(batch.output_tokens.ne(model_config.pad_token_id).sum().item())
            cumulative_step_seconds += step_elapsed_seconds
            cumulative_tokens += step_tokens
            loss_value = float(loss.item())
            losses.append(loss_value)
            final_step = step
            logger.log_scalar(step=step, split="train", value=loss_value)
            logger.log_scalar(step=step, split="train", metric="step_time_seconds", value=step_elapsed_seconds)
            logger.log_scalar(
                step=step,
                split="train",
                metric="tokens_per_second",
                value=step_tokens / max(step_elapsed_seconds, 1e-12),
            )

            if step % training_config.log_every == 0:
                print(
                    f"step={step} pretrain_loss={loss_value:.4f} "
                    f"step_time={step_elapsed_seconds:.4f}s "
                    f"tokens_per_second={step_tokens / max(step_elapsed_seconds, 1e-12):.1f} "
                    f"device={device}"
                )

            if validation_loader is not None and step % training_config.eval_every == 0:
                metrics = evaluate_decoder_language_model_metrics(
                    model,
                    validation_loader,
                    pad_token_id=model_config.pad_token_id,
                    device=device,
                    num_batches=training_config.num_eval_batches,
                )
                validation_losses.append((step, metrics.loss))
                logger.log_scalar(step=step, split="validation", value=metrics.loss)
                logger.log_scalar(step=step, split="validation", metric="perplexity", value=metrics.perplexity)
                logger.log_scalar(step=step, split="validation", metric="token_accuracy", value=metrics.token_accuracy)
                logger.log_scalar(step=step, split="validation", metric="top5_accuracy", value=metrics.top5_accuracy)
                print(
                    f"step={step} validation_loss={metrics.loss:.4f} "
                    f"perplexity={metrics.perplexity:.4f} "
                    f"token_acc={metrics.token_accuracy:.4f} "
                    f"top5_acc={metrics.top5_accuracy:.4f} "
                    f"device={device}"
                )

                improved = best_validation_loss is None or (
                    metrics.loss < best_validation_loss - training_config.early_stopping_min_delta
                )
                if improved:
                    best_validation_loss = metrics.loss
                    best_step = step
                    consecutive_non_improving_evals = 0
                    if training_config.save_best_checkpoint_path is not None:
                        save_checkpoint(
                            training_config.save_best_checkpoint_path,
                            _checkpoint_payload(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                model_config=model_config,
                                data_config=data_config,
                                training_config=training_config,
                                step=step,
                                best_validation_loss=best_validation_loss,
                            ),
                        )
                else:
                    consecutive_non_improving_evals += 1

                if (
                    training_config.early_stopping_patience is not None
                    and consecutive_non_improving_evals >= training_config.early_stopping_patience
                ):
                    stopped_due_to_early_stopping = True
                    print(
                        "early_stopping_triggered "
                        f"step={step} best_step={best_step} "
                        f"best_validation_loss={best_validation_loss:.4f} "
                        f"patience={training_config.early_stopping_patience} "
                        f"min_delta={training_config.early_stopping_min_delta}"
                    )
                    break
    finally:
        logger.close()

    elapsed_seconds = time.monotonic() - started_at
    if training_config.save_checkpoint_path is not None:
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            step=final_step,
            best_validation_loss=best_validation_loss,
        )
        payload["elapsed_seconds"] = elapsed_seconds
        payload["stopped_due_to_time_budget"] = stopped_due_to_time_budget
        payload["stopped_due_to_early_stopping"] = stopped_due_to_early_stopping
        save_checkpoint(training_config.save_checkpoint_path, payload)

    return DecoderPretrainingResult(
        losses=losses,
        validation_losses=validation_losses,
        device=device,
        checkpoint_path=training_config.save_checkpoint_path,
        best_checkpoint_path=training_config.save_best_checkpoint_path,
        best_validation_loss=best_validation_loss,
        final_step=final_step,
        resumed_from_checkpoint=training_config.resume_checkpoint_path,
        optimizer_state_loaded=optimizer_state_loaded,
        elapsed_seconds=elapsed_seconds,
        stopped_due_to_time_budget=stopped_due_to_time_budget,
        stopped_due_to_early_stopping=stopped_due_to_early_stopping,
        average_step_time_seconds=cumulative_step_seconds / max(len(losses), 1),
        average_tokens_per_second=cumulative_tokens / max(cumulative_step_seconds, 1e-12),
    )


def _checkpoint_payload(
    *,
    model: TransformerDecoderLM,
    optimizer: Adam,
    scheduler: LambdaLR | None,
    model_config: DecoderLanguageModelConfig,
    data_config: LmxDataConfig,
    training_config: TrainingConfig,
    step: int,
    best_validation_loss: float | None,
) -> dict:
    return {
        "model_type": "decoder_lm",
        "model_config": model_config.to_dict(),
        "data_config": data_config.to_dict(),
        "training_config": training_config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "step": step,
        "validation_loss": best_validation_loss,
        "best_validation_loss": best_validation_loss,
    }


def _validate_setup(
    model_config: DecoderLanguageModelConfig,
    data_config: LmxDataConfig,
) -> None:
    if model_config.max_length < data_config.max_length:
        raise ValueError("model.max_length must be >= data.max_length.")
