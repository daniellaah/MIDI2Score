from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from midi2score.data import LanguageModelDataConfig, build_language_model_dataloader
from midi2score.models import DecoderLanguageModelConfig, TransformerDecoderLM
from midi2score.trainers.checkpoint import save_checkpoint
from midi2score.trainers.config import TrainingConfig
from midi2score.trainers.device import resolve_device
from midi2score.trainers.logging import TrainingLogger
from midi2score.trainers.resume import load_checkpoint_for_resume


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


def run_decoder_pretraining_loop(
    model_config: DecoderLanguageModelConfig,
    data_config: LanguageModelDataConfig,
    training_config: TrainingConfig,
) -> DecoderPretrainingResult:
    _validate_pretraining_setup(model_config, data_config)
    device = resolve_device(training_config.device)
    train_loader = build_language_model_dataloader(
        data_config,
        batch_size=training_config.batch_size,
    )
    validation_loader = None
    if training_config.eval_every > 0:
        validation_loader = build_language_model_dataloader(
            replace(data_config, split="validation", random_crop=False),
            batch_size=training_config.batch_size,
            shuffle=False,
    )
    model = TransformerDecoderLM(model_config).to(device)
    optimizer = Adam(model.parameters(), lr=training_config.learning_rate)
    scheduler = build_lr_scheduler(optimizer, training_config)
    logger = TrainingLogger(
        csv_path=training_config.csv_log_path,
        tensorboard_log_dir=training_config.tensorboard_log_dir,
    )

    losses: list[float] = []
    validation_losses: list[tuple[int, float]] = []
    start_step = 0
    best_validation_loss: float | None = None
    best_step = start_step
    consecutive_non_improving_evals = 0
    optimizer_state_loaded = False
    scheduler_state_loaded = False
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
        scheduler_state_loaded = resume_state.scheduler_loaded
        print(
            f"resumed_from={training_config.resume_checkpoint_path} "
            f"start_step={start_step} optimizer_state_loaded={optimizer_state_loaded} "
            f"scheduler_state_loaded={scheduler_state_loaded}"
        )
    data_iterator = iter(train_loader)
    loop_started_at = time.monotonic()
    final_step = start_step
    stopped_due_to_time_budget = False
    stopped_due_to_early_stopping = False

    try:
        for step in range(start_step + 1, training_config.num_steps + 1):
            if (
                training_config.max_duration_seconds is not None
                and step > start_step + 1
                and time.monotonic() - loop_started_at >= training_config.max_duration_seconds
            ):
                stopped_due_to_time_budget = True
                break
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
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
            if scheduler is not None:
                scheduler.step()

            loss_value = float(loss.detach().item())
            losses.append(loss_value)
            final_step = step
            logger.log_scalar(step=step, split="train", loss=loss_value)

            if step % training_config.log_every == 0:
                print(f"step={step} pretrain_loss={loss_value:.4f} device={device}")

            if validation_loader is not None and step % training_config.eval_every == 0:
                validation_loss = evaluate_decoder_language_model(
                    model,
                    validation_loader,
                    pad_token_id=model_config.pad_token_id,
                    device=device,
                    num_batches=training_config.num_eval_batches,
                )
                validation_losses.append((step, validation_loss))
                logger.log_scalar(step=step, split="validation", loss=validation_loss)
                print(f"step={step} validation_loss={validation_loss:.4f} device={device}")

                improved = best_validation_loss is None or (
                    validation_loss < best_validation_loss - training_config.early_stopping_min_delta
                )
                if (
                    training_config.save_best_checkpoint_path is not None
                    and improved
                ):
                    best_validation_loss = validation_loss
                    best_step = step
                    consecutive_non_improving_evals = 0
                    save_checkpoint(
                        training_config.save_best_checkpoint_path,
                        {
                            "model_type": "decoder_lm",
                            "model_config": model_config.to_dict(),
                            "data_config": data_config.to_dict(),
                            "training_config": training_config.to_dict(),
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                            "step": step,
                            "validation_loss": validation_loss,
                            "best_validation_loss": best_validation_loss,
                        },
                    )
                elif improved:
                    best_validation_loss = validation_loss
                    best_step = step
                    consecutive_non_improving_evals = 0
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

    elapsed_seconds = time.monotonic() - loop_started_at

    if training_config.save_checkpoint_path is not None:
        save_checkpoint(
            training_config.save_checkpoint_path,
            {
                "model_type": "decoder_lm",
                "model_config": model_config.to_dict(),
                "data_config": data_config.to_dict(),
                "training_config": training_config.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                "step": final_step,
                "validation_loss": best_validation_loss,
                "best_validation_loss": best_validation_loss,
                "elapsed_seconds": elapsed_seconds,
                "stopped_due_to_time_budget": stopped_due_to_time_budget,
                "stopped_due_to_early_stopping": stopped_due_to_early_stopping,
            },
        )

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
    )


def evaluate_decoder_language_model(
    model: TransformerDecoderLM,
    loader,
    *,
    pad_token_id: int,
    device: torch.device | str,
    num_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            logits = model(batch.input_tokens, padding_mask=batch.padding_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.output_tokens.reshape(-1),
                ignore_index=pad_token_id,
            )
            total_loss += float(loss.item())
            total_batches += 1

            if num_batches is not None and batch_index >= num_batches:
                break

    if total_batches == 0:
        raise ValueError("Validation loader produced zero batches.")
    return total_loss / total_batches


def build_lr_scheduler(optimizer: Adam, training_config: TrainingConfig) -> LambdaLR | None:
    if training_config.scheduler == "none" and training_config.warmup_steps == 0:
        return None

    total_steps = training_config.num_steps
    warmup_steps = training_config.warmup_steps
    min_lr_ratio = training_config.min_lr_ratio
    scheduler_name = training_config.scheduler

    def lr_lambda(step: int) -> float:
        # LambdaLR counts the first scheduler.step() as step=1 in practice.
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
        if scheduler_name == "cosine":
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        raise ValueError(f"Unsupported scheduler {scheduler_name!r}.")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _validate_pretraining_setup(
    model_config: DecoderLanguageModelConfig,
    data_config: LanguageModelDataConfig,
) -> None:
    expected_specials = {
        "pad_token_id": (model_config.pad_token_id, data_config.pad_token_id),
        "bos_token_id": (model_config.bos_token_id, data_config.bos_token_id),
        "eos_token_id": (model_config.eos_token_id, data_config.eos_token_id),
    }
    for field_name, (model_value, data_value) in expected_specials.items():
        if model_value != data_value:
            raise ValueError(
                f"Mismatch between model and data for {field_name}: {model_value} != {data_value}."
            )

    if model_config.max_length < data_config.max_length:
        raise ValueError(
            f"model.max_length ({model_config.max_length}) must be >= "
            f"data.max_length ({data_config.max_length})."
        )

    tokenizer_vocab_size = data_config.tokenizer_vocab_size()
    if tokenizer_vocab_size is not None and tokenizer_vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer_vocab_size} does not match "
            f"model vocab size {model_config.vocab_size}."
        )

    dataset_path = Path(data_config.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
