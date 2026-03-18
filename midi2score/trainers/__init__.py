"""Training utilities for MIDI2Score."""

from midi2score.trainers.checkpoint import save_checkpoint
from midi2score.trainers.config import TrainingConfig
from midi2score.trainers.pretrain_loop import DecoderPretrainingResult, run_decoder_pretraining_loop
from midi2score.trainers.train_loop import TrainingRunResult, resolve_device, run_training_loop
from midi2score.trainers.train_step import TrainStepOutput, run_train_step

__all__ = [
    "DecoderPretrainingResult",
    "TrainingConfig",
    "TrainingRunResult",
    "TrainStepOutput",
    "resolve_device",
    "run_decoder_pretraining_loop",
    "run_train_step",
    "run_training_loop",
    "save_checkpoint",
]
