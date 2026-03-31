"""MIDI2Score package."""

from midi2score.data import (
    HuggingFaceLanguageModelDataset,
    LanguageModelBatch,
    LanguageModelDataConfig,
    LengthBucketBatchSampler,
    build_language_model_dataloader,
    collate_language_model_batch,
)
from midi2score.model import DecoderLanguageModelConfig, TransformerDecoderLM
from midi2score.train import (
    DecoderEvaluationMetrics,
    DecoderPretrainingResult,
    TrainingConfig,
    build_lr_scheduler,
    evaluate_decoder_language_model,
    evaluate_decoder_language_model_metrics,
    run_decoder_pretraining_loop,
)

__all__ = [
    "DecoderEvaluationMetrics",
    "DecoderLanguageModelConfig",
    "DecoderPretrainingResult",
    "HuggingFaceLanguageModelDataset",
    "LanguageModelBatch",
    "LanguageModelDataConfig",
    "LengthBucketBatchSampler",
    "TrainingConfig",
    "TransformerDecoderLM",
    "build_language_model_dataloader",
    "build_lr_scheduler",
    "collate_language_model_batch",
    "evaluate_decoder_language_model",
    "evaluate_decoder_language_model_metrics",
    "run_decoder_pretraining_loop",
]
