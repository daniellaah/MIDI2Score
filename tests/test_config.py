import pytest

from midi2score.model import DecoderLanguageModelConfig
from midi2score.train import TrainingConfig


def test_training_config_rejects_non_positive_time_budget() -> None:
    with pytest.raises(ValueError, match="max_duration_seconds"):
        TrainingConfig(max_duration_seconds=0.0)


def test_training_config_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        TrainingConfig(weight_decay=-0.1)


def test_training_config_rejects_invalid_beta1() -> None:
    with pytest.raises(ValueError, match="beta1"):
        TrainingConfig(beta1=1.0)


def test_training_config_rejects_invalid_beta2() -> None:
    with pytest.raises(ValueError, match="beta2"):
        TrainingConfig(beta2=0.0)


def test_training_config_rejects_non_positive_grad_clip_norm() -> None:
    with pytest.raises(ValueError, match="grad_clip_norm"):
        TrainingConfig(grad_clip_norm=0.0)


def test_training_config_rejects_negative_warmup_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingConfig(warmup_steps=-1)


def test_training_config_rejects_invalid_min_lr_ratio() -> None:
    with pytest.raises(ValueError, match="min_lr_ratio"):
        TrainingConfig(min_lr_ratio=1.5)


def test_training_config_rejects_non_positive_early_stopping_patience() -> None:
    with pytest.raises(ValueError, match="early_stopping_patience"):
        TrainingConfig(early_stopping_patience=0, eval_every=1)


def test_training_config_requires_eval_for_early_stopping() -> None:
    with pytest.raises(ValueError, match="eval_every > 0"):
        TrainingConfig(early_stopping_patience=3, eval_every=0)


def test_training_config_rejects_negative_target_validation_loss() -> None:
    with pytest.raises(ValueError, match="target_validation_loss"):
        TrainingConfig(target_validation_loss=-0.1, eval_every=1)


def test_training_config_requires_eval_for_target_validation_loss() -> None:
    with pytest.raises(ValueError, match="eval_every > 0"):
        TrainingConfig(target_validation_loss=1.0, eval_every=0)

def test_training_config_rejects_non_positive_eval_batch_size() -> None:
    with pytest.raises(ValueError, match="eval_batch_size"):
        TrainingConfig(eval_batch_size=0)


def test_decoder_language_model_config_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        DecoderLanguageModelConfig(vocab_size=128, activation="swish")


def test_decoder_language_model_config_rejects_unknown_positional_encoding() -> None:
    with pytest.raises(ValueError, match="positional_encoding"):
        DecoderLanguageModelConfig(vocab_size=128, positional_encoding="rope")
