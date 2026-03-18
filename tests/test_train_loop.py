from midi2score.data import FakeDataConfig
from midi2score.models import ModelConfig
from midi2score.trainers import TrainingConfig, resolve_device, run_training_loop


def test_resolve_device_returns_supported_option() -> None:
    assert resolve_device("cpu") == "cpu"


def test_run_training_loop_returns_loss_history() -> None:
    model_config = ModelConfig(
        src_vocab_size=64,
        tgt_vocab_size=64,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        max_source_length=32,
        max_target_length=32,
    )
    data_config = FakeDataConfig(
        num_samples=6,
        src_vocab_size=model_config.src_vocab_size,
        tgt_vocab_size=model_config.tgt_vocab_size,
        min_source_length=5,
        max_source_length=9,
        min_target_length=5,
        max_target_length=9,
        seed=29,
    )
    training_config = TrainingConfig(
        batch_size=4,
        num_steps=3,
        log_every=10,
        device="cpu",
    )

    result = run_training_loop(model_config, data_config, training_config)

    assert result.device == "cpu"
    assert len(result.losses) == training_config.num_steps
    assert all(loss > 0.0 for loss in result.losses)
