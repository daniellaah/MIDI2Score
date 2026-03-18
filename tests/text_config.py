from mid2sscore.models.config import ModelConfig


def test_model_config_validate():
    cfg = ModelConfig(
        src_vocab_size=128,
        tgt_vocab_size=256,
        d_model=256,
        nhead=8,
    )
    cfg.validate()
