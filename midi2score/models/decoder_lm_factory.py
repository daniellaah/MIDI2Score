from __future__ import annotations

from torch import nn

from midi2score.models.decoder_config import DecoderLanguageModelConfig
from midi2score.models.decoder_lm import TransformerDecoderLM
from midi2score.models.decoder_lm_block_attention import BlockLocalDecoderLM
from midi2score.models.decoder_lm_landmark_attention import LandmarkDecoderLM
from midi2score.models.decoder_lm_local_attention import LocalWindowDecoderLM


def build_decoder_language_model(config: DecoderLanguageModelConfig) -> nn.Module:
    if config.model_type == "standard":
        return TransformerDecoderLM(config)
    if config.model_type == "local_window":
        return LocalWindowDecoderLM(config)
    if config.model_type == "block_local":
        return BlockLocalDecoderLM(config)
    if config.model_type == "landmark":
        return LandmarkDecoderLM(config)
    raise ValueError(f"Unsupported decoder model_type {config.model_type!r}.")
