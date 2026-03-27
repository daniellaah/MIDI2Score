from __future__ import annotations

from torch import Tensor

from midi2score.models.decoder_config import DecoderLanguageModelConfig
from midi2score.models.decoder_lm_common import BaseDecoderLanguageModel
from midi2score.models.modules import build_local_window_causal_mask


class LocalWindowDecoderLM(BaseDecoderLanguageModel):
    """Decoder LM with causal local-window self-attention."""

    def __init__(self, config: DecoderLanguageModelConfig) -> None:
        super().__init__(config)

    def build_attention_mask(self, input_tokens: Tensor) -> Tensor:
        return build_local_window_causal_mask(
            sequence_length=input_tokens.size(1),
            window_size=self.config.attention_window_size,
            device=input_tokens.device,
        )
