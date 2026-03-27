from __future__ import annotations

from torch import Tensor

from midi2score.models.decoder_config import DecoderLanguageModelConfig
from midi2score.models.decoder_lm_common import BaseDecoderLanguageModel
from midi2score.models.modules import build_block_local_causal_mask


class BlockLocalDecoderLM(BaseDecoderLanguageModel):
    """Decoder LM with block-local causal self-attention."""

    def __init__(self, config: DecoderLanguageModelConfig) -> None:
        super().__init__(config)

    def build_attention_mask(self, input_tokens: Tensor) -> Tensor:
        return build_block_local_causal_mask(
            sequence_length=input_tokens.size(1),
            block_size=self.config.attention_block_size,
            lookback_blocks=self.config.attention_lookback_blocks,
            device=input_tokens.device,
        )
