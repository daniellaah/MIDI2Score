from __future__ import annotations

from torch import Tensor

from midi2score.models.decoder_config import DecoderLanguageModelConfig
from midi2score.models.decoder_lm_common import BaseDecoderLanguageModel
from midi2score.models.modules import build_alibi_causal_mask, build_causal_mask


class TransformerDecoderLM(BaseDecoderLanguageModel):
    """Baseline decoder-only Transformer with full causal self-attention."""

    def __init__(self, config: DecoderLanguageModelConfig) -> None:
        super().__init__(config)

    def build_attention_mask(self, input_tokens: Tensor) -> Tensor:
        if self.config.position_encoding_type == "alibi":
            return build_alibi_causal_mask(
                sequence_length=input_tokens.size(1),
                num_heads=self.config.nhead,
                batch_size=input_tokens.size(0),
                device=input_tokens.device,
            )
        return build_causal_mask(input_tokens.size(1), device=input_tokens.device)
