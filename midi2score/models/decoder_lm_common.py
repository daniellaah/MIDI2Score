from __future__ import annotations

import math
from abc import ABC, abstractmethod

from torch import Tensor, nn

from midi2score.models.decoder_config import DecoderLanguageModelConfig
from midi2score.models.modules import TransformerDecoderStack, build_positional_encoding


class BaseDecoderLanguageModel(nn.Module, ABC):
    """Shared decoder-LM skeleton for baseline and long-context attention variants."""

    def __init__(self, config: DecoderLanguageModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_encoding = build_positional_encoding(
            encoding_type=config.position_encoding_type,
            d_model=config.d_model,
            max_length=config.max_length,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.decoder = TransformerDecoderStack(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def decode(self, input_tokens: Tensor) -> Tensor:
        return self.dropout(
            self.tgt_embedding(input_tokens) * math.sqrt(self.config.d_model)
            + self.position_encoding(input_tokens)
        )

    @abstractmethod
    def build_attention_mask(self, input_tokens: Tensor) -> Tensor:
        """Return the attention mask expected by nn.MultiheadAttention."""

    def forward(
        self,
        input_tokens: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        decoded_inputs = self.decode(input_tokens)
        causal_mask = self.build_attention_mask(input_tokens)
        hidden_states = self.decoder(
            decoded_inputs,
            tgt_causal_mask=causal_mask,
            tgt_padding_mask=padding_mask,
        )
        return self.output_projection(hidden_states)
