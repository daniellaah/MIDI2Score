from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from midi2score.models.config import ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positions from the original Transformer paper."""

    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )

        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)

        # Register as a buffer so it moves with the module but is not trained.
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.size(1)
        return self.encoding[:, :sequence_length]


class TransformerSeq2Seq(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.src_embedding = nn.Embedding(
            num_embeddings=config.src_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.tgt_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_length=max(config.max_source_length, config.max_target_length),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier init is a common, stable default for Transformer baselines.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(self, src_tokens: Tensor) -> Tensor:
        return self.dropout(
            self.src_embedding(src_tokens) * math.sqrt(self.config.d_model)
            + self.position_encoding(src_tokens)
        )

    def decode(self, tgt_input_tokens: Tensor) -> Tensor:
        return self.dropout(
            self.tgt_embedding(tgt_input_tokens) * math.sqrt(self.config.d_model)
            + self.position_encoding(tgt_input_tokens)
        )

    def forward(
        self,
        src_tokens: Tensor,
        tgt_input_tokens: Tensor,
        *,
        src_padding_mask: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        src_embeddings = self.encode(src_tokens)
        tgt_embeddings = self.decode(tgt_input_tokens)

        # The decoder must not attend to future target positions during training.
        causal_mask = self.build_causal_mask(
            sequence_length=tgt_input_tokens.size(1),
            device=tgt_input_tokens.device,
        )

        hidden_states = self.transformer(
            src=src_embeddings,
            tgt=tgt_embeddings,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output_projection(hidden_states)

    @staticmethod
    def build_causal_mask(sequence_length: int, device: torch.device | str) -> Tensor:
        return torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
            diagonal=1,
        )
