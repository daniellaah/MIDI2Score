from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch
from torch import Tensor, nn


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
        return self.encoding[:, : tokens.size(1)]


def build_causal_mask(sequence_length: int, device: torch.device | str) -> Tensor:
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


def get_activation(name: str) -> Callable[[Tensor], Tensor]:
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return torch.nn.functional.gelu
    raise ValueError(f"Unsupported activation {name!r}.")


class TransformerDecoderLayer(nn.Module):
    """A decoder layer that can optionally skip cross-attention during LM pretraining."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = get_activation(activation)

    def forward(
        self,
        tgt: Tensor,
        *,
        tgt_causal_mask: Tensor,
        tgt_padding_mask: Tensor | None = None,
        memory: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = tgt
        self_attn_output = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_causal_mask,
            key_padding_mask=tgt_padding_mask,
            need_weights=False,
        )[0]
        x = self.norm1(x + self.dropout1(self_attn_output))

        if memory is not None:
            cross_attn_output = self.cross_attn(
                x,
                memory,
                memory,
                key_padding_mask=memory_padding_mask,
                need_weights=False,
            )[0]
            x = self.norm2(x + self.dropout2(cross_attn_output))

        feedforward_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm3(x + self.dropout3(feedforward_output))


class TransformerDecoderStack(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        *,
        tgt_causal_mask: Tensor,
        tgt_padding_mask: Tensor | None = None,
        memory: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = tgt
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                tgt_causal_mask=tgt_causal_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory=memory,
                memory_padding_mask=memory_padding_mask,
            )
        return self.norm(hidden_states)
