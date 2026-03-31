from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class DecoderLanguageModelConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "relu"
    position_encoding_type: str = "sinusoidal"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_length: int = 512

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu.")
        if self.position_encoding_type not in {"sinusoidal", "learned", "alibi"}:
            raise ValueError("position_encoding_type is unsupported.")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.encoding[:, : tokens.size(1)]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_length, d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        return self.embedding(positions)


class ZeroPositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: Tensor) -> Tensor:
        return torch.zeros((1, tokens.size(1), self.d_model), device=tokens.device)


def build_positional_encoding(encoding_type: str, d_model: int, max_length: int) -> nn.Module:
    if encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_length)
    if encoding_type == "learned":
        return LearnedPositionalEncoding(d_model, max_length)
    if encoding_type == "alibi":
        return ZeroPositionalEncoding(d_model)
    raise ValueError(f"Unsupported positional encoding {encoding_type!r}.")


def build_causal_mask(sequence_length: int, device: torch.device | str) -> Tensor:
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


def build_alibi_causal_mask(
    *,
    sequence_length: int,
    num_heads: int,
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    slopes = torch.tensor(
        [2 ** (-(8.0 * (index + 1) / num_heads)) for index in range(num_heads)],
        dtype=torch.float32,
        device=device,
    ).view(num_heads, 1, 1)
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    distance = (positions[:, None] - positions[None, :]).clamp_min(0.0)
    future = positions[:, None] < positions[None, :]
    bias = (-slopes * distance).masked_fill(future.unsqueeze(0), float("-inf"))
    return bias.repeat_interleave(batch_size, dim=0)


def _activation(name: str):
    return torch.relu if name == "relu" else torch.nn.functional.gelu


def _normalize_padding_mask(mask: Tensor | None, attn_mask: Tensor) -> Tensor | None:
    if mask is None or attn_mask.dtype == torch.bool:
        return mask
    return mask.to(dtype=attn_mask.dtype).masked_fill(mask, float("-inf"))


class TransformerDecoderLayer(nn.Module):
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
        self.activation = _activation(activation)

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
            key_padding_mask=_normalize_padding_mask(tgt_padding_mask, tgt_causal_mask),
            need_weights=False,
        )[0]
        x = self.norm1(x + self.dropout1(self_attn_output))
        if memory is not None:
            cross_attn_output = self.cross_attn(
                x,
                memory,
                memory,
                key_padding_mask=_normalize_padding_mask(memory_padding_mask, tgt_causal_mask),
                need_weights=False,
            )[0]
            x = self.norm2(x + self.dropout2(cross_attn_output))
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm3(x + self.dropout3(ff_output))


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
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                tgt_causal_mask=tgt_causal_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory=memory,
                memory_padding_mask=memory_padding_mask,
            )
        return self.norm(x)


class TransformerDecoderLM(nn.Module):
    def __init__(self, config: DecoderLanguageModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_encoding = build_positional_encoding(
            config.position_encoding_type,
            config.d_model,
            config.max_length,
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
        embeddings = self.tgt_embedding(input_tokens) * math.sqrt(self.config.d_model)
        return self.dropout(embeddings + self.position_encoding(input_tokens))

    def forward(self, input_tokens: Tensor, *, padding_mask: Tensor | None = None) -> Tensor:
        decoded_inputs = self.decode(input_tokens)
        if self.config.position_encoding_type == "alibi":
            causal_mask = build_alibi_causal_mask(
                sequence_length=input_tokens.size(1),
                num_heads=self.config.nhead,
                batch_size=input_tokens.size(0),
                device=input_tokens.device,
            )
        else:
            causal_mask = build_causal_mask(input_tokens.size(1), device=input_tokens.device)
        hidden_states = self.decoder(
            decoded_inputs,
            tgt_causal_mask=causal_mask,
            tgt_padding_mask=padding_mask,
        )
        return self.output_projection(hidden_states)
