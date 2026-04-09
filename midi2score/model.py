from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
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
    norm_type: str = "layernorm"
    residual_layout: str = "post_norm"
    tie_embeddings: bool = False
    position_encoding_type: str = "sinusoidal"
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 2.0
    rope_original_max_length: int | None = None
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_length: int = 512

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if self.activation not in {"relu", "gelu", "swiglu", "geglu"}:
            raise ValueError("activation must be relu, gelu, swiglu, or geglu.")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be layernorm or rmsnorm.")
        if self.residual_layout not in {"post_norm", "pre_norm"}:
            raise ValueError("residual_layout must be post_norm or pre_norm.")
        if self.position_encoding_type not in {"sinusoidal", "learned", "alibi", "rope", "rope_ntk"}:
            raise ValueError(
                "position_encoding_type must be sinusoidal, learned, alibi, rope, or rope_ntk."
            )
        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive.")
        if self.rope_scaling_factor <= 0.0:
            raise ValueError("rope_scaling_factor must be positive.")
        if self.rope_original_max_length is not None and self.rope_original_max_length <= 0:
            raise ValueError("rope_original_max_length must be positive when provided.")

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
        return torch.zeros(
            (1, tokens.size(1), self.d_model),
            dtype=torch.float32,
            device=tokens.device,
        )


def build_positional_encoding(encoding_type: str, d_model: int, max_length: int) -> nn.Module:
    if encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_length)
    if encoding_type == "learned":
        return LearnedPositionalEncoding(d_model, max_length)
    if encoding_type in {"alibi", "rope", "rope_ntk"}:
        return ZeroPositionalEncoding(d_model)
    raise ValueError(f"Unsupported positional encoding {encoding_type!r}.")


def build_causal_mask(sequence_length: int, device: torch.device | str) -> Tensor:
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


def _get_alibi_slopes(num_heads: int, *, device: torch.device | str) -> Tensor:
    return torch.tensor(
        [2 ** (-(8.0 * (head_index + 1) / num_heads)) for head_index in range(num_heads)],
        dtype=torch.float32,
        device=device,
    )


def build_alibi_causal_mask(
    *,
    sequence_length: int,
    num_heads: int,
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    distance = positions[:, None] - positions[None, :]
    future = distance < 0
    distance = distance.clamp_min(0.0)
    slopes = _get_alibi_slopes(num_heads, device=device).view(num_heads, 1, 1)
    bias = -slopes * distance
    bias = bias.masked_fill(future.unsqueeze(0), float("-inf"))
    return bias.repeat_interleave(batch_size, dim=0)


def normalize_key_padding_mask(
    key_padding_mask: Tensor | None,
    *,
    attn_mask: Tensor,
) -> Tensor | None:
    if key_padding_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        return key_padding_mask
    return key_padding_mask.to(dtype=attn_mask.dtype).masked_fill(key_padding_mask, float("-inf"))


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dropout: float,
        rope_type: str,
        rope_theta: float,
        rope_scaling_factor: float,
        rope_original_max_length: int | None,
    ) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        self.dropout = dropout
        self.rope_type = rope_type
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_original_max_length = rope_original_max_length
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _rope_base(self, sequence_length: int) -> float:
        if self.rope_type != "rope_ntk":
            return self.rope_theta
        original_length = self.rope_original_max_length or sequence_length
        if sequence_length <= original_length:
            return self.rope_theta
        exponent = self.head_dim / max(self.head_dim - 2, 1)
        scale = (
            (self.rope_scaling_factor * sequence_length / original_length)
            - (self.rope_scaling_factor - 1.0)
        ) ** exponent
        return self.rope_theta * scale

    def _cos_sin(
        self,
        *,
        sequence_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        inv_freq = 1.0 / (
            self._rope_base(sequence_length)
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        cos = torch.cos(angles).to(dtype=dtype)[None, None, :, :]
        sin = torch.sin(angles).to(dtype=dtype)[None, None, :, :]
        return cos, sin

    @staticmethod
    def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def forward(
        self,
        x: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        q = self.q_proj(x).view(batch_size, sequence_length, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, sequence_length, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, sequence_length, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self._cos_sin(sequence_length=sequence_length, device=x.device, dtype=q.dtype)
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        attn_mask = torch.full(
            (batch_size, 1, sequence_length, sequence_length),
            0.0,
            dtype=q.dtype,
            device=x.device,
        )
        causal_mask = build_causal_mask(sequence_length, device=x.device)
        attn_mask = attn_mask.masked_fill(causal_mask.view(1, 1, sequence_length, sequence_length), float("-inf"))
        if key_padding_mask is not None:
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.out_proj(attn_output)


def _activation(name: str):
    return torch.relu if name == "relu" else torch.nn.functional.gelu


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_norm(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"Unsupported norm_type {norm_type!r}.")


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, dim_feedforward: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.activation_name = activation
        hidden_dim = dim_feedforward * 2 if activation in {"swiglu", "geglu"} else dim_feedforward
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.linear1(x)
        if self.activation_name == "swiglu":
            value, gate = hidden.chunk(2, dim=-1)
            hidden = value * F.silu(gate)
        elif self.activation_name == "geglu":
            value, gate = hidden.chunk(2, dim=-1)
            hidden = value * F.gelu(gate)
        else:
            hidden = _activation(self.activation_name)(hidden)
        return self.linear2(self.dropout(hidden))


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_type: str,
        residual_layout: str,
        position_encoding_type: str,
        max_length: int,
        rope_theta: float,
        rope_scaling_factor: float,
        rope_original_max_length: int | None,
    ) -> None:
        super().__init__()
        self.position_encoding_type = position_encoding_type
        if position_encoding_type in {"rope", "rope_ntk"}:
            self.self_attn = RotarySelfAttention(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                rope_type=position_encoding_type,
                rope_theta=rope_theta,
                rope_scaling_factor=rope_scaling_factor,
                rope_original_max_length=rope_original_max_length or max_length,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
        self.feedforward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)
        self.residual_layout = residual_layout

    def forward(
        self,
        tgt: Tensor,
        *,
        tgt_causal_mask: Tensor,
        tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = tgt
        residual_input = self.norm1(x) if self.residual_layout == "pre_norm" else x
        if self.position_encoding_type in {"rope", "rope_ntk"}:
            self_attn_output = self.self_attn(
                residual_input,
                key_padding_mask=tgt_padding_mask,
            )
        else:
            self_attn_output = self.self_attn(
                residual_input,
                residual_input,
                residual_input,
                attn_mask=tgt_causal_mask,
                key_padding_mask=normalize_key_padding_mask(
                    tgt_padding_mask,
                    attn_mask=tgt_causal_mask,
                ),
                need_weights=False,
            )[0]
        x = x + self.dropout1(self_attn_output)
        if self.residual_layout == "post_norm":
            x = self.norm1(x)
        ff_input = self.norm2(x) if self.residual_layout == "pre_norm" else x
        ff_output = self.feedforward(ff_input)
        x = x + self.dropout2(ff_output)
        if self.residual_layout == "post_norm":
            x = self.norm2(x)
        return x


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
        norm_type: str,
        residual_layout: str,
        position_encoding_type: str,
        max_length: int,
        rope_theta: float,
        rope_scaling_factor: float,
        rope_original_max_length: int | None,
    ) -> None:
        super().__init__()
        layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            residual_layout=residual_layout,
            position_encoding_type=position_encoding_type,
            max_length=max_length,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rope_original_max_length=rope_original_max_length,
        )
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.norm = build_norm(norm_type, d_model)

    def forward(
        self,
        tgt: Tensor,
        *,
        tgt_causal_mask: Tensor,
        tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                tgt_causal_mask=tgt_causal_mask,
                tgt_padding_mask=tgt_padding_mask,
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
            norm_type=config.norm_type,
            residual_layout=config.residual_layout,
            position_encoding_type=config.position_encoding_type,
            max_length=config.max_length,
            rope_theta=config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_original_max_length=config.rope_original_max_length,
        )
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        self.reset_parameters()
        if config.tie_embeddings:
            self.output_projection.weight = self.tgt_embedding.weight

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
