from __future__ import annotations

import math

from torch import Tensor, nn

from midi2score.models.config import ModelConfig
from midi2score.models.decoder_config import DecoderLanguageModelConfig
from midi2score.models.modules import (
    SinusoidalPositionalEncoding,
    TransformerDecoderStack,
    build_causal_mask,
)


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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
        )
        self.decoder = TransformerDecoderStack(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
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

    def encode_memory(self, src_tokens: Tensor, *, src_padding_mask: Tensor | None = None) -> Tensor:
        src_embeddings = self.encode(src_tokens)
        return self.encoder(
            src_embeddings,
            src_key_padding_mask=src_padding_mask,
        )

    def forward(
        self,
        src_tokens: Tensor,
        tgt_input_tokens: Tensor,
        *,
        src_padding_mask: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
    ) -> Tensor:
        memory = self.encode_memory(src_tokens, src_padding_mask=src_padding_mask)
        tgt_embeddings = self.decode(tgt_input_tokens)

        # The decoder must not attend to future target positions during training.
        causal_mask = build_causal_mask(
            sequence_length=tgt_input_tokens.size(1),
            device=tgt_input_tokens.device,
        )

        hidden_states = self.decoder(
            tgt_embeddings,
            tgt_causal_mask=causal_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory=memory,
            memory_padding_mask=src_padding_mask,
        )
        return self.output_projection(hidden_states)

    def load_pretrained_decoder_state(self, decoder_state_dict: dict[str, Tensor]) -> list[str]:
        transferred_keys: list[str] = []
        own_state = self.state_dict()

        # Only move the decoder-side parameters learned during LM pretraining.
        for key, value in decoder_state_dict.items():
            if not key.startswith(("tgt_embedding.", "decoder.", "output_projection.")):
                continue
            if key not in own_state:
                continue
            if own_state[key].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for decoder parameter {key}: "
                    f"{tuple(value.shape)} != {tuple(own_state[key].shape)}."
                )
            own_state[key].copy_(value)
            transferred_keys.append(key)

        if not transferred_keys:
            raise ValueError("No compatible decoder parameters were found in the checkpoint.")
        return transferred_keys

    def load_pretrained_decoder_checkpoint(self, checkpoint_path: str) -> list[str]:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint["model_state"]
        raw_decoder_config = checkpoint.get("model_config")
        if raw_decoder_config is not None:
            decoder_config = DecoderLanguageModelConfig(**raw_decoder_config)
            decoder_config.assert_compatible_with_seq2seq(self.config)
        return self.load_pretrained_decoder_state(model_state)
