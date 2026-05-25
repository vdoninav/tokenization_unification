"""Variate-wise tokenizer (iTransformer-style): one token per channel, built from full history."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import TokenBatch, Tokenizer


class VariateTokenizer(Tokenizer):
    """One token per variable; token embedding is an MLP over the variable's full history."""

    def __init__(self, hidden_dim: int, num_channels: int, lookback: int, horizon: int):
        super().__init__(hidden_dim, num_channels, lookback, horizon)
        self.in_mlp = nn.Sequential(
            nn.Linear(lookback, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, horizon),
        )

    def sequence_length(self) -> int:
        return self.num_channels

    def encode(self, x: Tensor) -> TokenBatch:
        B, C, T = x.shape
        emb = self.in_mlp(x)
        pad = torch.zeros(B, C, dtype=torch.bool, device=x.device)
        return TokenBatch(embeddings=emb, padding_mask=pad, metadata={"batch": B, "channels": C})

    def decode(self, latent: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        B, C, d = latent.shape
        y = self.out_mlp(latent)
        return y.transpose(1, 2).contiguous()
