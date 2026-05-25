"""Point-wise tokenizer: one token per (timestep, channel) via channel-independent projection."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import TokenBatch, Tokenizer


class PointTokenizer(Tokenizer):
    """Channel-independent per-timestep linear projection.

    encode: (B, C, T) -> (B*C, T, d), each channel's scalar trajectory linear-embedded
    decode: (B*C, T, d) -> (B, H, C) by linear projecting H latent timesteps per channel
    """

    def __init__(self, hidden_dim: int, num_channels: int, lookback: int, horizon: int):
        super().__init__(hidden_dim, num_channels, lookback, horizon)
        self.in_proj = nn.Linear(1, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def sequence_length(self) -> int:
        return self.lookback

    def encode(self, x: Tensor) -> TokenBatch:
        B, C, T = x.shape
        assert C == self.num_channels and T == self.lookback
        x = x.reshape(B * C, T, 1)
        emb = self.in_proj(x)
        pad = torch.zeros(B * C, T, dtype=torch.bool, device=x.device)
        return TokenBatch(embeddings=emb, padding_mask=pad, metadata={"batch": B, "channels": C})

    def decode(self, latent: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        BC, L, d = latent.shape
        assert L == self.lookback
        B = BC // self.num_channels
        tail = latent[:, -self.horizon :, :]
        y = self.out_proj(tail).squeeze(-1)
        y = y.reshape(B, self.num_channels, self.horizon)
        return y.transpose(1, 2).contiguous()
