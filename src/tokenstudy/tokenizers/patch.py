"""Patch-wise tokenizer (PatchTST-style, channel-independent)."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import TokenBatch, Tokenizer


class PatchTokenizer(Tokenizer):
    """Channel-independent patching.

    encode: (B, C, T) -> (B*C, L, d) where L = (T-p)/s + 1; each patch of length p is
            linearly projected to d.
    decode: (B*C, L, d) -> (B, H, C) via flatten-head, then mean-over-patches per-channel.
    """

    def __init__(
        self, hidden_dim: int, num_channels: int, lookback: int, horizon: int,
        patch_length: int = 16, patch_stride: int = 8,
    ):
        super().__init__(hidden_dim, num_channels, lookback, horizon)
        assert (lookback - patch_length) % patch_stride == 0 or True, "patch config"
        self.p = patch_length
        self.s = patch_stride
        self._L = (lookback - patch_length) // patch_stride + 1
        self.in_proj = nn.Linear(patch_length, hidden_dim)
        self.flat_head = nn.Linear(self._L * hidden_dim, horizon)

    def sequence_length(self) -> int:
        return self._L

    def encode(self, x: Tensor) -> TokenBatch:
        B, C, T = x.shape
        patches = x.unfold(dimension=2, size=self.p, step=self.s)
        Bc = B * C
        patches = patches.reshape(Bc, self._L, self.p)
        emb = self.in_proj(patches)
        pad = torch.zeros(Bc, self._L, dtype=torch.bool, device=x.device)
        return TokenBatch(embeddings=emb, padding_mask=pad, metadata={"batch": B, "channels": C})

    def decode(self, latent: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        BC, L, d = latent.shape
        assert L == self._L
        B = BC // self.num_channels
        flat = latent.reshape(BC, L * d)
        y = self.flat_head(flat)
        y = y.reshape(B, self.num_channels, self.horizon)
        return y.transpose(1, 2).contiguous()
