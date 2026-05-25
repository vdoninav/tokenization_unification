"""Discrete scalar (Chronos-style) tokenizer: per-channel mean-scale + uniform binning in [-15,+15]."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import TokenBatch, Tokenizer


class DiscreteScalarTokenizer(Tokenizer):
    """Chronos-style quantization: mean-scale each channel, uniform bins on [-15, +15], shared vocab."""

    LO = -15.0
    HI = 15.0

    DECODE_CHUNK_SIZE = 64

    def __init__(
        self, hidden_dim: int, num_channels: int, lookback: int, horizon: int,
        vocab_size: int = 4096,
    ):
        super().__init__(hidden_dim, num_channels, lookback, horizon)
        self.V = vocab_size
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.out_logits = nn.Linear(hidden_dim, vocab_size)
        bin_centers = torch.linspace(self.LO, self.HI, vocab_size)
        self.register_buffer("bin_centers", bin_centers)

    def sequence_length(self) -> int:
        return self.lookback

    def _binning(self, x: Tensor) -> Tensor:
        """x: (..., ) -> ids in [0, V). Uniform bins in [LO, HI]."""
        normed = (x - self.LO) / (self.HI - self.LO)
        ids = torch.clamp((normed * self.V).long(), 0, self.V - 1)
        return ids

    def encode(self, x: Tensor) -> TokenBatch:
        B, C, T = x.shape
        scale = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        scaled = x / scale
        scaled = scaled.reshape(B * C, T)
        ids = self._binning(scaled)
        emb = self.emb(ids)
        pad = torch.zeros(B * C, T, dtype=torch.bool, device=x.device)
        return TokenBatch(
            embeddings=emb, padding_mask=pad,
            metadata={"batch": B, "channels": C, "scale": scale},
        )

    def decode(self, latent: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        BC, L, d = latent.shape
        B = BC // self.num_channels
        tail = latent[:, -self.horizon :, :]
        cs = max(1, self.DECODE_CHUNK_SIZE)
        if BC <= cs:
            logits = self.out_logits(tail)
            probs = torch.softmax(logits, dim=-1)
            y_scaled = (probs * self.bin_centers).sum(dim=-1)
        else:
            chunks: list[Tensor] = []
            for i in range(0, BC, cs):
                ch = tail[i : i + cs]
                logits_ch = self.out_logits(ch)
                probs_ch = torch.softmax(logits_ch, dim=-1)
                chunks.append((probs_ch * self.bin_centers).sum(dim=-1))
            y_scaled = torch.cat(chunks, dim=0)
        y_scaled = y_scaled.reshape(B, self.num_channels, self.horizon)
        return y_scaled.transpose(1, 2).contiguous()
