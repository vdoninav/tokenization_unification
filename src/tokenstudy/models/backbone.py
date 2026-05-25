"""Shared encoder-only Transformer backbone (PatchTST large-dataset config defaults)."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
        )
        self.pos_table = nn.Parameter(torch.zeros(1, 4096, d_model))
        nn.init.normal_(self.pos_table, std=0.02)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        B, L, d = x.shape
        if L > self.pos_table.shape[1]:
            raise ValueError(f"sequence L={L} exceeds pos table size {self.pos_table.shape[1]}")
        x = x + self.pos_table[:, :L, :]
        out = self.encoder(x, src_key_padding_mask=padding_mask)
        return out
