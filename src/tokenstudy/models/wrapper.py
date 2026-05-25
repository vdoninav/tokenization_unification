"""Glue module: Tokenizer + Backbone -> forecast."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..tokenizers.base import Tokenizer
from .backbone import TransformerBackbone


class ForecastingModel(nn.Module):
    def __init__(self, tokenizer: Tokenizer, backbone: TransformerBackbone):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone

    def forward(self, x: Tensor) -> Tensor:
        tb = self.tokenizer.encode(x)
        latent = self.backbone(tb.embeddings, padding_mask=tb.padding_mask)
        y = self.tokenizer.decode(latent, padding_mask=tb.padding_mask)
        scale = tb.metadata.get("scale")
        if scale is not None:
            y = y * scale.squeeze(-1).unsqueeze(1)
        return y
