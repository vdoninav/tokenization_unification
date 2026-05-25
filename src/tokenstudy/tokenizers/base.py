"""Tokenizer abstraction: encode (B,C,T) -> (B',L,d), decode (B',L,d) -> (B,H,C)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class TokenBatch:
    embeddings: Tensor
    padding_mask: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class Tokenizer(nn.Module, ABC):
    """Base class. Subclasses implement encode + decode.

    Shape contract:
        encode:  (B, C, T)  -> TokenBatch with embeddings (B', L, d)
        decode:  (B', L, d) -> (B, H, C)

    B' == B for variate-wise; B' == B*C for channel-independent modes.
    """

    def __init__(self, hidden_dim: int, num_channels: int, lookback: int, horizon: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.lookback = lookback
        self.horizon = horizon

    @abstractmethod
    def encode(self, x: Tensor) -> TokenBatch:
        """x: (B, C, T) -> TokenBatch."""

    @abstractmethod
    def decode(self, latent: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """latent: (B', L, d) -> forecast (B, H, C) in normalized space."""

    def sequence_length(self) -> int:
        """L for this tokenizer at the current lookback/channels (static where possible)."""
        raise NotImplementedError
