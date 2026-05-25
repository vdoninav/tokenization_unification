"""Base class for long-horizon forecasting datasets with canonical literature splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Split:
    """Index bounds for train/val/test."""
    start: int
    end: int


class ForecastingDataset(Dataset):
    """Sliding-window forecasting dataset, channel-independent normalization.

    Raw array shape: (T_total, C). Windows yield (x: (C, lookback), y: (C, horizon)).
    """

    def __init__(
        self,
        raw: np.ndarray,
        split: Split,
        lookback: int,
        horizon: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        if raw.ndim != 2:
            raise ValueError(f"raw must be (T, C); got {raw.shape}")
        self.raw = raw
        self.split = split
        self.lookback = lookback
        self.horizon = horizon
        self.mean = mean
        self.std = std

        self.starts = np.arange(
            split.start,
            split.end - lookback - horizon + 1,
        )
        if len(self.starts) == 0:
            raise ValueError(
                f"No windows fit: split {split}, lookback {lookback}, horizon {horizon}"
            )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.starts[idx])
        x = self.raw[t : t + self.lookback]
        y = self.raw[t + self.lookback : t + self.lookback + self.horizon]
        x = (torch.from_numpy(x).float() - self.mean) / self.std
        y = (torch.from_numpy(y).float() - self.mean) / self.std
        return x.T.contiguous(), y.T.contiguous()
