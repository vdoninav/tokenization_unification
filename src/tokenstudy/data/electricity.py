"""Electricity dataset (321 variates, hourly, 70/10/20 split per Autoformer/PatchTST)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .base import ForecastingDataset, Split

_STATS_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}


def _load(path: Path) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    key = str(path.resolve())
    if key in _STATS_CACHE:
        mean, std, raw = _STATS_CACHE[key]
        return raw, mean, std
    df = pd.read_csv(path)
    df = df.iloc[:, 1:]
    raw = df.to_numpy(dtype=np.float32)
    n = len(raw)
    train_end = int(n * 0.7)
    train_slice = raw[:train_end]
    mean = torch.from_numpy(train_slice.mean(axis=0)).float()
    std = torch.from_numpy(train_slice.std(axis=0)).float()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    _STATS_CACHE[key] = (mean, std, raw)
    return raw, mean, std


def _split_bounds(raw_len: int, split: str, lookback: int) -> Split:
    """PatchTST canonical: val/test start `lookback` before split boundary (border1 -= seq_len)."""
    train_end = int(raw_len * 0.7)
    val_end = int(raw_len * 0.8)
    return {
        "train": Split(0, train_end),
        "val": Split(train_end - lookback, val_end),
        "test": Split(val_end - lookback, raw_len),
    }[split]


class ElectricityDataset(ForecastingDataset):
    NUM_VARIATES = 321

    def __init__(self, split: str, lookback: int, horizon: int, data_path: str | Path):
        data_path = Path(data_path)
        raw, mean, std = _load(data_path)
        super().__init__(
            raw=raw,
            split=_split_bounds(len(raw), split, lookback),
            lookback=lookback,
            horizon=horizon,
            mean=mean,
            std=std,
        )
