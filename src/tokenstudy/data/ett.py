"""ETTh1 loader with PatchTST-canonical splits (12/4/4 months)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .base import ForecastingDataset, Split

_ETT_TRAIN = 12 * 30 * 24
_ETT_VAL = 4 * 30 * 24
_ETT_TEST = 4 * 30 * 24

_STATS_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}


def _load_etth1(path: Path) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    key = str(path.resolve())
    if key in _STATS_CACHE:
        mean, std, raw = _STATS_CACHE[key]
        return raw, mean, std

    df = pd.read_csv(path)
    cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    raw = df[cols].to_numpy(dtype=np.float32)
    train_slice = raw[:_ETT_TRAIN]
    mean = torch.from_numpy(train_slice.mean(axis=0)).float()
    std = torch.from_numpy(train_slice.std(axis=0)).float()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    _STATS_CACHE[key] = (mean, std, raw)
    return raw, mean, std


def _split_bounds(split: str, lookback: int) -> Split:
    """PatchTST canonical: val/test windows start `lookback` before the split boundary
    so the first window's input uses the tail of the prior split (border1 -= seq_len).
    This matches yuqinie98/PatchTST and thuml/Autoformer reference implementations.
    """
    if split == "train":
        return Split(0, _ETT_TRAIN)
    if split == "val":
        return Split(_ETT_TRAIN - lookback, _ETT_TRAIN + _ETT_VAL)
    if split == "test":
        return Split(
            _ETT_TRAIN + _ETT_VAL - lookback,
            _ETT_TRAIN + _ETT_VAL + _ETT_TEST,
        )
    raise ValueError(f"split must be train/val/test; got {split!r}")


class ETTh1Dataset(ForecastingDataset):
    """ETTh1 with PatchTST canonical 12/4/4-month split (channel-indep normalization on train)."""

    NUM_VARIATES = 7

    def __init__(
        self,
        split: str,
        lookback: int,
        horizon: int,
        data_path: str | Path,
    ):
        data_path = Path(data_path)
        raw, mean, std = _load_etth1(data_path)
        super().__init__(
            raw=raw,
            split=_split_bounds(split, lookback),
            lookback=lookback,
            horizon=horizon,
            mean=mean,
            std=std,
        )
