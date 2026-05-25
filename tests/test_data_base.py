"""Tests for ForecastingDataset base class using synthetic data (no CSV required)."""
import numpy as np
import pytest
import torch

from tokenstudy.data.base import ForecastingDataset, Split


def test_forecasting_dataset_shapes_on_synthetic():
    raw = np.random.RandomState(0).randn(500, 3).astype(np.float32)
    mean = torch.from_numpy(raw[:400].mean(0)).float()
    std = torch.from_numpy(raw[:400].std(0)).float()
    ds = ForecastingDataset(raw=raw, split=Split(0, 400), lookback=96, horizon=24, mean=mean, std=std)
    x, y = ds[0]
    assert x.shape == (3, 96)
    assert y.shape == (3, 24)
    assert x.dtype == torch.float32


def test_forecasting_dataset_last_window_uses_full_split():
    raw = np.zeros((200, 2), dtype=np.float32)
    mean = torch.zeros(2)
    std = torch.ones(2)
    ds = ForecastingDataset(raw=raw, split=Split(0, 200), lookback=64, horizon=16, mean=mean, std=std)
    assert ds.starts[-1] == 120
    assert ds.starts[-1] + 64 + 16 == 200


def test_forecasting_dataset_empty_window_raises():
    raw = np.zeros((50, 3), dtype=np.float32)
    mean = torch.zeros(3)
    std = torch.ones(3)
    with pytest.raises(ValueError, match="No windows fit"):
        ForecastingDataset(raw=raw, split=Split(0, 50), lookback=96, horizon=24, mean=mean, std=std)


def test_forecasting_dataset_bad_ndim_raises():
    with pytest.raises(ValueError, match="raw must be"):
        ForecastingDataset(
            raw=np.zeros(10, dtype=np.float32), split=Split(0, 10),
            lookback=1, horizon=1, mean=torch.zeros(1), std=torch.ones(1),
        )


def test_forecasting_dataset_normalizes_constant_column_safely():
    raw = np.zeros((100, 2), dtype=np.float32)
    raw[:, 0] = 5.0
    raw[:, 1] = np.arange(100, dtype=np.float32)
    train = raw[:80]
    mean = torch.from_numpy(train.mean(0)).float()
    std = torch.from_numpy(train.std(0)).float()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    ds = ForecastingDataset(raw=raw, split=Split(0, 80), lookback=24, horizon=8, mean=mean, std=std)
    x, y = ds[0]
    assert torch.allclose(x[0], torch.zeros(24))
    assert not torch.isnan(x).any() and not torch.isnan(y).any()
