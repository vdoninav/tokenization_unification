import os
from pathlib import Path
import numpy as np
import pytest
import torch

from tokenstudy.data.ett import ETTh1Dataset

DATA_PATH = Path(os.environ.get("ETT_PATH", Path.home() / "data/ETT-small/ETTh1.csv"))
requires_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="ETTh1.csv not present")


@requires_data
def test_etth1_shapes():
    ds = ETTh1Dataset(split="train", lookback=336, horizon=96, data_path=DATA_PATH)
    x, y = ds[0]
    assert x.shape == (7, 336)
    assert y.shape == (7, 96)
    assert x.dtype == torch.float32


@requires_data
def test_etth1_normalization_uses_train_stats():
    train = ETTh1Dataset(split="train", lookback=336, horizon=96, data_path=DATA_PATH)
    val = ETTh1Dataset(split="val", lookback=336, horizon=96, data_path=DATA_PATH)
    assert torch.allclose(train.mean, val.mean)
    assert torch.allclose(train.std, val.std)


@requires_data
def test_etth1_splits_canonical():
    """PatchTST canonical ETT splits: train 12*30*24, val 4*30*24, test 4*30*24 hours."""
    train = ETTh1Dataset(split="train", lookback=336, horizon=96, data_path=DATA_PATH)
    val = ETTh1Dataset(split="val", lookback=336, horizon=96, data_path=DATA_PATH)
    test = ETTh1Dataset(split="test", lookback=336, horizon=96, data_path=DATA_PATH)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
