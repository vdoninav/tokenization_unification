import os
from pathlib import Path
import pytest
import torch

from tokenstudy.data.weather import WeatherDataset

DATA_PATH = Path(os.environ.get("WEATHER_PATH", Path.home() / "data/weather/weather.csv"))
requires_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="weather.csv not present")


@requires_data
def test_weather_shape():
    ds = WeatherDataset(split="train", lookback=336, horizon=96, data_path=DATA_PATH)
    x, y = ds[0]
    assert x.shape == (21, 336)
    assert y.shape == (21, 96)


@requires_data
def test_weather_70_10_20_split():
    train = WeatherDataset(split="train", lookback=336, horizon=96, data_path=DATA_PATH)
    val = WeatherDataset(split="val", lookback=336, horizon=96, data_path=DATA_PATH)
    test = WeatherDataset(split="test", lookback=336, horizon=96, data_path=DATA_PATH)
    total = len(train.starts) + len(val.starts) + len(test.starts)
    assert total > 0
    assert torch.allclose(train.mean, val.mean)
