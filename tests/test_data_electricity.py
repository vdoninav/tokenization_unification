import os
from pathlib import Path
import pytest
import torch

from tokenstudy.data.electricity import ElectricityDataset

DATA_PATH = Path(os.environ.get("ELEC_PATH", Path.home() / "data/electricity/electricity.csv"))
requires_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="electricity.csv not present")


@requires_data
def test_elec_shape():
    ds = ElectricityDataset(split="train", lookback=336, horizon=96, data_path=DATA_PATH)
    x, y = ds[0]
    assert x.shape == (321, 336)
    assert y.shape == (321, 96)
