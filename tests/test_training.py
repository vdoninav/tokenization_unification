import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tokenstudy.training.determinism import set_seed
from tokenstudy.training.metrics import compute_mae, compute_rmse
from tokenstudy.training.train import train_one_run


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_compute_mae_rmse():
    y = torch.tensor([[1.0, 2.0]])
    yhat = torch.tensor([[1.5, 1.5]])
    assert abs(compute_mae(yhat, y) - 0.5) < 1e-6
    assert abs(compute_rmse(yhat, y) - 0.5) < 1e-6


def test_train_one_run_smoke():
    """End-to-end smoke test: train 2 epochs on random data, verify metrics JSON returned."""
    torch.manual_seed(0)
    X = torch.randn(16, 3, 48)
    Y = torch.randn(16, 3, 12)
    train_ds = TensorDataset(X, Y)
    val_ds = TensorDataset(X, Y)
    test_ds = TensorDataset(X, Y)

    metrics = train_one_run(
        tokenizer="patch",
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        num_channels=3, lookback=48, horizon=12,
        d_model=16, n_heads=2, d_ff=32, n_layers=1, dropout=0.0,
        vocab_size=32, bpe_base=16, bpe_final=24,
        patch_length=8, patch_stride=4,
        batch_size=8, max_epochs=2, patience=10,
        lr=1e-3, weight_decay=0.0, grad_clip=1.0, warmup_frac=0.1,
        seed=0, device="cpu",
    )
    assert "test_mae" in metrics and "test_rmse" in metrics
    assert "train_time_sec" in metrics
    assert metrics["epochs_trained"] == 2
