import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from tokenstudy.config import RunConfig
from tokenstudy.training.runner import execute_run


def test_execute_run_writes_artifacts(tmp_path, monkeypatch):
    """Monkeypatch dataset registry to inject synthetic data, verify run artifacts."""
    import tokenstudy.training.runner as runner_mod

    def _fake_build_datasets(cfg, data_root):
        X = torch.randn(16, 7, cfg.lookback)
        Y = torch.randn(16, 7, cfg.horizon)
        ds = TensorDataset(X, Y)
        return ds, ds, ds, 7

    monkeypatch.setattr(runner_mod, "build_datasets", _fake_build_datasets)

    cfg = RunConfig(
        dataset="ETTh1", horizon=12, lookback=48, tokenizer="patch", seed=0,
        batch_size=4, max_epochs=2, patience=10, lr=1e-3,
        d_model=16, n_heads=2, d_ff=32, n_layers=1, dropout=0.0,
        vocab_size=32, bpe_final_vocab=24, bpe_base_symbols=16,
        patch_length=8, patch_stride=4,
    )
    runs_root = tmp_path / "runs"
    execute_run(cfg, runs_root=runs_root, data_root=tmp_path, device="cpu")
    run_dir = runs_root / cfg.run_name()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "env.json").exists()
    m = json.loads((run_dir / "metrics.json").read_text())
    assert "test_mae" in m and "config_hash" in m and "git_sha" in m
