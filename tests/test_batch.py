import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch.utils.data import TensorDataset

from tokenstudy.batch import run_manifest
from tokenstudy.config import RunConfig


def _fake_build_datasets(cfg, data_root):
    X = torch.randn(8, 7, cfg.lookback)
    Y = torch.randn(8, 7, cfg.horizon)
    ds = TensorDataset(X, Y)
    return ds, ds, ds, 7


@pytest.fixture
def tiny_manifest_dir(tmp_path):
    import yaml
    manifest = {
        "runs": [
            {"dataset": "ETTh1", "horizon": 12, "lookback": 48, "tokenizer": "patch",
             "seed": 0, "batch_size": 4, "max_epochs": 1, "patience": 10, "lr": 1e-3,
             "d_model": 16, "n_heads": 2, "d_ff": 32, "n_layers": 1, "dropout": 0.0,
             "vocab_size": 32, "bpe_final_vocab": 24, "bpe_base_symbols": 16,
             "patch_length": 8, "patch_stride": 4},
            {"dataset": "ETTh1", "horizon": 12, "lookback": 48, "tokenizer": "point",
             "seed": 0, "batch_size": 4, "max_epochs": 1, "patience": 10, "lr": 1e-3,
             "d_model": 16, "n_heads": 2, "d_ff": 32, "n_layers": 1, "dropout": 0.0,
             "vocab_size": 32, "bpe_final_vocab": 24, "bpe_base_symbols": 16,
             "patch_length": 8, "patch_stride": 4},
        ],
    }
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(manifest))
    return p, tmp_path


def test_batch_runs_all_cells(tiny_manifest_dir, monkeypatch):
    path, tmp = tiny_manifest_dir
    import tokenstudy.training.runner as runner_mod
    monkeypatch.setattr(runner_mod, "build_datasets", _fake_build_datasets)
    run_manifest(
        manifest_path=path, runs_root=tmp / "runs",
        data_root=tmp, device="cpu", skip_git_check=True, skip_venv_check=True,
    )
    completed = list((tmp / "runs").glob("*/metrics.json"))
    assert len(completed) == 2


def test_batch_idempotent(tiny_manifest_dir, monkeypatch, tmp_path):
    path, tmp = tiny_manifest_dir
    import tokenstudy.training.runner as runner_mod
    monkeypatch.setattr(runner_mod, "build_datasets", _fake_build_datasets)
    run_manifest(
        manifest_path=path, runs_root=tmp / "runs",
        data_root=tmp, device="cpu", skip_git_check=True, skip_venv_check=True,
    )
    log_before = (tmp / "runs" / "batch.log").read_text()
    run_manifest(
        manifest_path=path, runs_root=tmp / "runs",
        data_root=tmp, device="cpu", skip_git_check=True, skip_venv_check=True,
    )
    log_after = (tmp / "runs" / "batch.log").read_text()
    assert log_after.count("status=SKIPPED") >= 2
