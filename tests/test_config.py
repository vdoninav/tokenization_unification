import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from tokenstudy.config import RunConfig, config_hash, load_config


def test_runconfig_round_trip(tmp_path: Path):
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "dataset: ETTh1\n"
        "horizon: 96\n"
        "lookback: 336\n"
        "tokenizer: patch\n"
        "seed: 2021\n"
        "batch_size: 128\n"
        "max_epochs: 100\n"
        "patience: 10\n"
        "lr: 1.0e-4\n"
    )
    cfg = load_config(cfg_yaml)
    assert cfg.dataset == "ETTh1"
    assert cfg.horizon == 96
    assert cfg.tokenizer == "patch"
    assert cfg.seed == 2021


def test_config_hash_stable():
    data = dict(
        dataset="ETTh1", horizon=96, lookback=336, tokenizer="patch", seed=2021,
        batch_size=128, max_epochs=100, patience=10, lr=1e-4,
    )
    h1 = config_hash(RunConfig(**data))
    h2 = config_hash(RunConfig(**data))
    assert h1 == h2
    assert len(h1) == 64


def test_config_hash_differs_on_change():
    base = RunConfig(
        dataset="ETTh1", horizon=96, lookback=336, tokenizer="patch",
        seed=2021, batch_size=128, max_epochs=100, patience=10, lr=1e-4,
    )
    changed = base.model_copy(update={"seed": 2022})
    assert config_hash(base) != config_hash(changed)


def test_config_hash_insensitive_to_kwarg_order():
    """Actually exercises config_hash + model_dump + canonicalization pipeline."""
    a = RunConfig(dataset="ETTh1", horizon=96, tokenizer="patch", seed=2021)
    b = RunConfig(seed=2021, tokenizer="patch", horizon=96, dataset="ETTh1")
    assert config_hash(a) == config_hash(b)


def test_config_hash_cross_process_deterministic():
    """Hash must be identical across separate Python interpreters - critical for batch.py idempotency."""
    script = (
        "from tokenstudy.config import RunConfig, config_hash;"
        "print(config_hash(RunConfig(dataset='ETTh1', horizon=96, tokenizer='patch', seed=2021)))"
    )
    r1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    r2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    h1 = r1.stdout.strip()
    h2 = r2.stdout.strip()
    assert len(h1) == 64 and h1 == h2


def test_extra_forbid_rejects_unknown_key():
    """YAML typos (e.g. `learing_rate` instead of `lr`) must fail validation, not silently ignore."""
    with pytest.raises(ValidationError):
        RunConfig(
            dataset="ETTh1", horizon=96, tokenizer="patch", seed=1,
            learing_rate=0.001,
        )


def test_literal_rejects_unknown_dataset():
    """Typoed or out-of-scope dataset name must fail validation."""
    with pytest.raises(ValidationError):
        RunConfig(dataset="Traffic", horizon=96, tokenizer="patch", seed=1)


def test_literal_rejects_unknown_tokenizer():
    with pytest.raises(ValidationError):
        RunConfig(dataset="ETTh1", horizon=96, tokenizer="point_v2", seed=1)


def test_horizon_ge_1():
    with pytest.raises(ValidationError):
        RunConfig(dataset="ETTh1", horizon=0, tokenizer="patch", seed=1)
