import json
from pathlib import Path

import polars as pl
import pytest

from tokenstudy.analysis import aggregate_runs, compute_utility, pareto_mask


def _write_fake_run(runs_root: Path, **rec) -> None:
    name = f"{rec['dataset']}_h{rec['horizon']}_{rec['tokenizer']}_seed{rec['seed']}"
    d = runs_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(rec))


def test_aggregate_runs(tmp_path):
    for tok in ["point", "patch"]:
        for seed in [0, 1]:
            _write_fake_run(
                tmp_path, dataset="ETTh1", horizon=96, tokenizer=tok, seed=seed,
                test_mae=0.3 + 0.01 * seed, test_rmse=0.4,
                train_time_sec=60.0, peak_vram_mb=1000,
                tokens_per_input=41, val_mae=0.25, val_rmse=0.35,
                inference_ms_per_batch=10.0,
                param_count={"tokenizer": 1, "backbone": 2, "head": 3},
                epochs_trained=50, best_epoch=40,
                lookback=336, config_hash="abc", git_sha="def",
            )
    df = aggregate_runs(tmp_path)
    assert df.height == 4
    assert set(df["tokenizer"].unique()) == {"point", "patch"}


def test_compute_utility():
    df = pl.DataFrame({
        "dataset": ["A", "A", "A", "A"],
        "horizon": [96, 96, 96, 96],
        "tokenizer": ["t1", "t2", "t3", "t4"],
        "test_mae_mean": [0.1, 0.2, 0.3, 0.4],
        "train_time_sec_mean": [100.0, 200.0, 300.0, 400.0],
        "peak_vram_mb_mean": [1000.0, 2000.0, 3000.0, 4000.0],
        "test_mae_cv": [0.01, 0.02, 0.03, 0.04],
    })
    out = compute_utility(df, weights=(0.25, 0.25, 0.25, 0.25))
    assert out.filter(pl.col("tokenizer") == "t1")["U"].item() > \
        out.filter(pl.col("tokenizer") == "t4")["U"].item()


def test_pareto_mask():
    x = [0.1, 0.2, 0.15, 0.3]
    y = [100, 80, 50, 200]
    mask = pareto_mask(x, y)
    assert mask == [True, False, True, False]
