"""Per-run orchestration: configs -> datasets -> training -> artifacts."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from ..config import RunConfig, config_hash
from ..data.registry import build_dataset, num_variates
from .train import train_one_run


def _git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        ) != 0
        return sha, dirty
    except Exception:
        return "unknown", False


def _env_info(uv_lock_path: Path | None = None) -> dict:
    sha, dirty = _git_sha()
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
    }
    if uv_lock_path and uv_lock_path.exists():
        import hashlib
        info["uv_lock_sha256"] = hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
    return info


def build_datasets(cfg: RunConfig, data_root: Path):
    data_path = {
        "ETTh1": data_root / "ETT-small" / "ETTh1.csv",
        "Weather": data_root / "weather" / "weather.csv",
        "Electricity": data_root / "electricity" / "electricity.csv",
    }[cfg.dataset]
    train = build_dataset(cfg.dataset, "train", cfg.lookback, cfg.horizon, data_path)
    val = build_dataset(cfg.dataset, "val", cfg.lookback, cfg.horizon, data_path)
    test = build_dataset(cfg.dataset, "test", cfg.lookback, cfg.horizon, data_path)
    return train, val, test, num_variates(cfg.dataset)


def execute_run(
    cfg: RunConfig,
    runs_root: Path,
    data_root: Path,
    device: str = "cuda",
    uv_lock_path: Path | None = None,
) -> dict:
    run_dir = runs_root / cfg.run_name()
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=True))
    (run_dir / "env.json").write_text(json.dumps(_env_info(uv_lock_path), indent=2, sort_keys=True))

    train, val, test, C = build_datasets(cfg, data_root)

    metrics = train_one_run(
        tokenizer=cfg.tokenizer,
        train_ds=train, val_ds=val, test_ds=test,
        num_channels=C, lookback=cfg.lookback, horizon=cfg.horizon,
        d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff,
        n_layers=cfg.n_layers, dropout=cfg.dropout,
        vocab_size=cfg.vocab_size, bpe_base=cfg.bpe_base_symbols, bpe_final=cfg.bpe_final_vocab,
        patch_length=cfg.patch_length, patch_stride=cfg.patch_stride,
        batch_size=cfg.batch_size, max_epochs=cfg.max_epochs, patience=cfg.patience,
        lr=cfg.lr, weight_decay=cfg.weight_decay, grad_clip=cfg.grad_clip,
        warmup_frac=cfg.warmup_frac, seed=cfg.seed,
        device=device,
        progress_log_path=run_dir / "train.log",
        progress_tag=cfg.run_name(),
    )

    record = {
        "dataset": cfg.dataset, "horizon": cfg.horizon, "tokenizer": cfg.tokenizer,
        "seed": cfg.seed, "lookback": cfg.lookback,
        "config_hash": config_hash(cfg),
        "git_sha": _git_sha()[0],
        **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record
