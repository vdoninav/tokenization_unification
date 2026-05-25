"""Generates Matrix-A manifests."""
from __future__ import annotations

from pathlib import Path

import yaml

DATASETS_ORDER = ["ETTh1", "Weather", "Electricity"]
HORIZONS = [96, 336]
TOKENIZERS = ["point", "patch", "variate", "discrete_scalar", "discrete_bpe"]
SEEDS = [2021, 2022, 2023]


def _base(dataset: str, horizon: int, tokenizer: str, seed: int) -> dict:
    if dataset == "Electricity":
        batch_size = 8
        lr = 1.0e-4 * (batch_size / 128.0) ** 0.5
    else:
        batch_size = 128
        lr = 1.0e-4
    return {
        "dataset": dataset,
        "horizon": horizon,
        "lookback": 336,
        "tokenizer": tokenizer,
        "seed": seed,
        "batch_size": batch_size,
        "max_epochs": 100,
        "patience": 10,
        "lr": lr,
        "weight_decay": 1.0e-4,
        "grad_clip": 1.0,
        "warmup_frac": 0.10,
        "patch_length": 16,
        "patch_stride": 8,
        "vocab_size": 4096,
        "bpe_final_vocab": 2445,
        "bpe_base_symbols": 126,
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 16,
        "n_layers": 3,
        "dropout": 0.2,
    }


def matrix_a_runs() -> list[dict]:
    rows: list[dict] = []
    for ds in DATASETS_ORDER:
        for h in HORIZONS:
            for s in SEEDS:
                for tok in TOKENIZERS:
                    rows.append(_base(ds, h, tok, s))
    return rows


def pilot_runs() -> list[dict]:
    return [_base(ds, 96, "patch", 2021) for ds in DATASETS_ORDER]


def write_manifests(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix_a.yaml").write_text(yaml.safe_dump({"runs": matrix_a_runs()}, sort_keys=False))
    (out_dir / "pilot.yaml").write_text(yaml.safe_dump({"runs": pilot_runs()}, sort_keys=False))


if __name__ == "__main__":
    write_manifests(Path(__file__).parent)
    print("wrote pilot.yaml and matrix_a.yaml")
