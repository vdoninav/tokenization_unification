"""Unified dataset builder + metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import torch

from .electricity import ElectricityDataset
from .ett import ETTh1Dataset
from .weather import WeatherDataset

_NUM_VARIATES = {"ETTh1": 7, "Weather": 21, "Electricity": 321}


class _Builder(Protocol):
    def __call__(
        self, split: str, lookback: int, horizon: int, data_path: str | Path,
    ) -> torch.utils.data.Dataset: ...


_REGISTRY: dict[str, _Builder] = {
    "ETTh1": ETTh1Dataset,
    "Weather": WeatherDataset,
    "Electricity": ElectricityDataset,
}


def build_dataset(
    name: str, split: str, lookback: int, horizon: int, data_path: str | Path,
) -> torch.utils.data.Dataset:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; known: {list(_REGISTRY)}")
    return _REGISTRY[name](split=split, lookback=lookback, horizon=horizon, data_path=data_path)


def num_variates(name: str) -> int:
    if name not in _NUM_VARIATES:
        raise ValueError(f"Unknown dataset {name!r}")
    return _NUM_VARIATES[name]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(
    path: str | Path,
    expected: str | None,
    checksums_file: Path,
) -> str:
    """Return observed sha256 for ``path``.

    If ``expected`` is provided and mismatches the observed hash, raise ``RuntimeError``.

    Side-effect: the first time a given filename is seen, its observed hash is
    appended to ``checksums_file``. Subsequent calls are read-only w.r.t.
    existing entries - even if the file on disk has changed. This is intentional:
    provenance records are pinned by the first successful verification, and
    updates must be made deliberately (edit the JSON or delete the entry).
    """
    observed = file_sha256(path)
    if expected is not None and observed != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {expected}, got {observed}"
        )
    name = Path(path).name
    try:
        existing = json.loads(checksums_file.read_text())
    except FileNotFoundError:
        existing = {}
    if name not in existing:
        existing[name] = observed
        checksums_file.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return observed
