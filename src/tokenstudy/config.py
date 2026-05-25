"""Run configuration: pydantic model + canonical hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Tokenizer = Literal["point", "patch", "variate", "discrete_scalar", "discrete_bpe"]
Dataset = Literal["ETTh1", "Weather", "Electricity"]


class RunConfig(BaseModel):
    """Full spec of one run. Hashed to drive idempotency."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: Dataset
    horizon: int = Field(ge=1)
    lookback: int = 336
    tokenizer: Tokenizer
    seed: int

    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.10

    patch_length: int = 16
    patch_stride: int = 8
    vocab_size: int = 4096
    bpe_final_vocab: int = 2445
    bpe_base_symbols: int = 126

    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 16
    n_layers: int = 3
    dropout: float = 0.2

    def run_name(self) -> str:
        return f"{self.dataset}_h{self.horizon}_{self.tokenizer}_seed{self.seed}"


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return RunConfig(**data)


def config_hash(cfg: RunConfig) -> str:
    canonical = json.dumps(cfg.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
