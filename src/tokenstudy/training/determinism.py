"""Seed + deterministic algorithm setup for bit-exact reproducibility."""
from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_deterministic() -> None:
    """Must be called once before creating any CUDA tensors."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    warnings.filterwarnings(
        "ignore",
        message=r".*Memory Efficient attention defaults to a non-deterministic algorithm.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Flash Attention defaults to a non-deterministic algorithm.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
        category=UserWarning,
    )
