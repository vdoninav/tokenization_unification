"""Scalar forecasting metrics."""
from __future__ import annotations

import torch
from torch import Tensor


def compute_mae(pred: Tensor, target: Tensor) -> float:
    return (pred - target).abs().mean().item()


def compute_rmse(pred: Tensor, target: Tensor) -> float:
    return ((pred - target) ** 2).mean().sqrt().item()


def compute_mse(pred: Tensor, target: Tensor) -> float:
    return ((pred - target) ** 2).mean().item()
