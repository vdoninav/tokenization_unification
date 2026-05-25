"""End-to-end single-run training loop with early stopping + metrics collection.

Note: callers wanting bit-exact GPU determinism must call
``tokenstudy.training.determinism.configure_deterministic()`` once before any CUDA
tensor is allocated. ``train_one_run`` does NOT call it internally because the env
vars must be set before CUDA init, which is typically earlier than the first
training call.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..models.build import build_model
from .determinism import set_seed
from .metrics import compute_mae, compute_rmse


def _count_params(model: torch.nn.Module) -> dict[str, int]:
    """Split params into tokenizer/backbone/head.

    Under the current ForecastingModel wrapper (tokenizer + backbone only), every
    decoding parameter lives inside the tokenizer (each tokenizer owns its own
    encode/decode). Therefore ``head`` is always 0 today. The key is kept to
    future-proof against adding a separate head module.
    """
    tokenizer = sum(p.numel() for p in model.tokenizer.parameters())
    backbone = sum(p.numel() for p in model.backbone.parameters())
    total = sum(p.numel() for p in model.parameters())
    head = total - tokenizer - backbone
    return {"tokenizer": tokenizer, "backbone": backbone, "head": max(head, 0)}


def _infer_loop(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, float, float]:
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    n = 0
    total_time = 0.0
    amp_enabled = (device == "cuda")
    amp_device = "cuda" if amp_enabled else "cpu"
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y = y.transpose(1, 2).contiguous()
            start = time.perf_counter()
            with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                pred = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            pred_fp = pred.float()
            sum_abs += (pred_fp - y).abs().sum().item()
            sum_sq += ((pred_fp - y) ** 2).sum().item()
            n += y.numel()
    mae = sum_abs / max(n, 1)
    rmse = (sum_sq / max(n, 1)) ** 0.5
    ms_per_batch = total_time * 1000.0 / max(len(loader), 1)
    return mae, rmse, ms_per_batch


def train_one_run(
    tokenizer: str,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    num_channels: int,
    lookback: int,
    horizon: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    dropout: float,
    vocab_size: int,
    bpe_base: int,
    bpe_final: int,
    patch_length: int,
    patch_stride: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    warmup_frac: float,
    seed: int,
    device: str = "cuda",
    progress_log_path: "Path | None" = None,
    progress_tag: str = "",
) -> dict[str, Any]:
    set_seed(seed)

    def _progress(msg: str) -> None:
        """Print to stdout (flushed) and optionally append to progress_log_path."""
        line = f"{msg}"
        if progress_tag:
            line = f"[{progress_tag}] " + line
        print(line, flush=True)
        if progress_log_path is not None:
            progress_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_log_path, "a") as f:
                f.write(line + "\n")

    g = torch.Generator().manual_seed(seed)
    pin = (device == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, generator=g, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin,
    )

    model = build_model(
        tokenizer=tokenizer, num_channels=num_channels, lookback=lookback, horizon=horizon,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers, dropout=dropout,
        vocab_size=vocab_size, bpe_base=bpe_base, bpe_final=bpe_final,
        patch_length=patch_length, patch_stride=patch_stride,
    ).to(device)

    if tokenizer == "discrete_bpe":
        xs = [b[0] for b in itertools.islice(train_loader, 8)]
        fit_x = torch.cat(xs, dim=0).to(device)
        model.tokenizer.fit(fit_x, log_fn=_progress)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, max_epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=lr, total_steps=total_steps, pct_start=warmup_frac, anneal_strategy="cos",
    )

    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    patience_ctr = 0
    start = time.perf_counter()
    peak_vram_mb = 0
    amp_enabled = (device == "cuda")
    amp_device = "cuda" if amp_enabled else "cpu"

    n_batches = len(train_loader)
    _progress(
        f"training: tokenizer={tokenizer} dataset_C={num_channels} T={lookback} H={horizon} "
        f"bs={batch_size} batches/epoch={n_batches} max_epochs={max_epochs} lr={lr:.2e} "
        f"device={device} amp={amp_enabled}"
    )

    epoch = 0
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).transpose(1, 2).contiguous()
            with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
                pred = model(x)
                loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            sched.step()
            running_loss += float(loss.detach())
        if device == "cuda":
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() // (1024 * 1024))

        val_mae, val_rmse_tick, _ = _infer_loop(model, val_loader, device)
        train_loss = running_loss / max(n_batches, 1)
        epoch_time = time.perf_counter() - epoch_start
        cumulative = time.perf_counter() - start
        cur_lr = optim.param_groups[0]["lr"]
        improved = val_mae < best_val - 1e-6
        marker = "*" if improved else " "
        cum_m, cum_s = int(cumulative // 60), int(cumulative % 60)
        _progress(
            f"epoch {epoch:>3}/{max_epochs} train_loss={train_loss:.4f} val_mae={val_mae:.4f} "
            f"val_rmse={val_rmse_tick:.4f} lr={cur_lr:.2e} dt={epoch_time:.1f}s "
            f"elapsed={cum_m}m{cum_s:02d}s peak_vram={peak_vram_mb}MB "
            f"best={best_val:.4f}@{best_epoch}{marker}"
        )
        if improved:
            best_val = val_mae
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                _progress(f"early stop at epoch {epoch} (patience {patience} exhausted; best_val={best_val:.4f}@{best_epoch})")
                break

    train_time = time.perf_counter() - start

    test_mae, test_rmse, ms_per_batch = _infer_loop(model, test_loader, device)
    val_mae, val_rmse, _ = _infer_loop(model, val_loader, device)

    with torch.no_grad():
        x0, _ = next(iter(test_loader))
        x0 = x0.to(device)
        if tokenizer == "discrete_bpe":
            tb = model.tokenizer.encode(x0)
            tokens_per_input = int(tb.embeddings.shape[1])
        else:
            tokens_per_input = model.tokenizer.sequence_length()

    param_count = _count_params(model)
    result = {
        "val_mae": val_mae, "val_rmse": val_rmse,
        "test_mae": test_mae, "test_rmse": test_rmse,
        "train_time_sec": train_time,
        "inference_ms_per_batch": ms_per_batch,
        "peak_vram_mb": int(peak_vram_mb),
        "tokens_per_input": int(tokens_per_input),
        "param_count": param_count,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
    }

    import gc
    del model, optim, sched, train_loader, val_loader, test_loader, x0
    if tokenizer == "discrete_bpe":
        try:
            del fit_x, xs
        except NameError:
            pass
    for _ in range(3):
        gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return result
