"""Sequential batch orchestrator: manifest -> one run per cell, idempotent, resumable."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

from .config import RunConfig, config_hash
from .training.determinism import configure_deterministic
from .training.runner import execute_run


def _release_gpu_memory() -> None:
    """Force a clean GPU state between cells.

    The CUDA caching allocator keeps freed blocks for reuse - good for one cell, but
    accumulates fragmentation across many cells. After a few dozen cells, a later cell
    can OOM on a tensor allocation even though most VRAM is technically free, just
    spread across small blocks.

    Multi-pass: PyTorch tensors can have circular references (autograd graph nodes,
    optimizer state holding parameter refs, etc.) that need 2-3 GC passes to fully
    collect. We sync first to ensure all pending CUDA ops are done before releasing.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _in_venv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or "VIRTUAL_ENV" in os.environ
    )


def _git_dirty() -> bool:
    try:
        return subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        ) != 0
    except Exception:
        return False


def _load_manifest(path: Path) -> list[RunConfig]:
    data = yaml.safe_load(path.read_text())
    return [RunConfig(**entry) for entry in data["runs"]]


def _already_done(run_dir: Path, expected_hash: str) -> bool:
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return False
    try:
        rec = json.loads(mpath.read_text())
        return rec.get("config_hash") == expected_hash
    except Exception:
        return False


def _append_log(log_path: Path, line: str) -> None:
    """Append to the batch log AND echo to stdout (flushed) so tmux attach sees live progress."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_manifest(
    manifest_path: Path,
    runs_root: Path,
    data_root: Path,
    device: str = "cuda",
    skip_git_check: bool = False,
    skip_venv_check: bool = False,
    is_pilot: bool = False,
) -> None:
    configs = _load_manifest(manifest_path)
    log_path = runs_root / "batch.log"

    if not skip_venv_check and not _in_venv():
        raise RuntimeError("batch.py refuses to run outside a venv. Use `uv run python -m tokenstudy.batch ...`")
    if not skip_git_check and not is_pilot and _git_dirty():
        raise RuntimeError("git_dirty=true; commit or stash before launching main batch.")

    configure_deterministic()

    total = len(configs)
    for idx, cfg in enumerate(configs, 1):
        run_dir = runs_root / cfg.run_name()
        h = config_hash(cfg)
        if _already_done(run_dir, h):
            _append_log(log_path, f"[{idx}/{total}] {cfg.run_name()} status=SKIPPED")
            continue
        _release_gpu_memory()
        _append_log(log_path, f"[{idx}/{total}] {cfg.run_name()} status=STARTED (tail runs/{cfg.run_name()}/train.log for per-epoch progress)")
        t0 = time.perf_counter()
        try:
            rec = execute_run(
                cfg, runs_root=runs_root, data_root=data_root,
                device=device, uv_lock_path=Path("uv.lock"),
            )
            dt = time.perf_counter() - t0
            _append_log(
                log_path,
                f"[{idx}/{total}] {cfg.run_name()} "
                f"MAE={rec['test_mae']:.4f} RMSE={rec['test_rmse']:.4f} "
                f"time={int(dt // 60)}m{int(dt % 60):02d}s status=OK",
            )
        except Exception as e:
            dt = time.perf_counter() - t0
            (run_dir).mkdir(parents=True, exist_ok=True)
            (run_dir / "FAILED.log").write_text(traceback.format_exc())
            _append_log(
                log_path,
                f"[{idx}/{total}] {cfg.run_name()} "
                f"time={int(dt // 60)}m{int(dt % 60):02d}s status=FAILED err={type(e).__name__}",
            )
            _release_gpu_memory()
            continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenstudy.batch", description="Batch orchestrator")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--data-root", type=Path, default=Path.home() / "data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pilot", action="store_true", help="skip git-clean check")
    args = parser.parse_args(argv)

    run_manifest(
        manifest_path=args.manifest,
        runs_root=args.runs_root,
        data_root=args.data_root,
        device=args.device,
        is_pilot=args.pilot,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
