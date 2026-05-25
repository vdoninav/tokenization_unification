"""CLI: `uv run python -m tokenstudy.cli run <config.yaml>` for a single cell."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .config import load_config
from .training.determinism import configure_deterministic
from .training.runner import execute_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenstudy", description="Tokenization study CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run a single cell")
    run_p.add_argument("config", type=Path)
    run_p.add_argument("--runs-root", type=Path, default=Path("runs"))
    run_p.add_argument("--data-root", type=Path, default=Path.home() / "data")
    run_p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        configure_deterministic()
        cfg = load_config(args.config)
        uv_lock = Path("uv.lock")
        rec = execute_run(
            cfg, runs_root=args.runs_root, data_root=args.data_root,
            device=args.device, uv_lock_path=uv_lock,
        )
        print(f"[OK] {cfg.run_name()} test_mae={rec['test_mae']:.4f} test_rmse={rec['test_rmse']:.4f}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
