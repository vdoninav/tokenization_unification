"""Aggregates runs/<*>/metrics.json into master.parquet, emits tables and Pareto plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import friedmanchisquare, wilcoxon


def _iter_metrics(runs_root: Path):
    for p in sorted(runs_root.glob("*/metrics.json")):
        try:
            yield json.loads(p.read_text())
        except Exception:
            continue


def aggregate_runs(runs_root: Path) -> pl.DataFrame:
    rows = list(_iter_metrics(runs_root))
    if not rows:
        return pl.DataFrame()
    flat = []
    for r in rows:
        pc = r.pop("param_count", {}) or {}
        r2 = {**r, **{f"param_{k}": v for k, v in pc.items()}}
        flat.append(r2)
    return pl.DataFrame(flat)


def seed_aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """Mean/std/cv over seeds per (dataset, horizon, tokenizer).

    Single-seed groups (e.g., 2 of 3 seeds failed) yield ``None`` from polars' .std();
    we fill those with 0.0 so downstream formatters don't crash. ``seed_count`` is
    exposed so the paper can flag cells with fewer than the planned 3 seeds.
    """
    grouped = df.group_by(["dataset", "horizon", "tokenizer"]).agg([
        pl.col("test_mae").mean().alias("test_mae_mean"),
        pl.col("test_mae").std().alias("test_mae_std"),
        pl.col("test_rmse").mean().alias("test_rmse_mean"),
        pl.col("test_rmse").std().alias("test_rmse_std"),
        pl.col("train_time_sec").mean().alias("train_time_sec_mean"),
        pl.col("peak_vram_mb").mean().alias("peak_vram_mb_mean"),
        pl.col("tokens_per_input").mean().alias("tokens_per_input_mean"),
        pl.col("inference_ms_per_batch").mean().alias("inference_ms_mean"),
        pl.col("seed").count().alias("seed_count"),
    ]).with_columns([
        pl.col("test_mae_std").fill_null(0.0),
        pl.col("test_rmse_std").fill_null(0.0),
    ]).with_columns(
        (pl.col("test_mae_std") / pl.col("test_mae_mean").clip(lower_bound=1e-12)).alias("test_mae_cv"),
    ).with_columns(
        pl.col("test_mae_cv").fill_null(0.0).fill_nan(0.0),
    )
    return grouped


def _fmt(v: float | None, prec: int = 3) -> str:
    """Defensive float formatter: handles None/NaN gracefully so partial-matrix
    aggregations don't crash the table writers."""
    if v is None:
        return "n/a"
    try:
        if v != v:
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{v:.{prec}f}"


def _esc(s: str) -> str:
    """Pass-through identity for tokenizer/dataset names.

    Historically escaped LaTeX special characters; now retained for call-site
    compatibility while CSV/Markdown emitters write names verbatim.
    """
    if not isinstance(s, str):
        return str(s)
    return s


def _write_csv_md(out_path_no_ext: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write paired CSV + Markdown table files.

    ``out_path_no_ext`` is the path without extension; this writes
    ``<stem>.csv`` and ``<stem>.md`` side by side. CSV uses UTF-8 and the
    standard comma delimiter; Markdown is a GitHub-flavored pipe table with a
    ``| --- |`` separator row.
    """
    out_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path_no_ext.with_suffix(".csv")
    md_path = out_path_no_ext.with_suffix(".md")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _min_max(col: pl.Expr, inv: bool = False) -> pl.Expr:
    rng = col.max() - col.min()
    norm = (col - col.min()) / rng.fill_null(1).clip(lower_bound=1e-12)
    return (1 - norm) if inv else norm


def compute_utility(
    agg: pl.DataFrame, weights: tuple[float, float, float, float],
) -> pl.DataFrame:
    w_q, w_t, w_m, w_s = weights
    return agg.with_columns([
        _min_max(pl.col("test_mae_mean"), inv=True).over(["dataset", "horizon"]).alias("Q"),
        _min_max(pl.col("train_time_sec_mean"), inv=False).over(["dataset", "horizon"]).alias("T"),
        _min_max(pl.col("peak_vram_mb_mean"), inv=False).over(["dataset", "horizon"]).alias("M"),
        _min_max(pl.col("test_mae_cv"), inv=True).over(["dataset", "horizon"]).alias("S"),
    ]).with_columns(
        (w_q * pl.col("Q") - w_t * pl.col("T") - w_m * pl.col("M") + w_s * pl.col("S")).alias("U"),
    )


def pareto_mask(x: list[float], y: list[float]) -> list[bool]:
    """True where (x_i, y_i) is not strictly dominated by any other (both minimized)."""
    mask = [True] * len(x)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                mask[i] = False
                break
    return mask


def write_main_table(agg: pl.DataFrame, out_path: Path) -> None:
    """Emit the headline MAE/RMSE table as paired CSV + Markdown files.

    ``out_path`` is the legacy path (typically ending in ``.tex``); the writer
    strips the suffix and emits ``<stem>.csv`` plus ``<stem>.md``. A cell is
    flagged ``*`` in the tokenizer column when fewer than 3 seeds succeeded.
    """
    header = ["dataset_horizon", "tokenizer", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "seeds"]
    rows: list[list[str]] = []
    for (ds, h), group in agg.sort(["dataset", "horizon", "tokenizer"]).group_by(
        ["dataset", "horizon"], maintain_order=True,
    ):
        for r in group.iter_rows(named=True):
            n_seeds = r.get("seed_count", 0)
            marker = "*" if n_seeds < 3 else ""
            rows.append([
                f"{_esc(ds)}/H={h}",
                f"{_esc(r['tokenizer'])}{marker}",
                _fmt(r["test_mae_mean"]),
                _fmt(r["test_mae_std"]),
                _fmt(r["test_rmse_mean"]),
                _fmt(r["test_rmse_std"]),
                str(n_seeds),
            ])
    _write_csv_md(out_path.with_suffix(""), header, rows)


def write_utility_table(agg: pl.DataFrame, out_path: Path) -> None:
    """Emit the per-tokenizer utility ranking (3 weight configs) as CSV + Markdown."""
    header = ["tokenizer", "U_equal", "U_accuracy", "U_deployment"]
    weight_sets = [
        ((0.25, 0.25, 0.25, 0.25), "equal"),
        ((0.70, 0.10, 0.10, 0.10), "accuracy"),
        ((0.25, 0.40, 0.25, 0.10), "deployment"),
    ]
    U_by_weight: dict[str, pl.DataFrame] = {}
    for w, name in weight_sets:
        U_by_weight[name] = compute_utility(agg, weights=w).group_by("tokenizer").agg(
            pl.col("U").mean().alias(f"U_{name}"),
        )
    merged = U_by_weight["equal"].join(U_by_weight["accuracy"], on="tokenizer").join(
        U_by_weight["deployment"], on="tokenizer",
    )
    sortable = merged.with_columns(pl.col("U_equal").fill_null(float("-inf")).alias("_sort"))
    rows: list[list[str]] = []
    for r in sortable.sort("_sort", descending=True).iter_rows(named=True):
        rows.append([
            _esc(r["tokenizer"]),
            _fmt(r["U_equal"]),
            _fmt(r["U_accuracy"]),
            _fmt(r["U_deployment"]),
        ])
    _write_csv_md(out_path.with_suffix(""), header, rows)


def write_compute_table(agg: pl.DataFrame, out_path: Path) -> None:
    """Emit the per-cell training-time table as paired CSV + Markdown.

    Layout mirrors the paper's compute-cost table: cells (dataset/horizon) as
    rows, tokenizers as columns, with a ``TOTAL`` row at the bottom giving the
    sum across all 18 runs (3 datasets x 2 horizons x 3 seeds) per tokenizer.
    All values are in hours with 2 decimals; per-cell entries are the mean
    over the 3 seeds; ``TOTAL`` is ``mean_per_cell * 3`` summed across the
    six cells, which equals the raw sum of ``train_time_sec`` per tokenizer
    converted to hours.
    """
    cells = agg.sort(["dataset", "horizon"]).select(["dataset", "horizon"]).unique(
        maintain_order=True,
    ).rows()
    tokenizers = sorted(agg["tokenizer"].unique().to_list())

    header = ["cell"] + tokenizers
    rows: list[list[str]] = []
    totals: dict[str, float] = {t: 0.0 for t in tokenizers}
    for ds, h in cells:
        cell_label = f"{_esc(ds)}/H={h}"
        row = [cell_label]
        for tok in tokenizers:
            sub = agg.filter(
                (pl.col("dataset") == ds)
                & (pl.col("horizon") == h)
                & (pl.col("tokenizer") == tok),
            )
            if sub.is_empty():
                row.append("n/a")
                continue
            mean_sec = sub["train_time_sec_mean"].item()
            if mean_sec is None:
                row.append("n/a")
                continue
            n_seeds = int(sub["seed_count"].item() or 0)
            mean_hr = mean_sec / 3600.0
            row.append(_fmt(mean_hr, prec=2))
            totals[tok] += mean_sec * n_seeds
        rows.append(row)

    total_row = ["TOTAL"] + [_fmt(totals[t] / 3600.0, prec=2) for t in tokenizers]
    rows.append(total_row)

    _write_csv_md(out_path.with_suffix(""), header, rows)


def write_pareto_plots(agg: pl.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (ds, h), group in agg.group_by(["dataset", "horizon"], maintain_order=True):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        ax = axes[0]
        x = group["test_mae_mean"].to_list()
        y = group["train_time_sec_mean"].to_list()
        labels = group["tokenizer"].to_list()
        mask = pareto_mask(x, y)
        for xi, yi, lab, m in zip(x, y, labels, mask):
            ax.scatter(xi, yi, c="C0" if m else "gray", s=60)
            ax.annotate(lab, (xi, yi), fontsize=8)
        ax.set_xlabel("Test MAE"); ax.set_ylabel("Train time (s)")
        ax.set_title(f"{ds}/H={h}: MAE vs. time")
        ax = axes[1]
        y2 = group["peak_vram_mb_mean"].to_list()
        mask2 = pareto_mask(x, y2)
        for xi, yi, lab, m in zip(x, y2, labels, mask2):
            ax.scatter(xi, yi, c="C0" if m else "gray", s=60)
            ax.annotate(lab, (xi, yi), fontsize=8)
        ax.set_xlabel("Test MAE"); ax.set_ylabel("Peak VRAM (MB)")
        ax.set_title(f"{ds}/H={h}: MAE vs. VRAM")
        plt.tight_layout()
        out_path = out_dir / f"pareto_{ds}_h{h}.pdf"
        fig.savefig(out_path)
        plt.close(fig)


def write_stats_table(df: pl.DataFrame, out_path: Path) -> None:
    """Pairwise Wilcoxon per (dataset, horizon) on per-seed test_mae.

    Emits paired CSV + Markdown table files. p-values are rendered via
    ``_fmt`` (3 decimal places by default) so partial matrices with NaNs
    remain readable.
    """
    header = ["dataset_horizon", "tok_a", "tok_b", "p_value"]
    rows: list[list[str]] = []
    for (ds, h), g in df.group_by(["dataset", "horizon"], maintain_order=True):
        toks = g["tokenizer"].unique().to_list()
        for i, a in enumerate(toks):
            for b in toks[i + 1 :]:
                a_vals = g.filter(pl.col("tokenizer") == a).sort("seed")["test_mae"].to_list()
                b_vals = g.filter(pl.col("tokenizer") == b).sort("seed")["test_mae"].to_list()
                if len(a_vals) != len(b_vals) or len(a_vals) < 3:
                    continue
                try:
                    _, p = wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    p = float("nan")
                rows.append([
                    f"{_esc(ds)}/H={h}",
                    _esc(a),
                    _esc(b),
                    _fmt(p),
                ])
    _write_csv_md(out_path.with_suffix(""), header, rows)


_C_BY_DATASET = {"ETTh1": 7, "Weather": 21, "Electricity": 321}


def write_rq2_plot(agg: pl.DataFrame, out_path: Path) -> None:
    """Tokenizer rank vs. variate count C - split into one subplot per horizon.

    Single-axis version had two horizons sharing identical (C, rank) coordinates,
    so square markers (H=336) covered circle markers (H=96) pixel-for-pixel. With
    side-by-side subplots, every tokenizer's trajectory is visible and the colour
    legend is shared.
    """
    ranked = agg.with_columns(
        pl.col("test_mae_mean").rank("ordinal").over(["dataset", "horizon"]).alias("rank"),
    ).with_columns(
        pl.col("dataset").replace_strict(_C_BY_DATASET, return_dtype=pl.Int64).alias("C"),
    )
    horizons = sorted(ranked["horizon"].unique().to_list())
    n_h = max(len(horizons), 1)
    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, h in zip(axes, horizons):
        sub_h = ranked.filter(pl.col("horizon") == h)
        for tok in sorted(sub_h["tokenizer"].unique().to_list()):
            sub = sub_h.filter(pl.col("tokenizer") == tok).sort("C")
            ax.plot(sub["C"].to_list(), sub["rank"].to_list(),
                    marker="o", linewidth=1.5, markersize=7, label=tok)
        ax.set_xscale("log")
        ax.set_xlabel("Variate count C (log scale)")
        ax.set_title(f"H = {h}")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Rank within (dataset, horizon)\n(1 = best)")
    axes[-1].legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_rq3_rule_block(agg: pl.DataFrame, out_path: Path) -> None:
    """Per-cell empirical winners + textual decision rule, as paired CSV + Markdown.

    Replaces a sklearn decision tree as the primary RQ3 artifact. With only six
    cells of training data (3 datasets x 2 horizons), a depth-2 tree typically
    degenerates into "majority class everywhere", which is uninformative. A
    direct enumeration of per-cell winners plus an explicit majority-rule-with-
    exceptions sentence is more honest and more readable.

    Output: ``<stem>.csv`` and ``<stem>.md`` contain the per-cell winners table.
    The textual rule is appended only to the Markdown file (as a paragraph
    below the table) so the CSV stays purely tabular.
    """
    winners_df = agg.group_by(["dataset", "horizon"]).agg(
        pl.col("tokenizer").sort_by("test_mae_mean").first().alias("winner"),
    ).sort(["dataset", "horizon"])

    rows = list(winners_df.iter_rows(named=True))
    winner_list = [r["winner"] for r in rows]
    counts: dict[str, int] = {}
    for w in winner_list:
        counts[w] = counts.get(w, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    header = ["dataset", "C", "horizon", "empirical_winner"]
    table_rows: list[list[str]] = []
    for r in rows:
        c = _C_BY_DATASET.get(r["dataset"], "?")
        table_rows.append([
            _esc(r["dataset"]),
            str(c),
            str(r["horizon"]),
            _esc(r["winner"]),
        ])

    stem = out_path.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for tr in table_rows:
            w.writerow(tr)

    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for tr in table_rows:
        md_lines.append("| " + " | ".join(tr) + " |")
    md_lines.append("")

    if len(sorted_counts) == 1:
        winner = sorted_counts[0][0]
        md_lines.append(
            f"**Empirical decision rule:** within the observed matrix the "
            f"optimal strategy is `{winner}`: tau* = `{winner}` in every "
            f"studied (C, H) pair. At the current matrix coverage the features "
            f"(C, H, C/T) provide no information gain for a decision tree, so "
            f"the rule reduces to a constant. Extending the benchmark to more "
            f"datasets and horizons would let us test whether this dominance "
            f"persists in broader regimes."
        )
    else:
        majority, majority_n = sorted_counts[0]
        total = sum(counts.values())
        exceptions_parts: list[str] = []
        for tok, _cnt in sorted_counts[1:]:
            cell_strs = []
            for r in rows:
                if r["winner"] == tok:
                    cell_strs.append(f"{r['dataset']}/H={r['horizon']}")
            exceptions_parts.append(f"`{tok}` (in {', '.join(cell_strs)})")
        md_lines.append(
            f"**Empirical decision rule:** strategy `{majority}` is optimal "
            f"in {majority_n} of {total} observed cells. Exceptions where a "
            f"different tokenizer was best: {'; '.join(exceptions_parts)}. "
            f"To first order the strategy-selection rule reduces to "
            f"tau* = `{majority}` with the exception cells listed explicitly. "
            f"Six observed cells are too few to tell whether the exceptions "
            f"reflect structural properties of the data or statistical noise; "
            f"building a full classification tree would require extending the "
            f"matrix to additional datasets and horizons."
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_rq3_plot(agg: pl.DataFrame, out_path: Path) -> None:
    """Illustrative decision tree: features (C, H, C/T) -> winning tokenizer.

    With only 6 cells this is descriptive, not statistical. Rendered as a proper
    sklearn ``plot_tree`` diagram (boxes + arrows + thresholds + class labels)
    instead of the previous monospace ``export_text`` dump, which read like raw
    log output rather than a figure.
    """
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    per_cell = agg.group_by(["dataset", "horizon"]).agg(
        pl.col("tokenizer").sort_by("test_mae_mean").first().alias("winner"),
    ).with_columns(
        pl.col("dataset").replace_strict(_C_BY_DATASET, return_dtype=pl.Int64).alias("C"),
    )
    X = np.column_stack([
        per_cell["C"].to_numpy(),
        per_cell["horizon"].to_numpy(),
        (per_cell["C"].to_numpy() / 336),
    ])
    y = per_cell["winner"].to_list()

    unique_classes = sorted(set(y))
    if len(unique_classes) < 2:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            f"Empirical winner across all cells: {unique_classes[0]}\n"
            f"(no split learnable from the {len(y)} observed cells)",
            ha="center", va="center", fontsize=12, fontweight="bold",
        )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_tree(
        tree,
        feature_names=["C", "H", "C/T"],
        class_names=list(tree.classes_),
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        ax=ax,
        fontsize=10,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenstudy.analysis")
    parser.add_argument("--input", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    df = aggregate_runs(args.input)
    if df.is_empty():
        print("No runs found.")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output / "master.parquet")

    agg = seed_aggregate(df)
    (args.output / "tables").mkdir(exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    write_main_table(agg, args.output / "tables" / "main.csv")
    write_utility_table(agg, args.output / "tables" / "utility.csv")
    write_stats_table(df, args.output / "tables" / "stats.csv")
    write_compute_table(agg, args.output / "tables" / "compute.csv")
    write_pareto_plots(agg, args.output / "figures")
    write_rq2_plot(agg, args.output / "figures" / "rq2_dimensionality.pdf")
    write_rq3_plot(agg, args.output / "figures" / "rq3_rule.pdf")
    write_rq3_rule_block(agg, args.output / "tables" / "rq3_rule.csv")
    print(f"Wrote {args.output}/master.parquet + tables/ + figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
