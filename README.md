# Tokenization Unification

A controlled study of five tokenization strategies for long-horizon
multivariate time-series forecasting.

## Abstract

In modern transformer-based architectures, the choice of tokenization
strategy - the transformation of a continuous-valued signal into a sequence
of tokens passed to the transformer - has become one of the central
architectural decisions. The literature distinguishes four basic approaches:
point-wise, patch-based, channel-wise, and discrete. Comparisons between
them are typically performed within different architectures and under
different training budgets, which makes it impossible to isolate the effect
of tokenization itself. This project performs a controlled comparison of
five representatives of the four families (point, patch, variate, discrete
scalar, discrete BPE) on a fixed lightweight transformer backbone with
identical optimizer, schedule, and budget. The full campaign - 90 runs
across 3 datasets, 2 horizons, 3 seeds - completed on a single NVIDIA RTX
5090 (32 GB VRAM) at approximately 230 GPU-hours.

## Experimental matrix

| Axis | Values |
|---|---|
| Tokenizer | `point`, `patch`, `variate`, `discrete_scalar`, `discrete_bpe` |
| Dataset (channels) | ETTh1 (7), Weather (21), Electricity (321) |
| Look-back T | 336 (fixed) |
| Horizon H | 96, 336 |
| Seed | 2021, 2022, 2023 |

Total: 5 x 3 x 2 x 3 = **90 runs**

## Reproduction

```bash
uv sync
uv run python -m tokenstudy.batch src/tokenstudy/configs/pilot.yaml
uv run python -m tokenstudy.batch src/tokenstudy/configs/matrix_a.yaml
uv run python -m tokenstudy.analysis --input runs/ --output results/
```

Dataset CSVs are expected under `~/data/{ETT-small,weather,electricity}/`.
Override via the env vars `ETT_PATH`, `WEATHER_PATH`, `ELEC_PATH`

## Results

- `results/master.parquet` - one row per (dataset, horizon, tokenizer, seed)
- `results/figures/*.pdf` - Pareto plots per cell, RQ2 dimensionality plot,
  RQ3 decision rule
- `results/tables/*.csv` and `*.md` - main metrics, compute cost, utility
  ranking under three weight configurations, pairwise Wilcoxon, RQ3 winners

## Repository layout

```
src/tokenstudy/    Python package (config, batch runner, analysis, 5 tokenizers,
                   3 dataset loaders, transformer backbone + wrapper, training loop)
tests/             pytest suite; data-gated tests skip without CSVs
results/           master.parquet + figures + tables (committed)
runs/              per-cell artifacts (gitignored; recreated on run)
pyproject.toml     uv project metadata
uv.lock            pinned dependencies
```

## Hardware footprint

The full 90-run matrix consumed approximately **230 GPU-hours** on one
NVIDIA RTX 5090 (32 GB VRAM) - about 9.5 days of continuous compute.
Per-tokenizer breakdown ranges from ~1.3 h (variate, 18 runs) to ~133 h
(discrete_scalar, 18 runs); see `results/tables/compute.md` for details

## License

MIT License