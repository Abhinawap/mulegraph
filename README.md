# mulegraph

A reproducible benchmark and drift-monitoring toolkit answering one question: **when evaluated
the way a bank would have to deploy it, do graph neural networks beat a well-featured
gradient-boosted model at detecting money-mule and laundering networks — and can we tell when
either model stops working, without labels?**

The expected finding is that graph *features* matter more than graph *models*. Confirming or
overturning that under leakage-free evaluation is a result either way; nothing here is tuned
toward the expected answer.

See [`docs/project_spec.md`](docs/project_spec.md) for the full specification (requirements,
design decisions D1–D4, requirement register), [`docs/architecture.md`](docs/architecture.md)
for component contracts, and [`docs/project_status.md`](docs/project_status.md) for current
milestone progress.

## Status

**MVP** (due 31 Oct 2026): Elliptic++ only, two regimes, `{xgb, sage} × {base, base_gfp}` plus
the `xgb.raw165` reference row. Hyperparameter search, per-timestep curves and paired gaps are
v1a; the drift monitor and retraining simulator are v2.

## Install

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). The environment is pinned in
`uv.lock`; every dissertation number must be regenerable from a tagged commit (NFR-1).

```bash
uv sync
uv run mulegraph --help
```

The lockfile pulls `torch` built against CUDA 13.0 and `pyg-lib` from the PyG wheel index. On a
machine without a GPU everything still runs on CPU — set `MULEGRAPH_DEVICE=cpu`, or use
`sampler: {kind: full_batch}` in the config if `pyg-lib` is unavailable.

## Data

Datasets are **not** downloaded automatically and are never committed. Place (or symlink) the
Elliptic++ transaction files where the loader expects them:

```
data/raw/elliptic_pp/2023.1/
    txs_features.csv
    txs_classes.csv
    txs_edgelist.csv
```

The files come from [Elliptic++](https://github.com/git-disl/EllipticPlusPlus) (Elmougy & Liu,
KDD '23). If you already have them elsewhere:

```bash
mkdir -p data/raw/elliptic_pp/2023.1
for f in txs_features txs_classes txs_edgelist; do
  ln -s /path/to/elliptic/$f.csv data/raw/elliptic_pp/2023.1/$f.csv
done
```

Everything downstream is cached by content hash under `data/cache/`, so a change to a feature or
split definition never silently reuses a stale cache.

## Run

```bash
# Full MVP benchmark: 5 model configs x 2 regimes x 5 seeds, logged to MLflow
uv run mulegraph run --config configs/elliptic_mvp.yaml

# Two-minute end-to-end check on a synthetic Elliptic-shaped graph (used in CI)
uv run mulegraph smoke

# Browse the runs
uv run mlflow ui --backend-store-uri ./mlruns
```

Results are files, not stdout: tables land in `report/tables/`, figures in `report/figures/`.

## Development

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -m "not elliptic" --cov=mulegraph   # tests needing the real dataset are marked
uv run pytest -m elliptic                          # only once the raw CSVs are in place
```

Branch before making changes (`feature/…` or `fix/…`); commits to `main` are refused by a hook.

## Licence and data handling

Code is MIT. Elliptic++ is used under its published research licence. No personal data is
processed — the datasets are public and pseudonymous or synthetic — and no claim is made that
results on them generalise to real bank transactions (NFR-4).
