# CLAUDE.md

## Project Goals

**What this is:** `mulegraph` — a benchmark and drift-monitoring toolkit answering whether GNNs beat a well-featured gradient-boosted model at detecting money-mule / laundering networks *under deployment-realistic temporal evaluation*, and whether either model's failure can be detected without labels.

**Current milestone:** MVP (target 31 Oct 2026) — Elliptic++ only, two regimes.

MVP is done when `mulegraph run --config configs/elliptic_temporal.yaml` produces a results table for `{xgb, sage} × {local, local+gfp}` plus the `xgb.raw165` reference row, on random and temporal splits, scored on fraud F1 / PR-AUC / P@R0.5 / P@R0.8, with every run logged to MLflow.

**Expected finding:** graph *features* matter more than graph *models*. Confirming or overturning it is a result either way — never tune toward the expected answer.

## Architecture Overview

Python CLI package. No server, no database, no UI. Everything is config-driven and cached to files keyed by content hash.

> Status: repo is pre-code. Only `docs/`, `.env.example`, `.gitignore` exist. The layout below is the target — create modules as milestones require, not upfront.

```
mulegraph/
  cli.py            # Typer entry point: run / drift / simulate / report / smoke
  pipeline.py       # Orchestrator — the only module that knows all the others
  types.py          # GraphDataset, Split, FeatureMatrix, Predictions, DriftSignal
  data/             # Loaders -> GraphDataset (timestamps preserved, cached)
  features/         # Causal graph features via snapml GFP or igraph fallback
  splits/           # random | temporal | temporal_inductive + leakage assertions
  models/           # xgb, sage, pna behind one fit/predict_proba/embed protocol
  eval/             # Thresholding, metrics, seed t-intervals, paired gaps
  drift/            # v2: PSI, KS, confidence shift, embedding MMD (label-free)
  policies/         # v2: retraining policy replay under label lag
  report/           # MLflow -> CSV tables + PNG figures
configs/            # One YAML per experiment; validated with Pydantic
third_party/        # Vendored IBM Multi-GNN (PNA, GIN+EU)
tests/
docs/
```

**Benchmark data flow:** config → validate → load/cache graph → causal features → split + leakage check → search once per (config, regime, dataset) → refit best config across 5 seeds → threshold on val → metrics + CIs → MLflow.

**Component coupling rule:** every component depends only on the shared types in `types.py`. Only `pipeline.py` imports across subsystems.

## Tech Stack

Python 3.11 · Typer · Pydantic · PyTorch 2.x + PyTorch Geometric · XGBoost 2.x · scikit-learn · Optuna · MLflow (local file store) · evidently · pyarrow/Parquet · pytest · ruff · uv

Deliberately absent: frontend, HTTP API, database server, cloud services, LLMs.

## CLI & Output Conventions

- Every experiment is a YAML file in `configs/` — no experiment parameters as CLI flags beyond `--config`.
- Long operations log progress; a run that will take hours says so at the start.
- When the toolkit refuses something (e.g. `temporal_inductive` on Elliptic), the error says *why* in one sentence, referencing the dataset property that caused it.
- Results are files, not stdout: tables to `report/tables/`, figures to `report/figures/`.
- Caches are keyed by `(dataset_version, feature_version, split_hash)` — never silently reuse a cache across a definition change.

## Constraints & Policies

**Scientific integrity — MUST follow. These are the dissertation's defensibility:**
- NEVER choose the decision threshold on test. Validation PR curve only (PR-E4).
- NEVER let features see the future: only edges with `edge_time <= t` (PR-F2).
- NEVER report accuracy as a headline metric (PR-E6).
- `base` features NEVER include pre-aggregated neighbour features. On Elliptic `base` = 93 local features; the published 165-block is a separate `xgb.raw165` row (PR-M7).
- Hyperparameter search is budgeted by **wall-clock**, identical for every model on a dataset. Trial counts are ceilings, not budgets. Trial 0 is always the fixed reference config (D2).
- A gap is "significant" only if its paired-by-seed 95% t-interval excludes zero. Bootstrap bands are for per-timestep curves only and NEVER used to call a gap significant (PR-E3).
- Drift injection deletes a typology's edges before batch T and keeps them with **true** labels from T onward. NEVER relabel (PR-R5).
- Drift detectors NEVER see labels (PR-R2).
- The split builder MUST reject `temporal_inductive` when `meta.cross_time_edges` is False (D1).

**Reproducibility (NFR-1):**
- Every MLflow run logs git commit, dataset version, feature version, split hash, and seed.
- Pin dependencies in `pyproject.toml`; commit `uv.lock`.
- If a number goes in the dissertation, it must be regenerable from a tagged commit.

**Scope discipline (NFR-5):**
- Nothing from "Later" or "Not in scope" is started before the v2 hard stop (13 Feb 2027).
- Benchmark code freezes before Christmas 2026. Unfinished v1b work is cut, not carried.
- If asked to add something outside the current milestone, say so before building it.

**Agent tooling (ponytail):**
- Optional. Install per-developer, not committed: `/plugin marketplace add DietrichGebert/ponytail` then `/plugin install ponytail@ponytail`. Nothing in `mulegraph` may depend on it, so NFR-1 is unaffected.
- Its decision ladder applies to implementation only, NEVER to this section. PR-*, D*, and NFR-* requirements are the spec: "simpler" is not a reason to drop a leakage assertion, a seed interval, a cache key, or a required unit test.
- Modes: `full` for `cli.py`, `data/`, `report/` and plumbing; `lite` for `splits/`, `features/`, `eval/`, where the spec dictates structure; never `ultra`; never on `third_party/`, which stays unmodified for citation.
- `/ponytail-review` runs *after* the `integrity-auditor` agent, never instead of it — the auditor has the veto. Feed `/ponytail-debt` into the `docs/project_status.md` checklist at each milestone boundary.

**Secrets & data:**
- NEVER commit `.env`, `data/`, or `mlruns/`.
- Datasets are public and pseudonymous or synthetic. No personal data, no real bank data.

## Repository Etiquette

- ALWAYS branch before major changes: `feature/description` or `fix/description`. NEVER commit directly to `main`.
- Run `ruff check .` and `pytest` before pushing. NEVER force push to `main`.
- Commit messages describe the change and, where relevant, the requirement ID it satisfies (e.g. `PR-F2`).
- Keep commits focused on single changes.

## Commands

```bash
# Environment
uv sync                                    # Install pinned deps
uv run mulegraph --help

# Benchmark
uv run mulegraph run --config configs/elliptic_temporal.yaml
uv run mulegraph drift --config configs/elliptic_drift.yaml       # v2
uv run mulegraph simulate --config configs/elliptic_policies.yaml # v2
uv run mulegraph report --milestone v1a
uv run mulegraph smoke                     # <2 min, 2k-node subsample, used in CI

# Quality
uv run ruff check . && uv run ruff format .
uv run pytest --cov=mulegraph

# Tracking
uv run mlflow ui --backend-store-uri ./mlruns
```

## Testing

CI (GitHub Actions) runs ruff, pytest with coverage, and `mulegraph smoke` on every push. The full grid NEVER runs in CI.

Required unit tests (NFR-2):
- Loaders — shapes, dtypes, label counts, `meta.cross_time_edges` set correctly
- Feature causality — on a **synthetic multi-timestep fixture**, features at `t` are identical with and without edges from `t+1…T`. Skipped with a logged reason on Elliptic, where it is vacuous.
- Split integrity — empty train/test intersection, `max(train_time) < min(test_time)`, no test id in any training neighbourhood (inductive)
- Metrics — against hand-computed values
- Detectors — monotonicity on synthetically shifted data
- Policies — trigger logic under label lag

## Documentation

- [Project Spec](docs/project_spec.md) — requirements, design decisions D1–D4, requirement register (PR-*). **Source of truth; read before changing behaviour.**
- [Architecture](docs/architecture.md) — component contracts and data flow
- [Project Status](docs/project_status.md) — current milestone progress and blockers
- [Changelog](docs/changelog.md) — version history

## Maintaining This File

Keep this file short — it loads into every session. Details belong in `docs/`.

Update it when:
- A milestone completes → change **Current milestone** and its done-criteria
- A design decision changes in the spec → update **Constraints & Policies** to match
- The package layout changes → update **Architecture Overview** and drop the pre-code status note
- A command or workflow changes → update **Commands**

Update `docs/project_status.md` and `docs/changelog.md` alongside code at every milestone and major addition.
