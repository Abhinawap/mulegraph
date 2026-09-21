# CLAUDE.md

## Project Goals

**What this is:** `mulegraph` — a benchmark and drift-monitoring toolkit answering whether GNNs beat a well-featured gradient-boosted model at detecting money-mule / laundering networks *under deployment-realistic temporal evaluation*, and whether either model's failure can be detected without labels. A portfolio project: the repo, its README and its figures are the deliverable.

**Expected finding:** graph *features* matter more than graph *models*. Confirming or overturning it is a result either way — never tune toward the expected answer.

**Headline piece:** the label-free drift monitor's lead time on Elliptic's real t43 dark-market collapse (F1 0.85 → 0.02). The two-by-two benchmark on Elliptic++ and AMLworld HI-Small is the context for it.

## Architecture Overview

Python CLI package. No server, no database, no UI. Everything is config-driven and cached to files keyed by content hash.

```
mulegraph/
  cli.py            # Typer entry point: run / drift / score / smoke
  config.py         # Pydantic schema for configs/*.yaml (RunConfig, DriftRunConfig, ScoreConfig)
  pipeline.py       # Orchestrator — the only module that knows all the others
  types.py          # GraphDataset, Split, FeatureMatrix, Predictions; a "unit" is a node or an edge
  util.py           # Paths, hashing, seeding, provenance
  data/             # elliptic.py (node task), amlworld.py (edge task), synthetic.py -> GraphDataset
  features/         # Causal graph features via snapml GFP; node_agg_v1 fold on node tasks
  splits/           # random | temporal | temporal_rolling on batch_id + leakage assertions
  models/           # xgb, sage behind one fit/predict_proba protocol
  eval/             # Thresholding, metrics, seed t-intervals, curves.py (per-timestep)
  drift/            # detectors.py (PSI, KS, confidence shift, alert rate), monitor.py (batch scores, lead time)
  report/           # tables.py (MLflow -> CSV), figures.py (curves, drift PNG)
configs/            # One YAML per experiment; validated with Pydantic
tests/
docs/
```

**Benchmark data flow:** config → validate → load/cache graph → causal features → split + leakage check → fit each (regime, model, seed) → threshold on val → metrics + per-timestep curves → MLflow child run → tables + figures.

**Drift data flow:** same to the fit, then score every unit from the validation window on → PSI / KS / confidence shift per batch against the validation reference → lead time against the F1 curve → tables + figure.

**Score data flow (D5):** same load, features and split → fit once and cache the model plus its validation threshold → score one batch → ranked alert queue + label-free health check → exit 3 on a flag.

**Component coupling rule:** every component depends only on the shared types in `types.py`. Only `pipeline.py` imports across subsystems.

## Tech Stack

Python 3.11 · Typer · Pydantic · PyTorch 2.x + PyTorch Geometric · XGBoost 3.2 (newest supporting 3.11) · scikit-learn · scipy · matplotlib · MLflow (local SQLite file; the `file:` store is refused by MLflow 3) · pyarrow/Parquet · pytest · ruff · uv.

Deliberately absent: frontend, HTTP API, database server, cloud services, LLMs.

## CLI & Output Conventions

- Every experiment is a YAML file in `configs/` — no experiment parameters as CLI flags beyond `--config`.
- Long operations log progress; a run that will take hours says so at the start.
- When the toolkit refuses something, the error says *why* in one sentence, referencing the dataset property that caused it.
- Results are files, not stdout: tables to `report/tables/`, figures to `report/figures/`.
- Caches are keyed by `(dataset_version, feature_version, split_hash)` — never silently reuse a cache across a definition change.

## Constraints & Policies

**Scientific integrity — MUST follow. These are what make the numbers defensible:**
- NEVER choose the decision threshold on test. Validation PR curve only (PR-E4).
- NEVER let features see the future: only edges with `edge_time <= t` (PR-F2).
- NEVER report accuracy as a headline metric (PR-E6).
- `base` features NEVER include pre-aggregated neighbour features. On Elliptic `base` = 93 local features; the published 165-block is a separate `xgb.raw165` row (PR-M7).
- A gap is "significant" only if its paired-by-seed 95% t-interval excludes zero. Bootstrap bands are for per-timestep curves only and NEVER used to call a gap significant (PR-E3).
- Drift detectors NEVER see labels (PR-R2): they get features, scores and the validation-chosen threshold. Labels enter only in the lead-time and event evaluation.
- Every model runs its fixed reference configuration; runs log `trials_completed = 0` honestly (D2).

**Reproducibility (NFR-1):**
- Every MLflow run logs git commit, dataset version, feature version, split hash, and seed.
- Pin dependencies in `pyproject.toml`; commit `uv.lock`.
- If a number goes in the README, it must be regenerable from a tagged commit.

**Secrets & data:**
- NEVER commit `.env`, `data/`, or `mlruns/`.
- Datasets are public and pseudonymous or synthetic. No personal data, no real bank data.

## Building (ponytail)

All code is written under the **ponytail** skill (`/ponytail`, full level). Before writing anything, climb the ladder and stop at the first rung that holds: does it need to exist → already in this repo → stdlib → an installed dependency → one line → the minimum code that works.

- **No scaffolding for later.** No accepted-but-ignored config fields, no unreachable branches kept "as a home" for future work.
- **Docstrings are one line** plus the requirement id. Rationale lives in `docs/design.md`, `docs/project_status.md` → *Verified method notes*, or `docs/methods.md`.
- **Integrity overrides ponytail.** Never simplify away leakage assertions, threshold-on-validation, causality guards, shape/dtype validation at type boundaries, or an error message that says why.
- Run `/ponytail-review` on the diff before committing.

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
uv run mulegraph run --config configs/elliptic_mvp.yaml
uv run mulegraph run --config configs/amlworld_xgb.yaml       # XGBoost rows only, laptop
uv run mulegraph run --config configs/amlworld_hi_small.yaml  # full grid with SAGE, Kaggle
uv run mulegraph run --config configs/elliptic_rolling.yaml   # fixed vs rolling refit, ~35 min
uv run mulegraph drift --config configs/elliptic_drift.yaml
uv run mulegraph score --config configs/amlworld_score.yaml  # one day's alert queue + health check; exit 3 on a flag
uv run mulegraph smoke                     # ~15 s, synthetic graph, benchmark + one scored batch, used in CI

# Quality
uv run ruff check . && uv run ruff format .
uv run pytest --cov=mulegraph

# Tracking
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Testing

CI (GitHub Actions) runs ruff, pytest with coverage, and `mulegraph smoke` on every push. The full grid NEVER runs in CI.

Required unit tests (NFR-2):
- Loaders — shapes, dtypes, label counts, `meta.cross_time_edges` set correctly
- Feature causality — on a **synthetic multi-timestep fixture**, features at `t` are identical with and without edges from `t+1…T`. Skipped with a logged reason on Elliptic, where it is vacuous.
- Split integrity — empty train/test intersection, `max(train_time) < min(test_time)`
- Metrics — against hand-computed values
- Detectors — monotonicity on synthetically shifted data; no detector signature accepts labels

## Documentation

- [Design](docs/design.md) — the question, design decisions D1–D4, component contracts, requirement register (PR-*). **Source of truth; read before changing behaviour.**
- [Methods](docs/methods.md) — the write-up of the method, results and related work
- [Project Status](docs/project_status.md) — where things stand, verified method notes
- [Changelog](docs/changelog.md) — history

## Maintaining This File

Keep this file short — it loads into every session. Details belong in `docs/`.

Update it when a design decision changes in `docs/design.md`, the package layout changes, or a command changes. Update `docs/project_status.md` and `docs/changelog.md` alongside code at every major addition.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
