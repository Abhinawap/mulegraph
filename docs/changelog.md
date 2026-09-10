# Changelog

Project history, newest first. Spec revisions are tracked separately in [project_spec.md](project_spec.md) §Change log.

Add a dated entry for every milestone tag and every change that alters behaviour, requirements, or reproducibility. Milestone tags are `mvp`, `v1`, `v2`; each tag commits an MLflow CSV export under `report/exports/`.

---

## Unreleased

### 10 Sep 2026 — Ponytail audit: ahead-of-milestone scaffolding removed (NFR-5)
- **Removed** v1a search scaffolding: `mulegraph/search.py`, `search_space` on every model and on the protocol, and every `SearchConfig` field except `enabled`. Each model now merges its `trial0()` reference config under the config's `params` in its own constructor (XGBoost previously relied on `search.py` for this; D2).
- **Integrity-audit follow-ups:** the reporter raises when a child run lacks an identity or provenance tag instead of writing "unknown" (spec §2.3 now names the tags it reads: `regime`, `features`); XGBoost records `trial0_source` in its fit info; duplicate Elliptic++ txIds raise a one-sentence error; `feature_version` now also hashes `raw_sha256`.
- **Removed** v1a/v2 placeholders: `types.DriftSignal`, the accepted-but-ignored `eval.gap_pairs` / `bootstrap_samples` / `per_timestep`, and the unused `util.log_stage`.
- **Removed** dependencies `optuna` (returns with v1a) and `python-dotenv` (never imported).
- **Removed** `docs/architecture.md`, which duplicated spec §2.
- **Changed** `features.window` from a per-family mapping to one int (spec 0.7). **Every `feature_version` hash changes**, so existing feature caches are orphaned and recompute. No results had been logged.
- **Simplified** the reporter's tag lookup (no alias table), Elliptic++ id resolution (`pd.Index.get_indexer`), and SAGE fan-out handling: a fan-out whose length differs from `layers` is now an error instead of being silently extended.
- Module docstrings cut to one line plus requirement id. The verified findings they carried moved to `project_status.md` → *Verified method notes*.
- `CLAUDE.md` gains a *Building (ponytail)* section: every change climbs the ponytail ladder, builds for the current milestone only, and never simplifies away an integrity guard.

Next entry will be the component modules (loader, feature builder, splits, models, evaluator) and the remaining week-1 timing results (gates 2–5).

---

## 9 Sep 2026 — Python package foundation

### Added
- Repository pushed to GitHub (`Abhinawap/mulegraph`, private) with four Milestones carrying the spec's due dates and 25 issues, one per §1.3 milestone deliverable, each listing its requirement ids, done-criteria and applicable integrity constraints. Delivered MVP items are closed citing their commits; a closed "not planned" issue lists every Later / Not-in-scope item so the boundary is visible where work is tracked (NFR-5).
- `.claude/skills/issue`, `close-issue`, `issues-sync` — commands that create spec-aware issues (refusing out-of-scope work), close them with commit evidence while ticking `project_status.md`, and report drift between the two.
- `pyproject.toml` with fully pinned dependencies and a committed `uv.lock` (NFR-1), plus `.python-version`, `README.md` and a GitHub Actions workflow running ruff, format check and pytest on every push. Tests needing the real Elliptic++ files are marked `elliptic` and skipped in CI, which cannot download the dataset.
- `mulegraph/types.py` — the shared types every component depends on (`GraphDataset`, `DatasetMeta`, `FeatureMatrix`, `Split`, `Predictions`, `DriftSignal`), each validating its own shapes and dtypes on construction. Arrays are NumPy throughout; only the GNN converts to torch, which keeps features, splits and evaluation cheap to test.
- `mulegraph/config.py` — strict Pydantic schema for `configs/*.yaml` (`extra="forbid"`, so a mistyped key is an error rather than a silently ignored setting), with `MULEGRAPH_FEATURE_BACKEND` / `MULEGRAPH_DEVICE` overrides logged when applied.
- `mulegraph/util.py` — paths from the environment, content hashing, git provenance, seeding, and a `Timer` whose output feeds the week-one budget arithmetic.
- `mulegraph/data/synthetic.py` — a synthetic Elliptic-shaped generator with planted signal (shifted local features and fan-in stars on illicit nodes). It does two things the real dataset cannot: give CI something to run end to end, and provide a graph whose edges genuinely span timesteps, without which the PR-F2 causality test is vacuous.
- `mulegraph/models/base.py`, `mulegraph/search.py`, `mulegraph/cli.py` — the model protocol (PR-M5), the search phase, and the Typer entry point. `drift`, `simulate` and `report` exit with a one-sentence message naming the milestone that will deliver them.
- `configs/elliptic_mvp.yaml` (both regimes in one file, so one command produces the whole MVP table) and `configs/smoke.yaml`.

### Decided
- **Python 3.11 is kept, so XGBoost is pinned at 3.2.0** — 3.3+ require Python ≥3.12. The spec's tech stack says "XGBoost 2.x"; 3.2.0 is the newest release compatible with the pinned interpreter and its `XGBClassifier` API is unchanged for our use.
- **No search runs in the MVP** (`search.enabled: false`, and `true` is a validation error). The budget is wall-clock and must be identical for every model (D2), but `W` is not known until week-one gate 5. Runs therefore log `trials_completed: 0` rather than a nominal 1 — an honest zero, not a faked search.
- Torch installs from PyPI (already a CUDA 13.0 build, matching the driver here) with `pyg-lib` from the PyG wheel index via `find-links`; no custom PyTorch index and no extras.

### Fixed
- **The GFP drive pattern recorded in the gate-1 entry above was wrong.** `transform` inserts the batch into the in-memory graph itself, so the documented `partial_fit(batch_t)` then `transform(batch_t)` inserts every batch twice and doubles every degree, fan and histogram count (verified: a vertex with 2 out-edges reports 4). The correct pattern is `transform(batch_t)` alone, for t ascending. Corrected here and in `project_status.md`; the feature builder will expose no method that can call `partial_fit`, and a unit test pins the counts.

---

## 9 Sep 2026 — Week-1 gate 1: GFP backend confirmed

### Decided
- **Feature backend is IBM `snapml` GraphFeaturePreprocessor, not the `igraph` fallback.** `snapml==1.17.2` installs from a wheel on Python 3.11 / linux x86_64 with no build step. The spec §1.6 risk "GFP does not install or is unmaintained" is retired, and the fallback path in PR-F1 is now a contingency that is not being built.
- `lc-cycle_len` stays at the spec's bound of 10 for now, but is flagged as a tunable pending real-data timing in gate 2 (see below).

### Verified
- `GraphFeaturePreprocessor` natively provides every feature family in PR-F1: `fan`, `degree`, `scatter-gather`, `lc-cycle` (`lc-cycle_len` default 10), `temp-cycle`, `vertex_stats`, each with an independent `_tw` time window — so the per-dataset window configuration in spec §2.5 maps onto the API without wrapping.
- Output is deterministic across repeated runs and across thread counts (1 vs 12), satisfying NFR-1 for the feature stage.

### Constraint discovered (affects PR-F2)
- **GFP is causal only by usage, not by construction.** The preprocessor is stateful, and `transform` inserts the batch it is given before scoring it. Ingesting the full edge table before transforming leaks future edges into past rows — demonstrated on a toy fixture where a *t1* transaction acquired neighbour amount statistics produced by a *t2* edge.
- Consequence: `mulegraph/features/` MUST drive GFP strictly in time order, one batch at a time, calling **`transform(batch_t)` only**. `transform` inserts the batch itself, so calling `partial_fit` first inserts it twice and doubles every count. PR-F2 is therefore an implementation constraint on the feature builder, not merely a configuration setting, and the synthetic multi-timestep causality fixture is the test that guards it.

### Notes
- `lc-cycle` cost is superlinear in graph density: 4,000 edges over 400 nodes did not complete in 3 minutes, while the same edge count over 3,000 nodes finished in seconds. Elliptic's per-timestep components are sparse, but the cycle bound must be timed on real data before `W` is fixed.
- Local RTX 4060 Laptop GPU confirmed available, so the remaining week-1 timing gates are not blocked on BlueBEAR access.
- Repository remains pre-code; this gate was cleared in a throwaway virtualenv, and no dependency has been pinned into the project yet.

---

## 9 Sep 2026 — Project scaffolding

### Added
- `CLAUDE.md` — project memory loaded into every session: goals, architecture summary, tech stack, CLI conventions, scientific-integrity constraints, repository etiquette, commands, testing requirements
- `docs/architecture.md` — shared types, component contracts, the three Python protocols, benchmark and drift run sequences, storage layout, Elliptic-vs-AMLworld behaviour table
- `docs/project_status.md` — milestone table, accomplishments, next actions, week-1 gates, MVP definition of done, blockers, live risks
- `docs/changelog.md` — this file
- `.env.example` — paths, MLflow tracking, Kaggle download credentials, device and thread caps, feature backend, Optuna storage, reproducibility vars
- `.gitignore` — ignores `.env`, `data/`, `mlruns/`, Python and tooling caches; keeps `report/exports/` tracked so milestone exports are committed
- `.claude/` — project-scoped Claude Code tooling, committed so the environment versions with the code (NFR-1):
  - `hooks/guard_main.py` — refuses commits on `main`/`master`; allows the initial commit and an explicit `MULEGRAPH_ALLOW_MAIN_COMMIT=1` override
  - `hooks/project_state.py` — injects milestone, checklist counts and git state at session start, read from `project_status.md` so there is one source of truth
  - `skills/update-docs-and-commit/` — updates both docs from the working diff, then commits
  - `skills/req/` — looks up a `PR-*` / `D*` / `NFR-*` / `S*` id in the spec
  - `agents/integrity-auditor.md` — read-only diff audit against PR-E4, PR-F2, PR-M7, PR-E1, PR-E3, D2, PR-O1

### Notes
- Repository remains pre-code. No Python package, configs, or data yet.
- Week-1 timing gates (see [project_status.md](project_status.md)) must clear before the MVP schedule is confirmed — `W` and the v1a/v1b grid arithmetic depend on them.

---

## 8 Sep 2026 — Specification complete (spec v0.6)

### Added
- `docs/project_spec.md` — full specification: product requirements, milestone table, non-functional requirements, success criteria S1–S5, risk table, technical design, and the requirement register (PR-*)

### Decided
- **D1 Graph unit** — Elliptic++ modelled as the transaction graph; actor graph deferred to Later. Only two regimes are distinct on Elliptic; the three-regime contrast lives on AMLworld.
- **D2 Search budget** — wall-clock only, one cap `W` per dataset identical for every model; trial counts are per-model ceilings; trial 0 is always a fixed reference config.
- **D3 Feature × model** — two-by-two `{base, base+GFP} × {XGBoost, GraphSAGE}` plus a PNA/GIN+EU reference. `base` defined per dataset: Elliptic = 93 local features, with the published 165-block kept as a separate `xgb.raw165` row.
- **D4 Label lag and batch unit** — simulator carries lag `L` with a lagged-label oracle as ceiling; batch unit is one timestep on Elliptic, 6 hours on AMLworld; benchmark and drift splits differ on AMLworld.
- v1 split into v1a (Elliptic, 21 Nov) and v1b (AMLworld, 12 Dec, scope negotiable).
- Confidence intervals defined as seed-level t-intervals with paired-by-seed gaps; bootstrap restricted to per-timestep bands.
- AMLworld half of success criterion S3 recast as pre-registered hypothesis H1 — reported, not gated.
