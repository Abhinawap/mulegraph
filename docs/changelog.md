# Changelog

Project history, newest first. Spec revisions are tracked separately in [project_spec.md](project_spec.md) §Change log.

Add a dated entry for every milestone tag and every change that alters behaviour, requirements, or reproducibility. Milestone tags are `mvp`, `v1`, `v2`; each tag commits an MLflow CSV export under `report/exports/`.

---

## Unreleased

Next entry will be the Python package skeleton and the remaining week-1 timing results (gates 2–5).

---

## 9 Sep 2026 — Week-1 gate 1: GFP backend confirmed

### Decided
- **Feature backend is IBM `snapml` GraphFeaturePreprocessor, not the `igraph` fallback.** `snapml==1.17.2` installs from a wheel on Python 3.11 / linux x86_64 with no build step. The spec §1.6 risk "GFP does not install or is unmaintained" is retired, and the fallback path in PR-F1 is now a contingency that is not being built.
- `lc-cycle_len` stays at the spec's bound of 10 for now, but is flagged as a tunable pending real-data timing in gate 2 (see below).

### Verified
- `GraphFeaturePreprocessor` natively provides every feature family in PR-F1: `fan`, `degree`, `scatter-gather`, `lc-cycle` (`lc-cycle_len` default 10), `temp-cycle`, `vertex_stats`, each with an independent `_tw` time window — so the per-dataset window configuration in spec §2.5 maps onto the API without wrapping.
- Output is deterministic across repeated runs and across thread counts (1 vs 12), satisfying NFR-1 for the feature stage.

### Constraint discovered (affects PR-F2)
- **GFP is causal only by usage, not by construction.** The preprocessor is stateful: `partial_fit` accumulates the graph, `transform` reads accumulated state. Ingesting the full edge table before transforming leaks future edges into past rows — demonstrated on a toy fixture where a *t1* transaction acquired neighbour amount statistics produced by a *t2* edge.
- Consequence: `mulegraph/features/` MUST drive GFP strictly in time order, one batch at a time (`partial_fit(batch_t)` then `transform(batch_t)`), and must never fit globally before transforming. PR-F2 is therefore an implementation constraint on the feature builder, not merely a configuration setting, and the synthetic multi-timestep causality fixture is the test that guards it.

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
