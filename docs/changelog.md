# Changelog

Project history, newest first. Spec revisions are tracked separately in [project_spec.md](project_spec.md) §Change log.

Add a dated entry for every milestone tag and every change that alters behaviour, requirements, or reproducibility. Milestone tags are `mvp`, `v1`, `v2`; each tag commits an MLflow CSV export under `report/exports/`.

---

## Unreleased

Nothing yet. Next entry will be the Python package skeleton and the week-1 timing results.

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
