# Project Status

**Last updated:** 9 Sep 2026
**Current milestone:** MVP — due 31 Oct 2026 (52 days out)
**Spec version:** 0.6
**Overall state:** Pre-code. Spec and scaffolding done; no Python package yet.

---

## Milestones

| Milestone | Due | Status | Headline |
|---|---|---|---|
| Spec v0.6 | 8 Sep 2026 | ✅ Done | Requirements, design decisions D1–D4, requirement register |
| Scaffolding | 9 Sep 2026 | ✅ Done | `CLAUDE.md`, docs, `.env.example`, `.gitignore` |
| Week-1 gates | Sep 2026 | ⬜ Not started | Timing runs that set `W` and the grid arithmetic |
| **MVP** | **31 Oct 2026** | 🔵 **In progress** | Elliptic++, two regimes, two-by-two + `xgb.raw165`, MLflow logging |
| v1a | 21 Nov 2026 | ⬜ Not started | Elliptic complete: search decoupled from seeds, seed CIs, paired gaps, per-timestep curves, CI green |
| v1b | 12 Dec 2026 | ⬜ Not started | AMLworld HI-Small, three regimes, PNA reference row — **scope negotiable** |
| Benchmark freeze | before Christmas 2026 | ⬜ Not started | Code frozen; unfinished v1b work is cut, not carried |
| v2 | 5 Jan – 13 Feb 2027 | ⬜ Not started | Drift monitor, typology-shift events, retraining simulator, one-command reporting. **Hard stop.** |
| Later | — | ⬜ Deferred | PNA on remaining regimes, actor graph, TGN, Ethereum, explanations, cost-sensitive thresholds |

---

## What's been accomplished

**8 Sep 2026 — Specification (spec v0.6)**
- Full spec written: product requirements, milestone table, NFR-1…5, success criteria S1–S5, risk table with fallback ladders, technical design, requirement register (PR-*).
- Four design decisions resolved before any code: D1 graph unit, D2 search budget, D3 feature × model lineup, D4 label lag and batch unit.
- v1 split into v1a (Elliptic) and v1b (AMLworld) so the Elliptic result stands on its own if AMLworld slips.
- Statistics settled in advance: seed-level t-intervals headline, paired-by-seed gaps for significance, bootstrap confined to per-timestep bands.

**9 Sep 2026 — Repository scaffolding**
- `CLAUDE.md` written as project memory, with the scientific-integrity constraints (no test-set thresholding, no future edges, `base` excludes pre-aggregated neighbours, equal wall-clock budget, never relabel) stated as hard rules tied to their requirement ids.
- `docs/architecture.md`, `docs/project_status.md`, `docs/changelog.md` created.
- `.env.example` and `.gitignore` created; `report/exports/` deliberately kept tracked so milestone MLflow exports are committed.
- `.claude/` project tooling added and committed with the repo, so the environment versions alongside the code (NFR-1): a commit guard on the default branch, session-start state injection read from this file, `/update-docs-and-commit` and `/req` commands, and an `integrity-auditor` agent that audits diffs against the requirement register. The agent exists to compensate for having no second reader.

Nothing has been measured yet. No dataset loaded, no model fitted, no timing recorded.

---

## What's next

In order. Each item blocks the ones below it.

1. **Clear the week-1 gates** (see checklist below). `W` and the v1a/v1b grid sizes are unknown until these run, and the MVP date is not confirmable without them.
2. **Agree v1b scope with the supervisor, in writing, before v1a starts.** Minimum is the AMLworld two-by-two under temporal + inductive only.
3. **Bootstrap the package** — `pyproject.toml` with pinned deps and `uv.lock`, `mulegraph/` skeleton, `types.py`, Typer CLI stub, ruff + pytest config, GitHub Actions workflow.
4. **Elliptic++ loader** (PR-D1, PR-D4) — deterministic, caches to disk, sets `meta.cross_time_edges = False` and `meta.feature_blocks`.
5. **Causal feature builder** (PR-F1–F3) — GFP or igraph fallback, plus the synthetic multi-timestep fixture and its causality test.
6. **Split builder** (PR-E1) — random and temporal, with leakage assertions and `split_hash`.
7. **XGBoost then GraphSAGE** behind the shared protocol (PR-M1–M3, PR-M5).
8. **Evaluator and MLflow wiring** (PR-E2, PR-E4, PR-O1) — threshold on validation, metrics, run logging.
9. **Draft the methods chapter.** Spec §1.6 mitigates "Christmas writing slips" by drafting it at MVP, not at Christmas. Do not defer this.

---

## Week-1 gates

These set `W` and the grid arithmetic. Nothing downstream is reliable until they are done.

- [ ] `snapml` GraphFeaturePreprocessor installs — or the `igraph` fallback is chosen and recorded
- [ ] One GraphSAGE fit on full Elliptic timed **with our loader**
- [ ] One GraphSAGE fit and one PNA fit on AMLworld HI-Small timed **using IBM's Multi-GNN repo and its own preprocessing** (not our loader)
- [ ] AMLworld HI-Small real time span read off during that run
- [ ] `W` set to `max(120 min, 2 × slowest measured single fit)`; v1a/v1b grid arithmetic recomputed and the v1a/v1b dates confirmed

---

## MVP definition of done

- [ ] Elliptic++ transaction graph loads from one command and caches, with `meta.cross_time_edges = False`
- [ ] Causal graph features (fan-in/out, degree, scatter-gather, short cycles) via GFP or igraph fallback
- [ ] Causality unit test passes on a synthetic multi-timestep fixture
- [ ] Leak-checked temporal split: train ≤ t34, val t35–37, test t38–49; id-overlap assertion; `split_hash` logged
- [ ] Threshold chosen on the validation PR curve, never on test
- [ ] `mulegraph run --config configs/elliptic_temporal.yaml` produces the results table for `{xgb, sage} × {local, local+gfp}` + `xgb.raw165`, on random and temporal splits
- [ ] Metrics: fraud F1, PR-AUC, P@R0.5, P@R0.8
- [ ] Every run logs params, metrics, seed, feature version, git commit to MLflow

---

## Open blockers

Numbered per spec §2.6. These two gate work:

- **Compute access.** Is BlueBEAR (or equivalent GPU) available to final-year project students? Fallback is a personal GPU plus a capped cloud budget (target under £50).
- **v1b scope.** Must be agreed in writing *before v1a starts*. Minimum: AMLworld two-by-two under temporal + inductive only. Negotiable extras in priority order: the other two regimes, then PNA on temporal + inductive.

Also open with the supervisor: dissertation submission date and whether an autumn inspection exists; per-typology vs aggregate AMLworld reporting; whether to include ROC-AUC at all; whether the Multi-GNN node-task adaptation for Elliptic is worth the time; which typologies to hold out for injected drift events and how many events test H1; whether the 3-day/1-day/rest drift split leaves enough laundering edges in training.

---

## Active risks

Full table in spec §1.6. Currently live:

| Risk | Fallback if it fires |
|---|---|
| Elliptic grid does not fit after week-1 timing | Lower `W` (keeping it equal for all models), then GNN seeds 5 → 3 |
| AMLworld grid does not fit | In order: drop the random regime → GNN seeds to 3 → PNA trial 0 only → PNA to Later |
| `snapml` will not install | `igraph` fallback with identical feature definitions |
| Multi-GNN hard to adapt | Time-box two weeks, then GraphSAGE only |
| Scope creep into v2 before v1 is done | v2 cannot start before the benchmark freeze |
| Christmas writing slips | Methods chapter drafted at MVP — item 9 in *What's next* |

---

## Update rule

Revise this file at every milestone boundary and whenever a gate, blocker, or risk changes state. Move completed work into *What's been accomplished* with its date, and record the same event in [changelog.md](changelog.md). Keep this file about **now and next** — history belongs in the changelog.
