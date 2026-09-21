# Changelog

Project history, newest first. Add a dated entry for every tag and every change that alters behaviour, requirements, or reproducibility. Each tag commits an MLflow CSV export under `report/exports/`.

---

## Unreleased

### 21 Sep 2026 — Ponytail audit cuts
- **Removed** the `temporal_inductive` regime. The config accepted it and `build_split` refused it every time, on every dataset: accepted-but-ignored scaffolding. Design §1 already listed it as out of scope; methods §2 now says three regimes, and why an inductive regime would add nothing on Elliptic. A config that still asks for it fails at validation with the three valid regimes named, rather than with the old dataset-specific reason. Three tests went with it.
- **Removed** the `temp_cycle` feature family from the config schema: no config used it. The GFP driver still passes every snapml family explicitly (temp-cycle off), so the feature output and every recorded `feature_version` are unchanged.
- **Simplified** `plot_curves`, whose injectable interval nobody injected; `conf_shift`, which returned a p-value nobody read; and a one-entry tag alias in the results reporter.
- **Kept**, against the audit: the snapml family mapping (the explicit off is what pins the output across snapml upgrades), and the health file's own JSON writer (`util.write_json` sorts keys, which would bury `status`).

## v1.2 — 21 Sep 2026

`mulegraph score` and `mulegraph smoke` now write a static HTML report for each scored batch, beside the alert queue and health file: a banner saying whether a detector flagged the day, each detector's score against its flag level, the top 25 alerts with the features behind each, and the provenance. It is one self-contained page with no scripts and nothing loaded from outside. The rendered layout was not looked at in a browser before this tag (none was available), only its structure, its figure and its tests; see the entry below.

Tagged on the merge that adds this section. No new MLflow experiment and no new benchmark result: the exports and numbers listed under `v1.0` and `v1.1` are the ones this tag carries.

### 21 Sep 2026 — A static HTML report for every scored batch
- **Added** `report/html.py` and a third output of `mulegraph score`: `<name>_batch<d>_report.html`, one self-contained page (no scripts, nothing loaded from outside, the figure embedded) with a banner saying whether a detector flagged the day, each detector's score beside its flag level as a figure and a table, the top 25 alerts, and the provenance (scored-at and fitted-at commits, threshold, reference window, dataset, feature and split hashes). The caution from the health file, that no flag is not an all-clear, is on the page. `score` and `smoke` now print the report's path; `run_score` returns it as its third value. Design D5, §3.4a.
- **Checked** without a browser, since none is installed here: the tags balance (a parser walks both variants), there is no script and no external reference, and the embedded figure is a valid image showing the flagged detectors of day 10 in orange above their flag levels and day 8 entirely below. **The rendered layout has not been looked at**, so the CSS is unverified beyond that.
- **Tests** (`tests/test_score.py`): the page names every detector and both commits, says "Drift flagged" exactly when the health file does, and loads nothing; what the data supplies (feature names, the note) is escaped, and the queue is cut at 25 rows (turning escaping off fails it); the shifted-batch CLI test also checks the page.
- **Not added:** the day-8 and day-10 pages are not committed. GitHub shows HTML as source, so a visitor could not see them without cloning, and `mulegraph smoke` builds one in about 15 s with no download.

## v1.1 — 21 Sep 2026

`mulegraph score` becomes usable. In `v1.0` its health check flagged every AMLworld test day, so exit status 3 carried no signal there. `v1.1` adds a configurable health-check reference window (`reference: [0, 7]` in `configs/amlworld_score.yaml`): it exits 0 on days 8 and 9 and 3 on day 10, the laundering tail. The window was chosen after looking at those days, so that is a sensitivity check, not a held-out result, and day 10 is the easiest drift there is; the Elliptic result (no detector warned of t43) is unchanged. `mulegraph smoke` now also scores a batch, so CI exercises the score path, and the README opens with what `score` does, with real day-8 and day-10 outputs committed under `report/tables/`.

Tagged on the merge that adds this section. No new MLflow experiment: `score` writes files rather than runs (D5), so the exports listed under `v1.0` are the ones this tag carries.

### 21 Sep 2026 — README leads with `mulegraph score`
- **Changed** the README to open with what the tool does for a user: a new "What it does" section above "What I found" shows a real day-8 alert queue (top three rows with the features behind each alert) and the health check on day 8 (exit 0) and day 10 (exit 3), with the limits stated beside it: simulator data, day 10 is the easiest drift there is, and the reference window was chosen after seeing those days. The tagline now describes the score command first and the benchmark second, and the run list gains the `score` line.
- **Added** `report/tables/amlworld_batch{8,10}_{alerts.csv,health.json}`, from `configs/amlworld_score.yaml` on a clean tree at `b584125` (the health files carry that stamp): 654,467 transactions and 466 alerts on day 8, 396 and 136 on day 10. Threshold 0.9915 on both. Day 10 flags PSI, score shift and alert rate.
- **Found** while producing them: `git status --porcelain` counts untracked files, so the first day's output files made the next run's stamp `-dirty`. Each day's outputs are now moved out of the tree before the next run.

### 21 Sep 2026 — `mulegraph smoke` also runs the scoring path
- **Changed** `mulegraph smoke` to finish by scoring the first test batch with the model and settings of `configs/amlworld_score.yaml` (calibrated detectors, all history before the test window as the health reference) and to print the alert queue and health file it wrote. CI runs smoke on every push, so the product path is now exercised there, and the no-download first run produces an alert queue rather than only a results table. About 15 s; outputs land in the gitignored `report/tables/smoke_*`.
- **Found** while building it: with the score command's default detectors (fixed flag levels, validation window as reference), the synthetic graph's batch 38 reads `drift_flagged` on data with no drift. Each synthetic batch has 51 units, so PSI (1.18 against 0.2) and alert rate (0.86 against 0.5) are sampling noise. The smoke step uses the shipped settings instead, which report no drift (PSI 0.90 against a calibrated 2.3). That setting was fixed before looking at the result, not adjusted to it.

### 21 Sep 2026 — A longer health-check reference for `mulegraph score`; README heading
- **Added** `reference: [first, last]` to the score config: the batches the health check compares the scored day against. Unset it is the validation window, as before. It must be an ordered range wholly before `batch`, and a range reaching into training batches logs a warning (the model scored those in-sample). It is a health-check setting only: changing it never refits the model or moves the threshold (PR-E4), and no label reaches it (PR-R2). Design D5, §3.4a.
- **Measured** on AMLworld, which detectors flag (F1 on those days is 0.38 on day 8 and 0.45 on day 9, so days 8 and 9 are ordinary; day 10 is the laundering tail, where ordinary traffic stops and 396 units remain):

  | Reference | Day 8 | Day 9 | Day 10 |
  |---|---|---|---|
  | 6–7 (validation, the old default) | psi, ks, conf | all four | not run |
  | 4–7 or 4–8 | ks | ks, alert | psi, conf, alert (4–9) |
  | 2–7 | none | not run | not run |
  | 0–7 (all history before the test window) | none | none | psi, conf, alert |

  Wider windows raise every calibrated flag level (PSI 0.0002 → 1.6) and keep the response to the real event: on day 10 PSI is 8.7 against 1.6, and score shift and alert rate flag too. KS never flags day 10 (0.67 against 0.91): at 480k+ rows it rejects on almost every column, which is why it also flags ordinary days under narrow windows. The shipped `configs/amlworld_score.yaml` sets `reference: [0, 7]`; run through the CLI it exits 0 on days 8 and 9 and 3 on day 10. A day costs about 3 min (178 s, 6.2 GB peak) instead of 14 s, because calibration compares every reference day against the rest.
- **Caveats, recorded as found.** The window was chosen after looking at days 8 to 10, the days it is judged on, so treat the table as a sensitivity check rather than a held-out result. Day 10 is the easiest event there is; quiet on days 8 and 9 and loud on day 10 says nothing about a subtle shift or about model failure, and the Elliptic result (methods §12.4, no detector warned of t43) is unchanged. Day 9's alert share is 2.6 times day 8's and is not flagged (0.84 against 0.94). Days 0–5 are training days, so their scores are in-sample; their alert shares (0.0004–0.0019) are not inflated against validation days (0.0007–0.0008), but that is one model on one dataset.
- **Changed** the README heading "In 30 seconds" to "What I found".

## v1.0 — 21 Sep 2026

Portfolio release. The two-by-two benchmark runs end to end on both datasets, and the answer is the same on each: graph *features* help, a graph *model* does not. XGBoost with causal GFP features beats GraphSAGE on Elliptic++ and on AMLworld HI-Small, every paired-by-seed interval excluding zero (methods §12.3).

The headline drift result is a negative one, reported as measured: on Elliptic's real t43 dark-market collapse (F1 0.85 → 0.02), no label-free detector gives warning once a flag must hold for two timesteps. PSI, KS and the score-shift detector never hold; the alert-rate detector peaks at t43 but recovers at t44. The event diagnosis (methods §12.5) says why, and the `temporal_rolling` regime (methods §12.6) measures what post-shift labels buy instead: post-t43 F1 0.02 → 0.35, and only from t47.

Tagged on the merge of PR #39, after #38 (the exports) and #40 (the hardware correction). It also ships `mulegraph score` (D5, below). Carries an MLflow export for every experiment it reports: `elliptic_mvp_runs.csv` (153 runs, 3 parents), `elliptic_rolling_runs.csv` (82), `elliptic_drift_runs.csv` (105), `amlworld_xgb_runs.csv` (20) and `amlworld_hi_small_runs.csv` (Kaggle), all under `report/exports/`.

- **Fixed** a reproducibility gap (NFR-1): the reported Elliptic parent run `ba1e269a…` (§10.4) was in no committed export — `mvp_elliptic_mvp_runs.csv` holds two *other* parents, `341e0f93…` and `281caef1…`. The rolling, drift and AMLworld-XGBoost experiments had no export at all. `mvp_elliptic_mvp_runs.csv` is kept unchanged as the `mvp` tag's snapshot.
- **Added** the export command to methods §10.4, so the rule at the top of this file is a command and not a habit. `amlworld_dev` is scratch and stays unexported.

### 21 Sep 2026 — `mulegraph score`: the deployed scoring path (D5)
- **Added** `mulegraph score --config <yaml>`: one XGBoost model scores one batch and writes a ranked alert queue (account ids, top three TreeSHAP contributions per alert) and a label-free health check from the existing drift detectors. It exits 3 when a detector flags, so a scheduler can act on it. The model and its validation threshold are fitted once, cached under the dataset, feature and split hashes, and reused. Design D5, §3.4a; `configs/amlworld_score.yaml`.
- **Verified** on AMLworld HI-Small from a cold model cache: day 8 in 66 s at 5.8 GB peak RSS (654,467 transactions, 466 alerts), day 9 in 14 s from the saved model. Threshold 0.991463 and per-day F1 0.3837 and 0.4538, recomputed offline from the alert queues, match the `amlworld_xgb` benchmark to four decimals.
- **Found** that the health check flags every AMLworld test day. With a two-day reference (days 6–7), each calibrated flag level comes from a single leave-one-out comparison of two near-identical days (PSI flags at 0.0002 against the fixed default of 0.2), and KS at 650k rows per day rejects on almost any difference. Recorded as found; the reference window is the validation window by design (D4), and separating the two is a design change not yet made.
- **Fixed before merge, from the integrity audit:** alert explanations summed TreeSHAP over every tree in the booster, including the 50 that early stopping keeps past the best iteration, so an alert's listed reasons could differ from what drove its score by up to 4.2 log-odds; they now use the same trees as `predict_proba` (to 1e-6), with a test that fails on the old code. The model cache key now also covers the resolved device and the XGBoost version (GPU and CPU `hist` grow different trees). The fit's commit, seed, device, params and library version are written beside the model and copied into every health file as `model_fit`, because a cache hit otherwise stamps only the commit that scored, not the one that fitted.
- **Split** `DriftConfig` into `DetectorConfig` (the detectors) and `DriftConfig` (adds the label-side lead-time rule), so `score` takes no field it would ignore.
- **Tests** (`tests/test_score.py`): one threshold call on validation rows only (PR-E4); a second batch reuses the model; unlabelling the scored batch and fitting cold in a fresh directory leaves the threshold, queue and health file identical (PR-R2; a planted label leak fails it); a changed definition forces a refit; a shifted batch exits 3; and the config refuses SAGE, a batch outside the test window and a non-temporal regime, saying why.

### 21 Sep 2026 — Kaggle hardware corrected: T4, not P100
- **Fixed** the hardware named for the AMLworld SAGE run: it was a Kaggle **T4 x2** notebook, and the code uses one of the two GPUs (`device: cuda` on all 13 runs in `report/exports/amlworld_hi_small_runs.csv`; the export does not record the GPU model). README, methods §12.3, status, `kaggle/amlworld_sage.md` and one test comment said P100. No number changes. `v1.0` was moved to a commit that carries the corrected README.

### 20 Sep 2026 — README rewritten for a software-engineering audience
- **Rewrote** `README.md` from a research log (~260 lines) to a ~180-line page: a 30-second summary, quickstart with real smoke output, an engineering section (architecture diagram, test-enforced integrity rules, three bugs), results next to published numbers and the leaky random split, and one drift figure. Detector tables, footnotes and the full "what didn't work" list stay in `docs/methods.md`.
- **Added** `scripts/readme_figures.py`, which writes `report/figures/readme_alert_load.png`, `report/figures/readme_recovery.png` and `report/tables/amlworld_alert_load.csv` from the `amlworld_xgb` run (`2546822`) and `elliptic_rolling_curves.csv`. On the realistic AMLworld test days 8–9, GFP takes alerts per laundering case caught at recall 0.5 from 26.3 to 4.4 (whole window: 11.1 to 1.8, flattered by the tail). Methods §12.3.
- **Added** a limitation (methods §11): the AMLworld SAGE edge head is a lower bound, as transaction features never enter its message passing.
- **Fixed** the stale drift caption that said PSI held a flag at t48; that flag went away at `ea5ecce`.

### 20 Sep 2026 — AMLworld SAGE rows from the Kaggle T4 run
- **Ran** `configs/amlworld_hi_small.yaml` on a Kaggle T4 (the notebook's T4 x2 accelerator; the code uses one of the two GPUs) from a clean clone at `87f1f5a` (12 fits, 1 h 14 min; SAGE fits 10–13 min each, every seed early-stopped at best epoch 3–5 of 10). XGBoost rows are identical to the laptop's `amlworld_xgb` run.
- **Found** `sage.base` F1 0.052 ± 0.007 and `sage.base_gfp` 0.142 ± 0.031, against XGBoost's 0.209 and 0.539. Paired-by-seed F1 gaps: XGBoost − SAGE +0.157 on `base` and +0.397 on `base_gfp`, GFP − base within SAGE +0.090; every interval excludes zero. The ranking holds on the realistic test days 8–9 as well as over the laundering tail.
- **Added** `report/tables/amlworld_hi_small_*`, `report/figures/amlworld_hi_small_curves.png` and the MLflow export `report/exports/amlworld_hi_small_runs.csv`. README, methods §12.3, status.

### 19 Sep 2026 — First Kaggle attempt: a CUDA-only bug in the SAGE edge head
- **Fixed** `SAGEEdgeModel._infer` read `batch.input_id` after `_forward` had moved the batch to the GPU in place (PyG's `batch.to()`), so edge SAGE failed at its first validation pass on CUDA. CPU hides it: `.to("cpu")` is a no-op. The edge-SAGE fixture now also runs on CUDA when one is present (skipped in CI).
- **Changed** the Kaggle recipe: `MPLBACKEND=Agg` (Kaggle's inline backend is not in the uv env), the dataset CSV is found under `/kaggle/input` rather than assumed at one mount path, and grep is line-buffered so progress streams.
- The Kaggle XGBoost fits matched the laptop's to four decimals of val PR-AUC; GFP took 846 s there.

### 19 Sep 2026 — AMLworld XGBoost rows from a clean commit
- **Added** `configs/amlworld_xgb.yaml` (`2546822`; NFR-1), the XGBoost half of `amlworld_hi_small.yaml`, so the committed `amlworld_xgb_*` tables have a committed config.
- **Reran** it from a clean tree at `2546822`: every value in the results table, curves and figure is identical to the `7c22e3e-dirty` run; only the `commit` stamp changed. Methods §12.3, status.

### 19 Sep 2026 — Rolling refit, the alert-rate detector, and a PSI fix
- **Added** the `temporal_rolling` regime (`34db885`; PR-E7). The temporal window slides forward one batch per test batch; each step is an ordinary temporal split through the same builder, leakage assertions and cache, thresholds on its own validation window (PR-E4) and scores one test batch. The step test sets are stitched into one MLflow child run per (model, seed) with a per-row threshold, so `n_seeds` counts seeds, not refits. Rank metrics are not logged for a rolling run (twelve models do not share a score scale); per-batch PR-AUC is in the curve, and each step's threshold, val_f1, val_pr_auc and best_iteration are logged at `step=t`. `configs/elliptic_rolling.yaml` runs fixed vs rolling on the two-by-two (260 fits). Design §1 narrows the out-of-scope line from "retraining-policy simulation" to policies beyond this one zero-lag refit.
- **Added** the alert-rate detector (`ea5ecce`; PR-R1): the absolute log ratio of the share of scored units at or above the validation-chosen threshold against the reference share. It sees scores and the threshold only (PR-R2); `score_batches` takes `threshold` and `alert_flag`. `configs/elliptic_drift.yaml` runs it.
- **Fixed** PSI (`ea5ecce`). Importing snapml enables flush-to-zero for the process, so the subnormal cut placed above a constant reference value collapsed onto it and PSI scored 0 for any shift in a constant column, on every real run. Bins are now (a, b] at every distinct reference quantile but the maximum, which also gives binary columns a cut. Found as a test that passed alone and failed after the pipeline tests; the test now imports snapml first.
- **Ran** `configs/elliptic_rolling.yaml` from a clean tree at `46f9771` (260 fits, 35 min): rolling refit lifts pooled F1 over t43–49 from 0.019 / 0.033 to 0.352 / 0.358 for `xgb.base` / `xgb.base_gfp` and from 0.015 / 0.020 to 0.101 / 0.094 for the SAGE configs (SAGE paired-by-seed intervals exclude zero), with no recovery until t47 and no difference between `base` and `base_gfp`. A first run was stamped `edf06ea-dirty`, because regenerated report files had dirtied the tree before it started, and was discarded and rerun. Methods §12.6, README, status.
- **Reran** the drift config from a clean tree at `7a19cb9` (3 min 26 s). With the fixed bins PSI's calibrated flag is 1.90 (was 1.52) and it never holds a flag (the t48 hold is gone); KS and score KS never hold, as before. The alert rate is the one detector whose largest score sits at t43 (1.34 against a flag of 0.38: XGBoost's alerts fall from 4.4% of units to 1.5%) but it does not hold, and its held flag at t38–39 is alerts rising with the illicit share, which the lead rule credits as +5. XGBoost event rows are bit-identical to `8c28027`; SAGE rows move by at most 0.02. README, methods §12.4–12.5 and the status notes rewritten accordingly; a domain-classifier detector was probed and rejected (ROC-AUC ≥ 0.99 for every timestep against the reference).

### 19 Sep 2026 — Why t43 breaks everything: the event diagnosis
- **Added** `drift.event` and `eval/event.py` (`8c28027`; PR-R5). With it set, `mulegraph drift` writes `report/tables/<experiment>_event.csv`: per side of the event batch, the fitted model's ROC-AUC, recall at the validation threshold and median illicit score, beside a 5-fold refit probe of XGBoost defaults inside that window alone. `configs/elliptic_drift.yaml` sets `event: 43`.
- **Found** on Elliptic: after t43 the models are confidently wrong (median illicit score 0.000 for XGBoost, ROC-AUC 0.96 → 0.56, recall 0.71 → 0.02), yet the post-t43 window alone is as learnable as the pre-t43 one (probe ROC-AUC 0.985 against 0.993). The shutdown changed what illicit looks like, for 2.5% of labelled units; whole-batch detectors cannot see that, and the score distribution looks calmer at t43, not stranger. Written up as methods §12.5.
- **Reran** the drift config (3 min 23 s, 2.8 GB RSS). XGBoost rows are bit-identical; SAGE still breaks at t39 in three seeds and t43 in two, with different seeds in each group (GPU nondeterminism). Methods §12.4 now cites this run.

### 19 Sep 2026 — Graphify audit: dead options removed, methods caught up with the results
- **Removed** config options that were accepted only to be refused or ignored: model `pna` (the row was dropped on 17 Sep), `features.backend` with its `igraph` value and the `MULEGRAPH_FEATURE_BACKEND` override, `search.enabled`, and `eval.seed_ci`. `feature_definition()` still hashes `"backend": "gfp"`, so every `feature_version` is unchanged. The `NotImplementedError` for `temporal_inductive` on cross-time graphs is now a `RegimeNotSupportedError` that says why; the D1 refusal on Elliptic is unchanged. Dead `load_definition()` removed; stale v1a/v1b/gate wording and unused `.env.example` entries removed.
- **Fixed** `docs/methods.md`: §12.2 and §13.1 SAGE numbers came from the `ec4501f` run, while the committed table and README come from `afe6ef5` (same model, feature and split code; the gap is GPU nondeterminism). §10.4 now names that run. Added §12.3 (AMLworld XGBoost) and §12.4 (drift monitor), which were missing from the write-up; §8.5 renumbered to §8.4. §8.3 now states the XGBoost-vs-SAGE gap as the paired interval PR-E3 requires, not as non-overlapping intervals. AMLworld `xgb.base_gfp` F1 is 0.539 (0.5395 had been rounded twice to 0.540), and the post-day-10 PR-AUC floor is 0.92, not 0.93.
- **Fixed** `docs/design.md` §3.4: the score detector flags on the KS statistic (> 0.1), not p < 0.01, and `calibrate` is described. Noted that the validation reference F1 is measured at the threshold chosen on that window, so it is optimistic.
- **Removed** code only the tests called (ponytail audit, net −182 lines): `embed()` and `save()` on every model and the protocol, `Predictions.embeddings`, `subsample()`, `Split.sizes`, `GfpDriver.last_time`, the never-passed `force`/`use_cache` loader and builder flags, and an unreachable dataset-name check. The SAGE nets fold `embed` into `forward` with identical parameters, so seeded fits are unchanged. `score_batches` / `lead_time` take their tuning values explicitly instead of repeating `DriftConfig`'s defaults.
- **Found** that the committed AMLworld results were run from a dirty tree (`7c22e3e-dirty`); a clean rerun is open in the status checklist.

### 19 Sep 2026 — Lead time needs persistent flags
- **Changed** `lead_time` so a detector's first flag must start a `drop_run`-long run of flags, the same persistence the F1 drop already needed (PR-R4). Before, one isolated flag counted: on Elliptic PSI flagged t39 and t41, went silent through the t42–t45 collapse, then flagged t46 and t48–49, and KS never flagged two batches in a row, so the old lead of 4 / 3 timesteps rested on single flags. Rerun result (3 min 34 s, 2.9 GB RSS): PSI first holds a flag at t48 for every model and seed (XGBoost lead −5; SAGE −9 against its t39–40 dip in three seeds, −5 against t43 in the other two; the dip count varied between runs, since GPU SAGE training is not bit-reproducible); KS and the score-shift detector never hold one. README drift section, figure and status rewritten around the negative result, with the SAGE dip-versus-collapse caveat.
- **Measured** `mulegraph drift --config configs/elliptic_drift.yaml`: 3 min 39 s wall, 3.0 GB peak RSS, about 330 MiB GPU on the RTX 4060. The 10 fits take 86 s; scoring and logging after each fit take about 12 s. The five XGBoost seeds give identical fits, and PSI / KS see only features, so both repeat work across seeds.

### 17 Sep 2026 — Pivot to a portfolio project
- **Reframed** the repository as a portfolio project. The dissertation scaffolding is gone: milestone dates, week-1 gates, the wall-clock search budget `W`, supervisor questions, the retraining-policy simulator and typology-shift injection are out of scope. `docs/project_spec.md` is cut down to `docs/design.md` (question, D1–D4, contracts, requirement register); `docs/methods_draft.md` becomes `docs/methods.md` with the literature comparison folded in from the viability review, which is removed. The `issue` / `close-issue` / `issues-sync` skills and the `simulate` / `report` CLI stubs are removed.
- **Dropped** the IBM Multi-GNN PNA reference row. Its published configuration needs more than the laptop's 8 GB of GPU memory (11.88 GiB allocated at OOM), and replicating IBM's exact settings answered nothing the two-by-two does not.
- **Added** per-timestep F1 / PR-AUC curves (`eval/curves.py`), a per-regime figure (`report/figures.py`), and a `predictions.parquet` artifact on every child run (`afe6ef5`; PR-E5).
- **Added** the label-free drift monitor (`drift/detectors.py`, `drift/monitor.py`) and `mulegraph drift`: PSI and KS on features, KS on model scores, per batch against the validation reference; lead time against the F1 curve; `configs/elliptic_drift.yaml` (`d322082`; PR-R1–R4). Detector signatures are tested for the absence of labels.
- **Added** AMLworld HI-Small as an edge task (`data/amlworld.py`; PR-D2): a unit is an edge, `batch_id` is the day, GFP rows are used per edge, splits and chronology run on `batch_id`, `configs/amlworld_hi_small.yaml` runs `xgb.base` and `xgb.base_gfp` on IBM's day 0–5 / 6–7 / 8–17 split (`a10c53c`). `dataset.max_days` truncates for a small host and is part of the cache key.
- **Changed** the drift monitor after the first real run: with PSI ≥ 0.2 / KS p < 0.01 every Elliptic test timestep was flagged from t38. `drift.calibrate: true` sets each detector's threshold to its largest leave-one-out score inside the validation window; `drift.drop_run` makes the F1 drop persist for two batches (`df98d4f`). Result on `xgb.base_gfp`: PSI flags t39, KS t40, collapse t43 (lead 4 / 3); the score-shift detector fires only at t49.
- **Added** `models/sage_edge.py`: GraphSAGE over accounts with a learned embedding and an edge head, trained and scored through `LinkNeighborLoader` with `time_attr` so a seed edge never samples a later edge (`5546a8d`; PR-M3, PR-F2). `configs/amlworld_hi_small.yaml` gains the two SAGE rows; `kaggle/amlworld_sage.md` runs the grid on a Kaggle GPU notebook (T4 x2; first written as P100, corrected 21 Sep 2026).
- **Ran** AMLworld HI-Small XGBoost on the full data (8 min, 5.0 GB RSS): `xgb.base` F1 0.209 / PR-AUC 0.109, `xgb.base_gfp` F1 0.540 / PR-AUC 0.521 (`report/tables/amlworld_xgb_*`).
- **Pinned** `matplotlib` as a direct dependency. `report/figures/` and the final tables are now tracked (smoke and dev outputs stay ignored).
- **Repository made public.** The 13 dissertation-only GitHub issues were closed as not planned and the four milestones deleted.

### 16 Sep 2026 — Methods draft (#8) and AMLworld week-1 gates (#1)
- **Added** `docs/methods_draft.md`: first methods-chapter draft covering every #8 done-criterion, with definitions taken from the code at `ec4501f`, an Elliptic results preview led by the per-window numbers, and related work framing Elliptic as replication + decomposition of Maganti 2026. Šafář 2026 is marked abstract-only. Remaining `TODO`s: histogram-bin test, per-window numbers regenerated by #12, gate 3/5 numbers, two citations.
- **Gate 4 cleared:** HI-Small spans 2022-09-01 to 2022-09-18 (17.68 days nominal). Ordinary traffic stops after 10 Sep, and the remaining 1,108 transactions are 59% laundering. Recorded in *Verified method notes*; D4 branch choice raised on #9.
- **Gates 3 and 5 blocked on compute:** with IBM's Multi-GNN settings (`252b025`), PNA runs out of memory on the 8 GB RTX 4060 (11.88 GiB allocated) and SAGE runs only by spilling into shared memory. Batch size and neighbour counts were not reduced. The timings need a 16–24 GB GPU.
- **Changed** spec to 0.8: AMLworld SAGE is timed as a `SAGEConv` swap inside Multi-GNN's GIN class, because Multi-GNN ships no SAGE.

### 16 Sep 2026 — MVP merged to main and tagged `mvp`
- **Merged** PR #28 (`feature/mvp-elliptic`) into `main` as merge commit `dd85dec`. **Tagged** `mvp` on the merge of the closeout PR #29 (first placed on `dd85dec`, moved so the tagged tree carries the export). CI green on main (145 passed, 1 skipped, 83% coverage, smoke OK); #13 closed.
- **Added** `report/exports/mvp_elliptic_mvp_runs.csv`: the `elliptic_mvp` MLflow experiment, 102 runs across parents `341e0f93…` (`ec4501f`, reported) and `281caef1…` (`a64a16e`, pre-fix SAGE record). Ships with the closeout PR #29, whose merge commit carries the `mvp` tag.
- **Closed** PR #26 unmerged; `CLAUDE.md` *Building (ponytail)* already sets the policy.

### 15 Sep 2026 — Viability review after the first Elliptic run
- **Added** `docs/viability_review_2026-09.md`: the MVP numbers against Weber 2019 and Maganti 2026 (same picture: trees beat GNNs, everything collapses at t43), the 2023–2026 literature and its gaps, threats (Elliptic pre-empted by Maganti; Šafář 2026 leakage claim unread; synthetic AMLworld; schedule) and the recommendation: continue, reposition Elliptic as replication + decomposition, make the AMLworld inductive two-by-two the v1b floor in #9, protect v2. Linked from `CLAUDE.md` and #9.

### 15 Sep 2026 — MVP run on real Elliptic++ (#7, #1 gate 2)
- **Ran** `uv run mulegraph run --config configs/elliptic_mvp.yaml` at `ec4501f`: 50 fits (2 regimes × 5 configs × 5 seeds) in under 9 minutes on an RTX 4060 Laptop GPU. MLflow parent `341e0f93709f487187f8ca276ff60d6a` with 50 fully tagged children; `report/tables/elliptic_mvp_results.csv` (50 rows). The MVP definition of done is met.
- **Week-1 gate 2:** `sage.base_gfp` temporal seed 0 took 23.1 s; slowest fit 44.3 s; W = 120 min on Elliptic. Gates 3–5 (AMLworld) remain open.
- XGBoost rows are bit-identical to the pre-fix run at `a64a16e`; its five seeds are identical fits (library defaults leave `random_state` nothing to randomise), which #27 handles for paired gaps.
- Temporal test performance collapses after t43 (dark-market shutdown): F1 0.83–0.89 on t38–42, 0.02–0.03 on t43–49 for the XGBoost configs. Recorded in *Verified method notes*.

### 15 Sep 2026 — SAGE inputs z-scored with train-row statistics (PR-M3, PR-E1)
- **Changed** `SAGEModel.fit` computes per-column mean and std on `split.train` rows only and applies them to every node before message passing; columns constant on train keep std 1 (zero after centring, never NaN). The cached device graph is rebuilt at each fit so statistics from another split are never reused.
- Why: the first real Elliptic++ run (`a64a16e`, MLflow parent `281caef1ade34e42aecd51bf2ab1e8a8`) fed SAGE raw inputs — base columns up to |x| = 265, GFP counts up to 472 with 86% zeros. Unscaled inputs handicap the GNN and bias the benchmark toward the expected finding. Adopted on principle, before measuring its effect.
- **The SAGE rows of the `a64a16e` run are superseded**; its XGBoost rows are unaffected (trees are scale-invariant). The run stays in MLflow as the pre-fix record.
- **Added** two tests: statistics ignore poisoned val/test rows; an all-zero column yields finite predictions.
- Recorded split hash correction: the temporal split hash is `f576c23d95a59084` from `d0fbcc3` onward; the `b065845303bd6a35` quoted in #4 was the same partition hashed without `raw_sha256` (arrays verified bit-identical).

### 15 Sep 2026 — CI runs the smoke check and keeps its wheel cache (`a569285`)
- **Added** a `Smoke` step (`uv run mulegraph smoke`) after the tests in `.github/workflows/ci.yml`. CI previously never exercised the pipeline end to end, contrary to `CLAUDE.md`.
- **Changed** `prune-cache: false` on setup-uv. The default pruning stripped PyPI wheels before saving, leaving 0.1 MB caches and a full ~3 GB torch/CUDA download on every run (sync 1 min to 11½ min depending on PyPI). The two empty caches were deleted so the full 3.37 GB cache could be saved under the unchanged `uv.lock` key.

### 15 Sep 2026 — Pipeline wired end to end (#7; PR-E4, PR-O1, D2)
- **Added** `pipeline.py`: `run_benchmark` orchestrates load → causal features (built once per dataset + feature config) → per-model feature selection → per-regime split → fit each (regime, model, seed) → threshold on validation → score test → MLflow child run → results table. Splits are built before the first fit so an undefined regime (D1) fails immediately instead of after hours of fitting.
- **Added** `run_smoke`: runs `configs/smoke.yaml` with a throwaway graph cache and MLflow store, leaving the results table in the usual `report/tables/`. Ten fits complete in 10 seconds on CPU, inside the two-minute CI budget.
- Each child run carries the eight tags spec §2.3 makes mandatory (`dataset, dataset_version, regime, model, features, feature_version, split_hash, git_commit`) and logs `trials_completed = 0` with `trial0_source` — the honest zero for a milestone with no search (D2). Per-run counts and point precision/recall go to `diag_*`, keeping `metrics.test_*` — and so the results table — to the metrics the config requested. `dataset_version` is a tag on the child, not a param on the parent: nested runs inherit nothing and the reporter only reads children.
- **Changed** `report/tables.py`: `write_results_table` takes an optional `parent_run_id` and filters on `tags.mlflow.parentRunId`, and the pipeline passes the run it just finished. Without it, re-running a config into the same experiment pooled the earlier children as extra seeds — five seeds run twice reported `n_seeds = 10`, shrinking the t-interval to roughly 40% of its honest width on no new evidence. PR-E3 makes that interval the sole test of significance in v1a, so the n reported must be the n the protocol ran. Found by the integrity auditor and covered by a regression test.
- **Changed** a dirty working tree now stamps its runs `<sha>-dirty` instead of only logging a warning: the MLflow record and the CSV `commit` column must not assert provenance the run does not have (NFR-1, S5).
- A dirty working tree now logs a warning naming the commit the run is not regenerable from (NFR-1), and a relative SQLite tracking path has its directory created rather than failing on a fresh clone.
- **Added** `tests/test_pipeline.py`: results-table shape and provenance, every child run's tags and honest trial count, and a spy asserting `choose_threshold` receives the validation set — exactly, on every fit — and never the test set (PR-E4).

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

---

## 9 Sep 2026 — MVP component modules

### Added
- **Elliptic++ loader and splits** (`d0fbcc3`; PR-D1, PR-D4, PR-E1, D1) — `data/elliptic.py` reads the three raw CSVs with pyarrow (only the 165 published features plus id and time, as float32), drops the 17 Elliptic++ extras into `meta.dropped_columns` (PR-M7), computes `cross_time_edges` and refuses cross-timestep edges, and asserts the published 2023.1 counts. `splits/builder.py` builds stratified-random and temporal splits of labelled nodes, runs the leakage assertions on every build, rejects `temporal_inductive` when `cross_time_edges` is False, and caches by definition hash as a determinism check.
- **Causal graph features** (`e6625d7`; PR-F1–F3, PR-M7) — `features/gfp.py` drives snapml forward-only, one timestep per batch, against a probed output layout; `features/aggregate.py` folds edge features onto nodes (`node_agg_v1`) under the `node_time >= t` guard; `features/builder.py` caches Parquet under a hash of the full definition; `features/select.py` resolves `base` / `base_gfp` / `raw165` against the loader's feature blocks.
- **Models** — GraphSAGE with neighbour sampling and full-batch inference (`dfd13ab`; PR-M3, PR-M5); XGBoost with train-only class weighting and early stopping on validation PR-AUC (`0ccdfbf`; PR-M1, PR-M2).
- **Evaluation and reporting** (`0ccdfbf`; PR-E2–E4, PR-E6, PR-O1) — validation-only threshold selection, metrics that refuse accuracy, the across-seed Student-t interval, and the MLflow-to-CSV results table with a provenance-mismatch warning.

### Changed
- MLflow tracking uses a local SQLite file (`sqlite:///mlruns/mlflow.db`) instead of the `file:` store, which MLflow 3 refuses (`7c9133e`; NFR-1).

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
