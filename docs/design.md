# Design

**Project:** `mulegraph`, drift-aware graph detection of money-mule and laundering networks under realistic temporal evaluation
**Author:** Bambang Abhinawa Pinakasakti (Abhin)

This file is the authority for *what the toolkit must do and why*. The requirement ids (`PR-*`) at the end are cited by code, tests and docstrings. The write-up of the method and results is [methods.md](methods.md); where things stand is [project_status.md](project_status.md).

---

## 1. The question

*When evaluated the way a bank would have to deploy it, do graph neural networks beat a well-featured gradient-boosted model at detecting laundering and money-mule networks, and can we detect, without labels, the moment either model stops working?*

The expected finding, based on published benchmarks, is that graph **features** matter more than graph **models**. Confirming or overturning that under strict evaluation is a result either way; nothing is tuned toward the expected answer.

### Who it is for

A data scientist on a bank or payment firm's financial-crime team who has a transaction graph, a gradient-boosted model in production, and pressure to "add a GNN". They need to know whether that is worth doing, and they need to know when the model they already have has quietly stopped working because fraud patterns shifted.

### What is wrong with the usual answer

1. **Nobody can tell whether graph models earn their cost.** Published fraud-GNN results are mostly evaluated on random splits, which let the model see the future. When IBM evaluated its own GNNs against XGBoost with graph features on AMLworld, the tree model won. The protocols do not match deployment, so a practitioner cannot tell which result applies.
2. **Fraud shifts and the model does not notice.** Elliptic's Bitcoin data contains a dark-market shutdown that collapses model F1 overnight. Labels arrive weeks late, so a team cannot see the drop until it has cost money.

### Goals

- **G1.** Measure how much of the detection gain comes from graph *features* (base vs base + GFP, where base excludes any pre-aggregated neighbour features) and how much from graph *models* (XGBoost vs GraphSAGE), under leakage-free evaluation, with seed-level confidence intervals.
- **G2.** Show whether the ranking changes between a random (transductive) split and a temporal split.
- **G3.** Detect concept drift from unlabelled batches and show whether detection leads the measured performance decay on Elliptic's one natural drift event.

### Deliberately out of scope

Real bank data. Retraining policies beyond one rolling refit with zero label lag (`temporal_rolling`, PR-E7). Injected typology-shift events. PNA / GIN+EU via IBM's Multi-GNN code (its published configuration needs more than 8 GB of GPU memory; see [project_status.md](project_status.md)). The `temporal_inductive` regime. Transfer learning across datasets. Dashboards, web UI, HTTP serving, cloud deployment (the deployed-scoring path is a CLI command writing files, D5). LLM components. Oversampling on graphs. Random-split-only results, or accuracy as a headline metric. Any claim that synthetic results generalise to real transactions.

---

## 2. Design decisions

| ID | Decision | Consequence |
|---|---|---|
| **D1 Graph unit** | Elliptic++ is modelled as the **transaction graph** (node = transaction, 49 disconnected timestep components). AMLworld is an **edge task**: node = account, edge = transaction, label on the edge. | On Elliptic only two regimes are distinct: random (transductive, leaky) and temporal (inductive by construction, because no edge crosses a timestep). `GraphDataset.meta.cross_time_edges = False` on Elliptic; `True` on AMLworld, where accounts persist across days. The causality test runs on a synthetic fixture because it is vacuous on Elliptic. |
| **D2 Reference configuration** | No hyperparameter search. Every model runs its **fixed reference configuration** ("trial 0"): XGBoost library defaults with `scale_pos_weight` and early stopping on validation PR-AUC; a 2-layer 64-unit GraphSAGE. Every run logs `trials_completed = 0` and `trial0_source`. | Results compare untuned reference models under one protocol. A tuned comparison would need an equal wall-clock budget per model, which was not run. |
| **D3 Feature × model** | The lineup is a **two-by-two**: {base, base + GFP} × {XGBoost, GraphSAGE}. **What "base" means differs by dataset and is stated:** on **Elliptic**, the published 165-feature block is 93 *local* features plus 72 *one-hop aggregated neighbour* features, so `base = local` (93) and the full block is a separate `xgb.raw165` reference row; on **AMLworld**, `base = raw` transaction fields (amount, currency, payment format, hour of day), which contain no neighbour aggregates. | The feature gap (base vs base + GFP) measures added graph information cleanly on both datasets. On Elliptic, `sage.base` vs `xgb.base` is the cleanest model-effect comparison (SAGE's message passing does the one-hop aggregation the 72 features hard-code), and `xgb.raw165` shows how much of GFP's gain the published aggregates already capture. |
| **D4 Batch unit** | One Elliptic timestep (~2 weeks) on Elliptic. One **day** on AMLworld (`batch_id = hour // 24`), matching IBM's own Multi-GNN split of days 0–5 / 6–7 / 8–17; GFP causality runs at hour granularity. | Per-timestep curves, split bounds and drift batches all use `batch_id`. On Elliptic `batch_id == node_time`. The drift reference window is the validation batches; every later batch is scored against it. Lead time is expressed in the dataset's batch unit. |
| **D5 Deployed scoring** | `mulegraph score` runs one fitted XGBoost model on one batch, as a bank would each day: a ranked alert queue with each alert's top TreeSHAP contributions, plus the drift detectors as a label-free health check. The model and its validation threshold are fitted once and cached under a key of dataset version, feature version, split hash, model, params, resolved device and XGBoost version, then reused; the fit's commit and definition go into every health file as `model_fit`. XGBoost only: a GNN scores through the whole graph and has no saved model to carry from one day to the next. | The benchmark's integrity rules hold on the product path: the threshold comes from validation (PR-E4), no label reaches scoring or the health check (PR-R2), and features for batch *d* see only edges at or before *d* (PR-F2), because the cached causal features are the "as of *d*" rows. Exit status 3 when a detector flags, so a scheduler can act on it; the health reference defaults to the validation window and can be widened. A clean check is not an all-clear, and every health file says so (methods §12.5). Dashboards, HTTP and cloud stay out of scope. |

---

## 3. Technical design

### 3.1 Stack

Command-line research toolkit, not a web application.

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | Whole ML stack lives here |
| CLI / config | Typer; YAML under `configs/` validated with Pydantic (`extra="forbid"`) | Every experiment is a file, so every result is reproducible from its config |
| Graph and tabular ML | PyTorch 2.x, PyTorch Geometric 2.x, XGBoost 3.2, scikit-learn | GraphSAGE needs PyG; XGBoost is the tabular model; 3.2 is the newest XGBoost supporting 3.11 |
| Graph features | IBM `snapml` GraphFeaturePreprocessor | The published feature set; installs from a wheel |
| Drift detection | `scipy.stats` (KS) and a 30-line PSI | Reuse tested statistics; only the batching is custom |
| Experiment tracking | MLflow, local SQLite file | MLflow 3 refuses its own `file:` store; SQLite is one file, no server |
| Storage | Parquet, `torch.save`, `.npy`, all keyed by content hash | No database server |
| Figures | matplotlib (`Agg`) | Files under `report/figures/` |
| Testing / lint / CI | pytest, ruff, GitHub Actions | Lint, tests, smoke on every push; the full grid never runs in CI |
| Packaging | `pyproject.toml` pinned, `uv.lock` committed | Reproducible environment |

Deliberately absent: frontend, HTTP API, database server, cloud services, LLMs.

### 3.2 Architecture

```
   configs/*.yaml --> CLI (Typer): run / drift / smoke
                         |
                         v
                 pipeline.py  (orchestrator: the only module that imports across subsystems)
                  |     |      |       |
                data  features splits  models         benchmark path
                  \     |      |       /
                   +--> eval  <-------+               threshold on val, metrics, per-timestep curves
                         |
                         +--> drift                   label-free detectors, lead time
                         |
                         v
                 MLflow store --> report/  (CSV tables, PNG figures)
```

Every component depends only on the shared types in `mulegraph/types.py` (`GraphDataset`, `FeatureMatrix`, `Split`, `Predictions`). Only `pipeline.py` imports across subsystems.

| Component | Module | Responsibility | Inputs | Outputs |
|---|---|---|---|---|
| Loaders | `data/` | Read raw files, build a `GraphDataset` with timestamps preserved, cache it | Dataset name, version | `GraphDataset` |
| Feature builder | `features/` | Causal graph features via GFP, driven forward in time; version them | `GraphDataset`, feature config | `FeatureMatrix` with `feature_version` |
| Split builder | `splits/` | Random / temporal index sets on `batch_id`; leakage checks | `GraphDataset`, regime config | `Split` with `split_hash` |
| Models | `models/` | Train and score behind one `fit / predict_proba` protocol | `GraphDataset`, `FeatureMatrix`, `Split`, seed | scores |
| Evaluator | `eval/` | Threshold on validation, metrics, seed t-intervals, per-timestep curves | `Predictions` | metric records, curves |
| Drift monitor | `drift/` | PSI / KS / confidence shift per batch with no labels; lead time | features, scores, `batch_id` | scores, flags, lead-time table |
| Reporter | `report/` | Aggregate MLflow child runs over seeds; figures | parent run id | files under `report/` |

### 3.3 Benchmark run (`mulegraph run`)

1. Pydantic validates the config. The orchestrator seeds Python, NumPy, PyTorch, CUDA; opens an MLflow parent run; logs the config, git commit (stamped `-dirty` on an uncommitted tree) and dataset version.
2. Loader returns a cached `GraphDataset` or builds and caches it.
3. Feature builder checks the cache for `(dataset_version, feature_version)`; on a miss it computes causal features one batch at a time, forward in time.
4. Split builder builds indices for each regime, runs the leakage assertions, logs `split_hash`. All splits are built before the first fit so an undefined regime fails in seconds. A `temporal_rolling` regime is a list of temporal splits, one per test batch, the whole window shifted forward one batch at a time (PR-E7); every step passes the same assertions.
5. For each (regime, model, seed): fit on train with early stopping on validation; choose the threshold on the validation PR curve; score test once; compute metrics and per-timestep curves; log a child run with tags `dataset, dataset_version, regime, model, features, feature_version, split_hash, git_commit` and a `predictions.parquet` artifact. Under `temporal_rolling` this repeats per step, each step thresholding on its own validation window, and the step test sets are stitched into one child run per seed so the table's `n_seeds` counts seeds, not refits; `split_hash` is a hash over the step hashes.
6. Reporter aggregates the children of *this* parent run into `report/tables/<experiment>_results.csv`, `<experiment>_curves.csv` and `report/figures/<experiment>_curves.png`.

### 3.4 Drift run (`mulegraph drift`)

1. One temporal regime only (validated). Same load, features and split as above.
2. Each model × seed is fitted exactly as in the benchmark. The model then scores **every unit** whose `batch_id` falls in the validation or test window, labelled or not.
3. Detectors compare each post-reference batch to the validation batches: PSI per feature column (score = max; flag > 0.2), two-sample KS per column (score = fraction with p < 0.01; flag > 0.2), the KS statistic on the model's own scores (flag > 0.1), and the alert rate: the absolute log ratio of the share of units at or above the validation threshold against the reference share (flag > 0.5). With `calibrate`, each fixed flag is replaced by the detector's largest leave-one-out score inside the reference window. They receive feature values, scores and the validation threshold only.
4. Lead time = (first test batch starting a `drop_run`-long stretch where F1 is more than 20% below the validation mean) − (first batch starting a `drop_run`-long run of flags), per detector. The drop and the flag need the same persistence, so a one-batch flag is not a warning. Labels enter only here, in the evaluation of the detector. The validation mean is F1 at the threshold chosen on that same window, so it is optimistic and the drop level sits correspondingly high; this can make an F1 drop register earlier than it would against an out-of-sample reference.
5. With `drift.event` set, the labelled test units are split at that batch. For each side the run records prevalence, the fitted model's ROC-AUC, recall at the validation threshold and median illicit score, plus a probe: 5-fold stratified CV ROC-AUC of XGBoost library defaults refit inside that window alone. A probe near 1 where transfer fails means the event changed what illicit looks like rather than making it unlearnable. The probe's folds are random within the window, so it measures separability, never deployment performance.
6. Outputs: `report/tables/<experiment>_scores.csv`, `<experiment>_lead_time.csv`, `<experiment>_event.csv` (with `drift.event`), `report/figures/<experiment>_drift.png`, all logged as artifacts.

### 3.4a Scoring run (`mulegraph score`, D5)

1. One temporal regime, one XGBoost model, one seed, one `batch` inside the test window (validated, with the reason in the error).
2. Load, causal features and split as in the benchmark. The model and threshold are read from `<cache>/models/<key>.ubj` and `.json`, or fitted with the benchmark's own fit-and-threshold step and written there. The threshold file is written last, so a crash between the two files forces a refit.
3. The model scores every unit in the validation window and in `batch`, labelled or not. The detectors of §3.4 step 3 compare `batch` against the reference: the validation window unless `reference` names other batches, any range wholly before `batch`. The reference is a health-check setting only, so changing it never refits the model or moves the threshold (PR-E4). A reference reaching into training batches is logged, because the model scored those in-sample.
4. Outputs, no MLflow run: `report/tables/<name>_batch<d>_alerts.csv` (rank, unit, score, account or node ids, top contributing features, only rows at or above the threshold) and `<name>_batch<d>_health.json` (status, each detector's score and flag level, counts, threshold, commit, dataset and feature versions, split hash). Exit 0 = no flag, 3 = a detector flagged, 1 = error.

### 3.5 Storage and schemas

```
data/
  raw/<dataset>/<version>/...                 original downloads, never modified, never committed
  cache/<dataset>/<version>/graph.pt          GraphDataset
  cache/<dataset>/<version>/features/<feature_version>.parquet
  cache/<dataset>/<version>/splits/<definition_hash>/{train,val,test}.npy
```

**`GraphDataset`**

| Field | Notes |
|---|---|
| `x` float32 [N, F] | Elliptic++: 165 published node features with `meta.feature_blocks = {local: 0..92, agg1hop: 93..164}`; empty `[N, 0]` on AMLworld |
| `edge_index` int64 [2, E] | directed |
| `edge_attr` float32 [E, K] or None | AMLworld: amount paid/received, currency codes, payment format, hour of day |
| `node_time`, `edge_time` int64 | timestep on Elliptic; hour index on AMLworld |
| `batch_id` int64 [units] | timestep on Elliptic; day on AMLworld. Splits, curves and drift all use it |
| `y` int64 [units] | 1 illicit, 0 licit, −1 unknown (never enters a loss or a metric) |
| `task` | `"node"` or `"edge"`; a unit is a node or an edge accordingly (`num_units`, `unit_time`, `unit_features`) |
| `meta` | dataset, version, `cross_time_edges`, feature blocks and names, dropped columns, label counts, `raw_sha256` |

**Results CSV** (per config): `dataset, regime, model, features, metric, seed_mean, ci_low, ci_high, ci_kind=seed_t, n_seeds, feature_version, split_hash, commit`.

**Curves CSV**: `time, f1, pr_auc, n, n_pos, regime, model, features, seed`.

**Drift scores CSV**: `detector, batch_id, score, flagged, threshold, model, features, seed`. **Lead-time CSV**: `detector, first_flag, first_drop, lead, drop_level, ref_f1, model, features, seed`.

### 3.6 Interfaces

| Command | Purpose |
|---|---|
| `mulegraph run --config <yaml>` | Full benchmark run |
| `mulegraph drift --config <yaml>` | Fit, then run label-free detectors and report lead time |
| `mulegraph score --config <yaml>` | Score one batch with the deployed model: alert queue plus label-free health check (D5) |
| `mulegraph smoke` | Fifteen-second end-to-end check on a synthetic graph: a benchmark, then one scored batch with the settings of `configs/amlworld_score.yaml` (used in CI) |

```python
class BaseModel(Protocol):
    def fit(self, data: GraphDataset, feats: FeatureMatrix, split: Split, seed: int) -> FitInfo: ...
    def predict_proba(self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray: ...
```

Detectors are plain functions over `(reference, current)` arrays: `psi`, `ks_frac`, `conf_shift`. None has a label parameter, and a test asserts it.

### 3.7 Method details

**Features.** GFP with a fixed window per dataset: one timestep on Elliptic++, 24 hours on AMLworld; cycle length bound 10. GFP is stateful and `transform` inserts a batch before scoring it, so the only drive pattern is `transform(batch_t)` alone, for *t* ascending. On a node task the per-edge output is folded onto nodes (`node_agg_v1`); on an edge task the per-edge rows are the features. `feature_version` = sha256 of the full definition (backend, families, bins, window, cycle bound, aggregation, drive pattern, unit, dataset version, raw-file hash).

**Splits.** Random: stratified 70/15/15 on labelled units. Temporal: inclusive `train_end`, `val`, `test` bounds on `batch_id` (Elliptic: ≤ t34 / t35–37 / t38–49, spanning the dark-market shutdown at ~t43; AMLworld: days 0–5 / 6–7 / 8–17). Assertions on every build: pairwise disjoint; no empty part; no unlabelled unit; at least one illicit unit per part; chronological order; on a node task with cross-time edges, no edge joins train to test. The split seed is separate from the model seed, so seed intervals measure retraining variance only.

**Models.** `xgb`: `scale_pos_weight` from the training split, early stopping on validation PR-AUC, `n_estimators = 1000` ceiling. `sage`: inputs z-scored with train-row statistics, two `SAGEConv` layers, `NeighborLoader` fan-out [15, 10], class-weighted BCE, full-batch inference; early stopping on validation PR-AUC.

**Threshold.** Maximise F1 on the validation PR curve; ties go to the higher threshold. The pipeline scores test once with it; a test spies on `choose_threshold` to prove it only ever sees validation.

**Metrics.** Illicit-class F1 at the validation threshold, PR-AUC (average precision), ROC-AUC (secondary), precision at recall 0.5 and 0.8 (curve metrics that never set a threshold). Accuracy is refused at the source.

**Statistics.** Five seeds per config on one fixed split; headline interval is the across-seed mean ± *t*<sub>0.975, 4</sub>·*s*/√5. A gap is significant only if its paired-by-seed interval excludes zero. XGBoost trial 0 is deterministic, so XGBoost-vs-XGBoost pairs are degenerate and no such gap is called significant. Bootstrap bands, if ever drawn, are per-timestep only.

**Reproducibility.** Every child run carries git commit, dataset version, feature version, split hash and seed; the results table is scoped to one parent run so a rerun never pools old seeds; a dirty tree is stamped into the `commit` column; caches are keyed by content hash and a cached split is compared against a fresh build on every load.

---

## Appendix A: Requirement register

Stable ids cited by code, tests and docstrings.

**Data**
- **PR-D1.** Load the Elliptic++ transaction graph (nodes, edges, features, timesteps, labels) into a graph object with timestep preserved and `meta.cross_time_edges = False` recorded.
- **PR-D2.** Load AMLworld HI-Small as an edge-labelled transaction graph with persistent accounts.
- **PR-D4.** All loaders are deterministic and cache to disk.

**Features**
- **PR-F1.** Compute graph features per node/edge using IBM's Graph Feature Preprocessor: in/out degree, fan-in/fan-out counts, scatter-gather, simple cycles up to a bounded length, vertex statistics.
- **PR-F2.** Graph features are computed causally: only edges at or before the current timestamp/window are used.
- **PR-F3.** The feature set is versioned and logged with every run.

**Models**
- **PR-M1.** XGBoost on base features (Elliptic: 93 local; AMLworld: raw transaction fields), plus an `xgb.raw165` reference row on Elliptic's published 165-feature block.
- **PR-M2.** XGBoost on base + graph features.
- **PR-M3.** GraphSAGE with neighbour sampling, on base features and on base + graph features.
- **PR-M5.** All models expose the same `fit` / `predict_proba` interface.
- **PR-M7.** "Base" never includes pre-aggregated neighbour features; the loader exposes feature blocks so the selection is explicit and logged.

**Evaluation**
- **PR-E1.** Regimes: (a) stratified random split, (b) temporal split (train on earlier batches, test on later). Leakage assertions run on every build.
- **PR-E2.** Metrics: fraud-class F1, PR-AUC, ROC-AUC, precision at recall = 0.5 and 0.8.
- **PR-E3.** Five seeds per config per regime on a fixed split; tables report the across-seed mean with a 95% t-interval; a gap is significant only as a paired-by-seed difference whose interval excludes zero. Bootstrap over test ids is for per-timestep bands only.
- **PR-E4.** Threshold chosen on the validation PR curve, never on test.
- **PR-E5.** Per-timestep metric curves for every model, covering the dark-market shutdown on Elliptic.
- **PR-E6.** Accuracy is never reported as a headline metric.
- **PR-E7.** Rolling temporal regime: the temporal window slides forward one batch per test batch; each step fits on its own training window, thresholds on its own validation window (PR-E4) and scores one test batch; the stitched test set is one run per seed. Zero label lag beyond the validation window; it measures what post-shift labels buy, not a retraining policy.

**Drift monitoring**
- **PR-R1.** Detectors: Population Stability Index and Kolmogorov–Smirnov on input features; prediction-confidence distribution shift; alert rate at the validation threshold.
- **PR-R2.** Detectors run on batches with no access to labels.
- **PR-R3.** Output per batch: drift score, threshold flag, and batch id.
- **PR-R4.** Evaluation: lead time between the first drift flag and the first measured F1 drop beyond a set tolerance, on Elliptic's natural event, reported per detector.
- **PR-R5.** Diagnosis of a known event: per side of the event batch, the fitted model's transfer (ROC-AUC, recall, median illicit score) beside a within-window refit probe. Labels are used; this evaluates the event, it is not a detector.

**Reporting**
- **PR-O1.** Every run logs parameters, metrics, feature version, dataset version, git commit, and seed to MLflow.
- **PR-O3.** README documents how to reproduce every reported number.

**Non-functional**
- **NFR-1 Reproducibility.** Any reported number can be regenerated from a tagged commit with one command.
- **NFR-2 Testability.** Unit tests for loaders, feature causality, split integrity, metrics against hand-computed values, detector monotonicity. CI runs on every push.
- **NFR-4 Honesty.** Synthetic-data limitations are stated. No claim of real-world generalisation.
