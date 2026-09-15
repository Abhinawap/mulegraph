# Project Specification

**Project:** Drift-aware graph detection of money-mule and laundering networks under realistic temporal evaluation
**Author:** Bambang Abhinawa Pinakasakti (Abhin)

---

## Definition

### What we want to build

A reproducible benchmark and monitoring toolkit that answers one question: *when evaluated the way a bank would have to deploy it, do graph neural networks beat a well-featured gradient-boosted model at detecting laundering and money-mule networks, and can we detect when either model stops working?*

### How we want to build it

A single Python package (`mulegraph`) with three subsystems:

1. **Benchmark harness.** Loads public transaction-graph datasets, computes graph features, trains a two-by-two of feature set against model family (plus one reference GNN), and evaluates them under evaluation regimes matched to each dataset's graph structure, with repeated seeds and confidence intervals.
2. **Drift monitor.** Label-free drift detectors that run on incoming batches and a simulator that compares retraining policies.
3. **Reporting.** MLflow experiment tracking, result tables and plots generated from tracked runs, and a dissertation-ready results export.

Everything runs from configuration files and a command-line entry point. Every result in the dissertation must be regenerable from a tagged commit.

---

## Part 1: Product Requirements

### 1.1 Product purpose

**Who is the product for?**

The primary user is a data scientist on a bank or payment firm's fraud or financial-crime team who has a transaction graph, a gradient-boosted model already in production, and pressure from vendors and papers to "add a GNN". They need to know whether that is worth doing before spending a quarter on it, and they need to know when their existing model has quietly stopped working because fraud patterns shifted.

The secondary users are the project author (who needs a defensible dissertation and a placement portfolio piece), the supervisor, and the examiners, who need to see a scoped question, rigorous method, honest limitations, and reproducible numbers.

**What problems does it solve?**

1. **Nobody can tell whether graph models earn their cost.** Published fraud-GNN results are mostly evaluated on random splits, which let the model see the future. When IBM evaluated its own GNNs against XGBoost with graph features on AMLworld, the tree model won. A practitioner cannot tell which result applies to their situation because the evaluation protocols do not match deployment.
2. **Fraud shifts and the model does not notice.** Authorised push payment fraud in the UK rose to £576m in 2025 as criminals moved from technical attacks to manipulating victims; Elliptic's Bitcoin data shows a dark-market shutdown that collapses model F1 overnight. Deloitte finds most institutions have no ongoing monitoring for their ML models. Labels arrive weeks late, so a team cannot see the drop until it has already cost money.
3. **Nobody agrees when to retrain.** Retraining on a schedule wastes compute and analyst review time when nothing has changed, and misses the moment when everything has. There is no public comparison of retraining policies on fraud graphs.

**What does the product do?**

The product is a command-line toolkit and benchmark, not a web app.

- The user points the toolkit at a transaction dataset (Elliptic++ Bitcoin transactions, or IBM AMLworld synthetic bank transfers) and it builds a time-ordered graph, keeping every node and edge stamped with when it happened.
- The user picks an evaluation regime: random split, temporal split (train on the past, test on the future), or temporal + inductive (test on accounts the model has never seen, with their graph features built only from edges that existed at that moment). The toolkit builds the split, checks it for leakage, and records a hash of it. On Elliptic's transaction graph, where timesteps are disconnected components, the temporal split is already inductive, so the toolkit refuses to run a separate "inductive" regime there and says why; the three-way contrast is run on AMLworld, where accounts persist across time.
- The toolkit computes graph features for every account or transaction (how many counterparties send money in and out, whether money fans in and then out again within hours, whether it moves in short cycles) using only edges up to the current timestamp, so the features never peek forward.
- The user trains the model lineup with one command: XGBoost and GraphSAGE, each on base features and on base + graph features (a two-by-two, so the effect of the features can be separated from the effect of the model), plus PNA or GIN+EU on base features as IBM's published reference GNN. On Elliptic, "base" is the 93 local transaction features only, because the published 165-feature block already includes 72 one-hop neighbour aggregates; that full block is run as an extra XGBoost reference row so the reader can see what the published features already capture. Hyperparameters are searched once per model, regime, and dataset under one wall-clock budget that is identical for every model; trial zero of every search is a published or library-default configuration so a slow model still produces a valid result, and the best configuration is then rerun across five seeds.
- The toolkit evaluates every model five times with different seeds, reports fraud-class F1, PR-AUC, and precision at fixed recall with confidence intervals, and plots performance timestep by timestep so the user can see exactly where a model breaks.
- After training, the user streams later timesteps through the drift monitor. It compares each batch of new transactions to the training period using PSI and KS on the features, the shape of the model's confidence scores, and (for GNNs) the embedding space, and raises a flag when the distribution has moved. It does this with no labels.
- The user then runs the retraining simulator, which replays the test period under a label lag: at timestep *t* the model may use features up to *t* but labels only up to *t − L*, because fraud confirmations arrive weeks late. It compares five policies (never retrain, retrain every k steps, retrain when the drift monitor alarms, retrain on a sliding window, and an oracle that retrains when lagged-label F1 drops) and reports the fraud F1 achieved and the number of retrains each policy cost.
- Every run is logged to MLflow with its git commit, dataset and feature versions, split hash, and seed. One command turns the logged runs into the tables and figures that go into the dissertation, so any number in the write-up can be regenerated.

### 1.2 Q1: What are we really trying to do? What are the goals?

We are trying to find out whether graph neural networks beat a well-featured gradient-boosted model at detecting laundering and money-mule networks *when evaluated the way a bank would have to deploy them*, and whether we can detect, without labels, the moment either model stops working.

Goals:

- **G1.** Measure, with confidence intervals, how much of the detection gain comes from graph *features* (base vs base + GFP, where base excludes any pre-aggregated neighbour features) and how much from graph *models* (XGBoost vs GraphSAGE), plus where IBM's reference PNA/GIN+EU lands, under leakage-free evaluation.
- **G2.** Show whether the ranking changes between a random (transductive) split and a temporal split on Elliptic, and between random, temporal, and temporal + inductive splits on AMLworld, where the three are genuinely distinct.
- **G3.** Detect concept drift from unlabelled batches and show whether detection leads the measured performance decay on Elliptic's one natural drift event. On AMLworld, test the hypothesis (H1) that model-side detectors (confidence, embedding) lead injected typology-shift events while feature detectors on the full transaction distribution do not, because laundering edges are a fraction of a percent of the data.
- **G4.** Compare retraining policies on detection quality against retraining cost under a realistic label lag, including an oracle that sees lagged labels as the ceiling.

The expected finding, based on published benchmarks, is that graph *features* matter more than graph *models*. Confirming or overturning that under strict evaluation is a result either way.

### 1.3 Q2: What are the milestones of functionality?

| Version | Core functionality |
|---|---|
| **MVP** (by 31 Oct 2026) | **Week-1 gates:** GFP installs (or fallback chosen); one GraphSAGE fit on full Elliptic is timed with our loader; one GraphSAGE fit and one PNA fit on AMLworld HI-Small are timed **using IBM's Multi-GNN repository and its own AMLworld preprocessing**, not our loader (which is v1b work). Multi-GNN ships no GraphSAGE, so the SAGE fit swaps PyG `SAGEConv` into its GIN model class and keeps IBM's loader, sampler and training loop; the HI-Small time span is read off during that run; W and the v1a/v1b grid arithmetic are set from those numbers.<br><br>**Single dataset, two regimes.** Elliptic++ transaction graph loads into a time-stamped graph from one command and caches to disk, with a metadata flag recording that it has no cross-timestep edges.<br><br>**Core function:** User runs `mulegraph run --config elliptic_mvp.yaml` and gets a results table for the two-by-two (XGBoost and GraphSAGE, each on the 93 local features and on local + GFP graph features), plus an `xgb.raw165` reference row on the published 165-feature block, on both the random and the temporal split, scored on fraud F1, PR-AUC, and precision at recall 0.5/0.8.<br><br>**Causal graph features:** Fan-in/fan-out, degree, scatter-gather, and short-cycle features are computed with IBM's GFP (or the igraph fallback) using only edges up to each timestep. The causality unit test runs on a synthetic multi-timestep graph fixture (not on Elliptic, where it would be vacuous) and proves features at timestep *t* do not change when future edges are added.<br><br>**Leak-checked split:** Train ends at t34, test starts at t38; the toolkit asserts no id overlap and logs the split hash.<br><br>**Threshold on validation:** The decision threshold is picked on the validation PR curve, never on test.<br><br>**Every run logged:** Parameters, metrics, seed, feature version, and git commit go to MLflow. |
| **v1a** (by 21 Nov 2026) | **Elliptic complete.** The two-by-two runs under both regimes with search decoupled from seeds: search once per (config, regime) under one wall-clock cap shared by every model, with trial 0 a fixed reference config, then the best config rerun over five seeds. Cap, wall-clock used, trial ceiling, and trials completed are logged.<br><br>**Seed-level CIs:** Every table reports the across-seed mean with a 95% t-interval (n = 5), and every gap of interest (local vs local + GFP; XGBoost vs SAGE; raw165 vs local + GFP) as a paired-by-seed difference with its own interval, so "is the gap significant" has one defined answer.<br><br>**Per-timestep curves:** F1 and PR-AUC at each test timestep for every config, so the dark-market collapse at ~t43 is visible.<br><br>**CI green:** Lint, unit tests, and a two-minute smoke test on a 2,000-node subsample run on every push. |
| **v1b** (by 12 Dec 2026; **scope to be negotiated with supervisor**) | **Second dataset, all three regimes:** AMLworld HI-Small loads as an edge-labelled transaction graph with persistent accounts; its real time span is recorded in `meta` and the batch unit (6 hours by default) and the separate drift split are derived from it. Random, temporal, and temporal + inductive splits are selectable; inductive test accounts are unseen during training and their features are built without training-period leakage. The split builder raises an error if `temporal_inductive` is requested on a dataset with no cross-timestep edges.<br><br>**Two-by-two on AMLworld:** XGBoost and SAGE, raw and raw + GFP, under all three regimes, same search and seed protocol as v1a.<br><br>**Reference GNN, one regime:** PNA or GIN+EU via IBM's Multi-GNN code, raw features, temporal + inductive regime only, same wall-clock cap as every other model (trial 0 = IBM's published config, ceiling 10), 3 seeds, behind the same interface. Other regimes for PNA go to Later. Elliptic node-task adaptation goes to Later.<br><br>**Freeze:** Benchmark code frozen before Christmas whether or not v1b is complete; anything unfinished is cut, not carried. |
| **v2** (5 Jan to 13 Feb 2027, hard stop) | **Anything cut from v1b** is not resumed here unless the supervisor agrees it outranks the drift work.<br><br>**Typology-shift drift events (hypothesis H1, not a success gate):** On AMLworld, the user selects a laundering typology; its edges are **removed** from the data before an injection batch T and present with their true labels from T onward (never relabelled, so training never sees wrong labels). Repeated across typologies for several events. Feature detectors are scored on all transactions and on the model's top-1% scored subset, since a fraction-of-a-percent class shift is invisible in the full distribution. Every detector's lead time on every event is reported against H1.<br><br>**Label-free drift monitor:** User streams test timesteps as batches; PSI and KS on features, confidence-distribution shift, and embedding MMD (GNNs) each output a drift score and flag per batch, with no access to labels.<br><br>**Lead-time report:** For each detector, the toolkit reports how many timesteps before the measured F1 drop the first flag was raised.<br><br>**Retraining simulator with label lag:** User sets a label lag L (default 3 timesteps on Elliptic, roughly six weeks; 4 six-hour batches, one day, on AMLworld) and replays the test period under no-retrain, fixed-cadence, drift-triggered, sliding-window, and lagged-label-oracle policies; each reports cumulative F1 and retrain count. Drift-triggered is judged against the oracle, not against a policy that could see labels instantly.<br><br>**One-command reporting:** `mulegraph report --milestone v2` regenerates every dissertation table and figure from MLflow. README documents reproduction of every number. |
| **Later** | **PNA on the remaining AMLworld regimes** and on Elliptic (node-task adaptation).<br><br>**Elliptic++ actor graph:** Wallet-address graph where entities persist across timesteps, as a third view with a real inductive regime.<br><br>**Continuous-time GNN:** TGN or similar as an extra challenger.<br><br>**Third domain:** Ethereum phishing addresses for additional cross-dataset evidence.<br><br>**Explanations:** GNNExplainer / SHAP on the graph models, showing which neighbours drove a flag.<br><br>**Cost-sensitive thresholds:** Investigator-hours model to set the operating point.<br><br>**SynthAML:** Real-data-calibrated synthetic set for the drift study. |
| **Not in scope for now** | Real bank data, NDAs, or ethics approvals. Transfer learning with feature alignment across datasets. A heterogeneous transaction-plus-actor graph. GAT or any architecture published after project start. Dashboard, web UI, or cloud deployment. Payer-side or scam-message detection. LLM or generative components. SMOTE or oversampling on graphs. Random-split-only results or accuracy as a headline metric. Any claim that synthetic results generalise to real transactions. |

### 1.4 Non-functional requirements

- **NFR-1 Reproducibility.** Any table in the dissertation can be regenerated from a tagged commit with one command.
- **NFR-2 Testability.** Unit tests for loaders, feature causality, split integrity, and detector outputs. CI runs on every push.
- **NFR-3 Compute.** Hyperparameter search is decoupled from the seed loop and budgeted by wall-clock (D2). The v1a (Elliptic) grid must fit in two overnight runs on one GPU; the v1b (AMLworld) grid must fit in four overnight runs, with PNA limited to one regime. Both are sized from week-one timings. **HI-Medium rule (single source):** AMLworld uses HI-Small; HI-Medium is considered only if the complete HI-Small v1b grid finishes in under half its four-overnight budget, and then only for the temporal + inductive regime.
- **NFR-4 Honesty.** Synthetic-data limitations are stated in the dissertation. No claim of real-world generalisation.
- **NFR-5 Scope discipline.** Nothing in "Later" or "Not in scope" is started before the v2 hard stop unless v2 is complete. v1b scope is agreed with the supervisor in writing before v1a starts; whatever is unfinished at the Christmas freeze is cut.

### 1.5 Success criteria

- **S1.** Two-by-two results table (feature set × model) with seed-level CIs and paired gaps under random and temporal splits on Elliptic (v1a), and under the regimes agreed for v1b on AMLworld, plus the PNA/GIN+EU reference row on temporal + inductive (v1b).
- **S2.** A stated answer to "how much comes from graph features and how much from graph models under leakage-free evaluation", supported by the paired gaps in S1 (v1a for Elliptic, v1b for AMLworld).
- **S3.** At least one label-free detector with positive lead time on Elliptic's natural drift event (v2), reported as a single observation. The AMLworld injected events are **not** a pass/fail criterion: H1 is stated in advance, every detector's lead time on every event is reported, feature detectors are additionally scored on the model's top-1% scored subset, and the outcome is discussed whichever way it falls.
- **S4.** Retraining-policy comparison under label lag showing the F1 vs retrain-count trade-off, with drift-triggered compared against the lagged-label oracle (v2).
- **S5.** Every number in the dissertation reproducible from a tagged commit.

### 1.6 Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| GFP (`snapml`) does not install or is unmaintained | Medium | Week-1 install test; `igraph` fallback with identical feature definitions |
| Multi-GNN reference code is hard to adapt | Medium | Time-box to two weeks; fall back to GraphSAGE only |
| GNN training too slow on available hardware | Medium | Neighbour sampling; small/medium dataset variants; university cluster |
| Scope creep into v2 before v1 is done | High | v2 cannot start before benchmark freeze; hard stop 13 Feb |
| Inductive regime harder than expected on AMLworld | Medium | Define the protocol and leakage tests against a synthetic fixture in MVP; run for real in v1 |
| Elliptic grid does not fit after week-1 timing | Medium | Lower W (keeping it equal), then reduce GNN seeds to 3 |
| AMLworld grid does not fit after week-1 timing | High | In order: (1) drop the random regime on AMLworld (leakage inflation is already shown on Elliptic); (2) GNN seeds to 3; (3) PNA runs trial 0 only (no search), 3 seeds; (4) PNA moves to Later entirely |
| v1b scope too large for 12 Dec | High | Negotiate with supervisor in the first meeting: v1b minimum is the AMLworld two-by-two under temporal + inductive only; three regimes and PNA are the negotiable extras |
| Christmas writing slips | Medium | Methods chapter drafted at MVP, not Christmas |

### 1.7 Assumptions

- Elliptic++ and AMLworld remain publicly downloadable under their current licences.
- Access to a GPU (personal or university) for at least the v1 grid.
- Supervisor agrees to the v1/v2 cut-offs in the first meeting.

---

## Part 2: Technical Design

### 2.0 Design decisions (resolved before code)

| ID | Decision | Consequence |
|---|---|---|
| **D1 Graph unit** | Elliptic++ is modelled as the **transaction graph** (node = transaction, 49 disconnected timestep components). The actor graph is deferred to Later. | On Elliptic only two regimes are distinct: random (transductive, leaky) and temporal (inductive by construction). The three-regime contrast lives on AMLworld, whose accounts persist across time. `GraphDataset.meta.cross_time_edges = False` on Elliptic and the split builder rejects `temporal_inductive` there. The causality test runs on a synthetic fixture. The dark-market collapse and per-timestep curves stay on the transaction side, which is where they are documented. |
| **D2 Search budget** | Optuna search runs **once per (model config, regime, dataset)** at a fixed `search_seed`; the best config is rerun across seeds. **The budget is wall-clock only:** one cap W per dataset, identical for every model config. Trial count is a per-model *ceiling*, not a budget. **Trial 0 is always a fixed reference config** (XGBoost library defaults with `scale_pos_weight`; a 2-layer 64-unit SAGE; IBM's published PNA/GIN+EU hyperparameters) so every model has a valid result even if nothing else completes in W. GNN trials use a median pruner. Logged per search: `wallclock_cap_min`, `wallclock_used_min`, `trial_ceiling`, `trials_completed`, `trial0_source`, `best_trial`. | Fairness claim: equal wall-clock, not equal trials. If PNA completes only trial 0 within W, that is reported as such and the dissertation says its search was nominal. W is set in week one as max(120 min, 2 × slowest measured single fit on that dataset) so every model completes at least trial 0 and one searched trial. Grid sizes are in section 2.5 (compute plan). |
| **D3 Feature × model** | The lineup is a **two-by-two**: {base, base + GFP} × {XGBoost, GraphSAGE}, plus PNA/GIN+EU on base features as IBM's reference configuration. **What "base" means differs by dataset and is stated:** on **Elliptic**, the published 165-feature block is 93 *local* features plus 72 *one-hop aggregated neighbour* features, so `base = local` (93) and the full 165-feature block is a separate `raw165` reference row for `xgb` only; on **AMLworld**, `base = raw` transaction fields (amount, currency, payment format, timestamp-derived), which contain no neighbour aggregates. | The feature gap (base vs base + GFP) therefore measures added graph information cleanly on both datasets. On Elliptic, `sage.local` vs `xgb.local` is the cleanest model-effect comparison (SAGE's message passing is doing the one-hop aggregation the 72 features hard-code), and `xgb.raw165` shows how much of GFP's gain the published aggregates already capture. Six configs on Elliptic (PNA optional); five on AMLworld with PNA mandatory. |
| **D4 Label lag and batch unit** | The policy simulator carries a `label_lag` parameter L; at batch *t* training may use features ≤ *t* and labels ≤ *t − L*. A **lagged-label oracle** policy is the ceiling comparator. Drift events: one natural (Elliptic t43) plus injected typology shifts on AMLworld. **Batch unit:** one Elliptic timestep (~2 weeks) on Elliptic. On AMLworld the span is checked in week one; HI-Small is expected to cover roughly ten days, so the **benchmark split and the drift split differ**: the benchmark uses chronological 70/15/15 by day; the drift study uses a short training period (first ~3 days), one validation day, and the remaining ~6 days as test at **6-hour batches** (~24 test batches). Feature detectors on AMLworld compare each batch to reference batches of the **same hour-of-day block** so the day–night volume cycle is not read as drift. Lead time and L are always expressed in batches of the dataset's unit. | S3 on Elliptic is a single observation and is reported as such. Elliptic default L = 3 batches (~6 weeks, a realistic confirmation lag); sweep {0, 3, 6}. AMLworld default L = 4 batches (one day); sweep {0, 4, 8}. Fixed-cadence k on AMLworld ∈ {4, 8} batches. Because AMLworld spans days, its lag is a sensitivity parameter, not a calibrated one, and the dissertation says so. If the week-one span check shows a longer dataset, daily batches and a single split are used instead. |

### 2.1 Tech stack

This is a command-line research toolkit, not a web application. Several conventional layers are deliberately absent; they are listed so the omission is a decision, not an oversight.

| Layer | Choice | Reason |
|---|---|---|
| Programming language | Python 3.11 | Whole ML stack lives here; matches existing repos |
| Frontend | None | Users are data scientists running the CLI; a UI is out of scope |
| Backend / application layer | Python package `mulegraph` with a Typer CLI | One entry point, config-driven, importable for notebooks |
| Configuration | YAML files under `configs/`, validated with Pydantic | Every experiment is a file, so every result is reproducible from its config |
| Graph and tabular ML | PyTorch 2.x, PyTorch Geometric 2.x, XGBoost 2.x, scikit-learn | GraphSAGE and PNA/GIN+EU need PyG; XGBoost is the primary tabular model |
| Reference GNN code | IBM Multi-GNN (PNA, GIN+EU) vendored under `third_party/` | Published AMLworld baselines; avoids reimplementing edge-update layers |
| Graph features | IBM `snapml` GraphFeaturePreprocessor; fallback `igraph` implementation | GFP is the published feature set; fallback covers install failure |
| Hyperparameter search | Optuna, run once per (model config, regime, dataset) under one wall-clock cap shared by every model; trial 0 is a fixed reference config; best config rerun across seeds | Wall-clock is the only budget that can be made equal across a tree model and a GNN (D2); trials completed is logged, not promised (PR-M6) |
| Drift detection | `evidently` (PSI, KS) plus a small MMD implementation in PyTorch | Reuse tested statistics; only the batching and embedding-drift glue is custom |
| Experiment tracking | MLflow, local SQLite file (`sqlite:///mlruns/mlflow.db`) | Already in use across existing repos; queryable for report generation. MLflow 3 refuses its own `file:` store ("in maintenance mode"), and numbers must still regenerate from a tagged commit in 2027 (NFR-1, S5); SQLite is one file with no server, so §2.3's "no database server" still holds |
| Data storage | Parquet (pyarrow) for tabular caches, `torch.save` for PyG `Data`, NumPy `.npy` for split indices | No database server; everything is files keyed by content hash |
| Testing | pytest, pytest-cov | Unit and smoke tests (NFR-2) |
| Lint and format | ruff | Fast, single tool |
| CI | GitHub Actions | Runs lint, tests, smoke test on push |
| Packaging | `pyproject.toml` with pinned dependencies and a `uv` lockfile | Reproducible environment |
| Cloud / VMs | None by default. Personal GPU laptop or desktop for MVP; University of Birmingham HPC (BlueBEAR, if access granted) or a capped cloud GPU budget for the v1 grid and AMLworld | Compute is the main external dependency; see open question 2 |
| AI models | XGBoost (base; base + GFP; raw165 reference on Elliptic), GraphSAGE (base; base + GFP), PNA or GIN+EU (base). No LLMs, no foundation models | Model lineup fixed by Part 1; anything else is out of scope |

### 2.2 Technical architecture

#### System design overview

```
                +---------------------+
   configs/*.yaml --> |   CLI (Typer)       |  mulegraph run / report / drift / simulate
                +----------+----------+
                           |
                           v
        +------------------+------------------+
        |           Pipeline orchestrator      |  builds the run from config, seeds everything,
        +--+--------+--------+--------+-------+  opens an MLflow run, calls components in order
           |        |        |        |
           v        v        v        v
      [Data]   [Features] [Splits]  [Models]        core benchmark path
           \        |        |        /
            \       v        v       /
             +---> [Evaluator] <----+
                       |
                       +------> [Drift monitor] ---> [Policy simulator]     v2 path
                       |
                       v
                 [MLflow store] ----> [Reporter] ----> report/tables, report/figures
```

The orchestrator is the only component that knows about all the others. Every other component depends only on the shared data types in `mulegraph/types.py` (`GraphDataset`, `Split`, `FeatureMatrix`, `Predictions`, `DriftSignal`).

#### Key components

| Component | Module | Responsibility | Inputs | Outputs |
|---|---|---|---|---|
| Loaders | `mulegraph/data/` | Download or read raw files, build a `GraphDataset` with timestamps preserved, cache it | Dataset name, version | `GraphDataset` |
| Feature builder | `mulegraph/features/` | Compute causal graph features via GFP or igraph fallback; version them | `GraphDataset`, window config | `FeatureMatrix` (with `feature_version` hash) |
| Split builder | `mulegraph/splits/` | Build random / temporal / temporal+inductive index sets; run leakage checks | `GraphDataset`, regime config | `Split` (train/val/test ids + `split_hash`) |
| Models | `mulegraph/models/` | Train and predict behind one interface; expose a search-space for the orchestrator's single search phase | `GraphDataset`, `FeatureMatrix`, `Split`, seed | `Predictions`, embeddings (GNNs), fitted model |
| Evaluator | `mulegraph/eval/` | Threshold on validation, compute metrics, seed-level t-intervals and paired gaps, bootstrap bands for per-timestep curves | `Predictions`, `Split` | Metric records, figures |
| Drift monitor | `mulegraph/drift/` | Run detectors per batch with no labels; emit scores and flags | Reference window, batches, model (for confidence/embeddings) | `DriftSignal` per batch |
| Policy simulator | `mulegraph/policies/` | Replay test period under retraining policies; call model `fit` when a policy triggers | Model class, `Split`, `DriftSignal`s, policy config | Cumulative metrics, retrain count, cost |
| Reporter | `mulegraph/report/` | Query MLflow, aggregate over seeds, write CSV tables and PNG figures | Milestone tag | Files under `report/` |
| Orchestrator | `mulegraph/pipeline.py` | Wire components, seed RNGs, manage MLflow run lifecycle | Validated config | MLflow run |

#### How components interact (benchmark run)

1. CLI parses `mulegraph run --config configs/elliptic_mvp.yaml`; Pydantic validates the config.
2. Orchestrator sets global seeds (Python, NumPy, PyTorch, CUDA), starts an MLflow run, logs the config, git commit, and dataset version.
3. Loader returns a cached `GraphDataset` or builds and caches it.
4. Feature builder checks the cache for `(dataset_version, feature_version)`; on miss it computes causal features and caches them.
5. Split builder builds indices for the requested regime, runs the leakage assertions, logs `split_hash`, saves indices to `.npy`.
6. Search phase (once per model config, regime, dataset): trial 0 evaluates the fixed reference config; Optuna then runs further trials at `search_seed` until the shared wall-clock cap W or the per-model trial ceiling is hit, scoring on validation PR-AUC. The best config, trials completed, and wall-clock used are written to the run.
7. Seed phase: for each seed in `seeds`, the best config is refitted, `predict_proba` is called on validation and test, and the threshold is chosen on validation.
8. Evaluator computes metrics and CIs, writes per-timestep curves, logs everything as MLflow metrics and artifacts under a child run per seed.
9. Orchestrator ends the parent run. Reporter can be invoked later to aggregate.

#### How components interact (drift and policy run, v2)

1. `mulegraph drift --config configs/elliptic_drift.yaml` loads a fitted model artifact from a previous MLflow run by run id.
2. Drift monitor takes the training timesteps as the reference window, then iterates test timesteps as batches, calling each detector and, for confidence and embedding detectors, the model's `predict_proba` and `embed`.
3. `DriftSignal`s are logged per batch. Lead time is computed against the evaluator's per-timestep F1 from the original run.
4. `mulegraph simulate --config configs/elliptic_policies.yaml` replays the same timesteps under each policy; when a policy triggers, it calls `fit` on the data available up to that timestep and continues. Results are logged as child runs under one parent.

### 2.3 Data storage and schemas

There is no database server. Storage is files, keyed by content hashes, plus the MLflow file store.

**Raw and cached data**

```
data/
  raw/<dataset>/<version>/...                 original downloads, never modified
  cache/<dataset>/<version>/graph.pt          GraphDataset (PyG Data + metadata)
  cache/<dataset>/<version>/features/<feature_version>.parquet
  cache/<dataset>/<version>/splits/<split_hash>/{train,val,test}.npy
```

**`GraphDataset` (in-memory, saved with `torch.save`)**

| Field | Type | Notes |
|---|---|---|
| `x` | float tensor [N, F] | Elliptic++: 165 published node features with `meta.feature_blocks = {local: 0..92, agg1hop: 93..164}` so `base` can select the local block; None on AMLworld (edge task) |
| `edge_index` | long tensor [2, E] | directed |
| `edge_attr` | float tensor [E, K] | amount, currency id, payment format, etc. (AMLworld) |
| `edge_time` / `node_time` | long tensor | timestep or Unix timestamp; always present |
| `batch_id` | long tensor | derived batch index: timestep on Elliptic, floor(timestamp / batch_hours) on AMLworld; used by splits, drift, and simulator |
| `y` | long tensor | labels on nodes or edges; -1 for unknown |
| `task` | str | `"node"` or `"edge"` |
| `meta` | dict | dataset name, version, `cross_time_edges: bool`, typology labels (AMLworld), source URL |

**Feature matrix (parquet)**

| Column | Type | Notes |
|---|---|---|
| `id` | int | node or edge id |
| `time` | int | timestep the features are valid at |
| `fan_in`, `fan_out`, `deg_in`, `deg_out` | int | counted within the window |
| `scatter_gather`, `gather_scatter` | int | pattern counts within the window |
| `cycles_le_k` | int | simple cycles up to length k |
| `amt_in_sum`, `amt_out_sum`, `amt_in_mean`, `amt_out_mean` | float | AMLworld only |
| ... | | full list generated from the feature config and stored in `feature_version.json` |

**MLflow run schema (on every run; `git_commit` through `features` are tags — the reporter reads only tags and refuses a run missing one)**

| Key | Example |
|---|---|
| `git_commit` | `a1b2c3d` |
| `dataset`, `dataset_version` | `elliptic_pp`, `2023.1` |
| `feature_version` | sha256 prefix |
| `regime`, `split_hash` | `temporal`, sha256 prefix |
| `model`, `features` | `xgb`, `base_gfp` |
| `seed` | `3` |
| `wallclock_cap_min`, `wallclock_used_min` | `120`, `117.4` |
| `trial_ceiling`, `trials_completed`, `best_trial` | `40`, `31`, `17` |
| `trial0_source` | `xgboost_defaults` / `sage_default_2x64` / `ibm_multignn_published` |
| `threshold` | `0.31` |
| metrics | `test_f1`, `test_pr_auc`, `test_roc_auc`, `test_p_at_r50`, `test_p_at_r80`, `val_*`, `f1_t38` ... `f1_t49` |
| artifacts | per-timestep CSV, PR curve PNG, fitted model, config snapshot |

**Results export (CSV written by the Reporter)**

Per config: `dataset, regime, model, features, metric, seed_mean, ci_low, ci_high, ci_kind=seed_t, n_seeds, feature_version, split_hash, commit`
Per gap: `dataset, regime, comparison, metric, paired_diff_mean, ci_low, ci_high, n_seeds, significant`

**Drift signal (parquet)**

`run_id, detector, batch_id, batch_unit, score, flagged, threshold`

**Policy result (parquet)**

`run_id, policy, cumulative_f1, retrain_count, retrain_times, train_seconds`

### 2.4 Interfaces

The product has no HTTP API. Its interfaces are the CLI, the config schema, and three Python protocols.

**CLI**

| Command | Purpose |
|---|---|
| `mulegraph run --config <yaml>` | Full benchmark run (loader → features → split → models → eval) |
| `mulegraph drift --config <yaml>` | Run detectors on a fitted model from an MLflow run id |
| `mulegraph simulate --config <yaml>` | Retraining-policy replay |
| `mulegraph report --milestone <tag>` | Aggregate MLflow runs into `report/tables` and `report/figures` |
| `mulegraph smoke` | Two-minute end-to-end check on a 2,000-node subsample (used in CI) |

**Config schema (Pydantic; abbreviated)**

```yaml
dataset: {name: elliptic_pp, version: "2023.1"}
features:
  backend: gfp            # gfp | igraph
  window: 1               # timesteps on Elliptic++; per-family windows (AMLworld 6h scatter-gather) return in v1b
  cycle_len: 10
split:
  regime: temporal             # random | temporal | temporal_inductive (rejected if meta.cross_time_edges is False)
  train_end: 34
  val: [35, 37]
  test: [38, 49]
models:                          # two-by-two + reference rows
  - {name: xgb,  features: base}          # Elliptic: 93 local; AMLworld: raw txn fields
  - {name: xgb,  features: base_gfp}
  - {name: sage, features: base,     sampler: {fanout: [15, 10]}}
  - {name: sage, features: base_gfp, sampler: {fanout: [15, 10]}}
  - {name: xgb,  features: raw165}        # Elliptic only: published block incl. 72 one-hop aggregates
  - {name: pna,  features: base}          # optional on Elliptic, required on AMLworld
search:
  wallclock_cap_minutes: 120      # THE budget; identical for every model config on this dataset; set in week 1
  trial_ceiling: {xgb: 40, sage: 20, pna: 10}   # upper bound only, not a budget
  trial0: {xgb: xgboost_defaults, sage: sage_default_2x64, pna: ibm_multignn_published}
  pruner: median
  search_seed: 0                  # search once, then rerun best config over `seeds`
seeds: [0, 1, 2, 3, 4]
eval:
  metrics: [f1, pr_auc, roc_auc, p_at_r50, p_at_r80]
  seed_ci: t95                  # headline interval across seeds
  gap_pairs: [[xgb.base, xgb.base_gfp], [sage.base, sage.base_gfp], [xgb.base, sage.base], [xgb.base_gfp, sage.base_gfp], [xgb.raw165, xgb.base_gfp], [xgb.base_gfp, pna.base]]
  bootstrap_samples: 1000       # per-timestep bands only
  per_timestep: true
mlflow: {experiment: elliptic_v1}
```

**Python protocols**

```python
class BaseModel(Protocol):
    def fit(self, data: GraphDataset, feats: FeatureMatrix, split: Split, seed: int) -> None: ...
    def predict_proba(self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray: ...
    def embed(self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray | None: ...

class Detector(Protocol):
    def fit_reference(self, ref: Batch) -> None: ...
    def score(self, batch: Batch) -> DriftSignal: ...       # no labels available

class Policy(Protocol):
    def should_retrain(self, t: int, signals: list[DriftSignal], history: PolicyState) -> bool: ...
    def training_window(self, t: int, label_lag: int) -> tuple[int, int]: ...   # labels available only up to t - label_lag
```

### 2.5 Other technical details

#### Feature computation
- GFP configured with a fixed time window per dataset: one timestep on Elliptic++; one day for degree and fan features and six hours for scatter-gather on AMLworld; cycle length bound 10.
- Fallback implements the same definitions over `igraph` with an explicit `edge_time <= t` filter.
- Causality test: on a synthetic fixture with edges spanning timesteps, features at timestep *t* are asserted identical with and without edges from *t+1…T*. On Elliptic this property holds trivially (no cross-timestep edges) and the test is skipped with a logged reason.
- `feature_version` = sha256 of (feature list, window config, backend, dataset version).

#### Split construction
- Random: stratified 70/15/15 on labelled ids.
- Temporal (Elliptic++): train ≤ t34, val t35–t37, test t38–t49, covering the dark-market shutdown at ~t43. Because timestep components are disconnected, this split is inductive by construction and is the only leakage-free regime on Elliptic. AMLworld: chronological 70/15/15 by timestamp.
- AMLworld drift split (v2, separate config): train first ~3 days, val 1 day, test remaining days at 6-hour batches; used only by the drift monitor and policy simulator, never for benchmark tables. Confirmed against the real span in week one.
- Temporal + inductive (AMLworld only; rejected on datasets with `cross_time_edges = False`): as temporal, plus test ids are removed from the graph used during training; at test time message passing uses the graph up to *t* but never test labels; test features are computed only from edges at or before their own timestamp.
- Leakage assertions: empty train/test intersection; max train time < min test time; for inductive, no test id appears in any training neighbourhood.

#### Model specifics
- `xgb` (base / base_gfp / raw165 on Elliptic): `scale_pos_weight` from class ratio, early stopping on validation PR-AUC, Optuna over depth, learning rate, subsample, colsample, min_child_weight.
- `sage` (base / base_gfp): inputs z-scored with train-row statistics, two `SAGEConv` layers, `NeighborLoader` with fan-out [15, 10], class-weighted BCE, Optuna over hidden size, dropout, learning rate, layers ∈ {2, 3}. On Elliptic, sampling is within the timestep component by construction.
- `pna` / `gin_eu` (base only): Multi-GNN reference implementation, edge task on AMLworld. Elliptic node-head adaptation is optional and time-boxed to one week.
- Search: once per (model config, regime, dataset) at `search_seed`; best config rerun over `seeds`. Budget is the shared wall-clock cap W; trial ceilings 40 / 20 / 10 for xgb / sage / pna bound the search but are not the budget; trial 0 is the fixed reference config; median pruner for GNNs. Logged: cap, wall-clock used, ceiling, trials completed, trial-0 source, best trial.
- Threshold: maximise F1 on the validation PR curve; stored with the run.

#### Evaluation
- Positive-class F1, PR-AUC, ROC-AUC, precision at recall 0.5 and 0.8.
- Five seeds per (config, regime, dataset), same split for every seed (seed controls model init and sampling only).
- **Primary interval (all headline tables):** across-seed mean with a 95% t-interval, n = 5. This captures "would this ranking replicate if I retrained", which is what the tables are read as.
- **Gaps:** every comparison of interest (base vs base + GFP within a model; XGBoost vs SAGE within a feature set; raw165 vs base + GFP on Elliptic; PNA vs XGBoost + GFP) is reported as the paired-by-seed difference with its own 95% t-interval. A gap is called significant only if that interval excludes zero.
- **Secondary interval (per-timestep curves and single-seed figures only):** 1,000-sample bootstrap over test ids, shown as a band. Never pooled with the seed interval and never used to call a gap significant.
- Per-timestep F1 and PR-AUC on Elliptic++ and per-batch on AMLworld.

#### Drift detection
- Batch unit: one timestep on Elliptic; 6 hours on AMLworld (`batch_hours: 6`) under the drift split (train ~3 days, val 1 day, test ~6 days ≈ 24 batches). The drift split is a separate config from the benchmark split and both hashes are logged. Reference window: the training period. On AMLworld, feature detectors compare a batch only to reference batches in the same hour-of-day block (`align_by: hour_block`) to remove the diurnal volume cycle; confidence and embedding detectors are also run with and without alignment and both are reported.
- PSI per feature, mean and max aggregation, flag at 0.2 and alarm at 0.25.
- Two-sample KS per feature; flag if more than 20% of features have p < 0.01.
- Confidence shift: KS between reference and batch `predict_proba`.
- Embedding drift: RBF-kernel MMD between reference and batch embeddings, threshold set by permutation test on reference batches.
- Lead time = (first batch with F1 drop > 20% relative to the reference mean) − (first flagged batch), in the dataset's batch unit.
- Drift events: Elliptic t43 (one natural event, reported as a single observation). AMLworld typology-shift events: choose a typology (e.g. cycles); **delete** its edges from all batches before injection batch T; keep them, with true labels, from T onward. Never relabel. Repeat per typology. Because laundering edges are a fraction of a percent of AMLworld, PSI/KS on the full feature distribution are expected not to fire; they are therefore also computed on the model's top-1% scored transactions per batch (`score_subset: top_1pct`). Pre-registered hypothesis H1: confidence and embedding detectors show positive lead time on most injected events; full-distribution feature detectors do not. Reported, not gated.

#### Retraining-policy simulation
- Label lag: `label_lag` L in batches; at batch *t* the training window may include features ≤ *t* but labels only ≤ *t − L*. Elliptic default L = 3 (~6 weeks), sweep {0, 3, 6}; AMLworld default L = 4 batches (one day at 6-hour batches), sweep {0, 4, 8}, treated as a sensitivity parameter.
- Policies: none; fixed cadence k ∈ {3, 6} batches on Elliptic and k ∈ {4, 8} on AMLworld; drift-triggered on a chosen detector's alarm; sliding window of width w retrained every step; lagged-label oracle (retrain when F1 on the most recent labelled timestep drops by more than 20%), used as the ceiling.
- Outputs: cumulative F1 over the test period, retrain count, total training seconds.
- Deterministic given seed; each policy run is an MLflow child run.

#### Testing and CI
- Unit tests: loaders (shapes, dtypes, label counts), feature causality, split leakage, metric functions against hand-computed values, detector monotonicity on synthetically shifted data, policy trigger logic.
- Smoke test: `mulegraph smoke` completes in under two minutes on CPU.
- GitHub Actions on every push: ruff, pytest with coverage, smoke test. The full grid never runs in CI.

#### Compute plan
- MVP: personal machine, CPU acceptable for XGBoost, small GPU for GraphSAGE.
- Week 1: time one full-data GraphSAGE fit and one XGBoost + GFP fit on Elliptic; recompute the grids below from those numbers before confirming the v1a and v1b dates.
- Week 1 also times one SAGE fit and one PNA fit on AMLworld HI-Small using IBM's Multi-GNN repo and its own preprocessing (the paper's reported run times are the prior; expect hours for PNA). Multi-GNN has no SAGE model, so SAGE is timed as a `SAGEConv` swap inside its GIN class with everything else unchanged, and records the dataset's real time span. The MVP has exactly one loader (Elliptic).
- v1a grid (Elliptic): 5 configs (two-by-two + `xgb.raw` reference) × 2 regimes × (one W-capped search + 5 seeds); with W = 120 min the ceilings imply at most ~90 XGBoost and ~50 SAGE fits per feature set; target two overnight runs on one GPU.
- v1b grid (AMLworld HI-Small): 4 configs × 3 regimes × (one W-capped search + 5 seeds) for XGBoost/SAGE, plus PNA on temporal + inductive only (one W-capped search + 3 seeds); target four overnight runs. Fits per search are whatever W allows and are logged, not assumed. If unavailable locally, BlueBEAR or a capped cloud budget (target under £50).
- Fallback orders are in the risk table (section 1.6): Elliptic prunes trials then seeds; AMLworld drops the random regime, then GNN seeds, then PNA search (trial 0 only), then PNA.
- v2: AMLworld HI-Small; HI-Medium escalation is governed solely by the rule in NFR-3.

#### Reproducibility and provenance
- `pyproject.toml` with pinned versions and a committed `uv.lock`.
- Every MLflow run carries git commit, dataset version, feature version, split hash, and seed.
- Milestone tags `mvp`, `v1`, `v2`; a CSV export of MLflow at each tag is committed under `report/exports/`.

#### Licensing and data handling
- Elliptic++: research use per its published licence. AMLworld: Community Data License Agreement. SynthAML: check licence before use.
- No personal data is processed; all datasets are public and either pseudonymous (Bitcoin) or synthetic. No ethics approval required (confirm with supervisor).

### 2.6 Open questions (to resolve with supervisor)

1. Exact dissertation submission date, and whether an autumn inspection or demo exists.
2. Whether BlueBEAR (or equivalent GPU access) is available to final-year project students.
3. Whether AMLworld results should be reported per laundering typology or aggregate only.
4. Whether to include ROC-AUC at all given class imbalance (leaning: include, never as headline).
5. Whether the Multi-GNN node-task adaptation for Elliptic++ is worth the time, or PNA/GIN+EU should be AMLworld-only from the start (spec default: AMLworld-only, Elliptic optional).
6. Which AMLworld typologies to hold out for the injected drift events, and how many events are needed for the AMLworld hypothesis (H1) to be considered tested.
8. The real time span of AMLworld HI-Small (week-one check) and whether the 3-day / 1-day / rest drift split leaves enough laundering edges in training.
7. **v1b scope (first meeting).** Minimum: AMLworld two-by-two under temporal + inductive only. Extras, in priority order: the other two regimes; PNA on temporal + inductive. Agree which extras are in before v1a starts.

---

## Appendix A: Requirement register

Stable IDs for the functional requirements described in Part 1. Later questions and answers should reference these.

**Data**
- **PR-D1.** Load the Elliptic++ transaction graph (nodes, edges, features, timesteps, labels) into a graph object with timestep preserved and `meta.cross_time_edges = False` recorded.
- **PR-D2.** Load at least one AMLworld variant (HI-Small or HI-Medium) as an edge-labelled transaction graph.
- **PR-D3.** Optionally load SynthAML for the drift study (v2 stretch).
- **PR-D4.** All loaders are deterministic and cache to disk.

**Features**
- **PR-F1.** Compute graph features per node/edge using IBM's Graph Feature Preprocessor (GFP) or an equivalent `networkx`/`igraph` fallback: in/out degree, fan-in/fan-out counts, scatter-gather, simple cycles up to a bounded length, neighbourhood amount statistics.
- **PR-F2.** Graph features are computed causally: only edges at or before the current timestamp/window are used.
- **PR-F3.** The feature set is versioned and logged with every run.

**Models**
- **PR-M1.** XGBoost on base features (Elliptic: 93 local; AMLworld: raw transaction fields), plus an `xgb.raw165` reference row on Elliptic's published 165-feature block.
- **PR-M2.** XGBoost on base + graph features.
- **PR-M3.** GraphSAGE with neighbour sampling, on base features and on base + graph features (two configs).
- **PR-M4.** PNA or GIN with edge updates via the Multi-GNN reference code, base features only (IBM reference configuration); required on AMLworld, optional on Elliptic.
- **PR-M7.** "Base" never includes pre-aggregated neighbour features; the dataset loader exposes feature blocks so the selection is explicit and logged.
- **PR-M5.** All models expose the same `fit` / `predict_proba` interface.
- **PR-M6.** Hyperparameter search runs once per (model config, regime, dataset) under a single wall-clock cap that is identical for every model on that dataset; trial count is a per-model ceiling only; trial 0 is a fixed reference configuration; the best config is rerun across seeds; cap, wall-clock used, ceiling, trials completed, and trial-0 source are recorded.

**Evaluation**
- **PR-E1.** Regimes: (a) stratified random split, (b) temporal split (train on earlier timesteps, test on later), (c) temporal + inductive (test nodes/edges unseen during training and their features computed without training-period leakage). Regime (c) is only valid on datasets with cross-timestep edges; the split builder rejects it otherwise. On Elliptic, (b) is inductive by construction and (c) is not run.
- **PR-E2.** Metrics: fraud-class F1, PR-AUC, ROC-AUC, precision at recall = 0.5 and 0.8.
- **PR-E3.** Five seeds per config per regime on a fixed split; headline tables report the across-seed mean with a 95% t-interval, and each gap of interest as a paired-by-seed difference with its own interval. Bootstrap over test ids is used only for per-timestep bands.
- **PR-E4.** Threshold chosen on validation PR curve, never on test.
- **PR-E5.** Per-timestep metric curves for every model on Elliptic++, covering the dark-market shutdown.
- **PR-E6.** Accuracy is never reported as a headline metric.

**Drift monitoring**
- **PR-R1.** Detectors: Population Stability Index and Kolmogorov–Smirnov on input features; prediction-confidence distribution shift; embedding-space drift for GNN models (MMD or centroid distance).
- **PR-R2.** Detectors run on batches (one Elliptic timestep, or one `batch_hours` window on AMLworld) with no access to labels.
- **PR-R3.** Output per batch: drift score, threshold flag, and timestamp.
- **PR-R4.** Evaluation: lead time between first drift flag and first measured F1 drop beyond a set tolerance, on Elliptic's natural event (success criterion) and on injected typology-shift events in AMLworld (hypothesis H1, reported per detector per event).
- **PR-R5.** Typology-shift drift injection on AMLworld: delete a typology's edges from all batches before injection batch T; keep them with true labels from T onward; never relabel.
- **PR-R6.** Feature detectors (PSI, KS) are computed both on all transactions in a batch and on the model's top-1% scored subset; on AMLworld, reference comparison is aligned by hour-of-day block.

**Retraining policies**
- **PR-P1.** Policies: no retraining, fixed cadence (every k timesteps), drift-triggered, sliding window, lagged-label oracle.
- **PR-P4.** The simulator enforces a label lag L in batches: at batch *t*, labels are available only up to *t − L*.
- **PR-P2.** Each policy is scored on cumulative fraud F1 across the test period and on number of retrains.
- **PR-P3.** The simulator is deterministic given a seed.

**Reporting**
- **PR-O1.** Every run logs parameters, metrics, feature version, dataset version, git commit, and seed to MLflow.
- **PR-O2.** A single command regenerates all dissertation tables and figures from MLflow.
- **PR-O3.** README documents how to reproduce every reported number.


## Change log

- 0.1 (8 Sep 2026): initial draft from planning discussion.
- 0.2 (8 Sep 2026): Part 1 rewritten as product purpose + Q1/Q2 milestone table; functional requirement IDs moved to Appendix A.
- 0.3 (8 Sep 2026): Part 2 rewritten as tech stack, technical architecture (components, interactions, storage schemas, interfaces), and other technical details.
- 0.4 (8 Sep 2026): Design decisions D1–D4 resolved: Elliptic modelled as transaction graph with two regimes and AMLworld moved to v1 for the three-regime contrast; search decoupled from seed loop with equal trial + wall-clock budget; two-by-two feature × model lineup plus reference PNA; label lag and lagged-label oracle in the simulator; injected typology-shift drift events on AMLworld.
- 0.5 (8 Sep 2026): v1 split into v1a (Elliptic, 21 Nov) and v1b (AMLworld, 12 Dec, scope negotiable); PNA limited to one AMLworld regime with its own fallback ladder; AMLworld batch unit defined (one day) with label lag in batches; confidence intervals defined as seed-level t-intervals with paired-by-seed gaps, bootstrap restricted to per-timestep bands; stale Optuna-in-seed-loop text removed.
- 0.6 (8 Sep 2026): Budget redefined as wall-clock only with per-model trial ceilings and a fixed trial-0 reference config (D2, PR-M6, config, MLflow schema, fallback ladder). "Base" features defined per dataset; Elliptic base = 93 local features, published 165 block kept as `xgb.raw165` reference (D3, PR-M7). AMLworld: span check in week one, separate drift split (3d/1d/rest) at 6-hour batches, hour-of-day-aligned reference, lag in batches (D4). Typology injection defined as edge deletion before T; feature detectors also scored on top-1% subset; AMLworld half of S3 recast as pre-registered hypothesis H1. Week-one HI-Small timing uses IBM's pipeline; MLflow logs cap/used/ceiling/completed; single HI-Medium rule in NFR-3.
- 0.8 (16 Sep 2026): Week-1 gate 3 wording (§1.3 MVP row, §2.5): IBM's Multi-GNN ships only GIN, GAT, PNA and RGCN, so the AMLworld SAGE timing swaps PyG `SAGEConv` into its GIN class, keeping IBM's preprocessing, sampler and training loop. No behaviour change in `mulegraph`.
- 0.7 (10 Sep 2026): §2.4 config schema — `features.window` is a single int for the MVP; per-family windows (AMLworld's 6-hour scatter-gather) return with v1b. No behaviour change on Elliptic. §2.3 MLflow schema names the tags the reporter reads (`regime`, `model` + `features`) instead of `split_regime` / `xgb_gfp`.
