# Methods

The method and results as the code implements them. Elliptic numbers are from the run at `afe6ef5`, the committed results table (§10.4); definitions cite the requirement ids in [design.md](design.md). The non-obvious engineering facts behind these definitions are in [project_status.md](project_status.md) → *Verified method notes*.

---

## 1. Data

We use the Elliptic++ release 2023.1 transaction graph (Elmougy and Liu, 2023; https://github.com/git-disl/EllipticPlusPlus). The loader (`mulegraph/data/elliptic.py`) reads three CSV files and checks them against the published counts before returning a graph. A mismatch in any count stops the run and names what it observed.

| Property | Value |
|---|---|
| Nodes (transactions) | 203,769 |
| Directed edges | 234,355 |
| Timesteps | 49 |
| Illicit / licit / unknown | 4,545 / 42,019 / 157,205 |
| Raw-file hash (`raw_sha256`, sha256 prefix over the three files) | `4dd761d5af9cd4f6` |

Unknown-label nodes stay in the graph. They take part in graph feature computation and in GraphSAGE message passing, but never enter a split, a loss or a metric: the split builder, the models and the evaluator each reject a label of −1.

Each edge takes its source node's timestep as `edge_time`. That definition is causal only because no edge joins two timesteps, and the loader enforces this: it counts edges whose endpoints carry different timesteps and refuses the files if it finds any (D1, PR-F2). The loader caches the finished graph under `data/cache/elliptic_pp/2023.1/graph.pt` (PR-D4).

## 2. Graph unit and evaluation regimes

We model Elliptic++ as a transaction graph: a node is a transaction, an edge is a flow of bitcoin from one transaction's outputs to another's inputs (D1). The 49 timesteps form 49 disconnected components. The Elliptic++ actor (wallet) graph, where entities do persist across time, is out of scope.

This graph structure fixes how many evaluation regimes Elliptic can support. Four regimes are defined (PR-E1, PR-E7):

- **random**: a stratified partition of labelled nodes, ignoring time. Training and test nodes share timestep components, so a GNN aggregates test nodes' features (never their labels) during training. The regime is transductive and leaks the future; we run it to measure what a random split inflates.
- **temporal**: train on earlier timesteps, validate and test on later ones. The model is fitted once and never refreshed.
- **temporal_rolling**: the temporal window slides forward one batch per test batch. For test batch *t* the model is refit on labels up to *t* − 4 and thresholded on *t* − 3 … *t* − 1, so each step is an ordinary temporal fit and the twelve test batches are scored by twelve models. It is the deployment-realistic counterpart of `temporal`: what a model that is always as fresh as its labels allow can do, with zero label lag beyond the validation window (§5).
- **temporal_inductive**: as temporal, with test nodes removed from the training graph and their features built only from edges that existed at their own time.

On Elliptic the temporal split is already inductive. No edge crosses a timestep, so no test node can appear in any training node's neighbourhood and no training-period edge can reach a test node's features. A separate inductive regime would produce the same partition under a different name. The toolkit therefore treats `temporal_inductive` as undefined on Elliptic. `build_split` raises `RegimeNotSupportedError` whenever `meta.cross_time_edges` is False, with the message:

> temporal_inductive is rejected on elliptic_pp: meta.cross_time_edges is False, so its temporal split is already inductive by construction and a separate inductive regime is undefined (D1).

The pipeline builds every split before the first model fit, so a config that requests the regime fails in seconds instead of after hours of fitting. The `temporal_inductive` regime is not implemented on AMLworld either; both datasets are evaluated under random and temporal splits only.

## 3. Feature sets: `base`, `base_gfp` and `raw165`

The published Elliptic feature block has 165 columns: 93 *local* features describing the transaction itself, followed by 72 *aggregated* features that summarise its one-hop neighbours (Weber et al., 2019). The 72 aggregates already contain graph information. If `base` included them, the comparison between `base` and `base` plus graph features would measure the second helping of graph information, and GraphSAGE would aggregate neighbours that had been aggregated once already.

We therefore define three feature sets (D3, PR-M7):

| Selection | Columns | Width | Used by |
|---|---|---|---|
| `base` | `Local_feature_1` … `Local_feature_93` | 93 | XGBoost, GraphSAGE |
| `base_gfp` | `base` plus the causal GFP node features of §4 | 93 + 36 = 129 | XGBoost, GraphSAGE |
| `raw165` | the published 165-column block, local plus aggregated | 165 | XGBoost only |

`raw165` is a reference row. It shows how much of any graph-feature gain the published aggregates already capture, and it makes our tree results comparable with published Elliptic baselines. The config schema refuses `raw165` for any model other than XGBoost.

The loader records the block boundaries in `meta.feature_blocks` (`local` = columns 0–92, `agg1hop` = 93–164). `select_features` is the single place that resolves a selection name into columns, and it refuses to build `base` unless the local block ends exactly where the aggregated block begins. Models read only the selected matrix, never the raw `x`.

**Elliptic++ extras.** The 2023.1 features file carries 17 columns beyond the published 165: `in_txs_degree`, `out_txs_degree`, `total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`, and minimum, maximum, mean, median and total BTC for inputs and for outputs. The loader drops all 17 and records them in `meta.dropped_columns`. Two of them, `in_txs_degree` and `out_txs_degree`, count a transaction's neighbours in the transaction graph; keeping them would put neighbourhood structure into `base` and break PR-M7. We drop the remaining 15 so that `x` equals the published block exactly: `raw165` then matches the feature set of Weber et al. (2019) and Maganti (2026), and `base` stays a strict subset of it. These 15 are local attributes of the transaction, so their exclusion narrows `base`; §11 lists it as a limitation.

The loader reads only the 167 columns it keeps (identifier, timestep and the 165 features), as float32.

## 4. Causal graph features (GFP)

### 4.1 Backend and input

Graph features come from IBM's Graph Feature Preprocessor, `snapml.GraphFeaturePreprocessor`, version 1.17.2 (Blanuša et al., 2024). It installs from a wheel on Python 3.11, so the `igraph` fallback once planned was never built (PR-F1).

GFP works on edges. The builder (`mulegraph/features/builder.py`) passes it one row per edge, `[edge_id, source, target, timestamp, dummy]`. Elliptic has no transaction amounts, and GFP's vertex statistics need a numeric column to point at, so the fifth column is a constant 1.0.

Edge ids are the dataset's global edge index. GFP overwrites an edge whose id it has seen before, so ids that restarted in each batch would erase earlier timesteps from its graph state.

### 4.2 Drive pattern: transform only, forward in time

GFP holds a graph in memory, and `transform(batch)` inserts the batch into that graph before it scores the batch's edges. Causality therefore depends on how the caller drives it, not on anything inside GFP (PR-F2). We verified two ways to get it wrong on toy fixtures:

1. Inserting the whole edge table before transforming lets earlier edges collect statistics from later ones. A transaction at *t1* gained neighbour statistics generated by an edge at *t2*.
2. Calling `partial_fit(batch)` followed by `transform(batch)` inserts the batch twice and doubles every count. A vertex with two out-edges reports a degree of four. A unit test pins this failure.

The only drive pattern we use is `transform(batch_t)` alone, one batch per timestep, for *t* ascending. The driver class (`GfpDriver` in `features/gfp.py`) has no fit method, refuses a batch that spans more than one timestep, and refuses a batch older than the one before it. A test asserts that `features/gfp.py` never mentions `partial_fit`. Because `transform` inserts the whole batch before scoring any of it, row order within a timestep has no effect on the output, and a test confirms this. The name of the drive pattern (`transform_only`) is part of the feature-version hash, so changing it invalidates every cache.

GFP output is deterministic: bit-identical across repeated runs and across 1 and 12 threads.

### 4.3 Configuration

| Parameter | Value | Note |
|---|---|---|
| Families | fan, degree, scatter-gather, length-constrained cycle | temporal cycles disabled |
| Time window | 1 timestep for every family | an Elliptic timestep is its own component |
| Cycle length bound | 10 | GFP's default; the spec's bound |
| Histogram bins | `[2, 4, 8, 16, 32]` | coarser than GFP's default; finer bins stay at zero for most nodes on sparse timesteps |
| Vertex statistics | on, statistics 0 (fan), 1 (degree), 2 (ratio) | statistics 3–10 need a real amount column |

### 4.4 Histogram bin convention

Each pattern family produces a histogram whose column `bin{b}` is named after the upper edge of its bin. We probed the convention on snapml 1.17.2 (a vertex receiving *k* in-edges from *k* distinct sources, for *k* from 1 to 40). A pattern of size *S* lands in the bin with upper edge *b<sub>i</sub>* when *b<sub>i−1</sub>* < *S* ≤ *b<sub>i</sub>*:

| Pattern size *S* | Column |
|---|---|
| 1 | none (below the first edge) |
| 2 | `bin2` |
| 3–4 | `bin4` |
| 5–8 | `bin8` |
| 9–16 | `bin16` |
| 17 and above | `bin32` (open above: sizes 33 and 40 also land here) |

Bins are upper-inclusive. snapml's own docstring describes the opposite, lower-inclusive convention (*b<sub>i</sub>* ≤ *S* < *b<sub>i+1</sub>*); we report the behaviour we measured. In the same probe the value written into the bin equalled the number of edges in the pattern, not a pattern count of one: three in-edges from distinct senders wrote 3 into `fan_in_bin4`. Fan size counts distinct counterparties and degree counts edges, so a vertex with four in-edges from two senders wrote 4 into `fan_in_bin2` and 4 into `degree_in_bin4`.

The probe was run ad hoc; the block *order* is pinned by `tests/test_features_gfp_layout.py`, the bin boundaries are not.

### 4.5 From edge features to node features: `node_agg_v1`

GFP scores edges; Elliptic labels nodes. `node_agg_v1` (`features/aggregate.py`) folds each timestep's edge features onto nodes, in place, as the builder steps forward.

| Edge block | Describes | Fold onto nodes | Node columns |
|---|---|---|---|
| `fan_in`, `degree_in` | the target vertex | element-wise max onto the target | 5 + 5 |
| `fan_out`, `degree_out` | the source vertex | element-wise max onto the source | 5 + 5 |
| `scatter_gather`, `lc_cycle` | the edge's pattern | sum onto both endpoints | 5 + 5 |
| vertex statistics: source-out, source-in, target-out, target-in (3 statistics each) | the endpoint named | max onto that endpoint, merged by direction | `v_fan_out`, `v_degree_out`, `v_ratio_out`, `v_fan_in`, `v_degree_in`, `v_ratio_in` |

That gives 30 histogram columns and 6 vertex-statistic columns, 36 in all, stored with a `gfp_` prefix.

Two facts justify the fold rules. First, GFP scoring is batch-static: every edge touching a vertex within one batch carries the same vertex-side histogram, so taking the max recovers that histogram exactly instead of approximating it. Across batches the max keeps the largest window in which the node appeared. Second, a scatter-gather or cycle pattern belongs to both endpoints, and how many patterns a node sits in is the signal, so those blocks add up.

**Causality guard.** A batch at time *t* writes only into nodes with `node_time` ≥ *t*: a node's features may include edges up to its own timestamp and none after it. Nodes with no qualifying edge keep a zero row. On Elliptic the guard never excludes anything, because every edge's endpoints share its timestep; on a graph with cross-time edges it is what keeps later edges out of earlier nodes' rows.

**Vertex statistics accumulate.** snapml 1.17.2 ignores `vertex_stats_tw`: the vertex-statistic blocks accumulate over every batch inserted so far, while the histogram families respect their windows. A unit test pins this. The `v_*` columns are therefore cumulative, not windowed. They remain past-only, so PR-F2 holds. On Elliptic no vertex appears in two timesteps and the difference cannot show; on AMLworld it will.

### 4.6 Output layout check

Downstream code indexes GFP's output by position. `probe_layout` derives the expected width by arithmetic (five bins per histogram block, six histogram blocks, four vertex-statistic blocks of three) and compares it with the width of a real `transform` on a three-edge toy batch; a mismatch stops the run before any feature is trusted. The runtime probe checks width only. Block *order* is pinned by unit tests that feed GFP a fan-in star, a fan-out star and a three-cycle and assert which blocks light up.

### 4.7 Feature version

`feature_version` is the first 16 hex characters of a sha256 over the canonical JSON of the full feature definition: backend, snapml version, families, bins, window, cycle bound, vertex-statistic switch and codes, aggregation name, drive pattern, edge and node column lists, dataset name and version, and the raw-file hash (PR-F3). The thread count is excluded because output does not depend on it. The Elliptic features behind the §12 results hash to `d6e912c8fcb83f3b`; adding the `unit` key on 17 Sep moved the current version to `a22c7accccccd0ed` over the same columns. The builder caches the matrix as Parquet under that name, with a JSON sidecar holding the definition and per-timestep timings. Computing all 49 timesteps took 2.6 s.

Runs that use no graph features (`base`, `raw165`) log `feature_version = none`; the parent run logs the GFP version.

### 4.8 Causality test

On Elliptic no edge crosses a timestep, so a causality test there proves nothing. The test runs on a synthetic multi-timestep graph with cross-timestep edges, a window of three timesteps and cutoffs at *t* = 4, 7 and 10 (`tests/test_features_causality.py`). For each cutoff it builds features with and without every edge after the cutoff and asserts that the rows of nodes at or before the cutoff are identical. A paired positive control deletes batches inside the window and asserts that the same rows change, so the test cannot pass on features that ignore the graph. The Elliptic-shaped case is skipped with its reason logged (NFR-2).

## 5. Splits

Both regimes partition labelled nodes only (46,564 on Elliptic).

| Regime | Definition | Train | Val | Test | Test illicit | `split_hash` |
|---|---|---|---|---|---|---|
| random | stratified 70 / 15 / 15, split seed 0 | 32,594 | 6,985 | 6,985 | 682 | `408c823763f1f17f` |
| temporal | train ≤ t34, val t35–37, test t38–49 (bounds inclusive) | 29,894 | 3,547 | 13,123 | 828 | `f576c23d95a59084` |

The random split is two stratified cuts with scikit-learn's `train_test_split`: 70% for training, then the remainder divided evenly between validation and test. The temporal test window spans the dark-market shutdown at t43.

**Rolling refit** (`temporal_rolling`, PR-E7) reuses the temporal bounds as a template. For each test batch *t* from 38 to 49 the whole window shifts by *t* − 38: train ≤ 34 + (*t* − 38), val [35, 37] + (*t* − 38), test [*t*, *t*]. Each step is built by the same temporal builder, passes the same leakage assertions and is cached under its own definition hash, so the regime is twelve temporal splits, each with its own `split_hash`; the child run logs a hash over the twelve. The label lag is the validation window only: at *t* the newest label the model has seen is from *t* − 1, and the newest label it trained on is from *t* − 4. Elliptic timesteps are about two weeks apart, so this is an optimistic bound on what refitting can recover, not a retraining policy.

**Leakage assertions** run on every build (`splits/builder.py`), and a failure stops the run:

- train, validation and test are pairwise disjoint;
- no part is empty, no part contains an unlabelled node, and every part contains at least one illicit node (otherwise PR-AUC and threshold selection are undefined);
- in the temporal regimes, max(train time) < min(validation time) and max(validation time) < min(test time);
- in the temporal regimes on a graph with cross-time edges, no edge joins a training node to a test node. On Elliptic the time-order check already implies this, so the builder skips it there.

**Split seed and model seed are separate.** The random partition has its own seed in the regime config. All five model seeds refit on the same fixed partition, so the seed interval of §8 measures retraining variance and not partition variance.

**Split hash.** `split_hash` is the first 16 hex characters of a sha256 over the sorted index arrays (named, in order) followed by the canonical JSON of the split definition: dataset, version, regime, raw-file hash, and either the fractions and seed or the timestep bounds. The builder caches indices under a hash of the definition; when a cached split exists, it compares the freshly built arrays with the cache and stops if they differ, which doubles as a determinism check (PR-D4).

## 6. Models

Both models sit behind one protocol (`fit`, `predict_proba`, `trial0`; PR-M5). `fit` receives the split and may read training and validation rows; the protocol forbids it to touch test.

### 6.1 XGBoost

We use `xgboost.XGBClassifier` 3.2.0. Its reference configuration, trial 0 (D2), is the library defaults with two additions: `n_estimators = 1000` as a ceiling and `early_stopping_rounds = 50`. xgboost 2 and later take early stopping on the constructor, which is why trial 0 carries it. The model also sets `tree_method = "hist"` (the library default) and uses every CPU core for its thread pool.

Four parameters are fixed by the protocol and override any config value:

- `scale_pos_weight` = licit count / illicit count **on the training split only**; the validation and test ratios belong to the future;
- `eval_metric = "aucpr"`, evaluated on the validation split for early stopping;
- `random_state` = the model seed;
- `device` = the resolved device (CUDA for the Elliptic run, per the logged `device` parameter).

With library defaults (`subsample = 1`, `colsample_* = 1`) the seed has nothing to randomise. All five seeds produced identical fits: the same best iteration in every cell (temporal: 59 for `base`, 182 for `base_gfp`, 132 for `raw165`) and identical metrics, which is why XGBoost rows show an interval half-width of 0.0000. §8 returns to the consequence for gap testing.

### 6.2 GraphSAGE

The GraphSAGE model (`models/sage.py`, PyTorch 2.14.0 and PyTorch Geometric 2.8.0) is a stack of `SAGEConv` layers, each followed by ReLU, with dropout between layers (not after the last), then a linear head producing one logit. Trial 0, `sage_default_2x64`:

| Parameter | Value | Origin |
|---|---|---|
| Layers × hidden | 2 × 64 | spec (D2) |
| Dropout | 0.2 | our choice |
| Optimiser | Adam, learning rate 1e-3, weight decay 0 | our choice |
| Epochs | up to 200, patience 20 on validation PR-AUC | our choice |
| Loss | BCE with logits, `pos_weight` = licit / illicit on the training split | our choice |
| Graph | symmetrised (`to_undirected`) | our choice |
| Sampler | `NeighborLoader`, fan-out [15, 10], batch size 1,024 seed nodes, shuffled | spec fan-out; batch size ours |

Only the depth and width come from the spec. The remaining values are ours, and every run logs them verbatim as `model.*` parameters.

**Training.** Mini-batches seed on training nodes; the sampled neighbourhood, which can include validation, test and unlabelled nodes, supplies features as context, and only seed nodes contribute to the loss. On Elliptic sampling cannot leave a timestep component. After each epoch the model scores every node with a full-batch forward pass and computes validation PR-AUC (scikit-learn average precision); training keeps the best epoch's weights and stops after 20 epochs without improvement.

**Inference is full-batch.** Prediction runs the whole graph (203,769 nodes) through the network in one pass. That is exact at this size, and it keeps sampling noise out of both the early-stopping signal and the reported metrics.

**Input standardisation.** Before building the graph tensors, `fit` z-scores every feature column with the mean and standard deviation of the **training rows only** and applies them to all nodes (PR-E1). A column that is constant on the training rows keeps a standard deviation of 1 and maps to zero; 17% of GFP columns are constant on the temporal training set. Validation and test rows never shape the scaling.

The first real run (`a64a16e`) fed raw values: `base` columns reached |x| = 265 with column standard deviations from 0.01 to 2.55, and GFP counts reached 472 with 86% zeros. Trees are invariant to monotone rescaling of a feature and GNNs are not, so unscaled inputs handicap the GNN and bias the benchmark toward the expected finding. We decided to standardise on that principle before measuring the effect, fixed it in `ec4501f`, and reran the grid. XGBoost receives unscaled inputs and its rows were bit-identical across the two runs.

**Determinism.** Every fit seeds Python, NumPy and PyTorch (CPU and CUDA). CUDA scatter reductions are not bitwise deterministic, so SAGE seeds differ from each other through initialisation, sampling order and GPU arithmetic. The seed interval covers all three.

### 6.3 Run cost

The Elliptic grid ran 50 fits (5 configs × 2 regimes × 5 seeds) in under nine minutes on an RTX 4060 Laptop GPU. The slowest fit was `random sage.base` at 44.3 s; the slowest XGBoost fit took 1.5 s (fit times from the MLflow export).

## 7. Decision threshold and metrics

### 7.1 Threshold on validation

`choose_threshold` takes the validation labels and validation scores and nothing else; its signature has no way to receive test data (PR-E4). It builds the precision-recall curve on validation, computes positive-class F1 at every threshold on the curve, and returns the threshold with the highest F1. When several thresholds tie, it returns the highest one: same F1, fewer alerts. A node is predicted illicit when its score is at or above the threshold.

The pipeline scores test once, with that threshold. Under `temporal_rolling` every step chooses its own threshold on its own validation window and scores its one test batch with it; the stitched test set therefore carries one threshold per row, and the pooled F1 is computed from those per-step decisions. The rank metrics (PR-AUC, ROC-AUC, P@R) are not reported for a rolling run: twelve models do not share a score scale, so pooling their scores would rank a 0.6 from one against a 0.6 from another. Per-batch PR-AUC, where one model scores one batch, is in the per-timestep curve. A unit test spies on `choose_threshold` and asserts that, on every fit, it receives the validation set and nothing else. The validation F1 at the chosen threshold is logged as `val_f1`, a selection diagnostic that makes a validation/test disagreement visible; it never appears in a results table.

### 7.2 Metrics

All metrics are computed on labelled test nodes for the illicit class (PR-E2):

- **F1** at the validation threshold;
- **PR-AUC**, as scikit-learn's average precision (a step-wise sum, not trapezoidal interpolation);
- **ROC-AUC**, reported as secondary only: at this class balance it is dominated by the easy negatives;
- **P@R0.5** and **P@R0.8**: precision at the highest threshold on the test PR curve whose recall reaches 0.5 or 0.8, the cleanest alert queue that still catches that share of illicit cases.

P@R0.5 and P@R0.8 describe the score ranking. They never set or influence the deployed threshold.

Accuracy is never reported (PR-E6). At Elliptic's class balance it measures the imbalance, and the metric function raises an error if asked for it. Point precision, recall and the test counts are logged as `diag_*` diagnostics and kept out of the results table.

## 8. Statistics

### 8.1 Seed intervals

Every config runs five seeds on one fixed split. The results table reports, per metric, the across-seed mean with a 95% Student-t interval:

mean ± *t*<sub>0.975, n−1</sub> · *s* / √*n*, with *n* = 5 and *s* the sample standard deviation (ddof = 1).

With four degrees of freedom *t*<sub>0.975</sub> = 2.776, against 1.96 for a normal interval, so a normal interval would be about 30% too narrow. The interval answers the question a reader brings to the table: would this ranking replicate if the model were retrained (PR-E3)?

### 8.2 Per-timestep curves

Every fit also scores each test timestep separately (`eval/curves.py`): F1 at the validation threshold and PR-AUC, per timestep, with the count and positive count. The curves figure shows the across-seed mean per timestep with the same t-interval as a band. A bootstrap over test ids would be narrow whatever the training instability, because it resamples test nodes around one fitted model; it is not used, and it would never be pooled with the seed interval or used to call a gap significant (PR-E3).

### 8.3 Paired gaps and deterministic pairs

A comparison of interest (`base` against `base_gfp` within a model; XGBoost against SAGE within a feature set; `raw165` against `base_gfp`) is significant only as a paired-by-seed difference, *d<sub>s</sub>* = metric(A, *s*) − metric(B, *s*), whose t-interval over the five *d<sub>s</sub>* excludes zero.

XGBoost trial 0 is deterministic (§6.1). For an XGBoost-versus-XGBoost pair all five paired differences are identical, the paired standard deviation is zero, and the interval collapses to a point, so any nonzero difference would be "significant" with no variance behind it. Such pairs are degenerate and no significance is claimed for them. Adding `subsample < 1` to trial 0 to manufacture variance is ruled out: it would move XGBoost's point estimate to make an interval behave. The results tables therefore report point estimates and seed intervals; the only gap that could be tested is XGBoost against SAGE. Because XGBoost is constant across seeds, that paired interval is the SAGE seed interval shifted by the XGBoost value, so it excludes zero exactly when SAGE's interval does not reach the XGBoost point estimate, which holds for F1 on both splits.

### 8.4 One run, one table

The reporter builds the results table from the child runs of **one** MLflow parent run, the one the pipeline has just finished. Before we added this scoping, rerunning a config into the same experiment counted the earlier children as extra seeds: five seeds run twice reported `n_seeds = 10`, and the half-width became *t*<sub>9</sub>/√10 instead of *t*<sub>4</sub>/√5, about 40% of its honest width, from no new information. `n_seeds` in a table is the *n* the protocol ran.

## 9. No hyperparameter search

Every fit uses its model's fixed reference configuration, trial 0 (D2). The config schema has no search setting. Every child run logs `trials_completed = 0`, an honest zero rather than a nominal 1, and `trial0_source` (`xgboost_defaults` or `sage_default_2x64`). The comparison is therefore between untuned reference models under one protocol; a tuned comparison would need an equal wall-clock search budget per model, which was not run.

### 9.1 AMLworld HI-Small

`HI-Small_Trans.csv` holds 5,078,345 transactions between 515,088 accounts, 5,177 of them laundering (0.10%). Timestamps run from 2022-09-01 00:00 to 2022-09-18 16:18, a nominal span of 17.68 days, but ordinary traffic occupies only the first ten days: after 10 September only 1,108 transactions remain, and 655 of them (59%) are laundering, because the generator completes its laundering patterns after background activity stops. The first three calendar days hold 2,076,752 transactions and 1,121 laundering edges.

AMLworld is an edge task: an account is a node, a transaction is an edge, and the label sits on the edge. The loader (`data/amlworld.py`) reads the Kaggle CSV by column position (its header names both account columns `Account`), keys accounts as `bank:account`, indexes time in hours from the first transaction, and sets `batch_id` to the day. `base` is the six raw transaction fields (amount paid and received, currency codes, payment format, hour of day), which contain no neighbour aggregates (PR-M7); `base_gfp` adds the per-edge GFP histograms with a 24-hour window. The temporal split is IBM's own Multi-GNN split, days 0–5 / 6–7 / 8–17, whose laundering rates are 0.08%, 0.11% and 0.19%: a test period that reaches into the tail sees a higher positive rate than training, and the results say so.

### 9.2 Why there is no PNA reference row

IBM's Multi-GNN PNA at its published settings (batch 8192, 100 × 100 sampled neighbours, `--emlps --reverse_mp --ego --ports`, commit `252b025`) ran out of GPU memory in the first backward pass on the development laptop (RTX 4060, 8 GB) with 11.88 GiB already allocated, so it needs at least about 13 GiB. A `SAGEConv` swap into the same code ran only by spilling into shared memory (about 48 minutes per training pass, which measures the spill, not the model). Reducing the batch or the neighbour counts would have stopped the run being IBM's published configuration, and replicating that configuration answered nothing the two-by-two does not, so the reference row was dropped rather than run at reduced settings.

## 10. Reproducibility and provenance

### 10.1 Tracking

Runs are tracked with MLflow 3.16.0 in a local SQLite file, `sqlite:///mlruns/mlflow.db`. MLflow 3 refuses its own `file:` store, and a single SQLite file needs no server. The MVP experiment is `elliptic_mvp`.

The pipeline opens one **parent** run per invocation, tagged `kind`, `dataset` and `git_commit`, with parameters `dataset_version`, `feature_version`, `seeds`, `device`, `total_fits` and `trials_completed`, and the config file as an artifact. Each (regime, model, seed) fit is a **child** run with:

- tags: `kind`, `dataset`, `dataset_version`, `regime`, `model`, `features`, `feature_version`, `split_hash`, `git_commit`;
- parameters: `seed`, `device`, `trials_completed`, `trial0_source`, `sampler`, and every model parameter as `model.*`;
- metrics: `test_f1`, `test_pr_auc`, `test_roc_auc`, `test_p_at_r50`, `test_p_at_r80`, `threshold`, `val_f1`, `val_pr_auc`, `fit_seconds`, `best_iteration`, and the `diag_*` counts.

Nested runs inherit nothing from their parent, so the identity and provenance fields are set on each child. The reporter refuses any child run missing one of those tags instead of guessing, and warns when a provenance field differs within a config's seeds (PR-O1, NFR-1).

### 10.2 Dirty trees

If the working tree has uncommitted changes, the pipeline stamps every run's commit as `<sha>-dirty`, and the stamp flows into the `commit` column of the results CSV. A warning on stderr disappears; the stamp stays with the numbers. The Elliptic results table carries the clean stamp `afe6ef5`.

### 10.3 Environment

`pyproject.toml` pins every dependency to one version, and `uv.lock` is committed. The lock resolves for Linux only.

| Component | Version |
|---|---|
| Python | 3.11 (`requires-python = "==3.11.*"`) |
| xgboost | 3.2.0, the newest release supporting Python 3.11 |
| torch | 2.14.0, CUDA 13.0 build (`pyg-lib 0.9.0+pt214cu130`; `nvidia-cuda-runtime 13.0.96` in `uv.lock`) |
| torch-geometric | 2.8.0.post1 |
| snapml | 1.17.2 |
| scikit-learn / scipy / numpy | 1.9.0 / 1.17.1 / 2.4.6 |
| pandas / pyarrow | 3.0.5 / 25.0.1 |
| mlflow | 3.16.0 |
| pydantic / typer | 2.13.5 / 0.27.2 |
| Hardware | RTX 4060 Laptop GPU, WSL2 host with 7 GB RAM |

### 10.4 Provenance of the Elliptic numbers

| Item | Value |
|---|---|
| Code that produced the results | `afe6ef5` (models, features and splits identical to `ec4501f`) |
| MLflow parent run | `ba1e269a1401499aa54f6557e505a564` (50 children) |
| First tagged run | `341e0f93709f487187f8ca276ff60d6a` at `ec4501f`, tag `mvp`; XGBoost rows identical, SAGE rows within GPU nondeterminism |
| Superseded first run | `281caef1ade34e42aecd51bf2ab1e8a8` at `a64a16e` (unscaled SAGE inputs) |
| Results table | `report/tables/elliptic_mvp_results.csv`, committed |
| Committed export | `report/exports/mvp_elliptic_mvp_runs.csv`, all 102 runs of both parents (`d16baab`) |
| Feature version | `d6e912c8fcb83f3b` |
| Split hashes | random `408c823763f1f17f`; temporal `f576c23d95a59084` |

To regenerate: check out `mvp`, run `uv sync`, place the three Elliptic++ 2023.1 CSVs in `data/raw/elliptic_pp/2023.1/`, and run `uv run mulegraph run --config configs/elliptic_mvp.yaml`. A run from the tag stamps the tagged merge commit. XGBoost rows reproduce exactly; SAGE rows reproduce within GPU nondeterminism, which is the whole difference between the `mvp` run and the `afe6ef5` run in §12.2.

## 11. Limitations

- **AMLworld is synthetic** (NFR-4). Its transactions and laundering typologies come from a simulator (Altman et al., 2023). Results on AMLworld describe the simulator's world; no claim is made that they generalise to real bank transactions.
- **Elliptic has one natural drift event.** The dark-market shutdown at t43 is a single observation. Any lead time a detector shows on it (S3) is reported as one observation, not an estimate of detector performance.
- **The Elliptic temporal test mixes two regimes.** Test F1 averages a period where every model works (t38–42) with one where none does (t43–49), and validation (t35–37) precedes the shutdown. The headline mean needs the per-window numbers of §12 beside it.
- **No search.** Every result is trial 0. The XGBoost reference is library defaults and the SAGE reference is partly our choice; neither is tuned.
- **Deterministic XGBoost.** Trial 0 shows no retraining variance, so XGBoost's zero-width intervals mean "retraining changes nothing", not certainty (§8.3).
- **GFP has little to see on Elliptic.** Timestep components are disconnected and the window is one timestep, so graph patterns are small and sparse. This limits what the feature gap can show on Elliptic.
- **`base` omits local Elliptic++ attributes.** Dropping the 15 non-graph extras keeps `raw165` comparable with published work but withholds local information from `base`.
- **Opaque features.** The 165 Elliptic features are anonymised and their construction is undocumented. Šafář et al. (2026) argue this hides leakage (§13).
- **One machine.** Timings come from one laptop GPU.

## 12. Results

### 12.1 Elliptic: per-window F1 on the temporal test

The temporal test window contains the t43 shutdown, and the mean over t38–49 hides it. We refitted the three deterministic XGBoost configs at `a64a16e` (seed 0; these rows are bit-identical to `ec4501f`) and scored each test window separately:

| Config | F1, t38–42 | F1, t43–49 |
|---|---|---|
| `xgb.raw165` | 0.894 | 0.032 |
| `xgb.base` | 0.848 | 0.019 |
| `xgb.base_gfp` | 0.829 | 0.033 |

169 of the 828 illicit test nodes fall in the post-shutdown window, where no config reaches an F1 above 0.033. Validation F1 at the chosen threshold was 0.931–0.949 for the XGBoost configs, measured on t35–37, before the shutdown. The same effect caps P@R0.8 near 0.2 for every config in Table 12.2: reaching 80% recall means reaching into cases no model detects.

These per-window numbers come from an ad hoc refit; the per-timestep curves written by every run (`report/tables/<experiment>_curves.csv`, §8.2) are the committed code path and supersede them.

### 12.2 Elliptic: headline means (`afe6ef5`)

Across-seed mean ± half-width of the 95% t-interval, *n* = 5, from `report/tables/elliptic_mvp_results.csv`.

**Temporal**

| Config | F1 | PR-AUC | P@R0.5 | P@R0.8 |
|---|---|---|---|---|
| `xgb.raw165` | 0.778 ± 0.000 | 0.739 ± 0.000 | 0.995 ± 0.000 | 0.194 ± 0.000 |
| `xgb.base` | 0.722 ± 0.000 | 0.726 ± 0.000 | 0.983 ± 0.000 | 0.196 ± 0.000 |
| `xgb.base_gfp` | 0.718 ± 0.000 | 0.715 ± 0.000 | 0.998 ± 0.000 | 0.159 ± 0.000 |
| `sage.base` | 0.591 ± 0.033 | 0.550 ± 0.038 | 0.705 ± 0.068 | 0.200 ± 0.011 |
| `sage.base_gfp` | 0.560 ± 0.023 | 0.577 ± 0.031 | 0.652 ± 0.062 | 0.186 ± 0.014 |

**Random**

| Config | F1 | PR-AUC | P@R0.5 | P@R0.8 |
|---|---|---|---|---|
| `xgb.raw165` | 0.959 ± 0.000 | 0.991 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `xgb.base` | 0.950 ± 0.000 | 0.987 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `xgb.base_gfp` | 0.950 ± 0.000 | 0.987 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `sage.base` | 0.907 ± 0.008 | 0.953 ± 0.005 | 0.994 ± 0.003 | 0.986 ± 0.005 |
| `sage.base_gfp` | 0.900 ± 0.009 | 0.948 ± 0.004 | 0.994 ± 0.004 | 0.978 ± 0.006 |

The random split inflates F1 by 0.18 to 0.34 over the temporal split for every config. On both splits XGBoost scores above SAGE, `raw165` scores highest, and adding GFP features does not raise F1 for either model. These are readings of point estimates; §8.3 says which gaps can be tested.

The GFP result is coherent with the field: on Elliptic the timestep components are disconnected, so a one-timestep GFP window has little to see, and Maganti (2026) finds the real edges carry less signal than shuffled ones under shift. On AMLworld, where GFP is IBM's own headline, the same two-by-two shows whether that reverses.

### 12.3 AMLworld HI-Small (`7c22e3e-dirty`)

Temporal split, days 0–5 / 6–7 / 8–17 (§9.1), three seeds, from `report/tables/amlworld_xgb_results.csv`. XGBoost is deterministic, so the intervals are zero (§8.3).

| Config | F1 | PR-AUC | P@R0.5 | P@R0.8 |
|---|---|---|---|---|
| `xgb.base` | 0.209 | 0.109 | 0.090 | 0.034 |
| `xgb.base_gfp` | 0.539 | 0.521 | 0.570 | 0.099 |

On AMLworld the GFP features more than double F1, the reverse of Elliptic. The test window reaches into the post-day-10 laundering tail (§9.1), so the per-day curve (`report/tables/amlworld_xgb_curves.csv`) matters here as it does on Elliptic: on the two realistic test days (8–9) GFP takes F1 from 0.10–0.20 to 0.38–0.45, and from day 10 every config scores PR-AUC above 0.92. The SAGE rows need the Kaggle run in `kaggle/amlworld_sage.md`. The run was made from a working tree with uncommitted changes, and its `-dirty` stamp (§10.2) says so; it is rerun from a clean commit before these numbers are tagged.

### 12.4 Drift monitor on the t43 shutdown (`8c28027`)

`mulegraph drift --config configs/elliptic_drift.yaml` fits `xgb.base_gfp` and `sage.base_gfp` on the temporal split, then scores every unit from t35 to t49 with no labels (design §3.4, PR-R2). The reference is the validation window t35–37. Three detectors run per batch: maximum PSI over feature columns, the fraction of feature columns whose KS test rejects at 0.01, and the KS statistic on the model's scores. With textbook flags (PSI > 0.2, KS p < 0.01) every test batch is flagged, because at 2,500–7,000 rows a batch the tests reject almost anything. Each flag is therefore calibrated to the detector's largest leave-one-out score inside the reference window.

A model counts as broken at the first timestep that starts two consecutive batches of F1 more than 20% below its validation mean. A detector warns at the first timestep that starts two consecutive flags. Lead time is the first minus the second. Both need the same persistence; an earlier rule that counted a single flag reported 3–4 timesteps of warning that disappear under the symmetric rule. The validation mean is F1 at the threshold chosen on that window, so it is optimistic (design §3.4).

| Model | Detector | First held flag | F1 drop | Lead |
|---|---|---|---|---|
| `xgb.base_gfp` | PSI on features | t48 | t43 | −5 |
| `xgb.base_gfp` | KS on features | never | t43 | — |
| `xgb.base_gfp` | KS on scores | never | t43 | — |
| `sage.base_gfp` | PSI on features | t48 | t39 (3 seeds), t43 (2 seeds) | −9 / −5 |
| `sage.base_gfp` | KS on features, KS on scores | never | as above | — |

No detector warns before the collapse. The SAGE drop at t39 is a dip below the drop level that recovers by t42, not the shutdown. Feature detectors see only features, so their flags are identical across models and seeds, and XGBoost's seeds give identical fits: this is one observation of one event (§11). Tables: `report/tables/elliptic_drift_scores.csv`, `report/tables/elliptic_drift_lead_time.csv`.

### 12.5 Why t43 breaks every model and every detector (`8c28027`)

The same drift run, with `drift.event: 43`, splits the labelled test units at t43 (design §3.4 step 5, PR-R5). For each side it records how the fitted model transfers, and it refits a probe: XGBoost with library defaults, 5-fold stratified CV inside that window alone. From `report/tables/elliptic_drift_event.csv`; SAGE is the mean over five seeds with the range in brackets, and XGBoost's seeds are identical.

| | t38–42 | t43–49 |
|---|---|---|
| Labelled units (illicit) | 6,436 (659, 10.2%) | 6,687 (169, 2.5%) |
| `xgb.base_gfp` ROC-AUC | 0.957 | 0.556 |
| `xgb.base_gfp` recall at the validation threshold | 0.713 | 0.018 |
| `xgb.base_gfp` median score of illicit units | 0.999 | 0.000 |
| `sage.base_gfp` ROC-AUC | 0.929 [0.926, 0.937] | 0.681 [0.639, 0.727] |
| `sage.base_gfp` recall at the validation threshold | 0.595 [0.549, 0.651] | 0.020 [0.012, 0.030] |
| `sage.base_gfp` median score of illicit units | 0.966 [0.942, 0.986] | 0.058 [0.008, 0.130] |
| Probe ROC-AUC, refit inside the window | 0.993 | 0.985 |

Three readings follow.

**The models are confidently wrong, not uncertain.** After t43 the median illicit unit scores 0.000 under XGBoost, and ROC-AUC falls to near chance for XGBoost (0.56), so no threshold rescues it: the ranking itself is gone. SAGE keeps slightly more of the ranking but recalls no more.

**The new illicit behaviour is learnable; it is just different.** Refit inside t43–49 alone, the probe separates illicit from licit about as well as it does before the shutdown (0.985 against 0.993). The labels after t43 are consistent; the rule that fits t1–34 no longer applies to them. The benchmark's random split shows the same from the other side: with some post-t43 units in training, its per-timestep F1 recovers to 0.89 and 0.95 at t48–49 for `xgb.base_gfp` (0.74 and 0.85 for SAGE), while every temporal config stays at or below 0.04 (`report/tables/elliptic_mvp_curves.csv`). Those random-split timesteps hold only 5 and 11 illicit test units, so the recovery is indicative. The probe's folds are random within the window, so 0.985 measures separability, not what a deployed model would reach.

**Nothing in the model class can fix this, and nothing label-free sees it.** Every config learns from the same t1–34 illicit pattern. Graph structure cannot supply the new one: Elliptic's timesteps are disconnected, so GFP windows and SAGE neighbourhoods see only the same timestep (§11). Weber et al. (2019) report the same collapse for a random forest retrained after each test step. The detectors, meanwhile, watch whole batches. At t43 the 24 illicit units are under 2% of the 1,370 labelled units and a smaller share of all scored units. From t42 to t43, PSI falls from 1.20 to 0.93 against a flag at 1.52. Feature KS rises from 0.64 to 0.71, which flags t43 alone, above its 0.64 flag, but falls to 0.53 at t44, so the flag does not hold. Score KS for XGBoost falls from 0.094 to 0.084 against a flag at 0.186. Because the model scores the new illicit units as licit, the score distribution loses high scores and looks calmer, not stranger. The change is in which feature patterns are illicit, for a small minority of rows. Marginal-distribution detectors are blind to that by construction, which is why §12.4 is negative for every detector rather than a matter of calibration.

## 13. Related work and positioning

### 13.1 Our Elliptic numbers against the literature

Train t1–34, test from t35 (ours starts at t38 after a validation block at t35–37, which is stricter). Illicit-class F1, mean over seeds where reported.

| Model | Ours (`afe6ef5`) | Weber 2019 | Maganti 2026 |
|---|---|---|---|
| Trees on all 165 features | 0.778 | 0.788 (RF) | 0.821 ± 0.003 (RF, 10 seeds) |
| Trees on 93 local features | 0.722 | 0.694 (RF) | — |
| Trees on local + causal GFP | 0.718 | — | — (not tried) |
| GraphSAGE | 0.591 ± 0.033 | — | 0.689 ± 0.017 |
| GCN / Skip-GCN | — | 0.628 / 0.705 | 0.503 (GCN) |
| MLP | — | 0.653 | 0.549 |

Trees beating GNNs on Elliptic under temporal evaluation is the 2019 result and the 2026 result; this repository reproduces it in nine minutes on a laptop with a cleaner protocol. Our SAGE is about 0.1 below Maganti's, and the three protocol differences listed under Maganti below explain most of it.

### 13.2 What others have done

| Paper | Showed | Did not do |
|---|---|---|
| Weber et al. 2019, KDD-ADF | RF beats GCN on Elliptic temporal split; t43 collapse first documented | No CIs, no engineered graph features, no drift detection |
| Altman, Egressy et al. 2023, NeurIPS D&B (AMLworld) | Synthetic bank data; GBT+GFP ≈ PNA (HI-Medium 59.5 vs 59.7) | No inductive regime as a contrast, no drift |
| Egressy et al. 2024, AAAI (Multi-GNN) | GIN+EU / PNA "closely match or outperform" tree baselines on AMLworld | Disagrees with the GFP paper; nobody has run both under one protocol |
| Blanuša et al. 2024, ICAIF (GFP) | GFP+XGBoost beats PNA on every AMLworld set: HI-Small 63.2 vs 56.8 | Temporal 60/20/20 only; IBM evaluating IBM |
| Maganti, Apr 2026, arXiv | Strict inductive Elliptic: RF 0.821 beats every GNN; shuffled edges beat real edges | Trees never get graph features; early-stops on test; single dataset; no drift |
| Šafář et al. 2026, FSI:DI / DFRWS | Elliptic feature construction is opaque; leakage across standard splits inflates results | Abstract only read |
| Heidrich et al. Mar 2026, arXiv | Significance-tested protocol for graph-derived signals in tabular ML on a crypto fraud set | GBT only, no GNN, no temporal framing |

What this repository adds: the feature × model two-by-two with *causal* GFP features on both datasets (Maganti compares models; IBM compares GBT+GFP to GNNs but never GNN+GFP); validation-only thresholding and paired-by-seed significance; and label-free lead time on a real drift event, which Weber and Maganti both name and stop at.

### 13.3 Papers

**Weber et al. (2019)** released the Elliptic dataset and trained on t1–34 and tested on t35–49. A random forest on all 165 features (F1 0.788) beat GCN (0.628) and Skip-GCN (0.705), and a random forest on the 93 local features reached 0.694. They documented the t43 collapse for every model, including a random forest retrained after each test step, without confidence intervals, engineered graph features or drift detection.

**Maganti (2026, arXiv 2604.19514)** re-evaluated GNNs on Elliptic under a strict inductive temporal protocol. A random forest (0.821 ± 0.003 over 10 seeds) beat every GNN, including GraphSAGE (0.689 ± 0.017), and shuffled edges outperformed real ones under shift. Our Elliptic work is a replication of that result with a feature-versus-model decomposition added. Three protocol differences separate the two:

1. Maganti early-stops on test F1; we early-stop on validation and choose the threshold on validation (PR-E4).
2. His tree models never receive engineered graph features; our two-by-two gives both model families the same causal GFP features.
3. His models use the 165-feature block, which already contains one hop of aggregation; our `base` is the 93 local features, so our SAGE has to learn that hop itself, and the 165 block appears only as the `xgb.raw165` reference.

Our SAGE trails his by about 0.1 F1; we have not measured how much of that each difference explains. Our XGBoost rows sit within 0.04 of both papers' tree results. He names retraining under drift as future work.

**Altman et al. (2023)** introduced AMLworld, a synthetic bank-transfer dataset with labelled laundering typologies. On HI-Medium, gradient-boosted trees with graph features (59.5) and PNA (59.7) finished level.

**Egressy et al. (2024)** proposed Multi-GNN, with edge updates (GIN+EU) and PNA for directed multigraphs, and reported that these GNNs closely match or outperform tree baselines on AMLworld.

**Blanuša et al. (2024)** introduced the Graph Feature Preprocessor and reported that GFP with XGBoost beats PNA on every AMLworld set (HI-Small 63.2 against 56.8), under a temporal 60/20/20 split with no inductive test. The two IBM groups therefore disagree on the same data, and nobody has run both model families under one protocol and one budget. The AMLworld two-by-two is designed to do that.

**Šafář et al. (2026)**, *The enemy of reproducibility is opacity: What's inside the Elliptic bitcoin dataset (and why it is wrong)*, FSI: Digital Investigation (DFRWS USA 2026). Read at abstract level only (paywalled). The abstract argues that Elliptic's feature construction is opaque and that leakage across the standard splits inflates reported performance. The open question is *where* the leakage sits: in the 72 aggregated features only, or in the 93 local features too. The design keeps the aggregates out of `base` (PR-M7), so leakage confined to them would affect only `xgb.raw165` and the published baselines. If it reaches the local features, Elliptic serves only as the drift dataset (the t43 event) and AMLworld carries the benchmark claim.

### References

- Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A. and Atasu, K. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. *NeurIPS Datasets and Benchmarks*. arXiv:2306.16424.
- Blanuša, J. et al. (2024). Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection. *ICAIF*. arXiv:2402.08593.
- Egressy, B. et al. (2024). Provably Powerful Graph Neural Networks for Directed Multigraphs. *AAAI*. arXiv:2306.11586.
- Elmougy, Y. and Liu, L. (2023). Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network. *KDD*. doi:10.1145/3580305.3599803.
- Heidrich et al. (2026). A Systematic Evaluation Protocol of Graph-Derived Signals for Tabular Machine Learning. arXiv:2603.13998.
- Maganti (2026). When Graph Structure Becomes a Liability: A Critical Re-Evaluation of GNNs for Bitcoin Fraud Detection under Temporal Distribution Shift. arXiv:2604.19514.
- Šafář, Pluskal, Veselý and Ryšavý (2026). The enemy of reproducibility is opacity: What's inside the Elliptic bitcoin dataset (and why it is wrong). *Forensic Science International: Digital Investigation* (DFRWS USA 2026).
- Weber, M., Domeniconi, G. et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. *KDD Workshop on Anomaly Detection in Finance*. arXiv:1908.02591.
