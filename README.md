# mulegraph

[![CI](https://github.com/Abhinawap/mulegraph/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhinawap/mulegraph/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Score a day of bank transactions and get a ranked alert queue, with the reasons behind each alert,
plus a check on whether today's scores can be trusted. Underneath is a benchmark that tests
money-laundering models the way a bank has to run them: trained on the past, scored on the future,
with no leakage.

## What it does

```bash
uv run mulegraph score --config configs/amlworld_score.yaml   # one day of AMLworld, ~3 min, ~6 GB RAM
```

The first call fits the model and saves it with its validation-chosen threshold; later days reuse
both. Day 8 is 654,467 transactions and produces 466 alerts, ranked by score
(`report/tables/amlworld_batch8_alerts.csv`; the last column is how much each feature pushed the
score toward laundering, in log-odds):

```text
rank  score   from -> to          why
1     0.9999   24281 -> 112152  payment_format +4.12; gfp_degree_out_bin2 +1.92; hour_of_day +0.90
2     0.9998   24281 -> 61426   payment_format +4.12; gfp_degree_out_bin2 +1.94; hour_of_day +0.84
3     0.9997  326565 -> 264765  payment_format +3.26; gfp_fan_in_bin8 +1.82; gfp_degree_in_bin8 +0.89
```

It also writes a health check (`report/tables/amlworld_batch8_health.json`) that asks whether
today looks like the days the model was fitted on. PSI and KS compare the input features, and
score-shift and alert-rate compare the model's own scores; none of them sees a label. The command
exits 3 when a detector flags the day, so a scheduler can stop trusting the queue. It also writes a
one-page HTML report of both (`..._report.html`: a figure, the detector table, the top 25 alerts and
the provenance; it opens in any browser and loads nothing from outside):

```text
day 8   exit 0   psi 0.82/1.65   ks 0.74/0.91   conf 0.07/0.10   alert 0.05/0.94
day 10  exit 3   psi 8.74/1.65*  ks 0.67/0.91   conf 0.74/0.10*  alert 5.99/0.94*
```

Each cell is the day's score over that detector's flag level, and `*` marks a flag.

**What this does and does not show.** The data is a bank simulator, not a real bank. Day 10 is the
simulator's laundering tail, where ordinary traffic stops and 396 transactions remain, so it is
the easiest drift there is. The reference window (every day before the test window) was chosen
after looking at days 8 to 10, so this shows the mechanism works, not that the check would warn of
a subtle shift. On the real collapse in the Bitcoin data, no detector warned in time (below). Scored
against the labels afterwards, the day-8 queue has F1 0.38
(`report/tables/amlworld_xgb_curves.csv`). Both days' outputs are committed and stamped with the
commit that produced them; set `batch: 10` in the config to reproduce day 10.

## What I found

- **Graph features cut an analyst's alert queue about 6×.** On 5 million simulated bank
  transactions, adding causal graph features (fan-in, fan-out, scatter-gather, computed only from
  past transactions) takes the alerts opened per real laundering case from 26 to 4.4 at 50% recall.
  The graph neural network I tested added nothing on top.
- **The evaluation method changes the answer more than the model does.** Scored on a random split,
  which lets the model see the future, every model gets F1 0.90–0.96. On the honest temporal split
  they get 0.56–0.78, and the tree models match published results. The toolkit only reports the
  honest version as the headline.
- **When criminal behaviour changed, no label-free alarm warned in time.** After a dark-market
  shutdown in the Bitcoin data, every model's F1 fell to near zero and none of four label-free drift
  detectors warned of it. Only retraining on fresh labels recovered it, and slowly. The practical
  lesson: budget for fast labelling and scheduled retraining, don't rely on drift alarms alone.

![Alerts per real case caught](report/figures/readme_alert_load.png)

*XGBoost on AMLworld HI-Small, test days 8–9 (862,792 transactions, 956 laundering). Days 10–17
are left out because the simulator stops ordinary traffic there and 59% of what remains is
laundering, which flatters every model. Regenerate with `scripts/readme_figures.py`.*

## Quickstart

Python 3.11 and [uv](https://docs.astral.sh/uv/). All dependencies are pinned in `uv.lock`.

```bash
uv sync
uv run mulegraph smoke        # about 15 s on a synthetic graph, no downloads
```

```text
INFO  loaded synthetic_elliptic smoke-0: 2000 nodes, 2751 edges, 165 features, 49 timesteps
INFO  features 941ccfa8cda3fe51: computing 24 columns for 2000 nodes over 49 timesteps
INFO  fit 3/10: temporal xgb.base_gfp seed 0
INFO  xgb fit: 632 train (57 illicit, spw=10.1) in 0.2s, best_iteration=7, val aucpr=0.8856
...
INFO  wrote report/tables/smoke_results.csv (25 rows)
INFO  deployed model .../models/1a0f4132a4721d15.ubj, validation threshold 0.9081
INFO  batch 38: 51 nodes scored, 3 alerts at threshold 0.9081, health no_drift_flagged
smoke OK: report/tables/smoke_results.csv
alerts: report/tables/smoke_batch38_alerts.csv
health: report/tables/smoke_batch38_health.json
report: report/tables/smoke_batch38_report.html
```

The smoke run ends by scoring one batch the way `mulegraph score` does: a ranked alert queue, a
health check on whether to trust the scores, and an HTML report of both. Open the report in a
browser to see what a person receives.

Every experiment is one YAML file, and results go to files, not stdout:

```bash
uv run mulegraph run   --config configs/elliptic_mvp.yaml      # 50 fits, ~9 min on an RTX 4060
uv run mulegraph run   --config configs/amlworld_xgb.yaml      # 5.1M transactions, ~8 min, 5 GB RAM
uv run mulegraph drift --config configs/elliptic_drift.yaml    # label-free monitor, ~3.5 min
uv run mulegraph score --config configs/amlworld_score.yaml    # one day's alert queue + health check
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

The datasets are public but not redistributed: put
[Elliptic++](https://github.com/git-disl/EllipticPlusPlus) under `data/raw/elliptic_pp/2023.1/` and
[AMLworld HI-Small](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
under `data/raw/amlworld/hi_small/`. The GraphSAGE grid on AMLworld needs more memory than a
laptop; `kaggle/amlworld_sage.md` runs it on a free Kaggle T4 in 74 minutes.

## Engineering

```mermaid
flowchart LR
    A[YAML config<br/>Pydantic-validated] --> B[Loader<br/>shape + count checks]
    B --> C[Causal graph features<br/>only edges at or before t]
    C --> D[Split<br/>leakage assertions]
    D --> E[Models<br/>XGBoost / GraphSAGE]
    E --> F[Threshold<br/>validation only]
    F --> G[Metrics + per-timestep curves]
    G --> H[MLflow run<br/>commit, data, feature, split hashes]
    H --> I[CSV tables + figures]
    E --> J[Drift monitor<br/>never sees labels]
```

- **The rules that make the numbers trustworthy are enforced by tests.** A spy on the threshold
  function fails the build if it ever receives test rows. A synthetic multi-timestep graph checks
  that features at time *t* are identical with and without later edges. A test checks that no
  drift detector's signature accepts labels. Asking for accuracy as a metric raises an error.
- **Config-driven and reproducible.** One Typer CLI with three commands, Pydantic-validated YAML,
  caches keyed by content hash (dataset version, feature version, split hash), and every MLflow run
  tagged with the git commit. A run from an uncommitted tree is stamped `-dirty` all the way into
  the results CSV.
- **Quality gates.** 186 tests (87% line coverage without the real-data tests), ruff lint and
  format, and the smoke run on every push through GitHub Actions. Every change goes through a
  branch and a pull request.
- **Built for a laptop.** AMLworld's 5.1M transactions and 515k accounts run through feature
  building and XGBoost on a 7 GB machine in 8 minutes (peak 5 GB), feeding the graph-feature
  engine 425 hourly batches in time order.
- **Small, decoupled modules.** About 4,400 lines. Every component depends only on shared types in
  `types.py`; `pipeline.py` is the only module that imports across subsystems.

### Bugs worth telling

- **A library silently switched off a detector.** Importing snapml turns on flush-to-zero for the
  whole process, so the drift detector's PSI binning, which relied on a tiny subnormal offset, went
  blind to any shift in a constant column. It showed up as a test that passed alone and failed after
  the pipeline tests. Fixed by rewriting the bins so they need no subnormal numbers, with a
  regression test that imports snapml first.
- **The feature engine leaks the future if you call it the obvious way.** Loading the whole
  transaction table before scoring lets early transactions see later ones, and `partial_fit` then
  `transform` counts every edge twice. The only correct pattern, one batch at a time in time order,
  is now the only one the code uses, and a causality test guards it.
- **Re-running an experiment inflated confidence.** Re-running a config counted the earlier runs as
  extra seeds: five seeds run twice reported ten, and a confidence interval about 40% too narrow.
  Results tables are now scoped to one parent run.

## Results in context

F1 on the illicit class. XGBoost uses library defaults and is deterministic, so it has no interval;
GraphSAGE shows the mean ± 95% interval over seeds.

**Elliptic++**: 203k Bitcoin transactions over 49 time steps, 2% illicit.

| Model | Features | Random split (leaky) | Temporal split | Published, temporal |
|---|---|---|---|---|
| XGBoost | 93 transaction features | 0.950 | 0.722 | 0.694 (random forest, Weber 2019) |
| XGBoost | + graph features | 0.950 | 0.718 | — |
| XGBoost | all 165 (adds neighbour aggregates) | 0.959 | **0.778** | 0.788 (random forest, Weber 2019) |
| GraphSAGE | 93 transaction features | 0.907 | 0.591 ± 0.033 | 0.689 ± 0.017 (Maganti 2026) |
| GraphSAGE | + graph features | 0.900 | 0.560 ± 0.023 | — |

Published results test from t35; this repo validates on t35–37 and tests from t38.

**AMLworld HI-Small**: 5.1M simulated bank transactions, 515k accounts, 0.10% laundering.

| Model | Features | Temporal split | Published |
|---|---|---|---|
| XGBoost | 6 transaction fields | 0.209 | — |
| XGBoost | + graph features | **0.539** | 0.632 (GFP + XGBoost, Blanuša 2024, different split) |
| GraphSAGE | 6 transaction fields | 0.052 ± 0.007 | 0.568 (PNA, Blanuša 2024) |
| GraphSAGE | + graph features | 0.142 ± 0.031 | — |

With 1 laundering transaction in 1,000, flagging everything scores F1 0.002, so 0.54 is far from
trivial and within 0.1 of the published GFP + XGBoost result. A score near 0.95 on this kind of data usually
means the evaluation leaked, which the random-split column shows directly.
The per-timestep curves, confidence intervals and significance tests are in
[`docs/methods.md` §12](docs/methods.md#12-results).

## When criminal behaviour changes

![Fixed vs retrained model around the t43 shutdown](report/figures/readme_recovery.png)

*XGBoost with graph features on Elliptic. The shaded area is after the dark-market shutdown at t43.
PSI and KS on the features and on the scores never held a flag for two time steps. The alert rate
held one at t38, but that was alerts rising with the share of illicit transactions, not a warning
of the shutdown. Retraining each step uses labels with no delay, so it is an upper bound on what a
real retraining policy would get.*

After t43 the models score the new illicit transactions as safe with high confidence. The new
pattern is learnable (a model refit inside the post-shutdown window separates it at ROC-AUC 0.985),
but it is too small a share of each batch to move a whole-batch drift statistic. Details:
[`docs/methods.md` §12.4–12.6](docs/methods.md#124-drift-monitor-on-the-t43-shutdown-7a19cb9).

## Limitations

- Both datasets are public: Bitcoin transactions with anonymised features, and a bank simulator.
  No claim is made that the numbers transfer to a real bank. The protocol does.
- The AMLworld GraphSAGE model is a lower bound. It passes messages between account embeddings
  only, and transaction features reach it just at the output layer. Edge-aware GNNs such as PNA do
  much better, so the result is "XGBoost beats this GNN", not "XGBoost beats GNNs".
- No hyperparameter tuning: every model runs a fixed reference configuration.
- The drift result rests on one real event.

## More

- [`docs/design.md`](docs/design.md): the question, design decisions and component contracts.
- [`docs/methods.md`](docs/methods.md): the full method, every result, and related work.
- [`docs/changelog.md`](docs/changelog.md): what changed and why.

**How this was built.** Claude Code was my pair programmer, which is why `CLAUDE.md` and the
`Co-Authored-By` trailers are in the history. I set the question, chose the datasets and wrote the
integrity rules; the tests in `tests/` enforce them.

Datasets are public and pseudonymous or synthetic; no personal or bank data. Code is MIT.
