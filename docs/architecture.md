# Architecture

Expands the summary in [CLAUDE.md](../CLAUDE.md). Authoritative source for design rationale is [project_spec.md](project_spec.md) §2.

## Principle

The orchestrator (`mulegraph/pipeline.py`) is the only module that knows about all the others. Every other component depends solely on the shared types in `mulegraph/types.py`. This keeps subsystems independently testable and lets the drift/policy work (v2) reuse the benchmark path without importing it.

## Shared types (`mulegraph/types.py`)

| Type | Carries |
|---|---|
| `GraphDataset` | `x`, `edge_index`, `edge_attr`, `edge_time`/`node_time`, `batch_id`, `y`, `task`, `meta` |
| `Split` | train/val/test index arrays + `split_hash` |
| `FeatureMatrix` | per-id feature table + `feature_version` |
| `Predictions` | probabilities, ids, optional embeddings |
| `DriftSignal` | detector, batch_id, score, flagged, threshold |

`meta` always records `dataset`, `version`, `cross_time_edges: bool`, `source_url`, and (AMLworld) typology labels. `meta.feature_blocks` exposes Elliptic's `local: 0..92` / `agg1hop: 93..164` split so `base` selection is explicit and logged.

## Component contracts

| Component | Module | In | Out |
|---|---|---|---|
| Loaders | `data/` | dataset name, version | `GraphDataset` (cached) |
| Feature builder | `features/` | `GraphDataset`, window config | `FeatureMatrix` |
| Split builder | `splits/` | `GraphDataset`, regime config | `Split` |
| Models | `models/` | `GraphDataset`, `FeatureMatrix`, `Split`, seed | `Predictions`, embeddings, fitted model |
| Evaluator | `eval/` | `Predictions`, `Split` | metric records, figures |
| Drift monitor | `drift/` | reference window, batches, model | `DriftSignal` per batch |
| Policy simulator | `policies/` | model class, `Split`, signals, policy config | cumulative metrics, retrain count |
| Reporter | `report/` | milestone tag | CSV tables, PNG figures |

## Python protocols

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
    def training_window(self, t: int, label_lag: int) -> tuple[int, int]: ...
```

## Benchmark run sequence

1. CLI parses `--config`; Pydantic validates.
2. Orchestrator seeds Python/NumPy/PyTorch/CUDA, opens an MLflow run, logs config + git commit + dataset version.
3. Loader returns cached `GraphDataset` or builds and caches it.
4. Feature builder checks cache on `(dataset_version, feature_version)`; on miss computes causal features.
5. Split builder builds indices, runs leakage assertions, logs `split_hash`, saves `.npy`.
6. **Search phase**, once per (model config, regime, dataset): trial 0 = fixed reference config, then Optuna at `search_seed` until the shared wall-clock cap `W` or the per-model trial ceiling. Scored on validation PR-AUC.
7. **Seed phase**: refit best config per seed; `predict_proba` on val and test; threshold from the validation PR curve.
8. Evaluator computes metrics, seed t-intervals, paired gaps, per-timestep curves. Logged as a child run per seed.
9. Reporter aggregates later, on demand.

## Drift / policy sequence (v2)

1. `mulegraph drift` loads a fitted model artifact by MLflow run id.
2. Training period is the reference window; test timesteps are iterated as batches; each detector scores without labels.
3. `DriftSignal`s logged per batch; lead time computed against the original run's per-timestep F1.
4. `mulegraph simulate` replays the same timesteps under each policy, calling `fit` on data available at that timestep under label lag `L`. One MLflow child run per policy.

## Storage

No database. Files keyed by content hash, plus the MLflow file store.

```
data/raw/<dataset>/<version>/...                          never modified
data/cache/<dataset>/<version>/graph.pt
data/cache/<dataset>/<version>/features/<feature_version>.parquet
data/cache/<dataset>/<version>/splits/<split_hash>/{train,val,test}.npy
```

- `feature_version` = sha256 of (feature list, window config, backend, dataset version).
- Drift signals → parquet: `run_id, detector, batch_id, batch_unit, score, flagged, threshold`
- Policy results → parquet: `run_id, policy, cumulative_f1, retrain_count, retrain_times, train_seconds`
- Results export → CSV, per config and per gap (see spec §2.3).

## Dataset-specific behaviour

| | Elliptic++ | AMLworld HI-Small |
|---|---|---|
| Task | node (transaction) | edge |
| `cross_time_edges` | `False` | `True` |
| Valid regimes | random, temporal | random, temporal, temporal+inductive |
| `base` features | 93 local | raw transaction fields |
| Batch unit | 1 timestep (~2 weeks) | 6 hours |
| Default label lag | 3 batches | 4 batches |
| Causality test | skipped (vacuous) | synthetic fixture |

Because Elliptic's timesteps are disconnected components, its temporal split is inductive by construction — the three-regime contrast only exists on AMLworld.
