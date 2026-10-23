"""End-to-end orchestration on a synthetic graph (PR-E4, PR-O1).

The integrity assertion is that every threshold is the optimum of validation rows
alone: ``pipeline.fit`` is the one place test data could reach it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulegraph import pipeline
from mulegraph.config import RunConfig
from mulegraph.data import load_dataset
from mulegraph.eval.threshold import choose_threshold
from mulegraph.features.select import select_features
from mulegraph.report.tables import CSV_COLUMNS
from mulegraph.splits.builder import build_split, rolling_steps
from mulegraph.types import FeatureMatrix, GraphDataset
from mulegraph.util import Paths

METRICS = ["f1", "pr_auc", "p_at_r50"]
SEEDS = [0, 1]


@pytest.fixture
def tiny_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    """Two xgb configs over two seeds, one temporal regime, on a 600-node synthetic graph."""
    monkeypatch.setenv("MULEGRAPH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MULEGRAPH_REPORT_DIR", str(tmp_path / "report"))
    return RunConfig.model_validate(
        {
            "dataset": {
                "name": "synthetic_elliptic",
                "version": "test",
                "synthetic": {"n_nodes": 600, "n_timesteps": 12, "illicit_rate": 0.2, "seed": 0},
            },
            "features": {"bins": [2, 4], "cycle_len": 4, "num_threads": 2},
            "split": {
                "regimes": [{"regime": "temporal", "train_end": 6, "val": [7, 8], "test": [9, 12]}]
            },
            "models": [
                {"name": "xgb", "features": "base", "params": {"n_estimators": 20}},
                {"name": "xgb", "features": "base_gfp", "params": {"n_estimators": 20}},
            ],
            "seeds": SEEDS,
            "eval": {"metrics": METRICS},
            "mlflow": {
                "experiment": "pipeline_test",
                "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            },
            "device": "cpu",
        }
    )


@pytest.fixture
def rolling_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    """One xgb config, four rolling refits per seed, on the same 600-node synthetic graph."""
    monkeypatch.setenv("MULEGRAPH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MULEGRAPH_REPORT_DIR", str(tmp_path / "report"))
    return RunConfig.model_validate(
        {
            "dataset": {
                "name": "synthetic_elliptic",
                "version": "test",
                "synthetic": {"n_nodes": 600, "n_timesteps": 12, "illicit_rate": 0.2, "seed": 0},
            },
            "features": {"bins": [2, 4], "cycle_len": 4, "num_threads": 2},
            "split": {
                "regimes": [
                    {"regime": "temporal_rolling", "train_end": 6, "val": [7, 8], "test": [9, 12]}
                ]
            },
            "models": [{"name": "xgb", "features": "base", "params": {"n_estimators": 20}}],
            "seeds": SEEDS,
            "eval": {"metrics": METRICS},
            "mlflow": {
                "experiment": "pipeline_rolling_test",
                "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            },
            "device": "cpu",
        }
    )


def test_rolling_regime_writes_one_child_run_per_seed(
    rolling_config: RunConfig, tmp_path: Path
) -> None:
    """PR-E7: n_seeds counts (model, seed) child runs, not the internal per-step refits."""
    table = pipeline.run_benchmark(rolling_config, pipeline.SMOKE_CONFIG)
    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert {r["regime"] for r in rows} == {"temporal_rolling"}
    assert {r["n_seeds"] for r in rows} == {str(len(SEEDS))}
    # Rank metrics over stitched rows would pool incomparable score scales (PR-E7).
    assert {r["metric"] for r in rows} == {"test_f1"}


def test_rolling_curve_has_one_row_per_seed_and_test_timestep(
    rolling_config: RunConfig, tmp_path: Path
) -> None:
    pipeline.run_benchmark(rolling_config, pipeline.SMOKE_CONFIG)
    with open(
        tmp_path / "report" / "tables" / "pipeline_rolling_test_curves.csv", newline=""
    ) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(SEEDS) * 4
    assert {r["time"] for r in rows} == {str(t) for t in range(9, 13)}
    assert {r["regime"] for r in rows} == {"temporal_rolling"}


def test_rolling_test_idx_covers_every_labelled_node_in_the_window(
    rolling_config: RunConfig, tmp_path: Path
) -> None:
    """The stitched test set across steps equals every labelled node in batches 9..12."""
    import mlflow

    pipeline.run_benchmark(rolling_config, pipeline.SMOKE_CONFIG)
    mlflow.set_tracking_uri(rolling_config.mlflow.resolved_uri())
    runs = mlflow.search_runs(
        experiment_names=["pipeline_rolling_test"], filter_string="tags.kind = 'child'"
    )
    path = mlflow.artifacts.download_artifacts(
        run_id=runs.iloc[0]["run_id"], artifact_path="predictions.parquet"
    )
    frame = pd.read_parquet(path)
    test_idx = set(frame.loc[frame["part"] == "test", "idx"].tolist())

    from mulegraph.util import Paths

    paths = Paths.from_env()
    data = pipeline.load_dataset(rolling_config.dataset, paths.data_dir)
    labelled = data.labelled_idx
    expected = set(
        labelled[(data.batch_id[labelled] >= 9) & (data.batch_id[labelled] <= 12)].tolist()
    )
    assert test_idx == expected


def _base_inputs(cfg: RunConfig) -> tuple[GraphDataset, FeatureMatrix, Path]:
    paths = Paths.from_env()
    data = load_dataset(cfg.dataset, paths.data_dir)
    cache_dir = paths.cache_dir(cfg.dataset.name, cfg.dataset.version)
    return data, select_features(data, None, "base"), cache_dir


def test_each_rolling_step_is_decided_at_its_own_validation_threshold(
    rolling_config: RunConfig,
) -> None:
    """PR-E4, PR-E7: a rolling fit is one single-step fit per test batch, stitched in order."""
    data, feats, cache_dir = _base_inputs(rolling_config)
    regime = rolling_config.split.regimes[0]
    steps = [build_split(data, s, cache_dir) for s in rolling_steps(regime)]
    model_cfg = rolling_config.models[0]

    rolling = pipeline.fit(data, feats, steps, model_cfg, 0, "cpu")

    assert list(rolling.steps) == list(range(9, 13))
    for step in steps:
        alone = pipeline.fit(data, feats, [step], model_cfg, 0, "cpu")
        assert alone.threshold == choose_threshold(alone.val.y, alone.val.proba)[0]
        t = step.params["test"][0]
        rows = rolling.test.time == t
        np.testing.assert_array_equal(rolling.test.idx[rows], alone.test.idx)
        np.testing.assert_array_equal(rolling.test.proba[rows], alone.test.proba)
        assert (rolling.test_threshold[rows] == alone.threshold).all()
        assert rolling.steps[t]["ts_threshold"] == alone.threshold
    # Drift and score apply one threshold: the last step's.
    assert rolling.threshold == alone.threshold


def test_run_benchmark_writes_the_results_table(tiny_config: RunConfig, tmp_path: Path) -> None:
    table = pipeline.run_benchmark(tiny_config, pipeline.SMOKE_CONFIG)

    assert table == tmp_path / "report" / "tables" / "pipeline_test_results.csv"
    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fh.seek(0)
        assert next(csv.reader(fh)) == CSV_COLUMNS

    # One row per (model, features, metric); both feature sets ran over both seeds.
    assert len(rows) == 2 * len(METRICS)
    assert {r["features"] for r in rows} == {"base", "base_gfp"}
    assert {r["metric"] for r in rows} == {f"test_{m}" for m in METRICS}
    assert {r["n_seeds"] for r in rows} == {str(len(SEEDS))}
    assert {r["regime"] for r in rows} == {"temporal"}
    # base_gfp carries the graph-feature version; base has none to carry.
    version_of = {r["features"]: r["feature_version"] for r in rows}
    assert version_of["base"] == "none"
    assert version_of["base_gfp"] != "none"
    assert len({r["split_hash"] for r in rows}) == 1


def test_run_benchmark_writes_per_timestep_curves_and_figure(
    tiny_config: RunConfig, tmp_path: Path
) -> None:
    """PR-E5: one curve row per (fit, test timestep); the figure lands beside the tables."""
    pipeline.run_benchmark(tiny_config, pipeline.SMOKE_CONFIG)

    with open(tmp_path / "report" / "tables" / "pipeline_test_curves.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    test_timesteps = 12 - 9 + 1
    assert len(rows) == tiny_config.total_fits() * test_timesteps
    assert {r["time"] for r in rows} == {str(t) for t in range(9, 13)}
    assert {r["seed"] for r in rows} == {str(s) for s in SEEDS}
    assert (tmp_path / "report" / "figures" / "pipeline_test_curves.png").stat().st_size > 0


@pytest.mark.parametrize("config", ["tiny_config", "rolling_config"])
def test_every_logged_test_f1_is_scored_at_its_runs_validation_optimum(
    config: str, request: pytest.FixtureRequest
) -> None:
    """PR-E4 end to end: each child run's threshold and test F1 re-derive from its predictions."""
    import mlflow
    from sklearn.metrics import f1_score

    cfg: RunConfig = request.getfixturevalue(config)
    pipeline.run_benchmark(cfg, pipeline.SMOKE_CONFIG)
    mlflow.set_tracking_uri(cfg.mlflow.resolved_uri())
    client = mlflow.MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=[cfg.mlflow.experiment], filter_string="tags.kind = 'child'"
    )
    assert len(runs) == len(cfg.models) * len(cfg.seeds)
    for run_id in runs["run_id"]:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="predictions.parquet"
        )
        frame = pd.read_parquet(path)
        val, test = frame[frame["part"] == "val"], frame[frame["part"] == "test"]
        logged = client.get_run(run_id).data.metrics
        # On a rolling run the stored val rows and the scalar threshold are the last step's.
        assert logged["threshold"] == choose_threshold(val["y"], val["proba"])[0]
        # Each rolling test batch is decided at its own step's threshold, logged at step=t.
        by_step = {m.step: m.value for m in client.get_metric_history(run_id, "ts_threshold")}
        thr = test["time"].map(by_step).to_numpy() if by_step else logged["threshold"]
        assert logged["test_f1"] == pytest.approx(f1_score(test["y"], test["proba"] >= thr))


def test_the_threshold_is_the_validation_optimum_applied_to_every_test_row(
    tiny_config: RunConfig,
) -> None:
    """PR-E4: fit() thresholds on its own validation rows; test rows are scored, never searched."""
    data, feats, cache_dir = _base_inputs(tiny_config)
    split = build_split(data, tiny_config.split.regimes[0], cache_dir)

    fitted = pipeline.fit(data, feats, [split], tiny_config.models[0], 0, "cpu")

    np.testing.assert_array_equal(fitted.val.idx, split.val)
    np.testing.assert_array_equal(fitted.test.idx, split.test)
    assert (fitted.threshold, fitted.val_f1) == choose_threshold(fitted.val.y, fitted.val.proba)
    assert (fitted.test_threshold == fitted.threshold).all()
    assert fitted.split_hash == split.split_hash
    assert fitted.steps == {}


def test_every_child_run_carries_its_provenance_and_an_honest_trial_count(
    tiny_config: RunConfig,
) -> None:
    """The reporter refuses a missing tag, so a run that logs one is caught here first (PR-O1)."""
    import mlflow

    pipeline.run_benchmark(tiny_config, pipeline.SMOKE_CONFIG)
    mlflow.set_tracking_uri(tiny_config.mlflow.resolved_uri())
    runs = mlflow.search_runs(
        experiment_names=["pipeline_test"], filter_string="tags.kind = 'child'"
    )

    assert len(runs) == tiny_config.total_fits()
    for tag in (
        "dataset",
        "dataset_version",
        "regime",
        "model",
        "features",
        "feature_version",
        "split_hash",
        "git_commit",
    ):
        assert runs[f"tags.{tag}"].notna().all()
    assert set(runs["params.trials_completed"]) == {"0"}
    assert runs["metrics.test_f1"].notna().all()


def test_rerunning_a_config_does_not_pool_the_old_seeds(tiny_config: RunConfig) -> None:
    """PR-E3: the same two seeds run twice are two seeds, not four.

    Pooling them would report n = 4 and divide the half-width by sqrt(4) with t(3)
    instead of t(1) — a much tighter interval on no new evidence.
    """
    pipeline.run_benchmark(tiny_config, pipeline.SMOKE_CONFIG)
    table = pipeline.run_benchmark(tiny_config, pipeline.SMOKE_CONFIG)

    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["n_seeds"] for r in rows} == {str(len(SEEDS))}
