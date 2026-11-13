"""End-to-end orchestration on a synthetic graph (PR-E4, PR-O1).

The integrity assertion is that every threshold is the optimum of validation rows
alone: ``pipeline.fit`` is the one place test data could reach it.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple
from unittest import mock

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

#: One xdist worker for the module, so each shared run happens once (--dist loadgroup).
pytestmark = pytest.mark.xdist_group("pipeline")

METRICS = ["f1", "pr_auc", "p_at_r50"]
SEEDS = [0, 1]


def _config(name: str, root: Path) -> RunConfig:
    """A fresh config, so no expectation reads an object a run could have edited (PR-E4)."""
    temporal = {"regime": "temporal", "train_end": 6, "val": [7, 8], "test": [9, 12]}
    xgb = {"name": "xgb", "params": {"n_estimators": 20}}
    rolling = {**temporal, "regime": "temporal_rolling"}
    experiment, regimes, models = {
        "tiny_config": ("pipeline_test", [temporal], ["base", "base_gfp"]),
        # Four rolling refits per seed.
        "rolling_config": ("pipeline_rolling_test", [rolling], ["base"]),
        # All three regimes at once, as the Elliptic configs mix them.
        "multi_config": (
            "pipeline_multi_test",
            [{"regime": "random"}, temporal, rolling],
            ["base", "base_gfp"],
        ),
    }[name]
    return RunConfig.model_validate(
        {
            "dataset": {
                "name": "synthetic_elliptic",
                "version": "test",
                "synthetic": {"n_nodes": 600, "n_timesteps": 12, "illicit_rate": 0.2, "seed": 0},
            },
            "features": {"bins": [2, 4], "cycle_len": 4, "num_threads": 2},
            "split": {"regimes": regimes},
            "models": [{**xgb, "features": f} for f in models],
            "seeds": SEEDS,
            "eval": {"metrics": METRICS},
            "mlflow": {"experiment": experiment, "tracking_uri": f"sqlite:///{root / 'mlflow.db'}"},
            "device": "cpu",
        }
    )


@pytest.fixture(scope="module")
def shared_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One data, report and MLflow root for the module's shared runs."""
    root = tmp_path_factory.mktemp("pipeline")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MULEGRAPH_DATA_DIR", str(root / "data"))
        mp.setenv("MULEGRAPH_REPORT_DIR", str(root / "report"))
        yield root


class Run(NamedTuple):
    """What one benchmark run left behind; its tables and MLflow rows live under the root."""

    table: Path
    #: The config's model_dump() was unchanged by the run.
    config_kept: bool
    #: Each curves frame plot_curves was called with.
    drawn: tuple[pd.DataFrame, ...]


@pytest.fixture(scope="module")
def benchmark(shared_root: Path) -> Callable[[str], Run]:
    """Run each config at most once per module and share its outputs, never its config."""
    done: dict[str, Run] = {}

    def run(name: str) -> Run:
        if name not in done:
            cfg = _config(name, shared_root)
            pinned = cfg.model_dump()  # frozen stops assignment, not in-place edits
            with mock.patch.object(pipeline, "plot_curves", wraps=pipeline.plot_curves) as draw:
                table = pipeline.run_benchmark(cfg, pipeline.SMOKE_CONFIG)
            drawn = tuple(c.args[0].copy() for c in draw.call_args_list)
            done[name] = Run(table, cfg.model_dump() == pinned, drawn)
        return done[name]

    return run


@pytest.fixture
def tiny_config(shared_root: Path) -> RunConfig:
    return _config("tiny_config", shared_root)


@pytest.fixture
def rolling_config(shared_root: Path) -> RunConfig:
    return _config("rolling_config", shared_root)


def test_rolling_regime_writes_one_child_run_per_seed(benchmark: Callable[[str], Run]) -> None:
    """PR-E7: n_seeds counts (model, seed) child runs, not the internal per-step refits."""
    with open(benchmark("rolling_config").table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert {r["regime"] for r in rows} == {"temporal_rolling"}
    assert {r["n_seeds"] for r in rows} == {str(len(SEEDS))}
    # Rank metrics over stitched rows would pool incomparable score scales (PR-E7).
    assert {r["metric"] for r in rows} == {"test_f1"}


def test_rolling_curve_has_one_row_per_seed_and_test_timestep(
    benchmark: Callable[[str], Run],
) -> None:
    benchmark("rolling_config")
    with open(Paths.from_env().tables_dir() / "pipeline_rolling_test_curves.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(SEEDS) * 4
    assert {r["time"] for r in rows} == {str(t) for t in range(9, 13)}
    assert {r["regime"] for r in rows} == {"temporal_rolling"}


def test_rolling_test_idx_covers_every_labelled_node_in_the_window(
    rolling_config: RunConfig, benchmark: Callable[[str], Run]
) -> None:
    """The stitched test set across steps equals every labelled node in batches 9..12."""
    import mlflow

    benchmark("rolling_config")
    mlflow.set_tracking_uri(rolling_config.mlflow.resolved_uri())
    runs = mlflow.search_runs(
        experiment_names=["pipeline_rolling_test"], filter_string="tags.kind = 'child'"
    )
    path = mlflow.artifacts.download_artifacts(
        run_id=runs.iloc[0]["run_id"], artifact_path="predictions.parquet"
    )
    frame = pd.read_parquet(path)
    test_idx = set(frame.loc[frame["part"] == "test", "idx"].tolist())

    data = load_dataset(rolling_config.dataset, Paths.from_env().data_dir)
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


def test_run_benchmark_writes_the_results_table(
    benchmark: Callable[[str], Run], shared_root: Path
) -> None:
    table = benchmark("tiny_config").table

    assert table == shared_root / "report" / "tables" / "pipeline_test_results.csv"
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
    tiny_config: RunConfig, benchmark: Callable[[str], Run], shared_root: Path
) -> None:
    """PR-E5: one curve row per (fit, test timestep); the figure lands beside the tables."""
    benchmark("tiny_config")

    with open(shared_root / "report" / "tables" / "pipeline_test_curves.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    test_timesteps = 12 - 9 + 1
    assert len(rows) == tiny_config.total_fits() * test_timesteps
    assert {r["time"] for r in rows} == {str(t) for t in range(9, 13)}
    assert {r["seed"] for r in rows} == {str(s) for s in SEEDS}
    assert (shared_root / "report" / "figures" / "pipeline_test_curves.png").stat().st_size > 0


@pytest.mark.parametrize("config", ["tiny_config", "rolling_config", "multi_config"])
def test_every_logged_f1_is_scored_at_its_runs_validation_optimum(
    config: str, benchmark: Callable[[str], Run], shared_root: Path
) -> None:
    """PR-E4 end to end: each child run's threshold and every F1 it logs re-derive from its rows."""
    import mlflow
    from sklearn.metrics import f1_score, precision_score, recall_score

    shared = benchmark(config)
    assert shared.config_kept, "the run rewrote its config"
    cfg = _config(config, shared_root)
    mlflow.set_tracking_uri(cfg.mlflow.resolved_uri())
    client = mlflow.MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=[cfg.mlflow.experiment], filter_string="tags.kind = 'child'"
    )
    assert len(runs) == len(cfg.models) * len(cfg.seeds) * len(cfg.split.regimes)
    curves = pd.read_csv(Paths.from_env().tables_dir() / f"{cfg.mlflow.experiment}_curves.csv")
    assert len(shared.drawn) == 1
    data, _, cache_dir = _base_inputs(cfg)
    regimes = {r.regime: r for r in cfg.split.regimes}
    for run_id in runs["run_id"]:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="predictions.parquet"
        )
        frame = pd.read_parquet(path)
        val, test = frame[frame["part"] == "val"], frame[frame["part"] == "test"]
        run = client.get_run(run_id)
        logged, tags = run.data.metrics, run.data.tags
        regime = regimes[tags["regime"]]
        if regime.regime == "random":
            # No window to check: the stored rows are the partition's own, and disjoint.
            split = build_split(data, regime, cache_dir)
            assert not set(val["idx"]) & set(test["idx"])
            np.testing.assert_array_equal(val["idx"], split.val)
            np.testing.assert_array_equal(test["idx"], split.test)
        else:
            # On a rolling run the stored val rows and the scalar threshold are the last step's.
            last = rolling_steps(regime)[-1] if regime.regime == "temporal_rolling" else regime
            assert val["time"].between(*last.val).all()
            assert test["time"].between(*regime.test).all()
        assert (logged["threshold"], logged["val_f1"]) == choose_threshold(val["y"], val["proba"])
        # Each rolling test batch is decided at its own step's threshold, logged at step=t.
        by_step = {m.step: m.value for m in client.get_metric_history(run_id, "ts_threshold")}
        thr = test["time"].map(by_step).to_numpy() if by_step else logged["threshold"]
        y, time = test["y"].to_numpy(), test["time"].to_numpy()
        hits = test["proba"].to_numpy() >= thr
        assert logged["test_f1"] == pytest.approx(f1_score(y, hits))
        assert logged["diag_precision"] == pytest.approx(precision_score(y, hits))
        assert logged["diag_recall"] == pytest.approx(recall_score(y, hits))
        # The per-timestep curve, which drift's lead time reads, is scored at it too (PR-E5).
        per_batch = {
            int(t): f1_score(y[time == t], hits[time == t], zero_division=0.0)
            for t in np.unique(time)
        }
        ts_f1 = {m.step: m.value for m in client.get_metric_history(run_id, "ts_f1")}
        assert ts_f1 == pytest.approx(per_batch)
        # So are the curves table and the curves the figure draws (PR-E5).
        for table in (curves, shared.drawn[0]):
            mine = table[
                (table["regime"] == tags["regime"])
                & (table["model"] == tags["model"])
                & (table["features"] == tags["features"])
                & (table["seed"] == int(run.data.params["seed"]))
            ]
            assert dict(zip(mine["time"], mine["f1"], strict=True)) == pytest.approx(per_batch)


def test_every_logged_rolling_step_threshold_is_its_own_validation_optimum(
    rolling_config: RunConfig, benchmark: Callable[[str], Run]
) -> None:
    """PR-E4, PR-E7 end to end: each logged ts_threshold and ts_val_f1 is its step's own."""
    import mlflow

    benchmark("rolling_config")
    data, feats, cache_dir = _base_inputs(rolling_config)
    regime = rolling_config.split.regimes[0]
    steps = [build_split(data, s, cache_dir) for s in rolling_steps(regime)]
    mlflow.set_tracking_uri(rolling_config.mlflow.resolved_uri())
    client = mlflow.MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=[rolling_config.mlflow.experiment], filter_string="tags.kind = 'child'"
    )
    assert len(runs) == len(rolling_config.seeds)
    for run_id, seed in zip(runs["run_id"], runs["params.seed"], strict=True):
        # Only the last step's validation rows are stored, so each step is refitted alone.
        expected: dict[str, dict[int, float]] = {"ts_threshold": {}, "ts_val_f1": {}}
        for step in steps:
            alone = pipeline.fit(data, feats, [step], rolling_config.models[0], int(seed), "cpu")
            assert (alone.threshold, alone.val_f1) == choose_threshold(alone.val.y, alone.val.proba)
            t = step.params["test"][0]
            expected["ts_threshold"][t], expected["ts_val_f1"][t] = alone.threshold, alone.val_f1
        for name, by_step in expected.items():
            history = client.get_metric_history(run_id, name)
            assert {m.step: m.value for m in history} == by_step, name


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
    tiny_config: RunConfig, benchmark: Callable[[str], Run]
) -> None:
    """The reporter refuses a missing tag, so a run that logs one is caught here first (PR-O1)."""
    import mlflow

    benchmark("tiny_config")
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


def test_rerunning_a_config_does_not_pool_the_old_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-E3: the same two seeds run twice are two seeds, not four.

    Pooling them would report n = 4 and divide the half-width by sqrt(4) with t(3)
    instead of t(1) — a much tighter interval on no new evidence.
    """
    # Its own root: the second run must not add to the shared runs' store.
    monkeypatch.setenv("MULEGRAPH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MULEGRAPH_REPORT_DIR", str(tmp_path / "report"))
    pipeline.run_benchmark(_config("tiny_config", tmp_path), pipeline.SMOKE_CONFIG)
    table = pipeline.run_benchmark(_config("tiny_config", tmp_path), pipeline.SMOKE_CONFIG)

    with open(table, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["n_seeds"] for r in rows} == {str(len(SEEDS))}
