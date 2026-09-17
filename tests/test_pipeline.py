"""End-to-end orchestration on a synthetic graph (PR-E4, PR-O1).

The integrity assertion is that ``choose_threshold`` only ever sees validation
rows: the pipeline is the one place test data could reach it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mulegraph import pipeline
from mulegraph.config import RunConfig
from mulegraph.report.tables import CSV_COLUMNS

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


def test_threshold_is_never_chosen_on_test_rows(
    tiny_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-E4: every threshold call must be the validation set, exactly."""
    from mulegraph.splits.builder import build_split
    from mulegraph.util import Paths

    seen: list[int] = []
    real = pipeline.choose_threshold

    def spy(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
        seen.append(int(np.asarray(y_val).size))
        return real(y_val, p_val)

    monkeypatch.setattr(pipeline, "choose_threshold", spy)
    pipeline.run_benchmark(tiny_config, pipeline.SMOKE_CONFIG)

    paths = Paths.from_env()
    data = pipeline.load_dataset(tiny_config.dataset, paths.data_dir)
    split = build_split(
        data,
        tiny_config.split.regimes[0],
        paths.cache_dir(tiny_config.dataset.name, tiny_config.dataset.version),
    )
    assert seen == [split.val.size] * tiny_config.total_fits()
    assert split.test.size not in seen


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
