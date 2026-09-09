"""Results export tests (PR-O1, spec §2.3).

Runs are logged to a throwaway MLflow file store rather than mocked: the column
naming the reporter has to survive (``metrics.``/``tags.``/``params.`` prefixes)
is MLflow's, not ours, so a mock would test the wrong thing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mulegraph.eval.intervals import seed_interval
from mulegraph.report.tables import CSV_COLUMNS, write_results_table

EXPERIMENT = "unit_test_experiment"
SEEDS = (0, 1, 2)
METRICS = ("test_f1", "test_pr_auc", "test_roc_auc", "test_p_at_r50", "test_p_at_r80")

#: Deterministic per-(model, seed, metric) values so the expected mean and CI can
#: be recomputed independently of the reporter.
VALUES = {
    ("xgb", 0): {
        "test_f1": 0.70,
        "test_pr_auc": 0.60,
        "test_roc_auc": 0.90,
        "test_p_at_r50": 0.80,
        "test_p_at_r80": 0.40,
    },
    ("xgb", 1): {
        "test_f1": 0.72,
        "test_pr_auc": 0.63,
        "test_roc_auc": 0.91,
        "test_p_at_r50": 0.82,
        "test_p_at_r80": 0.43,
    },
    ("xgb", 2): {
        "test_f1": 0.74,
        "test_pr_auc": 0.66,
        "test_roc_auc": 0.92,
        "test_p_at_r50": 0.84,
        "test_p_at_r80": 0.46,
    },
    ("sage", 0): {
        "test_f1": 0.60,
        "test_pr_auc": 0.50,
        "test_roc_auc": 0.85,
        "test_p_at_r50": 0.70,
        "test_p_at_r80": 0.30,
    },
    ("sage", 1): {
        "test_f1": 0.65,
        "test_pr_auc": 0.55,
        "test_roc_auc": 0.86,
        "test_p_at_r50": 0.71,
        "test_p_at_r80": 0.31,
    },
    ("sage", 2): {
        "test_f1": 0.55,
        "test_pr_auc": 0.45,
        "test_roc_auc": 0.87,
        "test_p_at_r50": 0.72,
        "test_p_at_r80": 0.32,
    },
}


@pytest.fixture(autouse=True)
def allow_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs are recorded in a local SQLite file (see ``MLflowConfig``).

    The project's whole tracking design is a local file store (spec §2.1, no
    server), so the same variable has to be set wherever the pipeline opens a
    run; it is set here so these tests do not depend on the developer's shell.
    """
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")


def log_runs(tracking_uri: str, commit: str = "abc1234") -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    for (model, seed), values in VALUES.items():
        with mlflow.start_run(run_name=f"{model}-{seed}"):
            mlflow.set_tags(
                {
                    "kind": "child",
                    "dataset": "elliptic_pp",
                    "regime": "temporal",
                    "model": model,
                    "features": "base",
                    "feature_version": "fv0123",
                    "split_hash": "sh4567",
                    "git_commit": commit,
                }
            )
            mlflow.log_param("seed", seed)
            mlflow.log_metrics(values)
    # A parent run with no `kind=child` tag must be ignored by the query.
    with mlflow.start_run(run_name="parent"):
        mlflow.set_tags({"dataset": "elliptic_pp", "regime": "temporal", "model": "xgb"})
        mlflow.log_metric("test_f1", 999.0)


@pytest.fixture
def store(tmp_path: Path) -> str:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    log_runs(uri)
    return uri


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_csv_has_exactly_the_spec_columns(store: str, tmp_path: Path) -> None:
    csv_path = write_results_table(EXPERIMENT, tmp_path / "tables", tracking_uri=store)
    assert csv_path.name == f"{EXPERIMENT}_results.csv"
    with open(csv_path, newline="") as fh:
        header = next(csv.reader(fh))
    assert header == CSV_COLUMNS


def test_one_row_per_config_and_metric(store: str, tmp_path: Path) -> None:
    rows = read_csv(write_results_table(EXPERIMENT, tmp_path / "tables", tracking_uri=store))
    # 2 models x 1 feature set x 1 regime x 1 dataset x 5 metrics
    assert len(rows) == 2 * len(METRICS)
    assert {r["model"] for r in rows} == {"xgb", "sage"}
    assert {r["metric"] for r in rows} == set(METRICS)
    assert {r["ci_kind"] for r in rows} == {"seed_t"}
    assert {r["n_seeds"] for r in rows} == {str(len(SEEDS))}
    assert {r["feature_version"] for r in rows} == {"fv0123"}
    assert {r["split_hash"] for r in rows} == {"sh4567"}
    assert {r["commit"] for r in rows} == {"abc1234"}
    assert {r["dataset"] for r in rows} == {"elliptic_pp"}
    assert {r["regime"] for r in rows} == {"temporal"}


def test_mean_and_ci_match_seed_interval(store: str, tmp_path: Path) -> None:
    rows = read_csv(write_results_table(EXPERIMENT, tmp_path / "tables", tracking_uri=store))
    for row in rows:
        expected_values = [VALUES[(row["model"], seed)][row["metric"]] for seed in SEEDS]
        mean, low, high = seed_interval(expected_values)
        assert float(row["seed_mean"]) == pytest.approx(mean)
        assert float(row["ci_low"]) == pytest.approx(low)
        assert float(row["ci_high"]) == pytest.approx(high)


def test_parent_run_is_excluded(store: str, tmp_path: Path) -> None:
    """The parent run's sentinel 999.0 would move every mean if the filter leaked."""
    rows = read_csv(write_results_table(EXPERIMENT, tmp_path / "tables", tracking_uri=store))
    assert all(float(r["seed_mean"]) <= 1.0 for r in rows)


def test_wide_markdown_is_written(store: str, tmp_path: Path) -> None:
    out_dir = tmp_path / "tables"
    write_results_table(EXPERIMENT, out_dir, tracking_uri=store)
    wide = out_dir / f"{EXPERIMENT}_results_wide.md"
    text = wide.read_text()
    assert "xgb.base" in text and "sage.base" in text
    assert "±" in text
    assert "test_f1" in text
    # 0.70, 0.72, 0.74 -> mean 0.72
    assert "0.7200" in text


def test_injected_interval_is_used(store: str, tmp_path: Path) -> None:
    def fixed(values):
        return (0.5, 0.4, 0.6)

    rows = read_csv(
        write_results_table(EXPERIMENT, tmp_path / "t", interval=fixed, tracking_uri=store)
    )
    assert {float(r["seed_mean"]) for r in rows} == {0.5}


def test_mixed_provenance_warns_and_is_written_into_the_cell(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Runs from different definitions are not seeds of one experiment (NFR-1)."""
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("mixed_experiment")
    for seed, version in enumerate(("fv_old", "fv_new")):
        with mlflow.start_run():
            mlflow.set_tags(
                {
                    "kind": "child",
                    "dataset": "elliptic_pp",
                    "regime": "temporal",
                    "model": "xgb",
                    "features": "base_gfp",
                    "feature_version": version,
                    "split_hash": "sh4567",
                    "git_commit": "abc1234",
                }
            )
            mlflow.log_param("seed", seed)
            mlflow.log_metric("test_f1", 0.5 + 0.1 * seed)

    with caplog.at_level("WARNING", logger="mulegraph"):
        rows = read_csv(write_results_table("mixed_experiment", tmp_path / "t", tracking_uri=uri))
    assert "feature_version differs" in caplog.text
    assert rows[0]["feature_version"] == "fv_new|fv_old"


def test_empty_experiment_raises_rather_than_writing_a_file(tmp_path: Path) -> None:
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("empty_experiment")
    out_dir = tmp_path / "tables"
    with pytest.raises(ValueError, match="no runs tagged"):
        write_results_table("empty_experiment", out_dir, tracking_uri=uri)
    assert not (out_dir / "empty_experiment_results.csv").exists()


def test_unknown_experiment_raises_a_clear_error(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with pytest.raises(ValueError, match="never_created"):
        write_results_table("never_created", tmp_path / "tables", tracking_uri=uri)
