"""Dump every results CSV, health file and MLflow child run, to diff across a refactor (NFR-1)."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

from mulegraph import pipeline
from mulegraph.config import DriftRunConfig, RunConfig, ScoreConfig, load_config
from mulegraph.util import setup_logging

# The 600-node graph and split the tests use.
SYNTHETIC: dict[str, Any] = {
    "dataset": {
        "name": "synthetic_elliptic",
        "version": "test",
        "synthetic": {"n_nodes": 600, "n_timesteps": 12, "illicit_rate": 0.2, "seed": 0},
    },
    "features": {"bins": [2, 4], "cycle_len": 4, "num_threads": 2},
    "device": "cpu",
}
SPLIT = {"train_end": 6, "val": [7, 8], "test": [9, 12]}
MODELS = [
    {"name": "xgb", "features": "base", "params": {"n_estimators": 20}},
    {"name": "xgb", "features": "base_gfp", "params": {"n_estimators": 20}},
]


def run_all() -> None:
    """Smoke benchmark and score, then a rolling benchmark, a drift run and two score runs."""
    smoke = load_config(pipeline.SMOKE_CONFIG)
    pipeline.run_benchmark(smoke, pipeline.SMOKE_CONFIG)
    pipeline._smoke_score(smoke)

    script = Path(__file__)
    fits = {**SYNTHETIC, "seeds": [0, 1], "models": MODELS}
    rolling = {"regimes": [{"regime": "temporal_rolling", **SPLIT}]}
    temporal = {"regimes": [{"regime": "temporal", **SPLIT}]}
    pipeline.run_benchmark(
        RunConfig.model_validate(
            {**fits, "split": rolling, "mlflow": {"experiment": "dump_rolling"}}
        ),
        script,
    )
    pipeline.run_drift(
        DriftRunConfig.model_validate(
            {
                **fits,
                "split": temporal,
                "drift": {"event": 11},
                "mlflow": {"experiment": "dump_drift"},
            }
        ),
        script,
    )
    score = ScoreConfig.model_validate(
        {
            **SYNTHETIC,
            "split": temporal,
            "name": "dump_score",
            "model": MODELS[1],
            "health": {"calibrate": True},
            "batch": 10,
        }
    )
    # The same config twice, so the second call reuses the first's deployed model.
    for call in ("fit", "reuse"):
        alerts, health, _, _ = pipeline.run_score(score)
        for path in (alerts, health):
            path.rename(path.with_name(f"{call}_{path.name}"))


def drop_commit(obj: Any) -> Any:
    """``obj`` without ``commit`` keys at any depth."""
    if isinstance(obj, dict):
        return {k: drop_commit(v) for k, v in obj.items() if k != "commit"}
    if isinstance(obj, list):
        return [drop_commit(v) for v in obj]
    return obj


def dump_tables(tables_dir: Path, out: Path) -> None:
    """Every CSV without its ``commit`` column, cells copied as text; every health JSON."""
    for path in sorted(tables_dir.glob("*.csv")):
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        keep = [i for i, column in enumerate(rows[0]) if column != "commit"]
        with open(out / path.name, "w", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerows([row[i] for i in keep] for row in rows)
    for path in sorted(tables_dir.glob("*_health.json")):
        report = drop_commit(json.loads(path.read_text()))
        (out / path.name).write_text(json.dumps(report, indent=2) + "\n")


def dump_children(tracking_uri: str, out: Path) -> None:
    """Every child run's params, tags and full metric histories, keyed by experiment/run name."""
    client = MlflowClient(tracking_uri)
    children: dict[str, Any] = {}
    for exp in client.search_experiments():
        for run in client.search_runs([exp.experiment_id], "tags.kind = 'child'"):
            key = f"{exp.name}/{run.info.run_name}"
            if key in children:
                raise ValueError(f"two child runs are named {key}, so one would hide the other")
            history = client.get_metric_history
            children[key] = {
                "params": run.data.params,
                "tags": {
                    k: v
                    for k, v in run.data.tags.items()
                    if k != "git_commit" and not k.startswith("mlflow.")
                },
                "metrics": {
                    name: sorted((m.step, m.value) for m in history(run.info.run_id, name))
                    for name in run.data.metrics
                    if name != "fit_seconds"
                },
            }
    text = json.dumps(children, indent=1, sort_keys=True)
    (out / "mlflow_children.json").write_text(text + "\n")


def main(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty, and a file left from an older dump would show in diff -r")
    setup_logging()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mulegraph-dump-") as tmp:
        tracking_uri = f"sqlite:///{tmp}/mlflow.db"
        os.environ.update(
            {
                "MULEGRAPH_DATA_DIR": f"{tmp}/data",
                "MULEGRAPH_REPORT_DIR": f"{tmp}/report",
                "MLFLOW_TRACKING_URI": tracking_uri,
            }
        )
        run_all()
        dump_tables(Path(tmp) / "report" / "tables", out)
        dump_children(tracking_uri, out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run python scripts/dump_outputs.py OUT_DIR")
    main(Path(sys.argv[1]))
