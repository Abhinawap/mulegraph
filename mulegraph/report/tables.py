"""Results table: MLflow child runs -> per-config CSV + wide Markdown (PR-O1, spec §2.3)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("mulegraph")

#: Bootstrap bands are per-timestep only and never appear in this column (PR-E3).
CI_KIND = "seed_t"

#: Exactly the spec's per-config export columns, in order.
CSV_COLUMNS = [
    "dataset",
    "regime",
    "model",
    "features",
    "metric",
    "seed_mean",
    "ci_low",
    "ci_high",
    "ci_kind",
    "n_seeds",
    "feature_version",
    "split_hash",
    "commit",
]

GROUP_FIELDS = ("dataset", "regime", "model", "features")
PROVENANCE_FIELDS = ("feature_version", "split_hash", "commit")

#: Metric ordering for the tables; anything else follows, alphabetically.
_METRIC_ORDER = ("test_f1", "test_pr_auc", "test_roc_auc", "test_p_at_r50", "test_p_at_r80")


def _field(runs: pd.DataFrame, field: str) -> pd.Series:
    """One identity/provenance tag per run; a missing one is an error, never a guess (PR-O1)."""
    # Spec §2.3 logs the commit as ``git_commit``; every other column is its own tag name.
    tag = "git_commit" if field == "commit" else field
    column = f"tags.{tag}"
    if column not in runs.columns or runs[column].isna().any():
        raise ValueError(
            f"some child runs do not carry the {tag!r} tag; every run must log it (PR-O1), "
            "and guessing would pool different regimes, configs or definitions into one cell"
        )
    return runs[column].astype("string")


def _provenance_value(values: pd.Series, field: str, group: tuple[str, ...]) -> str:
    """Collapse a provenance column over a group, warning if it is not constant (NFR-1)."""
    unique = sorted(set(values.tolist()))
    if len(unique) > 1:
        log.warning(
            "%s differs within %s: %s. These runs were produced from different "
            "definitions and averaging them across seeds compares definitions, not seeds.",
            field,
            ".".join(group),
            ", ".join(unique),
        )
        return "|".join(unique)
    return unique[0]


def _metric_sort_key(metric: str) -> tuple[int, str]:
    if metric in _METRIC_ORDER:
        return (_METRIC_ORDER.index(metric), "")
    return (len(_METRIC_ORDER), metric)


def _fetch_runs(
    experiment: str, tracking_uri: str | None, parent_run_id: str | None = None
) -> pd.DataFrame:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # PR-E3: scoped to one parent run. Re-running a config into the same experiment
    # would otherwise pool the same seeds twice, narrowing the interval on no new
    # evidence — t(9)/sqrt(10) instead of t(4)/sqrt(5) for a five-seed protocol.
    filters = ["tags.kind = 'child'"]
    if parent_run_id:
        filters.append(f"tags.`mlflow.parentRunId` = '{parent_run_id}'")
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment],
            filter_string=" and ".join(filters),
        )
    except Exception as exc:  # mlflow raises for an experiment that does not exist
        raise ValueError(
            f"cannot read MLflow experiment {experiment!r} from "
            f"{mlflow.get_tracking_uri()!r}: {exc}"
        ) from exc
    if runs is None or len(runs) == 0:
        raise ValueError(
            f"MLflow experiment {experiment!r} at {mlflow.get_tracking_uri()!r} has no "
            "runs tagged kind='child'; run `mulegraph run --config ...` before reporting"
        )
    return runs


def write_results_table(
    experiment: str,
    out_dir: Path,
    interval: Callable[[Sequence[float]], tuple[float, float, float]] | None = None,
    tracking_uri: str | None = None,
    parent_run_id: str | None = None,
) -> Path:
    """Aggregate an experiment's seed runs into ``<experiment>_results.csv``; returns its path."""
    if interval is None:
        # Injected rather than imported at module level: report depends on types only.
        from mulegraph.eval.intervals import seed_interval

        interval = seed_interval

    runs = _fetch_runs(experiment, tracking_uri, parent_run_id)
    metric_columns = sorted(
        (c for c in runs.columns if c.startswith("metrics.test_")),
        key=lambda c: _metric_sort_key(c[len("metrics.") :]),
    )
    if not metric_columns:
        raise ValueError(
            f"MLflow experiment {experiment!r} has child runs but none logged a test_* "
            "metric; nothing to tabulate"
        )

    frame = pd.DataFrame(
        {field: _field(runs, field) for field in (*GROUP_FIELDS, *PROVENANCE_FIELDS)}
    )
    for column in metric_columns:
        frame[column] = pd.to_numeric(runs[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for group, block in frame.groupby(list(GROUP_FIELDS), sort=True, dropna=False):
        group = tuple(str(g) for g in group)
        provenance = {
            field: _provenance_value(block[field], field, group) for field in PROVENANCE_FIELDS
        }
        for column in metric_columns:
            values = block[column].dropna().tolist()
            if not values:
                continue
            mean, ci_low, ci_high = interval(values)
            rows.append(
                {
                    **dict(zip(GROUP_FIELDS, group, strict=True)),
                    "metric": column[len("metrics.") :],
                    "seed_mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_kind": CI_KIND,
                    "n_seeds": len(values),
                    **provenance,
                }
            )

    if not rows:
        raise ValueError(
            f"MLflow experiment {experiment!r} has child runs but every test_* metric is "
            "empty; nothing to tabulate"
        )

    table = pd.DataFrame(rows, columns=CSV_COLUMNS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{experiment}_results.csv"
    table.to_csv(csv_path, index=False)
    _write_wide_markdown(table, out_dir / f"{experiment}_results_wide.md", experiment)
    log.info("wrote %s (%d rows)", csv_path, len(table))
    return csv_path


def _format_cell(mean: float, ci_low: float, ci_high: float) -> str:
    """``mean ± half-width``; the half-width is dropped when a single seed makes it undefined."""
    if pd.isna(ci_low) or pd.isna(ci_high):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {(ci_high - ci_low) / 2:.4f}"


def _write_wide_markdown(table: pd.DataFrame, path: Path, experiment: str) -> Path:
    """Human-readable pivot: one section per (dataset, regime), one row per config."""
    metrics = sorted(table["metric"].unique(), key=_metric_sort_key)
    lines = [
        f"# {experiment} — results",
        "",
        f"Mean ± half-width of the 95% across-seed t-interval (`ci_kind = {CI_KIND}`).",
        "Bootstrap bands are per-timestep only and never appear here (PR-E3).",
        "",
    ]
    for (dataset, regime), block in table.groupby(["dataset", "regime"], sort=True):
        lines += [
            f"## {dataset} — {regime}",
            "",
            "| config | seeds | " + " | ".join(metrics) + " |",
            "|---|---|" + "---|" * len(metrics),
        ]
        for (model, features), config_block in block.groupby(["model", "features"], sort=True):
            by_metric = config_block.set_index("metric")
            cells = []
            for metric in metrics:
                if metric not in by_metric.index:
                    cells.append("—")
                    continue
                row = by_metric.loc[metric]
                cells.append(_format_cell(row["seed_mean"], row["ci_low"], row["ci_high"]))
            n_seeds = int(by_metric["n_seeds"].max())
            lines.append(f"| {model}.{features} | {n_seeds} | " + " | ".join(cells) + " |")
        lines.append("")

    path.write_text("\n".join(lines))
    return path
