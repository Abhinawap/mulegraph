"""Results export: MLflow runs to the dissertation's per-config table (PR-O1, §2.3).

The reporter is the reason "regenerable from a tagged commit" (NFR-1) is a
property of the project rather than a promise. Nothing is recomputed here — the
table is a *query* over what the runs recorded, so a number in the write-up can
always be traced back to the run, split hash and commit that produced it.

Two deliberate constraints:

* **No import from ``mulegraph.eval``.** The interval is injected as a callable,
  so the reporter depends on the shared types and MLflow only (the component
  coupling rule in ``docs/architecture.md``). Swapping the interval definition is
  then a caller's decision, visible in the ``ci_kind`` column.
* **Provenance is checked, not assumed.** Rows that end up in the same table cell
  are averaged as if they were repeats of one experiment. If their feature
  version, split hash or commit differs, they are not, and the table would be
  comparing two definitions while claiming to compare two seeds — so that case
  warns loudly and the mixture is written into the cell rather than hidden.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("mulegraph")

#: The interval kind this table reports. Literal per spec §2.3; a bootstrap over
#: test ids is a per-timestep band and never appears in this column (PR-E3).
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

#: Where each field may have been logged. The MLflow schema in §2.3 names some of
#: these differently from the export columns, so the lookup is by alias rather
#: than by an exact key the pipeline would have to remember to match.
_ALIASES: dict[str, tuple[str, ...]] = {
    "dataset": ("dataset", "dataset_name"),
    "regime": ("regime", "split_regime"),
    "model": ("model", "model_name"),
    "features": ("features", "feature_set"),
    "feature_version": ("feature_version",),
    "split_hash": ("split_hash",),
    "commit": ("commit", "git_commit"),
}

#: Metric ordering for the tables; anything else follows, alphabetically.
_METRIC_ORDER = ("test_f1", "test_pr_auc", "test_roc_auc", "test_p_at_r50", "test_p_at_r80")

_MISSING = "unknown"


def _field(runs: pd.DataFrame, field: str) -> pd.Series:
    """Locate one logical field among MLflow's ``tags.``/``params.`` columns."""
    for alias in _ALIASES[field]:
        for column in (f"tags.{alias}", f"params.{alias}", alias):
            if column in runs.columns:
                return runs[column].astype("string").fillna(_MISSING)
    return pd.Series([_MISSING] * len(runs), index=runs.index, dtype="string")


def _provenance_value(values: pd.Series, field: str, group: tuple[str, ...]) -> str:
    """Collapse a provenance column over a group, warning if it is not constant."""
    unique = sorted(set(values.dropna().tolist()))
    if not unique:
        return _MISSING
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


def _fetch_runs(experiment: str, tracking_uri: str | None) -> pd.DataFrame:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment],
            filter_string="tags.kind = 'child'",
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
) -> Path:
    """Aggregate an experiment's seed runs into the per-config results table.

    Args:
        experiment: MLflow experiment name; its child runs (one per seed) are read.
        out_dir: Directory for ``<experiment>_results.csv`` and the wide Markdown
            pivot beside it. Created if absent.
        interval: ``values -> (mean, ci_low, ci_high)``. Defaults to
            ``mulegraph.eval.intervals.seed_interval`` (the 95% across-seed
            t-interval, PR-E3), imported lazily so ``report`` does not depend on
            ``eval``.
        tracking_uri: MLflow store; defaults to the ambient one.

    Returns:
        The path of the CSV written.

    Raises:
        ValueError: If the experiment has no child runs, or none of them logged a
            ``test_*`` metric.
    """
    if interval is None:
        from mulegraph.eval.intervals import seed_interval

        interval = seed_interval

    runs = _fetch_runs(experiment, tracking_uri)
    metric_columns = sorted(
        (c for c in runs.columns if c.startswith("metrics.test_")),
        key=lambda c: _metric_sort_key(c[len("metrics.") :]),
    )
    if not metric_columns:
        raise ValueError(
            f"MLflow experiment {experiment!r} has child runs but none logged a test_* "
            "metric; nothing to tabulate"
        )

    frame = pd.DataFrame({field: _field(runs, field) for field in _ALIASES})
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
