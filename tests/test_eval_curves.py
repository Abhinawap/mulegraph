"""Per-timestep curves equal the metric on each timestep's slice (PR-E5)."""

from __future__ import annotations

import numpy as np

from mulegraph.eval.curves import per_timestep
from mulegraph.eval.metrics import compute_metrics
from mulegraph.types import Predictions


def _pred() -> Predictions:
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int64)
    p = np.array([0.9, 0.2, 0.7, 0.4, 0.1, 0.3, 0.6, 0.2], dtype=np.float32)
    time = np.array([5, 5, 5, 5, 6, 6, 6, 6], dtype=np.int64)
    return Predictions(idx=np.arange(8, dtype=np.int64), proba=p, y=y, time=time)


def test_rows_match_compute_metrics_on_each_slice() -> None:
    pred = _pred()
    table = per_timestep(pred, threshold=0.5)

    assert table["time"].tolist() == [5, 6]
    first = compute_metrics(pred.y[:4], pred.proba[:4], 0.5, ("f1", "pr_auc"))
    assert table.loc[0, "f1"] == first["f1"]
    assert table.loc[0, "pr_auc"] == first["pr_auc"]
    assert table.loc[0, "n"] == 4 and table.loc[0, "n_pos"] == 2


def test_timestep_without_positives_is_zero_f1_and_nan_pr_auc() -> None:
    table = per_timestep(_pred(), threshold=0.5)
    row = table.set_index("time").loc[6]
    assert row["f1"] == 0.0
    assert np.isnan(row["pr_auc"])
    assert row["n_pos"] == 0
