"""Before/after-event rows match hand-computed values; the probe refuses a thin window (PR-R5)."""

from __future__ import annotations

import numpy as np
import pytest

from mulegraph.eval.event import event_windows, probe_auc
from mulegraph.types import Predictions


def test_windows_split_at_the_event_batch() -> None:
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    p = np.array([0.9, 0.2, 0.8, 0.3, 0.1, 0.6, 0.7, 0.2], dtype=np.float32)
    time = np.array([5, 5, 6, 6, 7, 7, 8, 8], dtype=np.int64)
    pred = Predictions(idx=np.arange(8, dtype=np.int64), proba=p, y=y, time=time)

    rows = event_windows(pred, threshold=0.5, event=7).set_index("window")

    before, after = rows.loc["before"], rows.loc["after"]
    assert (before["n"], before["n_pos"]) == (4, 2) and (after["n"], after["n_pos"]) == (4, 2)
    assert before["roc_auc"] == 1.0 and before["recall"] == 1.0
    assert before["pos_median"] == pytest.approx(0.85)
    # after: illicit 0.1, 0.6 vs licit 0.7, 0.2 -> 1 of 4 pairs ranked right
    assert after["roc_auc"] == 0.25 and after["recall"] == 0.5
    assert after["pos_median"] == pytest.approx(0.35)


def test_probe_separates_a_learnable_window_and_refuses_a_thin_one() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 50)
    x = rng.normal(size=(100, 3)) + y[:, None] * 3.0
    assert probe_auc(x, y, seed=0) > 0.95
    with pytest.raises(ValueError, match="3 illicit"):
        probe_auc(x[:53], y[:53], seed=0)
