"""Per-timestep metric curves from one set of predictions (PR-E5)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mulegraph.eval.metrics import compute_metrics
from mulegraph.types import Predictions

CURVE_METRICS = ("f1", "pr_auc")


def per_timestep(
    pred: Predictions, threshold: float, which: Sequence[str] = CURVE_METRICS
) -> pd.DataFrame:
    """One row per timestep in ``pred``; ``threshold`` is given, never chosen here (PR-E4)."""
    rows = []
    for t in np.unique(pred.time):
        mask = pred.time == t
        scored = compute_metrics(pred.y[mask], pred.proba[mask], threshold, which)
        rows.append(
            {
                "time": int(t),
                **{name: scored[name] for name in which},
                "n": int(mask.sum()),
                "n_pos": int(scored["n_pos"]),
            }
        )
    return pd.DataFrame(rows, columns=["time", *which, "n", "n_pos"])
