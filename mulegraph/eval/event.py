"""Test window on each side of a known event: model transfer and a refit probe (PR-R5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from mulegraph.eval.metrics import compute_metrics
from mulegraph.types import Predictions

PROBE_FOLDS = 5


def event_windows(pred: Predictions, threshold: float, event: int) -> pd.DataFrame:
    """Per side of ``event``: counts, ROC-AUC, recall at ``threshold``, median illicit score."""
    rows = []
    for window, mask in (("before", pred.time < event), ("after", pred.time >= event)):
        y, p = pred.y[mask], pred.proba[mask]
        scored = compute_metrics(y, p, threshold, ("roc_auc",))
        rows.append(
            {
                "window": window,
                "n": int(scored["n"]),
                "n_pos": int(scored["n_pos"]),
                "roc_auc": scored["roc_auc"],
                "recall": scored["recall"],
                "pos_median": float(np.median(p[y == 1])) if scored["n_pos"] else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def probe_auc(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Stratified CV ROC-AUC of XGBoost defaults refit inside one window alone."""
    n_pos = int((y == 1).sum())
    if min(n_pos, int((y == 0).sum())) < PROBE_FOLDS:
        raise ValueError(
            f"the window has {n_pos} illicit of {y.size} labelled units; the probe needs at "
            f"least {PROBE_FOLDS} of each class for {PROBE_FOLDS}-fold CV"
        )
    folds = StratifiedKFold(PROBE_FOLDS, shuffle=True, random_state=seed)
    clf = XGBClassifier(random_state=seed)
    return float(cross_val_score(clf, x, y, cv=folds, scoring="roc_auc").mean())
