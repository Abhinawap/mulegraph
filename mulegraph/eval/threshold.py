"""Decision threshold from the validation PR curve only, never test (PR-E4)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


# PR-E4: the signature admits validation arrays only, so test data has no way in.
def choose_threshold(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    """Return ``(threshold, val_f1)`` maximising positive-class F1 on validation."""
    y = np.asarray(y_val).ravel()
    p = np.asarray(p_val, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_val has shape {y.shape} but p_val has shape {p.shape}")
    if y.size == 0:
        raise ValueError("cannot choose a threshold on an empty validation set")
    if (y < 0).any():
        raise ValueError(
            f"{int((y < 0).sum())} unlabelled nodes (y == -1) in the validation set; "
            "the threshold must be chosen on labelled validation data only"
        )
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError(
            "validation set contains no illicit nodes, so positive-class F1 is undefined "
            "at every threshold; widen the validation window rather than falling back to test"
        )

    precision, recall, thresholds = precision_recall_curve(y, p)
    # The curve's last point (precision 1, recall 0) has no threshold behind it.
    precision = precision[: thresholds.size]
    recall = recall[: thresholds.size]

    denom = precision + recall
    f1 = np.zeros_like(denom)
    np.divide(2.0 * precision * recall, denom, out=f1, where=denom > 0)

    best = float(f1.max())
    # thresholds ascends, so the last tie is the highest threshold: fewest alerts.
    best_idx = int(np.flatnonzero(f1 == f1.max())[-1])
    return float(thresholds[best_idx]), best
