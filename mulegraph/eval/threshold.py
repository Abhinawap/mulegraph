"""Decision-threshold selection (PR-E4).

The single most safety-critical function in the project. Every headline number —
fraud F1, precision, recall — is read off one operating point, and if that point
is chosen by looking at the test set then the whole results table is a report of
how well the *threshold* was tuned, not how well the model detects fraud. Under a
temporal split it is worse than optimistic: it hands the model knowledge of a
future it is supposed to be predicting.

The defence is structural rather than procedural. ``choose_threshold`` takes
validation arrays only, so there is no parameter through which test data could
arrive, and no code path in which "just this once" is expressible.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def choose_threshold(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    """Pick the operating point that maximises positive-class F1 on validation.

    The threshold is chosen on the **validation** PR curve and never on test
    (PR-E4). Choosing it on test would invalidate every number in the
    dissertation, which is why this function's signature admits validation
    arrays only; the caller applies the returned threshold to test scores.

    On ties the **higher** threshold wins. Two thresholds can reach the same F1
    with different alert volumes, and the operationally sensible choice is fewer
    alerts for the same detection quality — an analyst queue is the scarce
    resource.

    Args:
        y_val: Validation labels, 1 illicit / 0 licit. Unlabelled nodes (-1)
            must already have been removed by the split builder.
        p_val: Validation scores, P(illicit), aligned with ``y_val``.

    Returns:
        ``(threshold, val_f1)`` — the chosen threshold and the F1 it achieved on
        validation. ``val_f1`` is a selection diagnostic, not a headline metric;
        it is logged so a run where validation and test disagree is visible.

    Raises:
        ValueError: If the arrays disagree in shape, contain an unlabelled node,
            or contain no positives (F1 is undefined with nothing to detect).
    """
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
    # The curve's last point (precision 1, recall 0) has no threshold behind it;
    # trim to the points that correspond to an actual decision rule.
    precision = precision[: thresholds.size]
    recall = recall[: thresholds.size]

    denom = precision + recall
    f1 = np.zeros_like(denom)
    np.divide(2.0 * precision * recall, denom, out=f1, where=denom > 0)

    best = float(f1.max())
    # thresholds is ascending, so the last index among the ties is the highest
    # threshold — the fewest-alerts tie-break.
    best_idx = int(np.flatnonzero(f1 == f1.max())[-1])
    return float(thresholds[best_idx]), best
