"""Detection metrics (PR-E2); accuracy is refused at the source (PR-E6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: The metrics ``eval.metrics`` may request, in table order.
ALL_METRICS = ("f1", "pr_auc", "roc_auc", "p_at_r50", "p_at_r80")

#: Always returned alongside the requested metrics.
EXTRA_KEYS = ("precision", "recall", "n", "n_pos")


def _as_labels_and_scores(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y).ravel()
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"y has shape {y_arr.shape} but p has shape {p_arr.shape}")
    if y_arr.size == 0:
        raise ValueError("cannot compute metrics on an empty set")
    # Unlabelled nodes never reach a metric.
    if (y_arr < 0).any():
        raise ValueError(
            f"{int((y_arr < 0).sum())} unlabelled nodes (y == -1) reached the evaluator; "
            "metrics are computed on labelled nodes only"
        )
    return y_arr, p_arr


def precision_at_recall(y: np.ndarray, p: np.ndarray, target_recall: float) -> float:
    """Precision at the highest threshold reaching ``target_recall``; never picks a threshold."""
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must lie in (0, 1], got {target_recall}")
    y_arr, p_arr = _as_labels_and_scores(y, p)
    if int((y_arr == 1).sum()) == 0:
        raise ValueError("precision_at_recall is undefined with no illicit nodes in y")

    precision, recall, thresholds = precision_recall_curve(y_arr, p_arr)
    # Trim the trailing (precision 1, recall 0) sentinel, which has no threshold.
    precision = precision[: thresholds.size]
    recall = recall[: thresholds.size]

    # thresholds ascends and recall is non-increasing along it, so the last index
    # that still reaches the target is the highest such threshold.
    reaches = np.flatnonzero(recall >= target_recall)
    if reaches.size == 0:  # pragma: no cover - recall is 1.0 at the lowest threshold
        return 0.0
    return float(precision[reaches[-1]])


def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float | np.ndarray,
    which: Sequence[str] | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Score ``p >= threshold`` (chosen on validation, PR-E4) plus :data:`EXTRA_KEYS`."""
    y_arr, p_arr = _as_labels_and_scores(y, p)
    names = list(ALL_METRICS if which is None else which)
    unknown = [n for n in names if n not in ALL_METRICS]
    if "accuracy" in unknown:
        raise ValueError(
            "accuracy is never reported (PR-E6): at this class balance it measures the "
            "imbalance, not the model. Use f1, pr_auc or precision at fixed recall."
        )
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; known: {list(ALL_METRICS)}")

    thr = np.asarray(threshold)
    if thr.ndim and thr.shape != p_arr.shape:
        raise ValueError(
            f"threshold has shape {thr.shape} but p has shape {p_arr.shape}: a per-row "
            "threshold must have one entry per scored row"
        )
    pred = (p_arr >= threshold).astype(np.int64)
    n_pos = int((y_arr == 1).sum())
    # ROC-AUC needs both classes present; PR-AUC and F1 degrade gracefully.
    both_classes = 0 < n_pos < y_arr.size

    available: dict[str, float] = {
        "f1": float(f1_score(y_arr, pred, zero_division=0.0)),
        "pr_auc": float(average_precision_score(y_arr, p_arr)) if n_pos else float("nan"),
        "roc_auc": float(roc_auc_score(y_arr, p_arr)) if both_classes else float("nan"),
        "p_at_r50": precision_at_recall(y_arr, p_arr, 0.5) if n_pos else float("nan"),
        "p_at_r80": precision_at_recall(y_arr, p_arr, 0.8) if n_pos else float("nan"),
    }
    out = {name: available[name] for name in names}
    out["precision"] = float(precision_score(y_arr, pred, zero_division=0.0))
    out["recall"] = float(recall_score(y_arr, pred, zero_division=0.0))
    out["n"] = float(y_arr.size)
    out["n_pos"] = float(n_pos)
    return {f"{prefix}{k}": v for k, v in out.items()}
