"""Metrics against hand-computed values (NFR-2, PR-E2, PR-E4, PR-E6).

Every expected number below is derived by hand in a comment, because a metric
tested against another implementation of itself is not tested at all — and these
are the numbers the dissertation reports.

The shared fixture throughout:

    y = [1, 0, 1, 1, 0, 0]
    p = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]

Ranked by score, descending, positives marked P:

    rank  1     2     3     4     5     6
    score 0.9   0.8   0.7   0.4   0.3   0.2
    label P     N     P     P     N     N
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mulegraph.eval.metrics import ALL_METRICS, compute_metrics, precision_at_recall
from mulegraph.eval.threshold import choose_threshold

Y = np.array([1, 0, 1, 1, 0, 0], dtype=np.int64)
P = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2], dtype=np.float64)

# Average precision = sum over ranks of (R_k - R_{k-1}) * P_k.
#   k=1: TP=1, P=1/1,   R=1/3.  dR=1/3 -> 1/3 * 1    = 0.333333
#   k=2: TP=1, P=1/2,   R=1/3.  dR=0   -> 0
#   k=3: TP=2, P=2/3,   R=2/3.  dR=1/3 -> 1/3 * 2/3  = 0.222222
#   k=4: TP=3, P=3/4,   R=1.    dR=1/3 -> 1/3 * 3/4  = 0.250000
#   k=5,6: dR=0
# AP = 1/3 + 2/9 + 1/4 = 29/36
AP = 29 / 36  # 0.8055555...

# ROC-AUC = (# positive/negative pairs the positive outranks) / (3 * 3).
#   0.9 beats 0.8, 0.3, 0.2  -> 3
#   0.7 beats 0.3, 0.2       -> 2
#   0.4 beats 0.3, 0.2       -> 2
# AUC = 7/9
ROC_AUC = 7 / 9  # 0.7777777...


def test_average_precision_matches_hand_computation() -> None:
    out = compute_metrics(Y, P, threshold=0.5)
    assert out["pr_auc"] == pytest.approx(AP)
    assert out["roc_auc"] == pytest.approx(ROC_AUC)


def test_f1_precision_recall_at_two_thresholds() -> None:
    # threshold 0.5 -> predicted positive {0.9 P, 0.8 N, 0.7 P}: TP=2, FP=1, FN=1
    #   precision = 2/3, recall = 2/3, F1 = 2/3
    out = compute_metrics(Y, P, threshold=0.5)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 / 3)

    # threshold 0.35 -> predicted positive {0.9, 0.8, 0.7, 0.4}: TP=3, FP=1, FN=0
    #   precision = 3/4, recall = 1, F1 = 2*(3/4)/(3/4 + 1) = 6/7
    out = compute_metrics(Y, P, threshold=0.35)
    assert out["precision"] == pytest.approx(0.75)
    assert out["recall"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(6 / 7)


def test_precision_at_recall_hand_computed() -> None:
    # PR points, highest threshold first:
    #   thr 0.9: TP=1 FP=0 -> P=1,    R=1/3
    #   thr 0.8: TP=1 FP=1 -> P=1/2,  R=1/3
    #   thr 0.7: TP=2 FP=1 -> P=2/3,  R=2/3
    #   thr 0.4: TP=3 FP=1 -> P=3/4,  R=1
    #   thr 0.3: TP=3 FP=2 -> P=3/5,  R=1
    #   thr 0.2: TP=3 FP=3 -> P=1/2,  R=1
    # Highest threshold reaching R >= 0.5 is 0.7 -> precision 2/3.
    # Highest threshold reaching R >= 0.8 is 0.4 -> precision 3/4.
    assert precision_at_recall(Y, P, 0.5) == pytest.approx(2 / 3)
    assert precision_at_recall(Y, P, 0.8) == pytest.approx(0.75)

    out = compute_metrics(Y, P, threshold=0.5)
    assert out["p_at_r50"] == pytest.approx(2 / 3)
    assert out["p_at_r80"] == pytest.approx(0.75)


def test_precision_at_recall_is_not_the_decision_threshold() -> None:
    """P@R is a curve metric: it is unchanged by whatever operating point is used."""
    at_low = compute_metrics(Y, P, threshold=0.05)
    at_high = compute_metrics(Y, P, threshold=0.95)
    assert at_low["p_at_r50"] == at_high["p_at_r50"]
    assert at_low["pr_auc"] == at_high["pr_auc"]
    assert at_low["f1"] != at_high["f1"]  # but the point metric does move


def test_perfect_and_inverted_rankings() -> None:
    y = np.array([1, 1, 0, 0], dtype=np.int64)
    perfect = compute_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]), threshold=0.5)
    assert perfect["pr_auc"] == pytest.approx(1.0)
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["f1"] == pytest.approx(1.0)

    # Exactly inverted: no positive outranks any negative, so ROC-AUC is 0 and
    # AP is the value of the worst possible ranking, 1/2 * (1/3 + 2/4)... by hand:
    #   ranks: N N P P -> k=3: TP=1, P=1/3, R=1/2, dR=1/2 -> 1/6
    #          k=4: TP=2, P=2/4, R=1,   dR=1/2 -> 1/4
    #   AP = 1/6 + 1/4 = 5/12
    inverted = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
    assert inverted["roc_auc"] == pytest.approx(0.0)
    assert inverted["pr_auc"] == pytest.approx(5 / 12)


def test_which_and_prefix_select_and_rename() -> None:
    out = compute_metrics(Y, P, threshold=0.5, which=["f1", "pr_auc"], prefix="test_")
    assert set(out) == {
        "test_f1",
        "test_pr_auc",
        "test_precision",
        "test_recall",
        "test_n",
        "test_n_pos",
    }
    assert out["test_n"] == 6.0
    assert out["test_n_pos"] == 3.0


def test_accuracy_is_refused() -> None:
    """PR-E6: accuracy is never a headline metric, so it is not obtainable here."""
    with pytest.raises(ValueError, match="accuracy is never reported"):
        compute_metrics(Y, P, threshold=0.5, which=["f1", "accuracy"])
    assert "accuracy" not in ALL_METRICS
    assert "accuracy" not in compute_metrics(Y, P, threshold=0.5)


def test_unlabelled_and_shape_errors() -> None:
    with pytest.raises(ValueError, match="unlabelled"):
        compute_metrics(np.array([1, -1, 0]), np.array([0.5, 0.5, 0.5]), threshold=0.5)
    with pytest.raises(ValueError, match="shape"):
        compute_metrics(Y, P[:3], threshold=0.5)
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(np.array([], dtype=np.int64), np.array([]), threshold=0.5)
    with pytest.raises(ValueError, match="target_recall"):
        precision_at_recall(Y, P, 0.0)


# --------------------------------------------------------------------------- #
# Threshold selection (PR-E4)
# --------------------------------------------------------------------------- #


def test_choose_threshold_maximises_validation_f1() -> None:
    # F1 at every candidate threshold on the shared fixture:
    #   0.2 -> P=1/2, R=1   -> 2/3
    #   0.3 -> P=3/5, R=1   -> 0.75
    #   0.4 -> P=3/4, R=1   -> 6/7   <- maximum
    #   0.7 -> P=2/3, R=2/3 -> 2/3
    #   0.8 -> P=1/2, R=1/3 -> 0.4
    #   0.9 -> P=1,   R=1/3 -> 0.5
    threshold, val_f1 = choose_threshold(Y, P)
    assert threshold == pytest.approx(0.4)
    assert val_f1 == pytest.approx(6 / 7)
    # And the threshold reproduces that F1 when applied.
    assert compute_metrics(Y, P, threshold)["f1"] == pytest.approx(6 / 7)


def test_choose_threshold_prefers_the_higher_threshold_on_ties() -> None:
    # y = [1, 0, 0, 1], p = [0.9, 0.8, 0.7, 0.6]
    #   thr 0.9: TP=1 FP=0 FN=1 -> F1 = 2/(2+0+1) = 2/3
    #   thr 0.8: TP=1 FP=1 FN=1 -> F1 = 2/(2+1+1) = 1/2
    #   thr 0.7: TP=1 FP=2 FN=1 -> F1 = 2/(2+2+1) = 2/5
    #   thr 0.6: TP=2 FP=2 FN=0 -> F1 = 4/(4+2+0) = 2/3
    # 0.9 and 0.6 tie at 2/3; fewer alerts wins, so 0.9.
    y = np.array([1, 0, 0, 1], dtype=np.int64)
    p = np.array([0.9, 0.8, 0.7, 0.6])
    threshold, val_f1 = choose_threshold(y, p)
    assert val_f1 == pytest.approx(2 / 3)
    assert threshold == pytest.approx(0.9)


def test_choose_threshold_rejects_degenerate_validation_sets() -> None:
    with pytest.raises(ValueError, match="no illicit nodes"):
        choose_threshold(np.zeros(5, dtype=np.int64), np.linspace(0, 1, 5))
    with pytest.raises(ValueError, match="unlabelled"):
        choose_threshold(np.array([1, -1, 0]), np.array([0.9, 0.5, 0.1]))
    with pytest.raises(ValueError, match="empty"):
        choose_threshold(np.array([], dtype=np.int64), np.array([]))


def test_choose_threshold_signature_admits_validation_only() -> None:
    """Structural guard for PR-E4: there is no parameter test data could arrive through."""
    params = list(inspect_signature_names(choose_threshold))
    assert params == ["y_val", "p_val"]


def inspect_signature_names(func: object) -> list[str]:
    import inspect

    return list(inspect.signature(func).parameters)  # type: ignore[arg-type]


def test_all_metrics_covers_the_spec_list() -> None:
    assert set(ALL_METRICS) == {"f1", "pr_auc", "roc_auc", "p_at_r50", "p_at_r80"}
    assert not math.isnan(compute_metrics(Y, P, 0.5)["pr_auc"])
