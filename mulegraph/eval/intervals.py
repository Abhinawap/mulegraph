"""Across-seed confidence intervals (PR-E3).

The project reports exactly two kinds of interval, and pooling them would be a
category error:

* **This one.** The across-seed Student-t interval, n = 5, over the metric each
  seed produced on the same fixed split. It answers "would this ranking survive a
  retrain?", which is what a headline table is read as. It is the **only**
  interval that may be used to call a gap significant, and a gap is significant
  only when its paired-by-seed interval excludes zero.
* **Not this one.** A bootstrap over test ids, which answers "would this number
  survive a different sample of transactions?". That is a per-timestep band on a
  curve and nothing else — never a significance test, never averaged in here
  (PR-E3, spec §2.5 "Evaluation").

They estimate different variances. A bootstrap band over 12,000 test nodes is
narrow almost regardless of how unstable training is, so treating it as evidence
about a model gap would declare noise significant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import t as student_t


def seed_interval(values: Sequence[float], conf: float = 0.95) -> tuple[float, float, float]:
    """Mean and Student-t confidence interval across seeds.

    ``mean +/- t.ppf((1 + conf) / 2, n - 1) * std(ddof=1) / sqrt(n)``. The t
    distribution rather than the normal because n is 5 (3 for the reference GNN):
    at those sample sizes a normal interval is roughly a third too narrow.

    Args:
        values: One metric value per seed.
        conf: Coverage, 0.95 for every headline table (``eval.seed_ci: t95``).

    Returns:
        ``(mean, ci_low, ci_high)``. With a single seed the interval is
        ``(mean, nan, nan)`` — one observation carries no information about
        spread, and a zero-width interval would claim it does.

    Raises:
        ValueError: On an empty sequence or a coverage outside (0, 1).
    """
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must lie in (0, 1), got {conf}")
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("seed_interval needs at least one value")
    if arr.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {arr.shape}")

    mean = float(arr.mean())
    if arr.size == 1:
        return mean, float("nan"), float("nan")

    sd = float(arr.std(ddof=1))
    half_width = float(student_t.ppf((1.0 + conf) / 2.0, arr.size - 1)) * sd / math.sqrt(arr.size)
    return mean, mean - half_width, mean + half_width
