"""Across-seed Student-t interval — the only interval used to call a gap significant (PR-E3)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import t as student_t


def seed_interval(values: Sequence[float], conf: float = 0.95) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)``; a single seed gives ``(mean, nan, nan)``."""
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
