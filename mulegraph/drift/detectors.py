"""Detectors over reference vs current arrays (PR-R1); no function here takes labels (PR-R2)."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

#: Floor on bin mass so an empty bin does not make the PSI log blow up.
EPS = 1e-4


def _check_2d(ref: np.ndarray, cur: np.ndarray) -> None:
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
        raise ValueError(f"ref and cur must be [n, K] with equal K, got {ref.shape}, {cur.shape}")
    if ref.shape[0] == 0 or cur.shape[0] == 0:
        raise ValueError("ref and cur must both be non-empty")


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> np.ndarray:
    """Population stability index per column, with bins cut at reference quantiles."""
    _check_2d(ref, cur)
    out = np.zeros(ref.shape[1], dtype=np.float64)
    q = np.linspace(0.0, 1.0, bins + 1)
    for k in range(ref.shape[1]):
        cuts = np.unique(np.quantile(ref[:, k], q))
        # A constant reference column gets one cut just above its value, so anything
        # larger lands in a bin the reference never occupied.
        inner = cuts[1:-1] if cuts.size >= 2 else np.nextafter(cuts, np.inf)
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        r = np.histogram(ref[:, k], edges)[0] / ref.shape[0]
        c = np.histogram(cur[:, k], edges)[0] / cur.shape[0]
        r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
        out[k] = float(np.sum((c - r) * np.log(c / r)))
    return out


def ks_frac(ref: np.ndarray, cur: np.ndarray, alpha: float = 0.01) -> float:
    """Fraction of columns whose two-sample KS test rejects at ``alpha``."""
    _check_2d(ref, cur)
    p = np.array([ks_2samp(ref[:, k], cur[:, k]).pvalue for k in range(ref.shape[1])])
    return float((p < alpha).mean())


def conf_shift(ref_p: np.ndarray, cur_p: np.ndarray) -> tuple[float, float]:
    """KS statistic and p-value between reference and current model scores."""
    if ref_p.size == 0 or cur_p.size == 0:
        raise ValueError("ref_p and cur_p must both be non-empty")
    res = ks_2samp(ref_p, cur_p)
    return float(res.statistic), float(res.pvalue)
