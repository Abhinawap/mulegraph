"""Per-batch scores and lead time (PR-R3, PR-R4); labels enter only through the F1 curve."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mulegraph.drift.detectors import conf_shift, ks_frac, psi

DETECTORS = ("psi", "ks", "conf")
SCORE_COLUMNS = ["detector", "batch_id", "score", "flagged", "threshold"]


def score_batches(
    values: np.ndarray,
    proba: np.ndarray,
    batch: np.ndarray,
    ref_batches: Sequence[int],
    detectors: Sequence[str] = DETECTORS,
    bins: int = 10,
    psi_flag: float = 0.2,
    ks_alpha: float = 0.01,
    ks_frac_flag: float = 0.2,
) -> pd.DataFrame:
    """Score every non-reference batch against the reference batches; rows are all units."""
    ref = np.isin(batch, np.asarray(ref_batches))
    if not ref.any():
        raise ValueError(f"no rows fall in the reference batches {list(ref_batches)}")
    rows = []
    for b in np.unique(batch[~ref]):
        cur = batch == b
        if "psi" in detectors:
            score = float(psi(values[ref], values[cur], bins).max())
            rows.append(("psi", int(b), score, score >= psi_flag, psi_flag))
        if "ks" in detectors:
            score = ks_frac(values[ref], values[cur], ks_alpha)
            rows.append(("ks", int(b), score, score > ks_frac_flag, ks_frac_flag))
        if "conf" in detectors:
            stat, p = conf_shift(proba[ref], proba[cur])
            rows.append(("conf", int(b), stat, p < ks_alpha, ks_alpha))
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def lead_time(
    curve: pd.DataFrame, scores: pd.DataFrame, ref_f1: float, drop: float = 0.2
) -> pd.DataFrame:
    """``first_drop`` (F1 below ``(1 - drop) * ref_f1``) minus ``first_flag``, per detector."""
    level = (1.0 - drop) * ref_f1
    dropped = curve.loc[curve["f1"] < level, "time"]
    first_drop = float(dropped.min()) if not dropped.empty else float("nan")
    rows = []
    for detector, block in scores.groupby("detector", sort=False):
        flagged = block.loc[block["flagged"], "batch_id"]
        first_flag = float(flagged.min()) if not flagged.empty else float("nan")
        rows.append(
            {
                "detector": detector,
                "first_flag": first_flag,
                "first_drop": first_drop,
                "lead": first_drop - first_flag,
                "drop_level": level,
            }
        )
    return pd.DataFrame(
        rows, columns=["detector", "first_flag", "first_drop", "lead", "drop_level"]
    )
