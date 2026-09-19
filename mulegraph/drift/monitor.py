"""Per-batch scores and lead time (PR-R3, PR-R4); labels enter only through the F1 curve."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from mulegraph.drift.detectors import conf_shift, ks_frac, psi


def _scorers(
    detectors: Sequence[str], bins: int, ks_alpha: float
) -> dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]]:
    """Each scorer maps ``(ref_x, cur_x, ref_p, cur_p)`` to one number that rises with drift."""
    table = {
        "psi": lambda rx, cx, rp, cp: float(psi(rx, cx, bins).max()),
        "ks": lambda rx, cx, rp, cp: ks_frac(rx, cx, ks_alpha),
        "conf": lambda rx, cx, rp, cp: conf_shift(rp, cp)[0],
    }
    return {name: table[name] for name in detectors}


def score_batches(
    values: np.ndarray,
    proba: np.ndarray,
    batch: np.ndarray,
    ref_batches: Sequence[int],
    *,
    detectors: Sequence[str],
    bins: int,
    psi_flag: float,
    ks_alpha: float,
    ks_frac_flag: float,
    conf_flag: float,
    calibrate: bool,
) -> pd.DataFrame:
    """Score every non-reference batch against the reference batches; rows are all units.

    With ``calibrate`` the threshold is each detector's largest leave-one-out score inside
    the reference window, so a batch is flagged only when it differs from the reference more
    than the reference batches differ from each other. Labels are never seen (PR-R2).
    """
    ref_batches = np.asarray(ref_batches)
    ref = np.isin(batch, ref_batches)
    if not ref.any():
        raise ValueError(f"no rows fall in the reference batches {ref_batches.tolist()}")
    scorers = _scorers(detectors, bins, ks_alpha)
    thresholds = {"psi": psi_flag, "ks": ks_frac_flag, "conf": conf_flag}
    if calibrate:
        present = [b for b in ref_batches if (batch == b).any()]
        if len(present) < 2:
            raise ValueError("calibrate needs at least two populated reference batches")
        for name, score in scorers.items():
            thresholds[name] = max(
                score(
                    values[ref & (batch != b)],
                    values[batch == b],
                    proba[ref & (batch != b)],
                    proba[batch == b],
                )
                for b in present
            )
    rows = []
    for b in np.unique(batch[~ref]):
        cur = batch == b
        for name, score in scorers.items():
            s = score(values[ref], values[cur], proba[ref], proba[cur])
            rows.append((name, int(b), s, s > thresholds[name], thresholds[name]))
    return pd.DataFrame(rows, columns=["detector", "batch_id", "score", "flagged", "threshold"])


def _first_run(times: np.ndarray, hits: np.ndarray, run: int) -> float:
    """First time that starts ``run`` consecutive hits, or NaN."""
    for i in range(len(hits) - run + 1):
        if hits[i : i + run].all():
            return float(times[i])
    return float("nan")


def lead_time(
    curve: pd.DataFrame,
    scores: pd.DataFrame,
    ref_f1: float,
    drop: float,
    drop_run: int,
) -> pd.DataFrame:
    """Per detector: first ``drop_run``-long F1 fall − first ``drop_run``-long flag run (PR-R4)."""
    level = (1.0 - drop) * ref_f1
    ordered = curve.sort_values("time")
    first_drop = _first_run(
        ordered["time"].to_numpy(), (ordered["f1"] < level).to_numpy(), drop_run
    )
    rows = []
    for detector, block in scores.groupby("detector", sort=False):
        block = block.sort_values("batch_id")
        first_flag = _first_run(block["batch_id"].to_numpy(), block["flagged"].to_numpy(), drop_run)
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
