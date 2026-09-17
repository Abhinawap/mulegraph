"""Per-timestep figures from the curves table (PR-E5); depends on types only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Interval = Callable[[Sequence[float]], tuple[float, float, float]]


def _interval(interval: Interval | None) -> Interval:
    if interval is not None:
        return interval
    from mulegraph.eval.intervals import seed_interval

    return seed_interval


def plot_curves(
    curves: pd.DataFrame, path: Path, metric: str = "f1", interval: Interval | None = None
) -> Path:
    """One panel per regime: mean ``metric`` per timestep per config, across-seed t band."""
    interval = _interval(interval)
    regimes = sorted(curves["regime"].unique())
    fig, axes = plt.subplots(
        1, len(regimes), figsize=(6 * len(regimes), 4), squeeze=False, sharey=True
    )
    for ax, regime in zip(axes[0], regimes, strict=True):
        block = curves[curves["regime"] == regime]
        for (model, features), cfg in block.groupby(["model", "features"], sort=True):
            stats = cfg.groupby("time")[metric].agg(lambda v: interval(v.dropna().tolist()))
            times = stats.index.to_numpy()
            mean, low, high = (
                pd.Series([s[i] for s in stats], dtype="float64").to_numpy() for i in range(3)
            )
            ax.plot(times, mean, marker="o", ms=3, label=f"{model}.{features}")
            if not pd.isna(low).all():
                ax.fill_between(times, low, high, alpha=0.15)
        ax.set_title(regime)
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(metric)
    axes[0][0].set_ylim(0, 1)
    axes[0][-1].legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
