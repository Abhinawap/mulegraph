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
    return _save(fig, path)


def plot_drift(curves: pd.DataFrame, leads: pd.DataFrame, path: Path) -> Path:
    """Per config: seed-mean test F1, the drop level, and each detector's median first flag."""
    configs = sorted(set(zip(curves["model"], curves["features"], strict=True)))
    fig, axes = plt.subplots(
        1, len(configs), figsize=(6 * len(configs), 4), squeeze=False, sharey=True
    )
    for ax, (model, features) in zip(axes[0], configs, strict=True):
        c = curves[(curves["model"] == model) & (curves["features"] == features)]
        lead = leads[(leads["model"] == model) & (leads["features"] == features)]
        mean = c.groupby("time")["f1"].mean()
        ax.plot(mean.index, mean.to_numpy(), marker="o", ms=3, color="black", label="test F1")
        ax.axhline(lead["drop_level"].mean(), color="grey", ls=":", label="drop level")
        first_drop = lead["first_drop"].median()
        if not pd.isna(first_drop):
            ax.axvspan(first_drop, mean.index.max(), color="red", alpha=0.08, label="F1 dropped")
        for i, (detector, block) in enumerate(lead.groupby("detector", sort=False)):
            flag = block["first_flag"].median()
            if not pd.isna(flag):
                ax.axvline(flag, ls="--", alpha=0.8, color=f"C{i}", label=f"{detector} first flag")
        ax.set_title(f"{model}.{features}")
        ax.set_xlabel("timestep")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("f1")
    axes[0][0].set_ylim(0, 1)
    axes[0][-1].legend(fontsize=8)
    return _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
