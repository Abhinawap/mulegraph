"""README figures from the latest amlworld_xgb run and elliptic_rolling_curves.csv (NFR-1)."""

from __future__ import annotations

import mlflow
import pandas as pd

from mulegraph.eval.metrics import precision_at_recall
from mulegraph.report.figures import _save, plt
from mulegraph.util import Paths

REALISTIC_DAYS = (8, 9)  # ordinary traffic stops on day 10; after it 59% of rows are laundering
RECALLS = (0.5, 0.8)
FEATURES = {"base": "Transaction fields only", "base_gfp": "+ graph features"}


def alert_load(paths: Paths) -> pd.DataFrame:
    """Alerts per laundering case caught (1 / precision) at fixed recall, test days 8-9 only."""
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    runs = mlflow.search_runs(experiment_names=["amlworld_xgb"], order_by=["start_time DESC"])
    parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
    children = runs[runs["tags.mlflow.parentRunId"] == parent["run_id"]]
    rows = []
    for features in FEATURES:
        # XGBoost trial 0 is deterministic, so seed 0 stands for every seed.
        child = children[children["tags.mlflow.runName"] == f"temporal.xgb.{features}.s0"].iloc[0]
        pred = pd.read_parquet(
            mlflow.artifacts.download_artifacts(
                run_id=child["run_id"], artifact_path="predictions.parquet"
            )
        )
        test = pred[(pred["part"] == "test") & pred["time"].isin(REALISTIC_DAYS)]
        for recall in RECALLS:
            p = precision_at_recall(test["y"].to_numpy(), test["proba"].to_numpy(), recall)
            rows.append(
                {
                    "features": features,
                    "recall": recall,
                    "precision": p,
                    "alerts_per_case": 1 / p,
                    "n": len(test),
                    "n_pos": int(test["y"].sum()),
                    "commit": parent["tags.git_commit"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(paths.tables_dir() / "amlworld_alert_load.csv", index=False)
    return table


def plot_alert_load(table: pd.DataFrame, paths: Paths) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    wide = table.pivot(index="recall", columns="features", values="alerts_per_case")
    wide[list(FEATURES)].rename(columns=FEATURES).plot.bar(
        ax=ax, rot=0, width=0.76, color=["#9aa5b1", "#1f5fa8"]
    )
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", padding=2)
    ax.set_xticks(range(len(RECALLS)), [f"Catch {r:.0%} of laundering" for r in RECALLS])
    ax.set(xlabel="", ylabel="Alerts an analyst opens\nper real case caught")
    ax.set_title("Graph features shrink the alert queue")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    _save(fig, paths.figures_dir() / "readme_alert_load.png")


def plot_recovery(paths: Paths) -> None:
    curves = pd.read_csv(paths.tables_dir() / "elliptic_rolling_curves.csv")
    xgb = curves[(curves["model"] == "xgb") & (curves["features"] == "base_gfp")]
    mean = xgb.groupby(["regime", "time"])["f1"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(42.5, mean.index.max() + 0.5, color="#d64545", alpha=0.08)
    ax.plot(mean.index, mean["temporal"], "o-", color="#9aa5b1", label="Trained once on the past")
    ax.plot(
        mean.index, mean["temporal_rolling"], "o-", color="#1f5fa8", label="Retrained each step"
    )
    ax.annotate(
        "Dark market shuts down:\nno drift alarm warned",
        xy=(43, 0.02),
        xytext=(43.4, 0.55),
        arrowprops={"arrowstyle": "->", "color": "#444"},
        fontsize=9,
    )
    ax.set_xlabel("Elliptic time step (about two weeks each)")
    ax.set_ylabel("F1, illicit transactions")
    ax.set_ylim(0, 1)
    ax.set_title("After criminal behaviour changes, only retraining recovers, and slowly")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    _save(fig, paths.figures_dir() / "readme_recovery.png")


if __name__ == "__main__":
    paths = Paths.from_env()
    table = alert_load(paths)
    print(table.to_string(index=False))
    plot_alert_load(table, paths)
    plot_recovery(paths)
