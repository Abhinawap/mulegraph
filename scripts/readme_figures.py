"""README figures from the latest amlworld_xgb run, elliptic_rolling_curves.csv and the cached
AMLworld graph (NFR-1)."""

from __future__ import annotations

import mlflow
import networkx as nx
import numpy as np
import pandas as pd

from mulegraph.config import DatasetConfig
from mulegraph.data import load_dataset
from mulegraph.eval.metrics import precision_at_recall
from mulegraph.report.figures import _save, plt
from mulegraph.util import Paths

REALISTIC_DAYS = (8, 9)  # ordinary traffic stops on day 10; after it 59% of rows are laundering
RECALLS = (0.5, 0.8)
MAX_ACCOUNTS = 30
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


def plot_laundering_subgraph(paths: Paths) -> None:
    """Largest laundering component of at most MAX_ACCOUNTS accounts, earliest on ties (PR-D2)."""
    data = load_dataset(DatasetConfig(name="amlworld", version="hi_small"), paths.data_dir)
    src, dst = data.edge_index
    bad = data.y == 1
    comps = [
        list(c)
        for c in nx.connected_components(nx.Graph(data.edge_index[:, bad].T.tolist()))
        if len(c) <= MAX_ACCOUNTS
    ]
    # Edges are sorted by hour, so the lowest laundering edge index is the earliest transaction.
    accounts = max(comps, key=lambda c: (len(c), -np.flatnonzero(bad & np.isin(src, c))[0]))
    inside = np.isin(src, accounts) & np.isin(dst, accounts) & (src != dst)  # self-payments off
    pairs = pd.DataFrame({"s": src[inside], "d": dst[inside], "bad": bad[inside]})
    pairs = pairs.groupby(["s", "d"])["bad"].agg(["any", "all"])
    g = nx.DiGraph(pairs.index.tolist())
    pos = nx.kamada_kawai_layout(g)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    grey, red = pairs.index[~pairs["all"]].tolist(), pairs.index[pairs["any"]].tolist()
    style = {"G": g, "pos": pos, "ax": ax}
    nx.draw_networkx_edges(
        **style, edgelist=grey, edge_color="#9aa5b1", connectionstyle="arc3,rad=0.25", arrowsize=8
    )
    nx.draw_networkx_edges(**style, edgelist=red, edge_color="#d64545", width=1.8, arrowsize=10)
    nx.draw_networkx_nodes(**style, node_size=90, node_color="#1f5fa8")
    n_bad = int(bad[inside].sum())
    ax.plot([], [], color="#d64545", lw=1.8, label=f"Laundering ({n_bad} transactions)")
    ax.plot([], [], color="#9aa5b1", label=f"Ordinary ({int(inside.sum()) - n_bad} transactions)")
    days = data.batch_id[inside & bad]
    ax.set_title(
        f"One laundering network in AMLworld HI-Small: {len(accounts)} accounts, "
        f"days {days.min()}-{days.max()}"
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=2)
    ax.axis("off")
    _save(fig, paths.figures_dir() / "readme_laundering_subgraph.png")


if __name__ == "__main__":
    paths = Paths.from_env()
    table = alert_load(paths)
    print(table.to_string(index=False))
    plot_alert_load(table, paths)
    plot_recovery(paths)
    plot_laundering_subgraph(paths)
