"""Elliptic++ transaction-graph loader (PR-D1, PR-D4). ``x`` is exactly the 165 published
features; Elliptic++'s own 17 extras are dropped and recorded (PR-M7)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

from mulegraph.types import (
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    DatasetMeta,
    GraphDataset,
)
from mulegraph.util import hash_files

log = logging.getLogger("mulegraph")

FEATURES_FILE = "txs_features.csv"
CLASSES_FILE = "txs_classes.csv"
EDGELIST_FILE = "txs_edgelist.csv"

SOURCE_URL = "https://github.com/git-disl/EllipticPlusPlus"

N_LOCAL = 93
N_AGG1HOP = 72

ID_COLUMN = "txId"
TIME_COLUMN = "Time step"

#: Published order: 93 local then 72 one-hop aggregates. ``base`` is the first block (D3, PR-M7).
LOCAL_COLUMNS = [f"Local_feature_{i + 1}" for i in range(N_LOCAL)]
AGG_COLUMNS = [f"Aggregate_feature_{i + 1}" for i in range(N_AGG1HOP)]
FEATURE_COLUMNS = LOCAL_COLUMNS + AGG_COLUMNS

#: Elliptic++ extras; the two degree columns are graph-derived and would contaminate ``base``.
EXTRA_COLUMNS = [
    "in_txs_degree",
    "out_txs_degree",
    "total_BTC",
    "fees",
    "size",
    "num_input_addresses",
    "num_output_addresses",
    "in_BTC_min",
    "in_BTC_max",
    "in_BTC_mean",
    "in_BTC_median",
    "in_BTC_total",
    "out_BTC_min",
    "out_BTC_max",
    "out_BTC_mean",
    "out_BTC_median",
    "out_BTC_total",
]

#: Raw ``class`` encoding -> the package-wide label encoding.
CLASS_MAP = {1: LABEL_ILLICIT, 2: LABEL_LICIT, 3: LABEL_UNKNOWN}

#: Published counts per release; a mismatch means a different revision and must fail loudly.
EXPECTED = {
    "2023.1": {
        "num_nodes": 203_769,
        "num_edges": 234_355,
        "num_timesteps": 49,
        "label_counts": {"illicit": 4_545, "licit": 42_019, "unknown": 157_205},
    }
}


def _require_files(raw_dir: Path) -> list[Path]:
    paths = [raw_dir / name for name in (FEATURES_FILE, CLASSES_FILE, EDGELIST_FILE)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Elliptic++ raw files missing from {raw_dir}: {', '.join(missing)}. "
            f"Expected {FEATURES_FILE}, {CLASSES_FILE} and {EDGELIST_FILE} in that directory; "
            "there is no auto-download in the MVP — fetch them from "
            f"{SOURCE_URL} and place (or symlink) them there."
        )
    return paths


def _read_header(path: Path) -> list[str]:
    with open(path) as fh:
        return fh.readline().rstrip("\n").split(",")


def _read_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return ``(node_ids, node_time, x, dropped)``; ``dropped`` comes from the header (PR-M7)."""
    wanted = [ID_COLUMN, TIME_COLUMN, *FEATURE_COLUMNS]
    header = _read_header(path)
    dropped = [name for name in header if name not in set(wanted)]
    unexpected = sorted(set(dropped) - set(EXTRA_COLUMNS))
    if unexpected:
        log.warning(
            "%s carries %d column(s) outside the known Elliptic++ extras: %s; "
            "they are excluded from x and recorded in meta.dropped_columns",
            path.name,
            len(unexpected),
            unexpected,
        )
    # Read only the 167 needed columns, typed during the parse: the file is 695 MB of text.
    column_types = {ID_COLUMN: pa.int64(), TIME_COLUMN: pa.int64()} | dict.fromkeys(
        FEATURE_COLUMNS, pa.float32()
    )
    table = pv.read_csv(
        path,
        convert_options=pv.ConvertOptions(include_columns=wanted, column_types=column_types),
    )
    if table.num_columns != len(wanted):
        found = set(table.column_names)
        raise ValueError(f"{path.name} is missing expected columns: {sorted(set(wanted) - found)}")

    node_ids = table.column(ID_COLUMN).to_numpy().astype(np.int64, copy=False)
    node_time = table.column(TIME_COLUMN).to_numpy().astype(np.int64, copy=False)

    # Filled column by column so the only full-size float32 allocation is x itself.
    x = np.empty((table.num_rows, len(FEATURE_COLUMNS)), dtype=np.float32)
    for j, name in enumerate(FEATURE_COLUMNS):
        x[:, j] = table.column(name).to_numpy(zero_copy_only=False)
    return node_ids, node_time, x, dropped


def _resolve_ids(ids: np.ndarray, index: pd.Index, source: str) -> np.ndarray:
    """Map original txIds to node indices."""
    rows = index.get_indexer(ids)
    unresolved = rows < 0
    if unresolved.any():
        raise ValueError(
            f"{source} references {int(unresolved.sum())} txIds absent from {FEATURES_FILE}, "
            f"e.g. {ids[unresolved][:5].tolist()}"
        )
    return rows.astype(np.int64)


def _read_labels(path: Path, index: pd.Index) -> np.ndarray:
    """Labels aligned to feature-file row order, in the package encoding."""
    table = pv.read_csv(
        path,
        convert_options=pv.ConvertOptions(
            include_columns=[ID_COLUMN, "class"],
            column_types={ID_COLUMN: pa.int64(), "class": pa.int64()},
        ),
    )
    label_ids = table.column(ID_COLUMN).to_numpy().astype(np.int64, copy=False)
    raw_class = table.column("class").to_numpy().astype(np.int64, copy=False)

    unknown_classes = sorted(set(np.unique(raw_class).tolist()) - set(CLASS_MAP))
    if unknown_classes:
        raise ValueError(f"{path.name} has unexpected class values {unknown_classes}")

    lut = np.full(max(CLASS_MAP) + 1, LABEL_UNKNOWN, dtype=np.int64)
    for raw, label in CLASS_MAP.items():
        lut[raw] = label

    y = np.full(len(index), LABEL_UNKNOWN, dtype=np.int64)
    y[_resolve_ids(label_ids, index, path.name)] = lut[raw_class]
    return y


def _read_edges(path: Path, index: pd.Index) -> np.ndarray:
    table = pv.read_csv(
        path,
        convert_options=pv.ConvertOptions(
            include_columns=["txId1", "txId2"],
            column_types={"txId1": pa.int64(), "txId2": pa.int64()},
        ),
    )
    src_ids = table.column("txId1").to_numpy().astype(np.int64, copy=False)
    dst_ids = table.column("txId2").to_numpy().astype(np.int64, copy=False)
    src = _resolve_ids(src_ids, index, path.name)
    dst = _resolve_ids(dst_ids, index, path.name)
    return np.stack([src, dst]).astype(np.int64)


def load_elliptic_raw(raw_dir: Path, version: str) -> GraphDataset:
    """Load the three raw CSVs; edges come back sorted by ``edge_time``."""
    raw_dir = Path(raw_dir)
    feature_path, class_path, edge_path = _require_files(raw_dir)

    node_ids, node_time, x, dropped = _read_features(feature_path)
    index = pd.Index(node_ids)
    if not index.is_unique:
        raise ValueError(
            f"{FEATURES_FILE} lists {int(index.duplicated().sum())} duplicate txIds, so labels "
            "and edges cannot be aligned to a single node"
        )
    y = _read_labels(class_path, index)
    edge_index = _read_edges(edge_path, index)

    src, dst = edge_index[0], edge_index[1]
    cross_time_edges = bool((node_time[src] != node_time[dst]).any())
    if cross_time_edges:
        # edge_time below is the source's timestep, causal only while both endpoints share it;
        # a cross-timestep edge would let an `edge_time <= t` filter admit the future (PR-F2).
        n_cross = int((node_time[src] != node_time[dst]).sum())
        raise ValueError(
            f"{edge_path.name} has {n_cross} of {edge_index.shape[1]} edges joining different "
            "timesteps, but the Elliptic++ transaction graph is defined as 49 disconnected "
            "timestep components (D1); edge_time would not be causal on these files."
        )
    edge_time = node_time[src]
    time_order = np.argsort(edge_time, kind="stable")
    edge_index = edge_index[:, time_order]
    edge_time = edge_time[time_order]

    label_counts = {
        "illicit": int((y == LABEL_ILLICIT).sum()),
        "licit": int((y == LABEL_LICIT).sum()),
        "unknown": int((y == LABEL_UNKNOWN).sum()),
    }
    num_timesteps = int(node_time.max())
    _check_version(
        version,
        num_nodes=node_ids.size,
        num_edges=edge_index.shape[1],
        num_timesteps=num_timesteps,
        label_counts=label_counts,
    )

    meta = DatasetMeta(
        dataset="elliptic_pp",
        version=version,
        cross_time_edges=cross_time_edges,
        source_url=SOURCE_URL,
        feature_blocks={"local": (0, N_LOCAL), "agg1hop": (N_LOCAL, N_LOCAL + N_AGG1HOP)},
        feature_names=list(FEATURE_COLUMNS),
        dropped_columns=dropped,
        num_timesteps=num_timesteps,
        label_counts=label_counts,
        raw_sha256=hash_files([feature_path, class_path, edge_path]),
        task="node",
    )
    log.info(
        "elliptic_pp %s: dropped %d Elliptic++ extras (incl. in_txs_degree/out_txs_degree) "
        "so x is the published 165-feature block (PR-M7)",
        version,
        len(dropped),
    )
    return GraphDataset(
        x=x,
        edge_index=edge_index,
        edge_attr=None,
        node_time=node_time,
        edge_time=edge_time,
        batch_id=node_time.copy(),
        y=y,
        node_ids=node_ids,
        task="node",
        meta=meta,
    )


def _check_version(
    version: str,
    *,
    num_nodes: int,
    num_edges: int,
    num_timesteps: int,
    label_counts: dict[str, int],
) -> None:
    """Assert the release's published counts, naming what was observed instead."""
    expected = EXPECTED.get(version)
    if expected is None:
        log.warning("no published counts registered for elliptic_pp %s; skipping checks", version)
        return
    observed = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_timesteps": num_timesteps,
        "label_counts": label_counts,
    }
    mismatched = {k: (expected[k], observed[k]) for k in expected if expected[k] != observed[k]}
    if mismatched:
        detail = "; ".join(
            f"{k}: expected {exp}, observed {obs}" for k, (exp, obs) in mismatched.items()
        )
        raise ValueError(
            f"{FEATURES_FILE} in this directory is not Elliptic++ {version} — {detail}. "
            "Pin the version the files actually are, or replace the files."
        )
