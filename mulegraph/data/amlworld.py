"""IBM AMLworld loader: accounts are nodes, transactions are labelled edges (PR-D2, PR-D4)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

from mulegraph.types import LABEL_ILLICIT, LABEL_LICIT, DatasetMeta, GraphDataset
from mulegraph.util import hash_files

log = logging.getLogger("mulegraph")

SOURCE_URL = (
    "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
)

FILES = {"hi_small": "HI-Small_Trans.csv"}

#: The raw header names both account columns ``Account``, so columns are read by position.
COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "currency_received",
    "amount_paid",
    "currency_paid",
    "payment_format",
    "is_laundering",
]
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
HOURS_PER_DAY = 24

#: ``x`` is empty on an edge task; these are the per-transaction raw features (``base``).
FEATURE_NAMES = [
    "amount_paid",
    "amount_received",
    "currency_paid",
    "currency_received",
    "payment_format",
    "hour_of_day",
]

#: Published HI-Small counts; checked only on a full load.
EXPECTED = {
    "hi_small": {"num_edges": 5_078_345, "num_nodes": 515_088, "num_illicit": 5_177},
}


def cache_version(version: str, max_days: int | None) -> str:
    """A truncated load must never share a cache with the full dataset (NFR-1)."""
    return version if max_days is None else f"{version}-d{max_days}"


def _read(path: Path) -> pa.Table:
    string_cols = ("from_bank", "from_account", "to_bank", "to_account")
    types = {
        "timestamp": pa.timestamp("s"),
        "amount_received": pa.float64(),
        "amount_paid": pa.float64(),
        "is_laundering": pa.int64(),
    } | dict.fromkeys(string_cols, pa.string())
    return pv.read_csv(
        path,
        read_options=pv.ReadOptions(column_names=COLUMNS, skip_rows=1),
        convert_options=pv.ConvertOptions(column_types=types, timestamp_parsers=[TIMESTAMP_FORMAT]),
    )


def load_amlworld_raw(raw_dir: Path, version: str, max_days: int | None = None) -> GraphDataset:
    """Load one AMLworld transaction file; edges come back sorted by hour."""
    if version not in FILES:
        raise ValueError(f"unknown amlworld version {version!r}; known: {sorted(FILES)}")
    path = Path(raw_dir) / FILES[version]
    if not path.is_file():
        raise FileNotFoundError(
            f"AMLworld file missing: {path}. Download {FILES[version]} from {SOURCE_URL} "
            "and place (or symlink) it there."
        )

    frame = _read(path).to_pandas()
    ts = frame["timestamp"].to_numpy().astype("datetime64[s]")
    hour = ((ts - ts.min()) // np.timedelta64(1, "h")).astype(np.int64)
    if max_days is not None:
        keep = hour < max_days * HOURS_PER_DAY
        frame, ts, hour = frame[keep], ts[keep], hour[keep]
        log.info("amlworld %s: max_days=%d keeps %d transactions", version, max_days, len(frame))

    order = np.argsort(hour, kind="stable")
    frame, ts, hour = frame.iloc[order], ts[order], hour[order]

    # Account = bank + account number; both zero-padded strings, so the key is never numeric.
    keys = pd.concat(
        [
            frame["from_bank"] + ":" + frame["from_account"],
            frame["to_bank"] + ":" + frame["to_account"],
        ],
        ignore_index=True,
    )
    codes, _ = pd.factorize(keys)
    e = len(frame)
    edge_index = np.stack([codes[:e], codes[e:]]).astype(np.int64)
    n = int(codes.max()) + 1 if e else 0

    # ponytail: ordinal codes for currency / format; one-hot them if the GNN needs it.
    edge_attr = np.column_stack(
        [
            frame["amount_paid"].to_numpy(),
            frame["amount_received"].to_numpy(),
            pd.factorize(frame["currency_paid"], sort=True)[0],
            pd.factorize(frame["currency_received"], sort=True)[0],
            pd.factorize(frame["payment_format"], sort=True)[0],
            (ts - ts.astype("datetime64[D]")) // np.timedelta64(1, "h"),
        ]
    ).astype(np.float32)
    y = frame["is_laundering"].to_numpy().astype(np.int64)
    if not np.isin(y, (LABEL_LICIT, LABEL_ILLICIT)).all():
        raise ValueError(f"{path.name}: Is Laundering must be 0/1, got {np.unique(y).tolist()}")

    node_time = np.full(n, hour.max() if e else 0, dtype=np.int64)
    np.minimum.at(node_time, edge_index[0], hour)
    np.minimum.at(node_time, edge_index[1], hour)

    num_illicit = int((y == LABEL_ILLICIT).sum())
    if max_days is None:
        _check_version(version, num_edges=e, num_nodes=n, num_illicit=num_illicit)

    meta = DatasetMeta(
        dataset="amlworld",
        version=cache_version(version, max_days),
        cross_time_edges=True,
        source_url=SOURCE_URL,
        feature_blocks={"local": (0, len(FEATURE_NAMES))},
        feature_names=list(FEATURE_NAMES),
        num_timesteps=int(hour.max() // HOURS_PER_DAY) + 1 if e else 0,
        label_counts={"illicit": num_illicit, "licit": e - num_illicit, "unknown": 0},
        raw_sha256=hash_files([path]),
        task="edge",
    )
    return GraphDataset(
        x=np.zeros((n, 0), dtype=np.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_time=node_time,
        edge_time=hour,
        batch_id=hour // HOURS_PER_DAY,
        y=y,
        node_ids=np.arange(n, dtype=np.int64),
        task="edge",
        meta=meta,
    )


def _check_version(version: str, **observed: int) -> None:
    """Assert the published counts, naming what was observed instead."""
    expected = EXPECTED[version]
    mismatched = {k: (expected[k], observed[k]) for k in expected if expected[k] != observed[k]}
    if mismatched:
        detail = "; ".join(
            f"{k}: expected {exp}, observed {obs}" for k, (exp, obs) in mismatched.items()
        )
        raise ValueError(
            f"{FILES[version]} in this directory is not AMLworld {version} — {detail}. "
            "Pin the version the file actually is, or replace the file."
        )
