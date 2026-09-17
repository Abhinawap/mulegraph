"""AMLworld loader tests on a synthetic CSV shaped like HI-Small_Trans.csv (PR-D2, PR-D4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mulegraph.config import DatasetConfig
from mulegraph.data import load_dataset
from mulegraph.data.amlworld import FILES, load_amlworld_raw

#: The real header names both account columns ``Account``; the loader reads by position.
HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
N_ROWS = 200
N_DAYS = 3


def write_csv(raw_dir: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    banks = ["010", "011", "1234"]
    accounts = [f"8000EB{i:03X}" for i in range(25)]
    currencies = ["US Dollar", "Euro", "Yuan"]
    formats = ["Reinvestment", "ACH", "Cheque", "Credit Card"]
    rows = []
    for _ in range(N_ROWS):
        minute = int(rng.integers(0, N_DAYS * 24 * 60))
        day, rest = divmod(minute, 24 * 60)
        hour, minute = divmod(rest, 60)
        amount = float(rng.uniform(1, 5000))
        rows.append(
            ",".join(
                [
                    f"2022/09/{1 + day:02d} {hour:02d}:{minute:02d}",
                    rng.choice(banks),
                    rng.choice(accounts),
                    rng.choice(banks),
                    rng.choice(accounts),
                    f"{amount:.2f}",
                    rng.choice(currencies),
                    f"{amount:.2f}",
                    rng.choice(currencies),
                    rng.choice(formats),
                    str(int(rng.random() < 0.1)),
                ]
            )
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / FILES["hi_small"]
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    write_csv(tmp_path)
    return tmp_path


def test_edge_task_shapes_dtypes_and_times(raw_dir: Path) -> None:
    data = load_amlworld_raw(raw_dir, "hi_small", max_days=N_DAYS)

    assert data.task == "edge" and data.meta.task == "edge"
    assert data.num_edges == N_ROWS and data.num_units == N_ROWS
    assert data.x.shape == (data.num_nodes, 0) and data.x.dtype == np.float32
    assert data.edge_attr is not None
    assert data.edge_attr.shape == (N_ROWS, 6) and data.edge_attr.dtype == np.float32
    assert data.y.shape == (N_ROWS,) and set(np.unique(data.y).tolist()) <= {0, 1}
    assert data.meta.label_counts["illicit"] == int(data.y.sum())
    assert data.meta.label_counts["unknown"] == 0

    # Edges sorted by hour; batches are days; an account exists from its first transaction.
    assert np.all(np.diff(data.edge_time) >= 0)
    assert np.array_equal(data.batch_id, data.edge_time // 24)
    assert data.edge_time.min() == 0 and data.batch_id.max() == N_DAYS - 1
    assert np.all(data.node_time[data.src] <= data.edge_time)
    assert np.all(data.node_time[data.dst] <= data.edge_time)
    assert np.unique(data.node_ids).size == data.num_nodes
    assert (data.edge_attr[:, 5] < 24).all()  # hour of day
    assert data.meta.cross_time_edges is True
    assert data.meta.feature_blocks == {"local": (0, 6)}
    assert data.meta.version == f"hi_small-d{N_DAYS}"
    assert data.meta.num_timesteps == N_DAYS


def test_max_days_keeps_a_prefix_of_days(raw_dir: Path) -> None:
    full = load_amlworld_raw(raw_dir, "hi_small", max_days=N_DAYS)
    first = load_amlworld_raw(raw_dir, "hi_small", max_days=1)

    assert 0 < first.num_edges < full.num_edges
    assert first.batch_id.max() == 0
    assert first.num_edges == int((full.batch_id == 0).sum())
    assert first.meta.version == "hi_small-d1"


def test_full_load_checks_the_published_counts(raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="not AMLworld hi_small"):
        load_amlworld_raw(raw_dir, "hi_small")


def test_truncated_load_is_cached_under_its_own_version(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_csv(data_dir / "raw" / "amlworld" / "hi_small")
    cfg = DatasetConfig(name="amlworld", version="hi_small", max_days=1)

    data = load_dataset(cfg, data_dir)
    assert (data_dir / "cache" / "amlworld" / "hi_small-d1" / "graph.pt").is_file()
    again = load_dataset(cfg, data_dir)
    assert again.task == "edge" and again.num_edges == data.num_edges
    assert np.array_equal(again.y, data.y)
    assert again.meta.feature_blocks == {"local": (0, 6)}


def test_max_days_is_amlworld_only() -> None:
    with pytest.raises(ValueError, match="max_days"):
        DatasetConfig(name="elliptic_pp", max_days=2)
