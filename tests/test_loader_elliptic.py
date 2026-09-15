"""Elliptic++ loader tests against the real CSVs (PR-D1, PR-M7).

Every test is marked ``elliptic`` and skips when the raw files are absent, so CI
— which cannot download the dataset — stays green. Parsing 695 MB takes a few
seconds, so the graph is loaded once and shared: ``_load`` is cached rather than
the fixture being module-scoped, because ``elliptic_raw_dir`` (which owns the
skip) is function-scoped and pytest forbids the wider scope depending on it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from mulegraph.data.elliptic import EXTRA_COLUMNS, load_elliptic_raw
from mulegraph.types import GraphDataset

pytestmark = pytest.mark.elliptic

VERSION = "2023.1"


@lru_cache(maxsize=1)
def _load(raw_dir: Path) -> GraphDataset:
    return load_elliptic_raw(raw_dir, VERSION)


@pytest.fixture
def elliptic(elliptic_raw_dir: Path) -> GraphDataset:
    return _load(elliptic_raw_dir)


def test_shapes_and_dtypes(elliptic: GraphDataset) -> None:
    assert elliptic.num_nodes == 203_769
    assert elliptic.num_edges == 234_355
    assert elliptic.x.shape == (203_769, 165)
    assert elliptic.x.dtype == np.float32
    assert elliptic.node_ids.dtype == np.int64
    assert np.unique(elliptic.node_ids).size == elliptic.num_nodes
    assert elliptic.edge_attr is None
    assert elliptic.task == "node"


def test_label_counts(elliptic: GraphDataset) -> None:
    assert elliptic.meta.label_counts == {"illicit": 4_545, "licit": 42_019, "unknown": 157_205}
    assert int((elliptic.y == 1).sum()) == 4_545
    assert int((elliptic.y == 0).sum()) == 42_019
    assert int((elliptic.y == -1).sum()) == 157_205
    assert elliptic.labelled_idx.size == 4_545 + 42_019


def test_timesteps(elliptic: GraphDataset) -> None:
    assert elliptic.meta.num_timesteps == 49
    assert np.array_equal(np.unique(elliptic.node_time), np.arange(1, 50))
    assert np.array_equal(elliptic.batch_id, elliptic.node_time)


def test_no_cross_time_edges(elliptic: GraphDataset) -> None:
    """The D1 property the split builder keys on — computed, not assumed."""
    assert elliptic.meta.cross_time_edges is False
    assert np.array_equal(elliptic.node_time[elliptic.src], elliptic.node_time[elliptic.dst])
    assert np.array_equal(elliptic.edge_time, elliptic.node_time[elliptic.src])


def test_edges_sorted_by_time(elliptic: GraphDataset) -> None:
    assert (np.diff(elliptic.edge_time) >= 0).all()


def test_features_are_finite(elliptic: GraphDataset) -> None:
    assert np.isfinite(elliptic.x).all()


def test_elliptic_pp_extras_are_dropped(elliptic: GraphDataset) -> None:
    """The 17 extras include graph-derived degrees; keeping them would answer the
    base-vs-GFP question by accident (PR-M7)."""
    assert len(elliptic.meta.dropped_columns) == 17
    assert elliptic.meta.dropped_columns == EXTRA_COLUMNS
    assert "in_txs_degree" in elliptic.meta.dropped_columns
    assert "out_txs_degree" in elliptic.meta.dropped_columns
    assert not set(elliptic.meta.dropped_columns) & set(elliptic.meta.feature_names)


def test_feature_blocks_and_names(elliptic: GraphDataset) -> None:
    assert elliptic.meta.feature_blocks == {"local": (0, 93), "agg1hop": (93, 165)}
    assert len(elliptic.meta.feature_names) == 165
    assert elliptic.meta.feature_names[0] == "Local_feature_1"
    assert elliptic.meta.feature_names[92] == "Local_feature_93"
    assert elliptic.meta.feature_names[93] == "Aggregate_feature_1"
    assert elliptic.meta.feature_names[164] == "Aggregate_feature_72"


def test_provenance(elliptic: GraphDataset) -> None:
    assert elliptic.meta.dataset == "elliptic_pp"
    assert elliptic.meta.version == VERSION
    assert "EllipticPlusPlus" in elliptic.meta.source_url
    assert len(elliptic.meta.raw_sha256) == 16


def test_missing_files_name_the_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        load_elliptic_raw(tmp_path, VERSION)
    message = str(excinfo.value)
    assert str(tmp_path) in message
    for name in ("txs_features.csv", "txs_classes.csv", "txs_edgelist.csv"):
        assert name in message
    assert "no auto-download" in message
