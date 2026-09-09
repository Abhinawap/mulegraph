"""Synthetic loader: shapes, dtypes, label counts, determinism, caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mulegraph.config import DatasetConfig
from mulegraph.data import load_dataset, subsample
from mulegraph.data.synthetic import N_LOCAL, make_synthetic_elliptic
from mulegraph.types import GraphDataset


def test_shapes_and_dtypes(synthetic_ds: GraphDataset) -> None:
    d = synthetic_ds
    assert d.x.shape == (600, 165)
    assert d.x.dtype == np.float32
    assert d.edge_index.shape[0] == 2
    for arr in (d.node_time, d.edge_time, d.batch_id, d.y, d.node_ids):
        assert arr.dtype == np.int64
    assert d.node_time.shape == (600,)
    assert d.edge_time.shape == (d.num_edges,)
    assert d.task == "node"


def test_feature_blocks_match_elliptic(synthetic_ds: GraphDataset) -> None:
    assert synthetic_ds.meta.feature_blocks == {"local": (0, N_LOCAL), "agg1hop": (N_LOCAL, 165)}
    assert len(synthetic_ds.meta.feature_names) == 165


def test_labels_are_the_three_expected_values(synthetic_ds: GraphDataset) -> None:
    assert set(np.unique(synthetic_ds.y)).issubset({-1, 0, 1})
    counts = synthetic_ds.meta.label_counts
    assert sum(counts.values()) == synthetic_ds.num_nodes
    assert counts["illicit"] > 0
    assert counts["unknown"] > 0


def test_labelled_idx_excludes_unknown(synthetic_ds: GraphDataset) -> None:
    idx = synthetic_ds.labelled_idx
    assert idx.dtype == np.int64
    assert (synthetic_ds.y[idx] >= 0).all()
    assert idx.size == synthetic_ds.num_nodes - synthetic_ds.meta.label_counts["unknown"]


def test_no_cross_time_edges_by_default(synthetic_ds: GraphDataset) -> None:
    d = synthetic_ds
    assert d.meta.cross_time_edges is False
    assert np.array_equal(d.node_time[d.src], d.node_time[d.dst])


def test_cross_time_variant_really_spans_timesteps(
    synthetic_cross_time_ds: GraphDataset,
) -> None:
    d = synthetic_cross_time_ds
    assert d.meta.cross_time_edges is True
    assert (d.node_time[d.src] != d.node_time[d.dst]).sum() > 0


def test_edge_time_is_the_later_endpoint(synthetic_cross_time_ds: GraphDataset) -> None:
    # A node must never carry an incident edge that predates its own appearance.
    d = synthetic_cross_time_ds
    assert np.array_equal(d.edge_time, np.maximum(d.node_time[d.src], d.node_time[d.dst]))
    assert np.all(np.diff(d.edge_time) >= 0), "edges must arrive in time order"


def test_no_self_loops(synthetic_ds: GraphDataset) -> None:
    assert not (synthetic_ds.src == synthetic_ds.dst).any()


def test_illicit_nodes_carry_planted_signal(synthetic_ds: GraphDataset) -> None:
    d = synthetic_ds
    illicit = d.y == 1
    licit = d.y == 0
    assert d.x[illicit, :10].mean() > d.x[licit, :10].mean() + 0.5
    # ... and the fan-in star makes them structurally distinguishable too.
    in_degree = np.bincount(d.dst, minlength=d.num_nodes)
    assert in_degree[illicit].mean() > in_degree[licit].mean()


def test_generation_is_deterministic() -> None:
    a = make_synthetic_elliptic(n_nodes=300, n_timesteps=8, seed=7)
    b = make_synthetic_elliptic(n_nodes=300, n_timesteps=8, seed=7)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.edge_index, b.edge_index)
    assert np.array_equal(a.y, b.y)


def test_different_seeds_differ() -> None:
    a = make_synthetic_elliptic(n_nodes=300, n_timesteps=8, seed=0)
    b = make_synthetic_elliptic(n_nodes=300, n_timesteps=8, seed=1)
    assert not np.array_equal(a.y, b.y)


def test_too_few_features_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 93"):
        make_synthetic_elliptic(n_features=10)


def test_cache_round_trip(tmp_data_dir: Path) -> None:
    cfg = DatasetConfig(name="synthetic_elliptic", version="t")
    first = load_dataset(cfg, tmp_data_dir)
    assert (tmp_data_dir / "cache" / "synthetic_elliptic" / "t" / "graph.pt").is_file()

    second = load_dataset(cfg, tmp_data_dir)  # served from cache this time
    assert np.array_equal(first.x, second.x)
    assert np.array_equal(first.edge_index, second.edge_index)
    assert np.array_equal(first.y, second.y)
    assert second.meta.feature_blocks == first.meta.feature_blocks
    assert isinstance(next(iter(second.meta.feature_blocks.values())), tuple)


def test_unknown_dataset_name_is_rejected(tmp_data_dir: Path) -> None:
    cfg = DatasetConfig(name="synthetic_elliptic")
    object.__setattr__(cfg, "name", "nope")
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset(cfg, tmp_data_dir)


def test_subsample_keeps_only_internal_edges(synthetic_ds: GraphDataset) -> None:
    keep = np.arange(0, synthetic_ds.num_nodes, 2, dtype=np.int64)
    sub = subsample(synthetic_ds, keep)
    assert sub.num_nodes == keep.size
    assert sub.num_edges <= synthetic_ds.num_edges
    if sub.num_edges:
        assert sub.edge_index.max() < sub.num_nodes
