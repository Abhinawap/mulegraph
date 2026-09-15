"""Split-builder tests (PR-E1, D1, PR-D4).

The synthetic fixtures stand in for Elliptic here: 600 nodes over 12 timesteps,
half of them unlabelled, which is enough to exercise every assertion the builder
makes without touching the real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mulegraph.config import RegimeConfig
from mulegraph.splits import builder
from mulegraph.splits.builder import (
    RegimeNotSupportedError,
    build_split,
    check_leakage,
    definition_hash,
    split_hash,
)
from mulegraph.types import GraphDataset

TEMPORAL = RegimeConfig(regime="temporal", train_end=6, val=(7, 9), test=(10, 12))
RANDOM = RegimeConfig(regime="random", fractions=(0.7, 0.15, 0.15), seed=0)


@pytest.mark.parametrize("cfg", [RANDOM, TEMPORAL], ids=["random", "temporal"])
def test_parts_are_disjoint_and_labelled(
    synthetic_ds: GraphDataset, cfg: RegimeConfig, tmp_path: Path
) -> None:
    split = build_split(synthetic_ds, cfg, tmp_path)
    parts = [split.train, split.val, split.test]

    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert np.intersect1d(parts[a], parts[b]).size == 0
    combined = np.concatenate(parts)
    assert np.unique(combined).size == combined.size
    assert (synthetic_ds.y[combined] >= 0).all()
    # Unlabelled nodes stay in the graph but never in a split.
    assert combined.size == synthetic_ds.labelled_idx.size


def test_temporal_split_is_chronological(synthetic_ds: GraphDataset, tmp_path: Path) -> None:
    split = build_split(synthetic_ds, TEMPORAL, tmp_path)
    time = synthetic_ds.node_time

    assert time[split.train].max() <= 6
    assert time[split.train].max() < time[split.val].min()
    assert time[split.val].min() >= 7 and time[split.val].max() <= 9
    assert time[split.val].max() < time[split.test].min()
    assert time[split.test].max() <= 12


def test_random_split_preserves_class_ratio(synthetic_ds: GraphDataset, tmp_path: Path) -> None:
    split = build_split(synthetic_ds, RANDOM, tmp_path)
    overall = float((synthetic_ds.y[synthetic_ds.labelled_idx] == 1).mean())

    for part in (split.train, split.val, split.test):
        assert float((synthetic_ds.y[part] == 1).mean()) == pytest.approx(overall, abs=0.05)
    assert split.train.size / synthetic_ds.labelled_idx.size == pytest.approx(0.7, abs=0.02)


def test_hash_is_stable_and_seed_sensitive(synthetic_ds: GraphDataset, tmp_path: Path) -> None:
    first = build_split(synthetic_ds, RANDOM, tmp_path)
    again = build_split(synthetic_ds, RANDOM, tmp_path)
    other_seed = build_split(
        synthetic_ds,
        RegimeConfig(regime="random", fractions=(0.7, 0.15, 0.15), seed=1),
        tmp_path,
    )

    assert first.split_hash == again.split_hash
    assert np.array_equal(first.train, again.train)
    assert other_seed.split_hash != first.split_hash


def test_temporal_inductive_rejected_without_cross_time_edges(
    synthetic_ds: GraphDataset, tmp_path: Path
) -> None:
    cfg = RegimeConfig(regime="temporal_inductive", train_end=6, val=(7, 9), test=(10, 12))
    assert synthetic_ds.meta.cross_time_edges is False

    with pytest.raises(RegimeNotSupportedError) as excinfo:
        build_split(synthetic_ds, cfg, tmp_path)
    message = str(excinfo.value)
    assert "cross_time_edges" in message
    assert synthetic_ds.meta.dataset in message


def test_temporal_inductive_is_v1b_where_it_is_defined(
    synthetic_cross_time_ds: GraphDataset, tmp_path: Path
) -> None:
    cfg = RegimeConfig(regime="temporal_inductive", train_end=6, val=(7, 9), test=(10, 12))
    assert synthetic_cross_time_ds.meta.cross_time_edges is True

    with pytest.raises(NotImplementedError, match="v1b"):
        build_split(synthetic_cross_time_ds, cfg, tmp_path)


def test_cache_is_written_and_reverified(synthetic_ds: GraphDataset, tmp_path: Path) -> None:
    split = build_split(synthetic_ds, TEMPORAL, tmp_path)
    cached = tmp_path / "splits" / definition_hash(split.params)

    assert {p.name for p in cached.iterdir()} == {
        "train.npy",
        "val.npy",
        "test.npy",
        "params.json",
    }
    assert np.array_equal(np.load(cached / "train.npy"), split.train)
    build_split(synthetic_ds, TEMPORAL, tmp_path)


def test_cache_is_keyed_on_the_definition_not_the_partition(
    synthetic_ds: GraphDataset, tmp_path: Path
) -> None:
    """A partition that changes under an unchanged definition must be caught.

    Keying the cache directory on the partition hash would send the changed
    partition to a *different* directory, miss the cache and write a second entry,
    so the comparison in ``_sync_cache`` could never fire (PR-D4).
    """
    build_split(synthetic_ds, RANDOM, tmp_path)
    original = builder._random_split

    def perturbed(data, cfg):
        train, val, test = original(data, cfg)
        return np.concatenate([train[:-1], val[:1]]), val[1:], test

    builder._random_split = perturbed
    try:
        with pytest.raises(ValueError, match="not deterministic"):
            build_split(synthetic_ds, RANDOM, tmp_path)
    finally:
        builder._random_split = original

    # ... and only one cache entry exists, because the key never moved.
    assert len(list((tmp_path / "splits").iterdir())) == 1


def test_check_leakage_rejects_overlap(synthetic_ds: GraphDataset, tmp_path: Path) -> None:
    split = build_split(synthetic_ds, TEMPORAL, tmp_path)
    leaky_train = np.sort(np.concatenate([split.train, split.test[:1]]))

    with pytest.raises(ValueError, match="train and test share"):
        check_leakage(synthetic_ds, "temporal", leaky_train, split.val, split.test)


def test_check_leakage_rejects_unlabelled_and_positive_free_parts(
    synthetic_ds: GraphDataset, tmp_path: Path
) -> None:
    split = build_split(synthetic_ds, TEMPORAL, tmp_path)
    unlabelled = np.flatnonzero(synthetic_ds.y < 0)[:1].astype(np.int64)

    with pytest.raises(ValueError, match="unlabelled"):
        check_leakage(
            synthetic_ds,
            "temporal",
            split.train,
            np.sort(np.append(split.val, unlabelled)),
            split.test,
        )

    negatives_only = split.val[synthetic_ds.y[split.val] == 0]
    with pytest.raises(ValueError, match="no positives"):
        check_leakage(synthetic_ds, "temporal", split.train, negatives_only, split.test)


def test_split_hash_depends_on_params(synthetic_ds: GraphDataset) -> None:
    idx = synthetic_ds.labelled_idx
    train, val, test = idx[:100], idx[100:150], idx[150:200]

    base = split_hash(train, val, test, {"regime": "random", "seed": 0})
    assert base == split_hash(train, val, test, {"regime": "random", "seed": 0})
    assert base != split_hash(train, val, test, {"regime": "random", "seed": 1})
    assert base != split_hash(train, val, test[:-1], {"regime": "random", "seed": 0})
