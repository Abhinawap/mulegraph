"""Feature selection, and the PR-M7 boundary it enforces.

``base`` on Elliptic is the 93 *local* transaction features and nothing else.
The published 165-feature block continues with 72 *one-hop aggregated neighbour*
features; if ``base`` reached any of them, both arms of the base vs base+GFP
comparison would already contain graph information and G1 — how much of the gain
comes from graph features — would be unanswerable. That boundary is asserted in
:mod:`mulegraph.features.select`, and these tests check the assertion actually
fires rather than sitting there decoratively.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mulegraph.config import FeaturesConfig
from mulegraph.features.builder import build_features
from mulegraph.features.select import select_features
from mulegraph.types import FeatureMatrix, GraphDataset

N_LOCAL = 93
N_PUBLISHED = 165
N_GFP = 36


@pytest.fixture
def gfp_matrix(synthetic_ds: GraphDataset, tmp_path) -> FeatureMatrix:
    """Real graph features under the MVP's default feature config (36 columns)."""
    return build_features(synthetic_ds, FeaturesConfig(num_threads=2), tmp_path)


def _with_blocks(data: GraphDataset, blocks: dict[str, tuple[int, int]]) -> GraphDataset:
    meta = dataclasses.replace(data.meta, feature_blocks=blocks)
    return dataclasses.replace(data, meta=meta)


def test_widths_are_93_129_and_165(synthetic_ds: GraphDataset, gfp_matrix: FeatureMatrix) -> None:
    assert gfp_matrix.num_features == N_GFP
    assert select_features(synthetic_ds, None, "base").num_features == N_LOCAL
    assert (
        select_features(synthetic_ds, gfp_matrix, "base_gfp").num_features == N_LOCAL + N_GFP == 129
    )
    assert select_features(synthetic_ds, None, "raw165").num_features == N_PUBLISHED


def test_base_is_exactly_the_local_block(synthetic_ds: GraphDataset) -> None:
    selected = select_features(synthetic_ds, None, "base")
    np.testing.assert_array_equal(selected.values, synthetic_ds.x[:, :N_LOCAL])
    assert selected.columns == synthetic_ds.meta.feature_names[:N_LOCAL]
    assert selected.name == "base"
    assert selected.feature_version == "none"
    assert selected.blocks == {"local": (0, N_LOCAL)}


def test_base_gfp_is_base_then_graph_features(
    synthetic_ds: GraphDataset, gfp_matrix: FeatureMatrix
) -> None:
    selected = select_features(synthetic_ds, gfp_matrix, "base_gfp")
    np.testing.assert_array_equal(selected.values[:, :N_LOCAL], synthetic_ds.x[:, :N_LOCAL])
    np.testing.assert_array_equal(selected.values[:, N_LOCAL:], gfp_matrix.values)
    assert selected.columns[N_LOCAL:] == gfp_matrix.columns
    # The gfp hash is what a run logs as its feature_version (PR-F3).
    assert selected.feature_version == gfp_matrix.feature_version
    assert selected.blocks == {"local": (0, N_LOCAL), "gfp": (N_LOCAL, N_LOCAL + N_GFP)}


def test_base_gfp_without_graph_features_is_an_error(synthetic_ds: GraphDataset) -> None:
    with pytest.raises(ValueError, match="base_gfp needs the causal graph features"):
        select_features(synthetic_ds, None, "base_gfp")


def test_raw165_keeps_the_published_block_intact(synthetic_ds: GraphDataset) -> None:
    selected = select_features(synthetic_ds, None, "raw165")
    np.testing.assert_array_equal(selected.values, synthetic_ds.x)
    assert selected.name == "raw165"
    assert selected.blocks == {"local": (0, N_LOCAL), "agg1hop": (N_LOCAL, N_PUBLISHED)}


def test_raw165_rejects_a_dataset_without_the_aggregate_block(
    synthetic_ds: GraphDataset,
) -> None:
    """It is the published 165-feature reference row or nothing (D3, PR-M1)."""
    data = _with_blocks(synthetic_ds, {"local": (0, N_PUBLISHED)})
    with pytest.raises(ValueError, match="raw165"):
        select_features(data, None, "raw165")


def test_pr_m7_assertion_fires_when_base_would_reach_the_aggregates(
    synthetic_ds: GraphDataset,
) -> None:
    """The mechanical enforcement of PR-M7: overlapping blocks must not select."""
    data = _with_blocks(synthetic_ds, {"local": (0, 100), "agg1hop": (N_LOCAL, N_PUBLISHED)})
    with pytest.raises(ValueError, match="PR-M7 violation"):
        select_features(data, None, "base")


def test_pr_m7_assertion_fires_on_a_gap_between_the_blocks(
    synthetic_ds: GraphDataset,
) -> None:
    data = _with_blocks(synthetic_ds, {"local": (0, 80), "agg1hop": (N_LOCAL, N_PUBLISHED)})
    with pytest.raises(ValueError, match="PR-M7 violation"):
        select_features(data, None, "base")


def test_base_needs_a_local_block(synthetic_ds: GraphDataset) -> None:
    data = _with_blocks(synthetic_ds, {"agg1hop": (0, N_PUBLISHED)})
    with pytest.raises(ValueError, match="'base' features are undefined"):
        select_features(data, None, "base")


def test_unknown_selection_is_rejected(synthetic_ds: GraphDataset) -> None:
    with pytest.raises(ValueError, match="unknown feature selection"):
        select_features(synthetic_ds, None, "everything")
