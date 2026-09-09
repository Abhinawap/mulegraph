"""``node_agg_v1``: the edge-to-node fold (PR-F1, PR-F2).

The fold is a feature definition, not plumbing — it decides what a node's
``fan_in`` column *means* — so it is checked against a hand-built graph with
hand-written edge features, where every expected number can be read off by eye.
The last test then shows on real GFP output why ``max`` is the exact answer for
vertex-side blocks rather than a convenient approximation.
"""

from __future__ import annotations

import numpy as np
import pytest

from mulegraph.config import FeaturesConfig
from mulegraph.features.aggregate import (
    empty_node_features,
    node_agg_v1,
    node_columns,
    node_width,
)
from mulegraph.features.gfp import GfpDriver, expected_layout, make_gfp_params, probe_layout

# Five connected nodes plus an isolated one. Edges: 1->0, 2->0, 3->0, 0->4.
N_NODES = 6
SRC = np.array([1, 2, 3, 0], dtype=np.int64)
DST = np.array([0, 0, 0, 4], dtype=np.int64)
ISOLATED = 5


@pytest.fixture
def tiny_cfg() -> FeaturesConfig:
    """Two families and two bins, so the hand-written matrix stays readable."""
    return FeaturesConfig(families=["fan", "lc_cycle"], bins=[2, 4], cycle_len=4, num_threads=1)


def test_layout_of_the_tiny_config(tiny_cfg: FeaturesConfig) -> None:
    layout = expected_layout(tiny_cfg)
    assert [b.name for b in layout.blocks] == [
        "fan_in",
        "fan_out",
        "lc_cycle",
        "vs_src_out",
        "vs_src_in",
        "vs_dst_out",
        "vs_dst_in",
    ]
    assert layout.width == 3 * 2 + 4 * 3
    assert node_width(layout) == 3 * 2 + 6


def _hand_edge_features(layout) -> np.ndarray:
    """Edge features chosen so every fold rule produces a distinguishable number."""
    ef = np.zeros((4, layout.width), dtype=np.float32)
    # fan_in describes the target: the three edges into node 0 all report its
    # fan-in of 3; the edge into node 4 reports nothing.
    ef[0:3, 0:2] = [3.0, 0.0]
    # fan_out describes the source: only node 0 sends more than one edge here.
    ef[3, 2:4] = [1.0, 0.0]
    # lc_cycle describes the edge itself.
    ef[:, 4:6] = [1.0, 0.0]
    # Vertex statistics, varied per edge so that max is actually exercised.
    for e in range(4):
        ef[e, 6:9] = [10 + e, 11 + e, 12 + e]  # vs_src_out
        ef[e, 9:12] = [20 + e, 21 + e, 22 + e]  # vs_src_in
        ef[e, 12:15] = [30 + e, 31 + e, 32 + e]  # vs_dst_out
        ef[e, 15:18] = [40 + e, 41 + e, 42 + e]  # vs_dst_in
    return ef


def _fold(layout, node_time: np.ndarray, t: int = 1) -> np.ndarray:
    node_feats = empty_node_features(N_NODES, layout)
    node_agg_v1(node_feats, _hand_edge_features(layout), SRC, DST, node_time, t, layout)
    return node_feats


def test_vertex_side_blocks_fold_by_max(tiny_cfg: FeaturesConfig) -> None:
    layout = expected_layout(tiny_cfg)
    columns = node_columns(layout)
    feats = _fold(layout, np.ones(N_NODES, dtype=np.int64))

    fan_in = feats[:, columns.index("fan_in_bin2")]
    assert fan_in[0] == pytest.approx(3.0)  # target of all three in-edges
    assert np.all(fan_in[[1, 2, 3, 4, ISOLATED]] == 0.0)

    fan_out = feats[:, columns.index("fan_out_bin2")]
    assert fan_out[0] == pytest.approx(1.0)  # source of 0->4
    assert np.all(fan_out[[1, 2, 3, 4, ISOLATED]] == 0.0)


def test_edge_pattern_blocks_fold_by_sum_onto_both_endpoints(tiny_cfg: FeaturesConfig) -> None:
    layout = expected_layout(tiny_cfg)
    columns = node_columns(layout)
    feats = _fold(layout, np.ones(N_NODES, dtype=np.int64))

    cycles = feats[:, columns.index("lc_cycle_bin2")]
    # Node 0 sits on all four edges; nodes 1-4 on one each.
    assert cycles[0] == pytest.approx(4.0)
    assert np.all(cycles[[1, 2, 3, 4]] == 1.0)
    assert cycles[ISOLATED] == 0.0


def test_vertex_statistics_fold_into_six_directional_columns(tiny_cfg: FeaturesConfig) -> None:
    layout = expected_layout(tiny_cfg)
    columns = node_columns(layout)
    feats = _fold(layout, np.ones(N_NODES, dtype=np.int64))
    out_cols = [columns.index(c) for c in ("v_fan_out", "v_degree_out", "v_ratio_out")]
    in_cols = [columns.index(c) for c in ("v_fan_in", "v_degree_in", "v_ratio_in")]

    # Node 0 is the source of edge 3 (vs_src_* -> [13, 14, 15] / [23, 24, 25]) and
    # the target of edges 0-2 (vs_dst_* -> max over e in {0,1,2}).
    assert feats[0, out_cols] == pytest.approx([32.0, 33.0, 34.0])
    assert feats[0, in_cols] == pytest.approx([42.0, 43.0, 44.0])
    # Node 1 is only the source of edge 0.
    assert feats[1, out_cols] == pytest.approx([10.0, 11.0, 12.0])
    assert feats[1, in_cols] == pytest.approx([20.0, 21.0, 22.0])
    # Node 4 is only the target of edge 3.
    assert feats[4, out_cols] == pytest.approx([33.0, 34.0, 35.0])
    assert feats[4, in_cols] == pytest.approx([43.0, 44.0, 45.0])


def test_isolated_node_stays_zero(tiny_cfg: FeaturesConfig) -> None:
    layout = expected_layout(tiny_cfg)
    feats = _fold(layout, np.ones(N_NODES, dtype=np.int64))
    assert np.all(feats[ISOLATED] == 0.0)


def test_a_batch_never_writes_into_a_node_that_predates_it(tiny_cfg: FeaturesConfig) -> None:
    """PR-F2 at the fold: node 1's row is valid at t=1 and may not see t=2 edges."""
    layout = expected_layout(tiny_cfg)
    columns = node_columns(layout)
    node_time = np.array([2, 1, 2, 2, 2, 2], dtype=np.int64)
    feats = _fold(layout, node_time, t=2)

    assert np.all(feats[1] == 0.0)
    # Its counterparty, which does exist at t=2, still gets its contribution.
    assert feats[0, columns.index("fan_in_bin2")] == pytest.approx(3.0)


def test_node_columns_match_the_folded_width(tiny_cfg: FeaturesConfig) -> None:
    layout = expected_layout(tiny_cfg)
    columns = node_columns(layout)
    assert len(columns) == node_width(layout)
    assert columns[-6:] == [
        "v_fan_out",
        "v_degree_out",
        "v_ratio_out",
        "v_fan_in",
        "v_degree_in",
        "v_ratio_in",
    ]


def test_max_is_exact_on_real_gfp_output(synthetic_ds, features_cfg: FeaturesConfig) -> None:
    """Every out-edge of a vertex carries identical source-side values.

    GFP inserts the whole batch before scoring any edge in it, so the source-side
    blocks are a property of the vertex, not of the edge. Folding them with
    ``max`` therefore reproduces the vertex's own value exactly; it is not a
    summary of differing values.
    """
    layout = probe_layout(features_cfg)
    driver = GfpDriver(make_gfp_params(features_cfg), layout)
    first_t = int(np.min(synthetic_ds.edge_time))
    mask = np.flatnonzero(synthetic_ds.edge_time == first_t)
    batch = np.column_stack(
        [
            mask.astype(np.float64),
            synthetic_ds.src[mask].astype(np.float64),
            synthetic_ds.dst[mask].astype(np.float64),
            np.full(mask.size, first_t, dtype=np.float64),
            np.ones(mask.size, dtype=np.float64),
        ]
    )
    edge_feats = driver.step(batch)

    src_side = [b for b in layout.blocks if b.side == "src"]
    assert src_side, "the layout must have source-side blocks for this to mean anything"
    src_columns = np.concatenate([np.arange(b.start, b.stop) for b in src_side])
    checked = 0
    for vertex in np.unique(synthetic_ds.src[mask]):
        rows = edge_feats[np.flatnonzero(synthetic_ds.src[mask] == vertex)][:, src_columns]
        if rows.shape[0] < 2:
            continue
        assert np.array_equal(rows, np.broadcast_to(rows[0], rows.shape))
        checked += 1
    assert checked > 0, "no vertex had two out-edges in this timestep; the test proved nothing"
