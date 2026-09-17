"""GFP's output layout, and the drive pattern that keeps it causal (PR-F1, PR-F2).

These tests exist because everything downstream — the aggregation, the column
names, the cached parquet — indexes into snapml's output by position. If a
snapml upgrade reorders or resizes a block, the features silently change meaning
and every logged number becomes unreproducible, so the layout is pinned here
rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from mulegraph.config import FeaturesConfig
from mulegraph.features.aggregate import empty_node_features, node_agg_v1, node_columns
from mulegraph.features.builder import build_features
from mulegraph.features.gfp import (
    RAW_WIDTH,
    VERTEX_STAT_NAMES,
    GfpDriver,
    expected_layout,
    make_gfp_params,
    probe_layout,
)

GFP_SOURCE = "mulegraph/features/gfp.py"


def _batch(edges: list[tuple[int, int, int]], first_id: int = 0) -> np.ndarray:
    """``[(src, dst, t), ...]`` as snapml input rows with globally unique edge ids."""
    return np.array(
        [[first_id + i, s, d, t, 1.0] for i, (s, d, t) in enumerate(edges)], dtype=np.float64
    )


def _block(layout, name):
    return next(b for b in layout.blocks if b.name == name)


def _cols(layout, out: np.ndarray, name: str) -> np.ndarray:
    block = _block(layout, name)
    return out[:, block.start : block.stop]


# --------------------------------------------------------------------------- #
# Width and column names
# --------------------------------------------------------------------------- #


def test_per_family_width_is_bins_times_block_count(features_cfg: FeaturesConfig) -> None:
    layout = expected_layout(features_cfg)
    n_bins = len(features_cfg.bins)
    # fan and degree each contribute an in- and an out-block; the pattern
    # families contribute one apiece.
    expected_blocks = {
        "fan": ["fan_in", "fan_out"],
        "degree": ["degree_in", "degree_out"],
        "scatter_gather": ["scatter_gather"],
        "lc_cycle": ["lc_cycle"],
    }
    for family, names in expected_blocks.items():
        assert family in features_cfg.families
        for name in names:
            assert _block(layout, name).width == n_bins
    assert len(layout.hist_blocks) == 6
    assert len(layout.vertex_stat_blocks) == 4
    assert layout.width == 6 * n_bins + 4 * len(VERTEX_STAT_NAMES)


def test_default_config_widths_are_42_edge_and_36_node() -> None:
    """The MVP's actual numbers, so a config drift is visible in one place."""
    layout = expected_layout(FeaturesConfig())
    assert layout.width == 42
    assert len(node_columns(layout)) == 36


def test_column_names_follow_block_order(features_cfg: FeaturesConfig) -> None:
    layout = expected_layout(features_cfg)
    assert layout.columns[:3] == ("fan_in_bin2", "fan_in_bin4", "fan_in_bin8")
    assert layout.columns[-len(VERTEX_STAT_NAMES) * 4] == "vs_src_out_fan"
    assert layout.columns[-1] == "vs_dst_in_ratio"
    assert len(set(layout.columns)) == len(layout.columns)


def test_probe_layout_matches_expected_layout(features_cfg: FeaturesConfig) -> None:
    """snapml's real output width equals the arithmetic, with no call needed to know it."""
    assert probe_layout(features_cfg) == expected_layout(features_cfg)


def test_disabled_families_are_absent_but_declared(features_cfg: FeaturesConfig) -> None:
    cfg = features_cfg.model_copy(update={"families": ["degree"]})
    layout = probe_layout(cfg)
    assert [b.name for b in layout.hist_blocks] == ["degree_in", "degree_out"]
    params = make_gfp_params(cfg)
    # Every family key is present so the param dict is a total definition.
    assert params["fan"] is False and params["degree"] is True
    assert params["lc-cycle_len"] == cfg.cycle_len


# --------------------------------------------------------------------------- #
# Structural fingerprints: does each block describe what its name claims?
# --------------------------------------------------------------------------- #


def test_fan_in_star_lights_only_the_in_blocks(features_cfg: FeaturesConfig) -> None:
    layout = probe_layout(features_cfg)
    driver = GfpDriver(make_gfp_params(features_cfg), layout)
    out = driver.step(_batch([(1, 0, 1), (2, 0, 1), (3, 0, 1)]))

    assert _cols(layout, out, "fan_in").any()
    assert _cols(layout, out, "degree_in").any()
    assert not _cols(layout, out, "fan_out").any()
    assert not _cols(layout, out, "degree_out").any()
    # The target's incoming statistics see three distinct senders; each sender's
    # own outgoing statistics see exactly one edge.
    assert _cols(layout, out, "vs_dst_in")[0, 0] == pytest.approx(3.0)
    assert np.all(_cols(layout, out, "vs_src_out")[:, 1] == 1.0)


def test_fan_out_star_lights_only_the_out_blocks(features_cfg: FeaturesConfig) -> None:
    layout = probe_layout(features_cfg)
    driver = GfpDriver(make_gfp_params(features_cfg), layout)
    out = driver.step(_batch([(0, 1, 1), (0, 2, 1), (0, 3, 1)]))

    assert _cols(layout, out, "fan_out").any()
    assert _cols(layout, out, "degree_out").any()
    assert not _cols(layout, out, "fan_in").any()
    assert not _cols(layout, out, "degree_in").any()
    assert _cols(layout, out, "vs_src_out")[0, 0] == pytest.approx(3.0)


def test_three_cycle_lights_the_lc_cycle_block(features_cfg: FeaturesConfig) -> None:
    layout = probe_layout(features_cfg)
    driver = GfpDriver(make_gfp_params(features_cfg), layout)
    out = driver.step(_batch([(0, 1, 1), (1, 2, 1), (2, 0, 1)]))

    assert _cols(layout, out, "lc_cycle").any()
    # A 3-cycle is not a scatter-gather pattern, and every vertex has one edge in
    # and one out, so no fan pattern reaches the two-neighbour threshold either.
    assert not _cols(layout, out, "scatter_gather").any()
    assert not _cols(layout, out, "fan_in").any()
    assert not _cols(layout, out, "fan_out").any()


# --------------------------------------------------------------------------- #
# The drive pattern (PR-F2)
# --------------------------------------------------------------------------- #


def test_no_double_count_and_why_partial_fit_is_forbidden(features_cfg: FeaturesConfig) -> None:
    """A vertex with two out-edges reports degree 2 — and 4 if driven wrongly.

    ``transform`` inserts the batch itself. Calling ``partial_fit(batch)`` first
    inserts it a second time, so every degree, fan and histogram count doubles.
    That is why :class:`GfpDriver` exposes no fit method at all; this test pins
    the failure mode so the constraint survives being read as an arbitrary
    stylistic choice.
    """
    from snapml import GraphFeaturePreprocessor

    layout = probe_layout(features_cfg)
    params = make_gfp_params(features_cfg)
    edges = _batch([(0, 1, 1), (0, 2, 1)])

    driver = GfpDriver(params, layout)
    edge_feats = driver.step(edges)
    node_feats = empty_node_features(3, layout)
    node_time = np.array([1, 1, 1], dtype=np.int64)
    node_agg_v1(
        node_feats,
        edge_feats,
        edges[:, 1].astype(np.int64),
        edges[:, 2].astype(np.int64),
        node_time,
        1,
        layout,
    )
    columns = node_columns(layout)
    right = node_feats[0, columns.index("v_degree_out")]
    assert right == pytest.approx(2.0)

    wrong_gp = GraphFeaturePreprocessor()
    wrong_gp.set_params(dict(params))
    wrong_gp.partial_fit(edges)
    wrong = wrong_gp.transform(edges)
    vs_src_out = _block(layout, "vs_src_out")
    degree_column = RAW_WIDTH + vs_src_out.start + VERTEX_STAT_NAMES.index("degree")
    assert wrong[0, degree_column] == pytest.approx(4.0)


def test_gfp_module_never_calls_partial_fit(repo_root) -> None:
    source = (repo_root / GFP_SOURCE).read_text()
    assert "partial_fit" not in source, (
        f"{GFP_SOURCE} must not reference the incremental-fit entry point: it inserts a batch "
        "that transform() then inserts again, doubling every count (PR-F2)"
    )


def test_batch_order_within_a_timestep_is_irrelevant(features_cfg: FeaturesConfig) -> None:
    """GFP inserts the whole batch before scoring any edge in it."""
    layout = probe_layout(features_cfg)
    edges = _batch([(1, 0, 1), (2, 0, 1), (3, 0, 1), (0, 4, 1)])

    forward = GfpDriver(make_gfp_params(features_cfg), layout).step(edges)
    reversed_out = GfpDriver(make_gfp_params(features_cfg), layout).step(edges[::-1].copy())
    assert np.array_equal(forward, reversed_out[::-1])


def test_window_bounds_how_far_back_a_batch_can_see(features_cfg: FeaturesConfig) -> None:
    """Positive control for the window: ``tw = k`` means batch *t* sees *t-k+1 … t*.

    Also records a snapml 1.17.2 quirk: the vertex-statistic block ignores
    ``vertex_stats_tw`` and accumulates over every batch inserted so far. That is
    still past-only, so PR-F2 holds, but ``vs_*`` columns are cumulative rather
    than windowed and the write-up says so.
    """
    first = _batch([(1, 0, 1), (2, 0, 1)], first_id=0)
    second = _batch([(3, 0, 2), (4, 0, 2)], first_id=10)

    results = {}
    for window in (1, 2):
        cfg = features_cfg.model_copy(update={"window": window})
        layout = probe_layout(cfg)
        driver = GfpDriver(make_gfp_params(cfg), layout)
        driver.step(first)
        out = driver.step(second)
        results[window] = (layout, out)

    layout_1, out_1 = results[1]
    layout_2, out_2 = results[2]
    # Two in-edges inside the window vs four: a wider window moves the fan-in
    # histogram into a higher bin.
    assert (
        _cols(layout_1, out_1, "fan_in")[0].argmax() < _cols(layout_2, out_2, "fan_in")[0].argmax()
    )
    assert not np.array_equal(_cols(layout_1, out_1, "fan_in"), _cols(layout_2, out_2, "fan_in"))
    # ... while the vertex statistics are identical, because they never evict.
    assert np.array_equal(_cols(layout_1, out_1, "vs_dst_in"), _cols(layout_2, out_2, "vs_dst_in"))
    assert _cols(layout_1, out_1, "vs_dst_in")[0, 1] == pytest.approx(4.0)


def test_step_rejects_a_batch_that_goes_back_in_time(features_cfg: FeaturesConfig) -> None:
    layout = probe_layout(features_cfg)
    driver = GfpDriver(make_gfp_params(features_cfg), layout)
    driver.step(_batch([(1, 0, 5)], first_id=0))
    with pytest.raises(ValueError, match="earlier than the previous batch"):
        driver.step(_batch([(2, 0, 4)], first_id=10))


def test_step_rejects_a_batch_spanning_two_timesteps(features_cfg: FeaturesConfig) -> None:
    layout = probe_layout(features_cfg)
    driver = GfpDriver(make_gfp_params(features_cfg), layout)
    with pytest.raises(ValueError, match="exactly one timestep"):
        driver.step(_batch([(1, 0, 1), (2, 0, 2)]))


def test_driver_exposes_no_way_to_insert_without_scoring() -> None:
    assert not hasattr(GfpDriver, "fit")
    assert not hasattr(GfpDriver, "partial_fit")


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def test_igraph_backend_is_retired(tmp_path, synthetic_ds, features_cfg: FeaturesConfig) -> None:
    cfg = features_cfg.model_copy(update={"backend": "igraph"})
    with pytest.raises(NotImplementedError, match="week-1 gate 1"):
        build_features(synthetic_ds, cfg, tmp_path)
