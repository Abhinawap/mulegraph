"""Feature causality (PR-F2) — the test the feature builder exists to pass.

The claim: a node's feature row, valid at timestep *t*, is identical whether or
not the graph contains any edge after *t*. GFP cannot give that for free — it is
stateful, and ``transform`` inserts the batch it is scoring — so causality is a
property of *how* the builder drives it, and only a graph whose edges genuinely
span timesteps can tell the difference.

Elliptic cannot: its timesteps are disconnected components (D1), so there is
nothing for a feature to peek at and the property holds vacuously. The spec
(§2.5) therefore requires the test to run on a synthetic multi-timestep fixture
and to be *skipped with a logged reason* on Elliptic-shaped data. Both happen
below.

The negative test alone would pass on a builder that ignored the graph entirely,
so each truncation is paired with a positive control: deleting batches that the
window does reach must change the same rows.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mulegraph.config import FeaturesConfig
from mulegraph.features.builder import build_features
from mulegraph.types import GraphDataset

log = logging.getLogger("mulegraph")

#: Wide enough that GFP state genuinely carries across batches, so truncating the
#: future is a real opportunity to leak rather than a no-op.
WINDOW = 3

CUTOFFS = (4, 7, 10)


@pytest.fixture
def causal_cfg(features_cfg: FeaturesConfig) -> FeaturesConfig:
    return features_cfg.model_copy(update={"window": WINDOW})


def _subgraph(data: GraphDataset, keep: np.ndarray) -> GraphDataset:
    """The same nodes, with only the edges selected by ``keep``.

    Nodes are untouched so that row *i* means the same node in every variant and
    the matrices can be compared elementwise.
    """
    return GraphDataset(
        x=data.x,
        edge_index=data.edge_index[:, keep],
        edge_attr=None if data.edge_attr is None else data.edge_attr[keep],
        node_time=data.node_time,
        edge_time=data.edge_time[keep],
        batch_id=data.batch_id,
        y=data.y,
        node_ids=data.node_ids,
        task=data.task,
        meta=data.meta,
    )


def _features(data: GraphDataset, cfg: FeaturesConfig, tmp_path, tag: str) -> np.ndarray:
    # A separate cache directory per variant: feature_version hashes the feature
    # *definition* and the dataset version, not the edge set, so three variants of
    # one dataset would otherwise collide on a single cache entry.
    return build_features(data, cfg, tmp_path / tag).values


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_features_ignore_edges_after_the_node_timestep(
    synthetic_cross_time_ds: GraphDataset, causal_cfg: FeaturesConfig, tmp_path, cutoff: int
) -> None:
    """PR-F2: rows valid at or before ``cutoff`` do not move when the future is added."""
    data = synthetic_cross_time_ds
    assert data.meta.cross_time_edges, "fixture must have cross-timestep edges to test anything"

    full = _features(data, causal_cfg, tmp_path, f"full_{cutoff}")
    truncated = _features(
        _subgraph(data, np.flatnonzero(data.edge_time <= cutoff)),
        causal_cfg,
        tmp_path,
        f"trunc_{cutoff}",
    )

    rows = np.flatnonzero(data.node_time <= cutoff)
    assert rows.size > 0
    np.testing.assert_array_equal(
        full[rows],
        truncated[rows],
        err_msg=(
            f"PR-F2 violation: features for nodes at t <= {cutoff} changed when edges after "
            f"{cutoff} were added to the graph"
        ),
    )


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_removing_recent_past_does_change_those_rows(
    synthetic_cross_time_ds: GraphDataset, causal_cfg: FeaturesConfig, tmp_path, cutoff: int
) -> None:
    """Positive control: without it, the test above would pass on constant features."""
    data = synthetic_cross_time_ds
    truncated = _features(
        _subgraph(data, np.flatnonzero(data.edge_time <= cutoff)),
        causal_cfg,
        tmp_path,
        f"trunc_{cutoff}",
    )
    # Delete the batches strictly inside the window the cutoff batch can still see.
    edge_time = data.edge_time
    keep = (edge_time <= cutoff) & ~((edge_time > cutoff - WINDOW) & (edge_time < cutoff))
    holed = _features(
        _subgraph(data, np.flatnonzero(keep)), causal_cfg, tmp_path, f"holed_{cutoff}"
    )

    rows = np.flatnonzero(data.node_time <= cutoff)
    assert not np.array_equal(truncated[rows], holed[rows]), (
        f"deleting batches in ({cutoff - WINDOW}, {cutoff}) left the features for t <= {cutoff} "
        "unchanged, so the causality test above is not measuring anything"
    )


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_edge_features_ignore_edges_after_their_own_time(
    causal_cfg: FeaturesConfig, tmp_path, cutoff: int
) -> None:
    """PR-F2 on an edge task: an edge's row does not move when later edges are added."""
    from tests.test_splits import make_edge_ds

    data = make_edge_ds()
    keep = np.flatnonzero(data.edge_time <= cutoff)
    # Edges are time-sorted, so the kept edges are a prefix and rows line up by position.
    assert keep.size > 0 and np.array_equal(keep, np.arange(keep.size))
    truncated = GraphDataset(
        x=data.x,
        edge_index=data.edge_index[:, keep],
        edge_attr=data.edge_attr[keep],  # type: ignore[index]
        node_time=data.node_time,
        edge_time=data.edge_time[keep],
        batch_id=data.batch_id[keep],
        y=data.y[keep],
        node_ids=data.node_ids,
        task=data.task,
        meta=data.meta,
    )

    full = _features(data, causal_cfg, tmp_path, f"edge_full_{cutoff}")
    trunc = _features(truncated, causal_cfg, tmp_path, f"edge_trunc_{cutoff}")
    assert full.shape[0] == data.num_edges
    np.testing.assert_array_equal(
        full[keep],
        trunc,
        err_msg=f"PR-F2 violation: edge rows at t <= {cutoff} changed when later edges were added",
    )


def test_causality_is_vacuous_without_cross_timestep_edges(
    synthetic_ds: GraphDataset, caplog: pytest.LogCaptureFixture
) -> None:
    """Elliptic-shaped case: skipped, with the reason logged as the spec requires."""
    if synthetic_ds.meta.cross_time_edges:  # pragma: no cover - fixture guarantees otherwise
        pytest.fail("the Elliptic-shaped fixture must not have cross-timestep edges")
    reason = "no cross-timestep edges (meta.cross_time_edges=False); causality is vacuous"
    with caplog.at_level(logging.INFO, logger="mulegraph"):
        log.info("skipping feature causality test on %s: %s", synthetic_ds.meta.dataset, reason)
    assert reason in caplog.text
    pytest.skip(reason)
