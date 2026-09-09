"""Edge-level GFP output folded down to node-level features (``node_agg_v1``).

GFP scores *edges*; Elliptic's task is a *node* task (D1), so something has to
map one to the other, and the choice is a feature definition rather than an
implementation detail. It is therefore versioned: ``node_agg_v1`` is named in
``FeaturesConfig.aggregation`` and hashed into ``feature_version``, so changing
it invalidates every cache and every logged run that used it (PR-F3).

The rules, one per kind of block:

* **Vertex-side histograms** (``fan_in``, ``degree_in`` describe the edge's
  target; ``fan_out``, ``degree_out`` its source) are folded by ``max`` onto the
  vertex they describe. Within one batch this is *exact*, not an approximation:
  GFP inserts the whole batch before scoring any edge in it, so every in-edge of
  a vertex reports the identical fan-in histogram for that vertex, and the max of
  a set of identical values is that value. ``max`` only starts to matter across
  batches, where it keeps the largest window a node was ever seen in.
* **Edge-pattern histograms** (``scatter_gather``, ``lc_cycle``, ``temp_cycle``)
  describe the edge, not either endpoint, so they are summed onto both endpoints:
  a node sitting on three laundering cycles is more interesting than one sitting
  on one, and multiplicity is the signal.
* **Vertex statistics** arrive as four edge-level blocks (source-out, source-in,
  target-out, target-in) and fold into six node columns: the two "outgoing"
  blocks both describe outgoing edges, of the source and target vertex
  respectively, so both feed the node's ``v_*_out`` columns via their own
  endpoint index; likewise for incoming.

The causality guard (PR-F2) is the ``node_time >= t`` mask. A node's feature row
is valid at its own timestamp, so a batch at *t* may only write into nodes that
already exist at *t*. On Elliptic this is always true — every edge sits inside
one timestep component — but on a graph with cross-timestep edges it is the
whole reason ``tests/test_features_causality.py`` passes: without it, an edge at
*t+3* would write into a node whose row claims to be valid at *t*.

Isolated nodes, and nodes all of whose incident edges the guard rejects, keep a
row of zeros. That is the honest answer — no graph evidence was available to
them causally — and it is what an unconnected transaction looks like.
"""

from __future__ import annotations

import numpy as np

from mulegraph.features.gfp import VERTEX_STAT_NAMES, GfpLayout

#: The six node columns the four edge-level vertex-statistic blocks fold into.
VERTEX_STAT_NODE_COLUMNS: tuple[str, ...] = tuple(
    f"v_{stat}_{direction}" for direction in ("out", "in") for stat in VERTEX_STAT_NAMES
)


def node_columns(layout: GfpLayout) -> list[str]:
    """Node-level column names, in the order :func:`node_agg_v1` writes them."""
    columns = [
        layout.columns[i] for block in layout.hist_blocks for i in range(block.start, block.stop)
    ]
    if layout.vertex_stat_blocks:
        columns.extend(VERTEX_STAT_NODE_COLUMNS)
    return columns


def node_width(layout: GfpLayout) -> int:
    """Width of one node feature row for this layout."""
    hist = sum(block.width for block in layout.hist_blocks)
    return hist + (len(VERTEX_STAT_NODE_COLUMNS) if layout.vertex_stat_blocks else 0)


def empty_node_features(num_nodes: int, layout: GfpLayout) -> np.ndarray:
    """Zeroed accumulator for :func:`node_agg_v1`."""
    return np.zeros((num_nodes, node_width(layout)), dtype=np.float32)


def node_agg_v1(
    node_feats: np.ndarray,
    edge_feats: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    node_time: np.ndarray,
    t: int,
    layout: GfpLayout,
) -> None:
    """Fold one batch of edge features into the node accumulator, in place.

    Args:
        node_feats: float32 ``[N, node_width(layout)]`` accumulator, mutated.
        edge_feats: float32 ``[B, layout.width]`` from ``GfpDriver.step``.
        src, dst: int64 ``[B]`` endpoint node indices for this batch.
        node_time: int64 ``[N]`` timestamp each node's row is valid at.
        t: the batch's timestamp.
        layout: the layout ``edge_feats`` was produced under.
    """
    if edge_feats.shape[1] != layout.width:
        raise ValueError(
            f"edge_feats has {edge_feats.shape[1]} columns, layout says {layout.width}"
        )
    if node_feats.shape[1] != node_width(layout):
        raise ValueError(
            f"node_feats has {node_feats.shape[1]} columns, layout implies {node_width(layout)}"
        )
    if not (src.shape == dst.shape == (edge_feats.shape[0],)):
        raise ValueError(f"src/dst must have shape [{edge_feats.shape[0]}]")

    # PR-F2: a batch at t may only write into nodes that already exist at t.
    allowed = node_time >= t
    src_ok = allowed[src]
    dst_ok = allowed[dst]

    cursor = 0
    for block in layout.hist_blocks:
        width = block.width
        out = node_feats[:, cursor : cursor + width]
        values = edge_feats[:, block.start : block.stop]
        if block.side == "dst":
            np.maximum.at(out, dst[dst_ok], values[dst_ok])
        elif block.side == "src":
            np.maximum.at(out, src[src_ok], values[src_ok])
        else:
            # The pattern belongs to the edge; both endpoints participate in it,
            # and how many such patterns touch a node is itself the signal.
            np.add.at(out, src[src_ok], values[src_ok])
            np.add.at(out, dst[dst_ok], values[dst_ok])
        cursor += width

    if not layout.vertex_stat_blocks:
        return
    stat_width = len(VERTEX_STAT_NAMES)
    offsets = {"out": cursor, "in": cursor + stat_width}
    for block in layout.vertex_stat_blocks:
        start = offsets[block.direction]
        out = node_feats[:, start : start + stat_width]
        values = edge_feats[:, block.start : block.stop]
        index, mask = (src, src_ok) if block.side == "src" else (dst, dst_ok)
        np.maximum.at(out, index[mask], values[mask])
