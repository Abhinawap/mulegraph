"""``node_agg_v1``: fold GFP edge output onto nodes; hashed into ``feature_version`` (PR-F3).

Vertex-side histograms fold by ``max`` onto their vertex, edge-pattern histograms sum onto both
endpoints, and the four vertex-statistic blocks fold into six ``v_*`` node columns.
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
    hist = sum(block.width for block in layout.hist_blocks)
    return hist + (len(VERTEX_STAT_NODE_COLUMNS) if layout.vertex_stat_blocks else 0)


def empty_node_features(num_nodes: int, layout: GfpLayout) -> np.ndarray:
    """Zeroed accumulator; nodes with no causal edges keep a zero row."""
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
    """Fold one batch at time ``t`` of edge features into ``node_feats``, in place."""
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
            # Edge patterns belong to both endpoints; multiplicity is the signal.
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
