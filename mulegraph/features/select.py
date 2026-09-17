"""The one place ``base``/``base_gfp``/``raw165`` is resolved from ``meta.feature_blocks`` (PR-M7).

Elliptic widths: ``base`` 93, ``base_gfp`` 93 + 36 = 129, ``raw165`` 165.
"""

from __future__ import annotations

import logging

import numpy as np

from mulegraph.types import FeatureMatrix, GraphDataset

log = logging.getLogger("mulegraph")

LOCAL_BLOCK = "local"
AGG1HOP_BLOCK = "agg1hop"


def _local_block(data: GraphDataset) -> tuple[int, int]:
    blocks = data.meta.feature_blocks
    if LOCAL_BLOCK not in blocks:
        raise ValueError(
            f"dataset {data.meta.dataset!r} declares feature blocks {sorted(blocks)} with no "
            f"{LOCAL_BLOCK!r} block, so 'base' features are undefined (PR-M7)"
        )
    start, stop = blocks[LOCAL_BLOCK]
    agg = blocks.get(AGG1HOP_BLOCK)
    if agg is not None and stop != agg[0]:
        # PR-M7. Read this if you are auditing whether `base` is clean.
        raise ValueError(
            f"PR-M7 violation: 'base' would select x[:, {start}:{stop}] but the pre-aggregated "
            f"one-hop neighbour block {AGG1HOP_BLOCK!r} starts at column {agg[0]}. 'base' must "
            f"end exactly where {AGG1HOP_BLOCK!r} begins, so that base features never include "
            "aggregated neighbour information and the base vs base+GFP gap measures added graph "
            f"information cleanly. Dataset {data.meta.dataset!r} declares blocks {dict(blocks)}."
        )
    return start, stop


def select_features(data: GraphDataset, gfp: FeatureMatrix | None, selection: str) -> FeatureMatrix:
    """Assemble the matrix named by ``selection``; ``gfp`` is required for ``base_gfp`` only."""
    names = data.meta.feature_names

    if selection in ("base", "base_gfp"):
        start, stop = _local_block(data)
        base_values = np.ascontiguousarray(data.unit_features[:, start:stop], dtype=np.float32)
        base_columns = list(names[start:stop])
        if selection == "base":
            return FeatureMatrix(
                values=base_values,
                columns=base_columns,
                time=data.unit_time,
                feature_version="none",
                name="base",
                blocks={LOCAL_BLOCK: (0, base_values.shape[1])},
            )
        if gfp is None:
            raise ValueError(
                "features: base_gfp needs the causal graph features, but none were built; "
                "call build_features() first (the pipeline does this once per dataset+config)"
            )
        if gfp.values.shape[0] != data.num_units:
            raise ValueError(
                f"graph features have {gfp.values.shape[0]} rows for {data.num_units} {data.task}s"
            )
        width = base_values.shape[1]
        return FeatureMatrix(
            values=np.hstack([base_values, gfp.values]).astype(np.float32),
            columns=base_columns + list(gfp.columns),
            time=data.unit_time,
            feature_version=gfp.feature_version,
            name="base_gfp",
            blocks={
                LOCAL_BLOCK: (0, width),
                "gfp": (width, width + gfp.num_features),
            },
        )

    if selection == "raw165":
        blocks = data.meta.feature_blocks
        if set(blocks) != {LOCAL_BLOCK, AGG1HOP_BLOCK}:
            raise ValueError(
                f"features: raw165 is the published Elliptic block — 93 local plus 72 one-hop "
                f"aggregates — so it needs feature blocks exactly {{{LOCAL_BLOCK!r}, "
                f"{AGG1HOP_BLOCK!r}}}; dataset {data.meta.dataset!r} declares {sorted(blocks)}"
            )
        expected = blocks[AGG1HOP_BLOCK][1]
        if data.x.shape[1] != expected:
            raise ValueError(
                f"features: raw165 expects x to end where {AGG1HOP_BLOCK!r} does (column "
                f"{expected}), but x has {data.x.shape[1]} columns"
            )
        return FeatureMatrix(
            values=np.ascontiguousarray(data.x, dtype=np.float32),
            columns=list(names),
            time=data.node_time,
            feature_version="none",
            name="raw165",
            blocks=dict(blocks),
        )

    raise ValueError(f"unknown feature selection {selection!r}")
