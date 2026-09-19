"""Causal, forward-only driving of snapml's ``GraphFeaturePreprocessor`` (PR-F1, PR-F2).

Layout and drive-pattern facts are in ``docs/project_status.md`` → Verified method notes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mulegraph.config import FeaturesConfig

log = logging.getLogger("mulegraph")

#: Input row width: ``[edge_id, source, target, timestamp, dummy_amount]``.
RAW_WIDTH = 5

#: Elliptic has no amounts, but ``vertex_stats`` needs a raw column to point at.
DUMMY_COLUMN = 4

#: Config spells families with underscores; snapml with hyphens.
SNAPML_FAMILY: dict[str, str] = {
    "fan": "fan",
    "degree": "degree",
    "scatter_gather": "scatter-gather",
    "temp_cycle": "temp-cycle",
    "lc_cycle": "lc-cycle",
}

#: snapml ``vertex_stats_feats`` codes 0 fan, 1 degree, 2 ratio; 3–10 need a raw amount column.
VERTEX_STAT_FEATS: tuple[int, ...] = (0, 1, 2)
VERTEX_STAT_NAMES: tuple[str, ...] = ("fan", "degree", "ratio")

#: Histogram blocks in snapml's output order: (name, family, which vertex it describes).
HIST_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("fan_in", "fan", "dst"),
    ("fan_out", "fan", "src"),
    ("degree_in", "degree", "dst"),
    ("degree_out", "degree", "src"),
    ("scatter_gather", "scatter_gather", "edge"),
    ("temp_cycle", "temp_cycle", "edge"),
    ("lc_cycle", "lc_cycle", "edge"),
)

#: Vertex-statistic blocks in snapml's output order: (which vertex, which direction).
VERTEX_STAT_BLOCKS: tuple[tuple[str, str], ...] = (
    ("src", "out"),
    ("src", "in"),
    ("dst", "out"),
    ("dst", "in"),
)


@dataclass(frozen=True)
class GfpBlock:
    """One contiguous column group of GFP's per-edge output."""

    name: str
    side: str  # "src" | "dst" | "edge"
    start: int
    stop: int
    kind: str  # "hist" | "vertex_stat"
    direction: str = ""  # "out" | "in" for vertex statistics

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class GfpLayout:
    """GFP's per-edge output shape for one feature config; hashed into ``feature_version``."""

    blocks: tuple[GfpBlock, ...]
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.columns) != self.width:
            raise ValueError(f"{len(self.columns)} column names for width {self.width}")

    @property
    def width(self) -> int:
        return self.blocks[-1].stop if self.blocks else 0

    @property
    def hist_blocks(self) -> tuple[GfpBlock, ...]:
        return tuple(b for b in self.blocks if b.kind == "hist")

    @property
    def vertex_stat_blocks(self) -> tuple[GfpBlock, ...]:
        return tuple(b for b in self.blocks if b.kind == "vertex_stat")


def make_gfp_params(cfg: FeaturesConfig) -> dict[str, object]:
    """Total snapml parameter dict (disabled families set False) so it hashes without defaults."""
    params: dict[str, object] = {
        "num_threads": cfg.num_threads,
        "time_window": cfg.window,
        "max_no_edges": -1,
        "vertex_stats": cfg.vertex_stats,
        "vertex_stats_tw": cfg.window,
        "vertex_stats_cols": [DUMMY_COLUMN],
        "vertex_stats_feats": list(VERTEX_STAT_FEATS),
    }
    for fam, key in SNAPML_FAMILY.items():
        params[key] = fam in cfg.families
        params[f"{key}_tw"] = cfg.window
        params[f"{key}_bins"] = list(cfg.bins)
    params["lc-cycle_len"] = cfg.cycle_len
    return params


def expected_layout(cfg: FeaturesConfig) -> GfpLayout:
    """Derive the output layout by arithmetic, without calling snapml."""
    blocks: list[GfpBlock] = []
    columns: list[str] = []
    cursor = 0
    n_bins = len(cfg.bins)
    for name, family, side in HIST_BLOCKS:
        if family not in cfg.families:
            continue
        blocks.append(GfpBlock(name, side, cursor, cursor + n_bins, "hist"))
        columns.extend(f"{name}_bin{edge}" for edge in cfg.bins)
        cursor += n_bins
    if cfg.vertex_stats:
        for vertex, direction in VERTEX_STAT_BLOCKS:
            width = len(VERTEX_STAT_FEATS)
            blocks.append(
                GfpBlock(
                    f"vs_{vertex}_{direction}",
                    vertex,
                    cursor,
                    cursor + width,
                    "vertex_stat",
                    direction,
                )
            )
            columns.extend(f"vs_{vertex}_{direction}_{s}" for s in VERTEX_STAT_NAMES)
            cursor += width
    return GfpLayout(tuple(blocks), tuple(columns))


def probe_layout(cfg: FeaturesConfig) -> GfpLayout:
    """Check :func:`expected_layout` against a real ``transform`` so a snapml change fails loud."""
    from snapml import GraphFeaturePreprocessor

    expected = expected_layout(cfg)
    preprocessor = GraphFeaturePreprocessor()
    preprocessor.set_params(make_gfp_params(cfg))
    toy = np.array(
        [[0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 2.0, 0.0, 1.0, 1.0], [2.0, 0.0, 3.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    observed = preprocessor.transform(toy).shape[1]
    if observed != RAW_WIDTH + expected.width:
        raise RuntimeError(
            f"snapml GraphFeaturePreprocessor returned {observed} columns for families "
            f"{cfg.families} with {len(cfg.bins)} bins and vertex_stats={cfg.vertex_stats}; "
            f"mulegraph expects {RAW_WIDTH + expected.width} "
            f"({RAW_WIDTH} input + {expected.width} graph features). "
            "The output layout has changed; update mulegraph/features/gfp.py before trusting "
            "any feature column."
        )
    return expected


class GfpDriver:
    """Stateful GFP that only steps forward in time; no fit or reset exists by design (PR-F2)."""

    def __init__(self, params: dict[str, object], layout: GfpLayout) -> None:
        from snapml import GraphFeaturePreprocessor

        self.layout = layout
        self.params = dict(params)
        self._gp = GraphFeaturePreprocessor()
        self._gp.set_params(dict(params))
        self._last_time: int | None = None

    def step(self, batch: np.ndarray) -> np.ndarray:
        """Insert and score one timestep's float64 ``[B, 5]`` edges; return float32 features."""
        if batch.ndim != 2 or batch.shape[1] != RAW_WIDTH:
            raise ValueError(f"batch must have shape [B, {RAW_WIDTH}], got {batch.shape}")
        if batch.dtype != np.float64:
            raise TypeError(f"batch must be float64 for snapml, got {batch.dtype}")
        if batch.shape[0] == 0:
            raise ValueError("batch is empty; the caller should skip timesteps with no edges")

        times = np.unique(batch[:, 3])
        if times.size != 1:
            raise ValueError(
                f"a batch must be exactly one timestep, got timestamps {times.tolist()}; "
                "GFP scores the whole batch against one graph state, so mixing timesteps "
                "would let later edges into earlier rows' features (PR-F2)"
            )
        now = int(times[0])
        if self._last_time is not None and now < self._last_time:
            raise ValueError(
                f"batch timestamp {now} is earlier than the previous batch's {self._last_time}; "
                "GFP is stateful and can only be driven forward in time, so a batch may never "
                "be scored against a graph that already contains its future (PR-F2)"
            )

        out = self._gp.transform(batch)
        if out.shape[1] != RAW_WIDTH + self.layout.width:
            raise RuntimeError(
                f"GFP returned {out.shape[1]} columns at t={now}, expected "
                f"{RAW_WIDTH + self.layout.width}"
            )
        self._last_time = now
        return np.ascontiguousarray(out[:, RAW_WIDTH:], dtype=np.float32)
