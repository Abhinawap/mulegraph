"""Causal driving of IBM's snapml ``GraphFeaturePreprocessor`` (GFP).

Why this module exists at all: GFP is **not causal by construction, only by
usage** (week-1 gate 1, 9 Sep 2026). It keeps an in-memory graph, and
``transform(batch)`` *inserts* that batch into the graph before scoring any edge
in it. Feeding it the whole edge table and then scoring leaks the future
backwards into every earlier row, which is exactly what PR-F2 forbids.

The only correct drive pattern, and the one :class:`GfpDriver` allows, is:

    for t ascending:  out = driver.step(edges_at_t)

Consequences of that pattern, each verified against snapml 1.17.2 on a toy
fixture before this module was written:

* **One insertion per edge.** The preprocessor's incremental-fit entry point
  inserts a batch too, so fitting *and* transforming the same batch inserts it
  twice and doubles every degree, fan and histogram count (a vertex with two
  out-edges reports degree 4). ``GfpDriver`` therefore exposes no method that
  can insert without scoring, and ``tests/test_features_gfp_layout.py`` pins the
  doubling so the reason survives the next person to read this.
* **Static within a batch.** The whole batch is inserted before any edge in it is
  scored, so every edge sees the same graph and row order inside one timestep
  does not change the output. Aggregating per-vertex blocks by ``max`` is
  therefore exact, not approximate (see :mod:`mulegraph.features.aggregate`).
* **Edge ids must be globally unique.** Reusing an edge id overwrites the earlier
  edge instead of adding one, which silently erases history. The builder uses the
  dataset's global edge index.
* **Pattern histograms are windowed; vertex statistics are not.** With
  ``<family>_tw = k`` the batch at *t* sees batches *t-k+1 … t* for the pattern
  histograms. The vertex-statistic block ignores ``vertex_stats_tw`` in this
  build and accumulates over every batch inserted so far. That is still causal —
  it is past-only — but it means ``vs_*`` columns are cumulative, not windowed.
  On Elliptic the distinction is invisible (timesteps are disconnected
  components, so no vertex recurs across batches); it is recorded here because
  it would matter on AMLworld and belongs in the dissertation's method section.

Layout of ``transform``'s output, also verified rather than assumed:

    [edge_id, source, target, timestamp, <raw cols>] + <graph features>

with the graph features ordered

    1. pattern histograms, only for enabled families, ``len(bins)`` columns each,
       in the order fan-in, fan-out, degree-in, degree-out, scatter-gather,
       temp-cycle, lc-cycle;
    2. vertex statistics, four blocks: source-outgoing, source-incoming,
       target-outgoing, target-incoming, each ``[fan, degree, ratio, ...]``
       filtered by ``vertex_stats_feats``.

Elliptic edges carry no attributes and no amounts, so the input rows carry a
single dummy raw column of ones and ``vertex_stats_feats = [0, 1, 2]``. Those
three statistics per side are the scalar ``fan_in``/``fan_out``/``deg_in``/
``deg_out`` columns the spec's §2.3 feature table names; the histograms are
IBM's published extras.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mulegraph.config import FeaturesConfig

log = logging.getLogger("mulegraph")

#: Input row width: ``[edge_id, source, target, timestamp, dummy_amount]``.
RAW_WIDTH = 5

#: Column index of the dummy raw column. Elliptic has no amounts, but
#: ``vertex_stats`` needs a column to point at before it will emit fan/degree/ratio.
DUMMY_COLUMN = 4

#: Config spells families with underscores; snapml with hyphens.
SNAPML_FAMILY: dict[str, str] = {
    "fan": "fan",
    "degree": "degree",
    "scatter_gather": "scatter-gather",
    "temp_cycle": "temp-cycle",
    "lc_cycle": "lc-cycle",
}

#: ``vertex_stats_feats`` codes we request. The full snapml mapping is
#: 0 fan, 1 degree, 2 ratio, 3 avg, 4 sum, 5 min, 6 max, 7 median, 8 var,
#: 9 skew, 10 kurt; 3–10 summarise a raw column, which Elliptic does not have.
VERTEX_STAT_FEATS: tuple[int, ...] = (0, 1, 2)
VERTEX_STAT_NAMES: tuple[str, ...] = ("fan", "degree", "ratio")

#: Histogram blocks in snapml's output order, with the vertex each one describes.
#: ``fan-in`` and ``degree-in`` describe the edge's target; ``*-out`` its source;
#: the pattern families describe the edge itself.
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
    """One contiguous group of columns in GFP's per-edge output.

    ``side`` says which endpoint of the edge the block describes — ``"edge"``
    means the pattern belongs to the edge itself and neither endpoint owns it —
    and is what :mod:`mulegraph.features.aggregate` dispatches on.
    """

    name: str
    #: ``"src"`` | ``"dst"`` | ``"edge"``
    side: str
    start: int
    stop: int
    #: ``"hist"`` | ``"vertex_stat"``
    kind: str
    #: ``"out"`` | ``"in"`` for vertex statistics, ``""`` for histograms.
    direction: str = ""

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class GfpLayout:
    """The shape of GFP's per-edge output for one feature config.

    Held explicitly rather than inferred at read time because the aggregation and
    the feature version both key off it: a change of families, bins or vertex
    statistics must move the ``feature_version`` hash, not silently reinterpret
    an old cache.
    """

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
    """Build a complete snapml parameter dict from a feature config.

    Every family key is set, disabled ones to ``False``, so the dict is a total
    description of the feature definition and can be hashed into
    ``feature_version`` without a silent dependency on snapml's defaults.

    ``time_window`` is the global eviction window and is set to the widest family
    window, so a family never loses edges its own ``_tw`` still wants.
    """
    family_windows = {fam: cfg.window.for_family(fam) for fam in SNAPML_FAMILY}
    enabled_windows = [family_windows[fam] for fam in cfg.families]
    params: dict[str, object] = {
        "num_threads": cfg.num_threads,
        "time_window": max([cfg.window.default, *enabled_windows]),
        "max_no_edges": -1,
        "vertex_stats": cfg.vertex_stats,
        "vertex_stats_tw": cfg.window.default,
        "vertex_stats_cols": [DUMMY_COLUMN],
        "vertex_stats_feats": list(VERTEX_STAT_FEATS),
    }
    for fam, key in SNAPML_FAMILY.items():
        params[key] = fam in cfg.families
        params[f"{key}_tw"] = family_windows[fam]
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
    """Check :func:`expected_layout` against a real ``transform`` on a toy batch.

    Cheap insurance against a snapml upgrade silently reordering or resizing the
    output: every downstream index — the aggregation, the column names, the
    cached parquet — is derived from the arithmetic, so a mismatch must be loud.
    """
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
    """Stateful GFP wrapper that can only be driven forward in time.

    There is deliberately no ``fit``, no incremental-fit and no reset: every
    method that could insert a batch without scoring it, or score a batch out of
    order, is absent, so PR-F2 is a property of the type rather than of the
    caller's discipline.
    """

    def __init__(self, params: dict[str, object], layout: GfpLayout) -> None:
        from snapml import GraphFeaturePreprocessor

        self.layout = layout
        self.params = dict(params)
        self._gp = GraphFeaturePreprocessor()
        self._gp.set_params(dict(params))
        self._last_time: int | None = None

    @property
    def last_time(self) -> int | None:
        """Timestamp of the most recent batch, or None before the first step."""
        return self._last_time

    def step(self, batch: np.ndarray) -> np.ndarray:
        """Insert and score one timestep's edges; return their graph features.

        Args:
            batch: float64 ``[B, 5]`` of ``[edge_id, source, target, timestamp,
                dummy]``. Edge ids must be globally unique — snapml overwrites an
                edge whose id it has already seen, so a repeated id deletes
                history instead of adding to it.

        Returns:
            float32 ``[B, layout.width]``, the graph features only.
        """
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
