"""Shared data types.

Every component depends only on this module; only ``pipeline.py`` imports across
subsystems (see ``docs/architecture.md``). Arrays are NumPy throughout — only
``models/sage.py`` converts to torch — which keeps the feature, split and eval
code torch-free and halves peak memory on a small machine.

Dataclasses are frozen and declare ``eq=False``: they hold NumPy arrays, for
which ``==`` is elementwise and a generated ``__eq__`` would raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

#: Half-open ``[start, stop)`` column range into ``GraphDataset.x``.
FeatureBlock = tuple[int, int]

Task = Literal["node", "edge"]

#: Label encoding used by every loader. ``-1`` means unlabelled, and unlabelled
#: nodes stay in the graph for message passing but never enter a loss or a metric.
LABEL_ILLICIT = 1
LABEL_LICIT = 0
LABEL_UNKNOWN = -1


@dataclass(frozen=True, eq=False)
class DatasetMeta:
    """Provenance and structure of a loaded dataset.

    ``cross_time_edges`` is the property design decision D1 turns on: when it is
    False the dataset's temporal split is inductive by construction and the
    ``temporal_inductive`` regime is undefined rather than merely unusual.
    """

    dataset: str
    version: str
    cross_time_edges: bool
    source_url: str
    #: e.g. Elliptic++ ``{"local": (0, 93), "agg1hop": (93, 165)}``. Exposed so
    #: "base" selection is explicit and logged (PR-M7).
    feature_blocks: dict[str, FeatureBlock]
    feature_names: list[str]
    #: Raw columns deliberately excluded from ``x``, recorded so the exclusion is
    #: auditable rather than invisible.
    dropped_columns: list[str] = field(default_factory=list)
    num_timesteps: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    raw_sha256: str = ""
    task: Task = "node"

    def __post_init__(self) -> None:
        for name, (start, stop) in self.feature_blocks.items():
            if not 0 <= start < stop:
                raise ValueError(f"feature_blocks[{name!r}] = {(start, stop)} is not a valid range")


@dataclass(frozen=True, eq=False)
class GraphDataset:
    """A time-stamped graph with labels.

    Node ``i`` is row ``i`` of ``x`` throughout the package; ``node_ids`` keeps the
    dataset's original identifiers for provenance only.
    """

    x: np.ndarray  # float32 [N, F]
    edge_index: np.ndarray  # int64 [2, E]; row 0 source, row 1 target
    edge_attr: np.ndarray | None  # float32 [E, K] or None (Elliptic has no edge features)
    node_time: np.ndarray  # int64 [N]
    edge_time: np.ndarray  # int64 [E]
    batch_id: np.ndarray  # int64 [N]; == node_time on Elliptic
    y: np.ndarray  # int64 [N]; 1 illicit, 0 licit, -1 unknown
    node_ids: np.ndarray  # int64 [N]; original ids
    task: Task
    meta: DatasetMeta

    def __post_init__(self) -> None:
        n, e = self.x.shape[0], self.edge_index.shape[1]
        if self.x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {self.x.shape}")
        if self.x.dtype != np.float32:
            raise TypeError(f"x must be float32, got {self.x.dtype}")
        if self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {self.edge_index.shape}")
        for name in ("edge_index", "node_time", "edge_time", "batch_id", "y", "node_ids"):
            arr = getattr(self, name)
            if arr.dtype != np.int64:
                raise TypeError(f"{name} must be int64, got {arr.dtype}")
        for name in ("node_time", "batch_id", "y", "node_ids"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape [{n}], got {arr.shape}")
        if self.edge_time.shape != (e,):
            raise ValueError(f"edge_time must have shape [{e}], got {self.edge_time.shape}")
        if e and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge_index refers to nodes outside [0, N)")
        if self.edge_attr is not None and self.edge_attr.shape[0] != e:
            raise ValueError(f"edge_attr must have {e} rows, got {self.edge_attr.shape[0]}")
        if len(self.meta.feature_names) != self.x.shape[1]:
            raise ValueError(
                f"meta.feature_names has {len(self.meta.feature_names)} entries "
                f"but x has {self.x.shape[1]} columns"
            )
        for name, (_, stop) in self.meta.feature_blocks.items():
            if stop > self.x.shape[1]:
                raise ValueError(
                    f"feature_blocks[{name!r}] runs past x's {self.x.shape[1]} columns"
                )

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def src(self) -> np.ndarray:
        return self.edge_index[0]

    @property
    def dst(self) -> np.ndarray:
        return self.edge_index[1]

    @property
    def labelled_idx(self) -> np.ndarray:
        """Indices of nodes carrying a real label. Splits only ever contain these."""
        return np.flatnonzero(self.y >= 0).astype(np.int64)


@dataclass(frozen=True, eq=False)
class FeatureMatrix:
    """A feature table aligned to node order: row ``i`` is node ``i``.

    ``feature_version`` is the sha256 prefix over the feature definition (list,
    window config, backend, dataset version) and is logged with every run
    (PR-F3). It is ``"none"`` for selections that add no computed features.
    """

    values: np.ndarray  # float32 [N, K]
    columns: list[str]
    time: np.ndarray  # int64 [N]; the timestep each row is valid at
    feature_version: str
    name: str  # "gfp" | "base" | "base_gfp" | "raw165"
    blocks: dict[str, FeatureBlock] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape {self.values.shape}")
        if self.values.dtype != np.float32:
            raise TypeError(f"values must be float32, got {self.values.dtype}")
        if len(self.columns) != self.values.shape[1]:
            raise ValueError(f"{len(self.columns)} column names for {self.values.shape[1]} columns")
        if self.time.shape != (self.values.shape[0],):
            raise ValueError(
                f"time must have shape [{self.values.shape[0]}], got {self.time.shape}"
            )
        if not np.isfinite(self.values).all():
            raise ValueError(f"feature matrix {self.name!r} contains NaN or infinity")

    @property
    def num_features(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, eq=False)
class Split:
    """Train/val/test node indices for one evaluation regime.

    Indices are sorted, disjoint, and contain labelled nodes only. ``split_hash``
    is logged with every run so a table can be traced to the exact partition that
    produced it (PR-O1).
    """

    regime: str
    train: np.ndarray  # int64, sorted
    val: np.ndarray
    test: np.ndarray
    split_hash: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("train", "val", "test"):
            arr = getattr(self, name)
            if arr.dtype != np.int64:
                raise TypeError(f"{name} must be int64, got {arr.dtype}")
            if arr.size and not np.all(np.diff(arr) > 0):
                raise ValueError(f"{name} must be sorted and free of duplicates")

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": self.train.size, "val": self.val.size, "test": self.test.size}


@dataclass(frozen=True, eq=False)
class Predictions:
    """Scores for a set of nodes, with the labels and times needed to evaluate them."""

    idx: np.ndarray  # int64 [n]; node indices scored
    proba: np.ndarray  # float32 [n]; P(illicit)
    y: np.ndarray  # int64 [n]
    time: np.ndarray  # int64 [n]; for per-timestep curves (v1a)
    embeddings: np.ndarray | None = None  # float32 [n, D] for GNNs

    def __post_init__(self) -> None:
        n = self.idx.shape[0]
        for name in ("proba", "y", "time"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape [{n}], got {arr.shape}")
        if self.proba.size and (self.proba.min() < 0.0 or self.proba.max() > 1.0):
            raise ValueError("proba must lie in [0, 1]")
        if self.embeddings is not None and self.embeddings.shape[0] != n:
            raise ValueError(f"embeddings must have {n} rows, got {self.embeddings.shape[0]}")


@dataclass(frozen=True, eq=False)
class DriftSignal:
    """One detector's verdict on one batch. Placeholder for v2 — nothing writes it yet."""

    detector: str
    batch_id: int
    batch_unit: str
    score: float
    flagged: bool
    threshold: float
