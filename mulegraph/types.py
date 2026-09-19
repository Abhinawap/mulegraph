"""Shared data types; every component depends only on this module. Arrays are NumPy throughout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

#: Half-open ``[start, stop)`` column range into ``GraphDataset.x``.
FeatureBlock = tuple[int, int]

Task = Literal["node", "edge"]

#: ``-1`` nodes stay in the graph for message passing but never enter a loss or a metric.
LABEL_ILLICIT = 1
LABEL_LICIT = 0
LABEL_UNKNOWN = -1


# Dataclasses declare eq=False: they hold NumPy arrays, whose == is elementwise.
@dataclass(frozen=True, eq=False)
class DatasetMeta:
    """Provenance and structure of a loaded dataset; ``cross_time_edges`` drives D1."""

    dataset: str
    version: str
    cross_time_edges: bool
    source_url: str
    #: e.g. Elliptic++ ``{"local": (0, 93), "agg1hop": (93, 165)}`` (PR-M7).
    feature_blocks: dict[str, FeatureBlock]
    feature_names: list[str]
    #: Raw columns excluded from ``x``, recorded so the exclusion is auditable.
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
    """A time-stamped graph with labels; the unit ``y`` labels is a node or, on ``task="edge"``,
    an edge (PR-D2)."""

    x: np.ndarray  # float32 [N, F]; F == 0 on an edge task
    edge_index: np.ndarray  # int64 [2, E]; row 0 source, row 1 target
    edge_attr: np.ndarray | None  # float32 [E, K] or None (Elliptic has no edge features)
    node_time: np.ndarray  # int64 [N]
    edge_time: np.ndarray  # int64 [E]
    batch_id: np.ndarray  # int64 [units]; == node_time on Elliptic, the day on AMLworld
    y: np.ndarray  # int64 [units]; 1 illicit, 0 licit, -1 unknown
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
        if self.task == "edge" and self.edge_attr is None:
            raise ValueError("an edge task needs edge_attr: the edge is the unit being scored")
        units = e if self.task == "edge" else n
        for name in ("node_time", "node_ids"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape [{n}], got {arr.shape}")
        for name in ("batch_id", "y"):
            arr = getattr(self, name)
            if arr.shape != (units,):
                raise ValueError(f"{name} must have shape [{units}], got {arr.shape}")
        if self.edge_time.shape != (e,):
            raise ValueError(f"edge_time must have shape [{e}], got {self.edge_time.shape}")
        if e and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge_index refers to nodes outside [0, N)")
        if self.edge_attr is not None and self.edge_attr.shape[0] != e:
            raise ValueError(f"edge_attr must have {e} rows, got {self.edge_attr.shape[0]}")
        width = self.unit_features.shape[1]
        if len(self.meta.feature_names) != width:
            raise ValueError(
                f"meta.feature_names has {len(self.meta.feature_names)} entries "
                f"but the unit features have {width} columns"
            )
        for name, (_, stop) in self.meta.feature_blocks.items():
            if stop > width:
                raise ValueError(f"feature_blocks[{name!r}] runs past the {width} feature columns")

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_units(self) -> int:
        """Rows of ``y``: edges on an edge task, nodes otherwise."""
        return self.num_edges if self.task == "edge" else self.num_nodes

    @property
    def unit_time(self) -> np.ndarray:
        return self.edge_time if self.task == "edge" else self.node_time

    @property
    def unit_features(self) -> np.ndarray:
        """Raw per-unit features: ``edge_attr`` on an edge task, ``x`` otherwise."""
        if self.task == "edge":
            assert self.edge_attr is not None
            return self.edge_attr
        return self.x

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
    """Node-aligned feature table; ``feature_version`` is logged with every run (PR-F3)."""

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
    """Sorted, disjoint, labelled-only train/val/test indices; ``split_hash`` is logged (PR-O1)."""

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


@dataclass(frozen=True, eq=False)
class Predictions:
    """Scores for a set of nodes, with the labels and times needed to evaluate them."""

    idx: np.ndarray  # int64 [n]; node indices scored
    proba: np.ndarray  # float32 [n]; P(illicit)
    y: np.ndarray  # int64 [n]
    time: np.ndarray  # int64 [n]; for per-timestep curves (PR-E5)

    def __post_init__(self) -> None:
        n = self.idx.shape[0]
        for name in ("proba", "y", "time"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape [{n}], got {arr.shape}")
        if self.proba.size and (self.proba.min() < 0.0 or self.proba.max() > 1.0):
            raise ValueError("proba must lie in [0, 1]")
