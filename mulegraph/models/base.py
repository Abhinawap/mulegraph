"""The model protocol every model is used through (PR-M5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mulegraph.types import FeatureMatrix, GraphDataset, Split


@dataclass
class FitInfo:
    """Cost and validation score of one fit; ``seconds`` is logged as ``fit_seconds`` (D2)."""

    seconds: float
    best_iteration: int | None = None
    val_pr_auc: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseModel(Protocol):
    """Fit on a split, score node indices, optionally expose embeddings."""

    name: str

    def fit(self, data: GraphDataset, feats: FeatureMatrix, split: Split, seed: int) -> FitInfo:
        """Fit on ``split.train``, early-stop on ``split.val``; never touch ``split.test``."""
        ...

    def predict_proba(
        self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray
    ) -> np.ndarray:
        """P(illicit) for ``idx`` as float32 in [0, 1]."""
        ...

    def embed(self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray | None:
        """Representation for ``idx``, or None for models without one."""
        ...

    def save(self, path: Path) -> Path:
        """Persist the fitted model under ``path``; returns the file written."""
        ...

    @classmethod
    def trial0(cls) -> tuple[dict[str, Any], str]:
        """Fixed reference config and its provenance name (D2)."""
        ...


def check_train_labelled(data: GraphDataset, split: Split) -> None:
    """Reject unlabelled nodes in train/val: a -1 in a loss is learned as a negative."""
    for name in ("train", "val"):
        idx = getattr(split, name)
        if idx.size and data.y[idx].min() < 0:
            n_bad = int((data.y[idx] < 0).sum())
            raise ValueError(
                f"{n_bad} unlabelled nodes (y == -1) found in split.{name}; "
                "splits must contain labelled nodes only"
            )
