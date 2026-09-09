"""The model interface.

Every model — tree or graph — is used through this protocol (PR-M5), so the
orchestrator never branches on model type and a new challenger slots in without
touching the pipeline.

``search_space`` is present but unused in the MVP: hyperparameter search is v1a
(D2, PR-M6). Defining it now means v1a adds a search loop rather than reworking
every model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from mulegraph.types import FeatureMatrix, GraphDataset, Split

if TYPE_CHECKING:  # pragma: no cover - typing only
    import optuna


@dataclass
class FitInfo:
    """What one fit cost and how well it did on validation.

    ``seconds`` is not diagnostic decoration: the wall-clock budget ``W`` is
    ``max(120 min, 2 x slowest single fit)`` (D2), so every fit is a measurement.
    """

    seconds: float
    best_iteration: int | None = None
    val_pr_auc: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseModel(Protocol):
    """Fit on a split, score arbitrary node indices, optionally expose embeddings."""

    name: str

    def fit(self, data: GraphDataset, feats: FeatureMatrix, split: Split, seed: int) -> FitInfo:
        """Fit on ``split.train``, early-stopping on ``split.val``.

        Must never look at ``split.test`` — not for early stopping, not for
        thresholding, not for feature statistics.
        """
        ...

    def predict_proba(
        self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray
    ) -> np.ndarray:
        """P(illicit) for the given node indices, as float32 in [0, 1]."""
        ...

    def embed(self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray | None:
        """Representation for the given nodes, or None for models without one."""
        ...

    def save(self, path: Path) -> Path:
        """Persist the fitted model under ``path``; returns the file written."""
        ...

    @classmethod
    def trial0(cls) -> tuple[dict[str, Any], str]:
        """The fixed reference configuration and a name for its provenance.

        Trial 0 is always a published or library-default configuration (D2), so
        every model has a valid result even when a search finds nothing better.
        """
        ...

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters. Unused in the MVP; the search phase is v1a."""
        ...


def check_train_labelled(data: GraphDataset, split: Split) -> None:
    """Guard that no unlabelled node reaches a loss.

    Unlabelled nodes belong in the graph — message passing over them is the point
    of a GNN — but a -1 label entering a loss would be silently learned as a
    negative.
    """
    for name in ("train", "val"):
        idx = getattr(split, name)
        if idx.size and data.y[idx].min() < 0:
            n_bad = int((data.y[idx] < 0).sum())
            raise ValueError(
                f"{n_bad} unlabelled nodes (y == -1) found in split.{name}; "
                "splits must contain labelled nodes only"
            )
