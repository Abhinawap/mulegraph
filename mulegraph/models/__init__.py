"""Model registry; implementations import lazily so an xgb-only run never loads torch."""

from __future__ import annotations

from typing import Any

from mulegraph.config import SamplerConfig
from mulegraph.models.base import BaseModel, FitInfo, check_train_labelled

__all__ = ["BaseModel", "FitInfo", "check_train_labelled", "get_model"]

KNOWN_MODELS = ("xgb", "sage", "pna")


def get_model(
    name: str,
    params: dict[str, Any],
    device: str = "cpu",
    sampler: SamplerConfig | None = None,
) -> BaseModel:
    """Instantiate a model by config name; ``sampler`` is ignored by xgb."""
    if name == "xgb":
        from mulegraph.models.xgb import XGBModel

        return XGBModel(params=params, device=device)
    if name == "sage":
        from mulegraph.models.sage import SAGEModel

        return SAGEModel(params=params, device=device, sampler=sampler or SamplerConfig())
    if name == "pna":
        raise NotImplementedError(
            "pna is a v1b reference row on AMLworld via IBM's Multi-GNN code; "
            "the MVP is Elliptic-only with xgb and sage"
        )
    raise ValueError(f"Unknown model {name!r}; known: {KNOWN_MODELS}")
