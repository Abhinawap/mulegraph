"""Model registry.

Implementations are imported lazily so that a run using only XGBoost never pays
torch's import cost, and so a broken optional dependency surfaces as an error
about that model rather than an import failure at startup.
"""

from __future__ import annotations

from typing import Any

from mulegraph.config import SamplerConfig
from mulegraph.models.base import BaseModel, FitInfo, check_train_labelled

__all__ = ["BaseModel", "FitInfo", "check_train_labelled", "get_model", "get_model_cls"]

KNOWN_MODELS = ("xgb", "sage", "pna")


def get_model_cls(name: str) -> type:
    """Look up a model class by config name."""
    if name == "xgb":
        from mulegraph.models.xgb import XGBModel

        return XGBModel
    if name == "sage":
        from mulegraph.models.sage import SAGEModel

        return SAGEModel
    if name == "pna":
        raise NotImplementedError(
            "pna is a v1b reference row on AMLworld via IBM's Multi-GNN code; "
            "the MVP is Elliptic-only with xgb and sage"
        )
    raise ValueError(f"Unknown model {name!r}; known: {KNOWN_MODELS}")


def get_model(
    name: str,
    params: dict[str, Any],
    device: str = "cpu",
    sampler: SamplerConfig | None = None,
) -> BaseModel:
    """Instantiate a model. ``sampler`` is ignored by models that do not sample."""
    cls = get_model_cls(name)
    if name == "sage":
        return cls(params=params, device=device, sampler=sampler or SamplerConfig())
    return cls(params=params, device=device)
