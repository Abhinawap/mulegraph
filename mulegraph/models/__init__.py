"""Model registry; implementations import lazily so an xgb-only run never loads torch."""

from __future__ import annotations

from typing import Any

from mulegraph.config import SamplerConfig
from mulegraph.models.base import BaseModel, FitInfo, check_train_labelled

__all__ = ["BaseModel", "FitInfo", "check_train_labelled", "get_model"]


def get_model(
    name: str,
    params: dict[str, Any],
    device: str = "cpu",
    sampler: SamplerConfig | None = None,
    task: str = "node",
) -> BaseModel:
    """Instantiate a model by config name; ``sampler`` is ignored by xgb."""
    if name == "xgb":
        from mulegraph.models.xgb import XGBModel

        return XGBModel(params=params, device=device)
    if name == "sage":
        if task == "edge":
            from mulegraph.models.sage_edge import SAGEEdgeModel

            return SAGEEdgeModel(params=params, device=device, sampler=sampler or SamplerConfig())
        from mulegraph.models.sage import SAGEModel

        return SAGEModel(params=params, device=device, sampler=sampler or SamplerConfig())
    raise ValueError(f"Unknown model {name!r}; known: xgb, sage")
