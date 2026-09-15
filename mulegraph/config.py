"""Experiment configuration: one strict (``extra="forbid"``) YAML file per experiment."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger("mulegraph")

Regime = Literal["random", "temporal", "temporal_inductive"]
FeatureSelection = Literal["base", "base_gfp", "raw165"]
MetricName = Literal["f1", "pr_auc", "roc_auc", "p_at_r50", "p_at_r80"]
GfpFamily = Literal["fan", "degree", "scatter_gather", "lc_cycle", "temp_cycle"]


class Strict(BaseModel):
    model_config = {"extra": "forbid"}


class SyntheticConfig(Strict):
    """Parameters of the synthetic Elliptic-shaped graph used by tests and smoke."""

    n_nodes: int = Field(2000, ge=10)
    n_timesteps: int = Field(49, ge=2)
    n_features: int = Field(165, ge=2)
    illicit_rate: float = Field(0.10, gt=0.0, lt=1.0)
    unknown_rate: float = Field(0.5, ge=0.0, lt=1.0)
    edges_per_node: float = Field(1.2, gt=0.0)
    cross_time_edges: bool = False
    cross_time_fraction: float = Field(0.3, ge=0.0, le=1.0)
    seed: int = 0


class DatasetConfig(Strict):
    name: Literal["elliptic_pp", "synthetic_elliptic"]
    version: str = "2023.1"
    synthetic: SyntheticConfig | None = None

    @model_validator(mode="after")
    def _synthetic_only_for_synthetic(self) -> DatasetConfig:
        if self.synthetic is not None and self.name != "synthetic_elliptic":
            raise ValueError("dataset.synthetic is only valid for name: synthetic_elliptic")
        if self.name == "synthetic_elliptic" and self.synthetic is None:
            object.__setattr__(self, "synthetic", SyntheticConfig())
        return self


class FeaturesConfig(Strict):
    backend: Literal["gfp", "igraph"] = "gfp"
    families: list[GfpFamily] = Field(
        default_factory=lambda: ["fan", "degree", "scatter_gather", "lc_cycle"]
    )
    #: Coarser than snapml's 2..30: finer bins are mostly zeros on Elliptic's sparse timesteps.
    bins: list[int] = Field(default_factory=lambda: [2, 4, 8, 16, 32])
    #: Time window for every family, in the dataset's time unit (timesteps on Elliptic).
    window: int = Field(1, ge=1)
    cycle_len: int = Field(10, ge=2)
    vertex_stats: bool = True
    num_threads: int = Field(8, ge=1)
    aggregation: Literal["node_agg_v1"] = "node_agg_v1"

    @model_validator(mode="after")
    def _check(self) -> FeaturesConfig:
        if not self.families:
            raise ValueError("features.families must not be empty")
        if len(set(self.families)) != len(self.families):
            raise ValueError("features.families contains duplicates")
        if len(self.bins) < 1:
            raise ValueError("features.bins must not be empty")
        if any(b < 2 for b in self.bins):
            raise ValueError("features.bins entries must all be >= 2")
        if any(b >= c for b, c in zip(self.bins, self.bins[1:], strict=False)):
            raise ValueError("features.bins must be strictly increasing")
        return self


class RegimeConfig(Strict):
    """One evaluation regime; ``temporal_inductive`` is rejected later by the split builder (D1)."""

    regime: Regime
    train_end: int | None = None
    val: tuple[int, int] | None = None
    test: tuple[int, int] | None = None
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    #: Seeds the random partition only; model seeds refit on the same partition (spec 2.5).
    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> RegimeConfig:
        if self.regime == "random":
            if abs(sum(self.fractions) - 1.0) > 1e-9:
                raise ValueError(f"split fractions must sum to 1, got {sum(self.fractions)}")
            if any(f <= 0 for f in self.fractions):
                raise ValueError("split fractions must all be positive")
            return self
        missing = [f for f in ("train_end", "val", "test") if getattr(self, f) is None]
        if missing:
            raise ValueError(f"regime {self.regime!r} requires {', '.join(missing)}")
        assert self.train_end is not None and self.val is not None and self.test is not None
        if not self.train_end < self.val[0] <= self.val[1] < self.test[0] <= self.test[1]:
            raise ValueError(
                f"regime {self.regime!r} needs train_end < val[0] <= val[1] < test[0] <= test[1], "
                f"got train_end={self.train_end}, val={self.val}, test={self.test}"
            )
        return self

    @property
    def key(self) -> str:
        return self.regime


class SplitConfig(Strict):
    regimes: list[RegimeConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> SplitConfig:
        names = [r.regime for r in self.regimes]
        if len(set(names)) != len(names):
            raise ValueError(f"split.regimes contains duplicates: {names}")
        return self


class SamplerConfig(Strict):
    """GNN neighbour sampling; ``full_batch`` is exact and needs no pyg-lib."""

    kind: Literal["neighbor", "full_batch"] = "neighbor"
    fanout: list[int] = Field(default_factory=lambda: [15, 10])
    batch_size: int = Field(1024, ge=1)


class ModelConfig(Strict):
    name: Literal["xgb", "sage", "pna"]
    features: FeatureSelection
    sampler: SamplerConfig | None = None
    #: Overrides on top of the model's fixed reference (trial-0) configuration.
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if self.features == "raw165" and self.name != "xgb":
            raise ValueError(
                "features: raw165 is an XGBoost-only reference row on Elliptic (D3); "
                f"got model {self.name!r}"
            )
        if self.sampler is not None and self.name == "xgb":
            raise ValueError("sampler is only meaningful for graph models, not xgb")
        if self.name == "sage" and self.sampler is None:
            object.__setattr__(self, "sampler", SamplerConfig())
        return self

    @property
    def key(self) -> str:
        return f"{self.name}.{self.features}"


class SearchConfig(Strict):
    """Hyperparameter search (D2, PR-M6) is v1a; the MVP fits trial 0 only."""

    enabled: bool = False

    @model_validator(mode="after")
    def _mvp_guard(self) -> SearchConfig:
        if self.enabled:
            raise ValueError(
                "search.enabled is v1a: the wall-clock cap W is set by week-one gate 5 and "
                "must be identical for every model (D2), so no search runs in the MVP"
            )
        return self


class EvalConfig(Strict):
    metrics: list[MetricName] = Field(
        default_factory=lambda: ["f1", "pr_auc", "roc_auc", "p_at_r50", "p_at_r80"]
    )
    seed_ci: Literal["t95"] = "t95"


class MLflowConfig(Strict):
    """Run tracking in a local SQLite file; MLflow 3 refuses the ``file:`` store (NFR-1)."""

    experiment: str
    tracking_uri: str | None = None

    def resolved_uri(self) -> str:
        return self.tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"
        )


class RunConfig(Strict):
    dataset: DatasetConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    split: SplitConfig
    models: list[ModelConfig] = Field(min_length=1)
    search: SearchConfig = Field(default_factory=SearchConfig)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4], min_length=1)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    mlflow: MLflowConfig
    device: Literal["auto", "cuda", "cpu"] = "auto"

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        keys = [m.key for m in self.models]
        if len(set(keys)) != len(keys):
            raise ValueError(f"models contains duplicate (name, features) pairs: {keys}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds contains duplicates: {self.seeds}")
        return self

    @property
    def needs_gfp(self) -> bool:
        return any(m.features == "base_gfp" for m in self.models)

    def total_fits(self) -> int:
        return len(self.models) * len(self.split.regimes) * len(self.seeds)


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a YAML config; ``MULEGRAPH_FEATURE_BACKEND``/``_DEVICE`` override it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}: {path}")

    backend = os.environ.get("MULEGRAPH_FEATURE_BACKEND")
    if backend:
        raw.setdefault("features", {})["backend"] = backend
        log.info("MULEGRAPH_FEATURE_BACKEND=%s overrides features.backend", backend)
    device = os.environ.get("MULEGRAPH_DEVICE")
    if device:
        raw["device"] = device
        log.info("MULEGRAPH_DEVICE=%s overrides device", device)

    return RunConfig.model_validate(raw)
