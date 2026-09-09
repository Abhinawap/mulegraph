"""Experiment configuration.

Every experiment is a YAML file under ``configs/``; there are no experiment
parameters as CLI flags beyond ``--config``, so a result is always traceable to a
file. Validation is strict (``extra="forbid"``): a typo in a key is an error, not
a silently ignored setting.

Some fields are accepted but not yet acted on (``eval.gap_pairs``,
``search.wallclock_cap_minutes``). They are validated so a v1a config parses
today and the schema does not churn.
"""

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


class WindowConfig(Strict):
    """Feature time windows, in the dataset's time unit (timesteps on Elliptic)."""

    default: int = Field(1, ge=1)
    scatter_gather: int | None = Field(None, ge=1)

    def for_family(self, family: str) -> int:
        if family == "scatter_gather" and self.scatter_gather is not None:
            return self.scatter_gather
        return self.default


class FeaturesConfig(Strict):
    backend: Literal["gfp", "igraph"] = "gfp"
    families: list[GfpFamily] = Field(
        default_factory=lambda: ["fan", "degree", "scatter_gather", "lc_cycle"]
    )
    #: Pattern-histogram bin edges. Coarser than snapml's default 2..30 because a
    #: 29-column histogram per family is mostly zeros on Elliptic's sparse timesteps.
    bins: list[int] = Field(default_factory=lambda: [2, 4, 8, 16, 32])
    window: WindowConfig = Field(default_factory=WindowConfig)
    cycle_len: int = Field(10, ge=2)
    #: Per-vertex fan/degree/ratio. These are the scalar columns the spec's feature
    #: table names; the histograms are IBM's published extras.
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
    """One evaluation regime. ``temporal_inductive`` parses here but is rejected by
    the split builder on datasets without cross-timestep edges (D1)."""

    regime: Regime
    train_end: int | None = None
    val: tuple[int, int] | None = None
    test: tuple[int, int] | None = None
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    #: Seeds the random partition only. The model seed never changes the split:
    #: every seed is refitted on the same partition (spec 2.5).
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
    """Neighbour sampling for GNNs. ``full_batch`` needs no pyg-lib and is exact
    on a graph this size, so it is the fallback when the sampler is unavailable."""

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
    """Hyperparameter search (D2, PR-M6).

    The MVP runs each model's fixed trial-0 reference configuration and no search:
    ``W`` is not known until week-one gate 5 sets it, and a wall-clock budget that
    is not yet measured cannot be applied equally. Enabling this is v1a.
    """

    enabled: bool = False
    wallclock_cap_minutes: float | None = Field(None, gt=0)
    trial_ceiling: dict[str, int] = Field(
        default_factory=lambda: {"xgb": 40, "sage": 20, "pna": 10}
    )
    trial0: dict[str, str] = Field(
        default_factory=lambda: {
            "xgb": "xgboost_defaults",
            "sage": "sage_default_2x64",
            "pna": "ibm_multignn_published",
        }
    )
    pruner: Literal["median"] = "median"
    search_seed: int = 0

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
    #: Paired-by-seed gaps are v1a; accepted here so a v1a config validates today.
    gap_pairs: list[tuple[str, str]] = Field(default_factory=list)
    bootstrap_samples: int = Field(1000, ge=1)
    per_timestep: bool = False

    @model_validator(mode="after")
    def _warn_unimplemented(self) -> EvalConfig:
        if self.per_timestep:
            log.warning("eval.per_timestep is v1a; per-timestep curves are not written in the MVP")
        if self.gap_pairs:
            log.warning("eval.gap_pairs is v1a; paired gaps are not computed in the MVP")
        return self


class MLflowConfig(Strict):
    experiment: str
    tracking_uri: str | None = None

    def resolved_uri(self) -> str:
        return self.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


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
    """Read and validate a YAML config, applying environment overrides.

    ``MULEGRAPH_FEATURE_BACKEND`` and ``MULEGRAPH_DEVICE`` override the file so a
    machine-specific setting need not be committed; both are logged when applied,
    because an override that changes results silently would break provenance.
    """
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
