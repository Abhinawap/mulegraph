"""Experiment configuration: one strict (``extra="forbid"``) YAML file per experiment."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger("mulegraph")

Regime = Literal["random", "temporal", "temporal_rolling", "temporal_inductive"]
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
    name: Literal["elliptic_pp", "synthetic_elliptic", "amlworld"]
    version: str = "2023.1"
    synthetic: SyntheticConfig | None = None
    #: AMLworld only: keep the first N days so the loader fits a small host; cached separately.
    max_days: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _synthetic_only_for_synthetic(self) -> DatasetConfig:
        if self.synthetic is not None and self.name != "synthetic_elliptic":
            raise ValueError("dataset.synthetic is only valid for name: synthetic_elliptic")
        if self.name == "synthetic_elliptic" and self.synthetic is None:
            object.__setattr__(self, "synthetic", SyntheticConfig())
        if self.max_days is not None and self.name != "amlworld":
            raise ValueError("dataset.max_days is only valid for name: amlworld")
        return self


class FeaturesConfig(Strict):
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
    #: Seeds the random partition only; every model seed refits on this same partition.
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

    @property
    def fits(self) -> int:
        """Fits per (model, seed): one, or one refit per test batch when rolling (PR-E7)."""
        if self.regime == "temporal_rolling":
            assert self.test is not None
            return self.test[1] - self.test[0] + 1
        return 1


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
    name: Literal["xgb", "sage"]
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


class EvalConfig(Strict):
    metrics: list[MetricName] = Field(
        default_factory=lambda: ["f1", "pr_auc", "roc_auc", "p_at_r50", "p_at_r80"]
    )


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
        return len(self.models) * len(self.seeds) * sum(r.fits for r in self.split.regimes)


class DetectorConfig(Strict):
    """Detector thresholds (PR-R1); the reference is always the validation window."""

    detectors: list[Literal["psi", "ks", "conf", "alert"]] = Field(
        default_factory=lambda: ["psi", "ks", "conf", "alert"], min_length=1
    )
    bins: int = Field(10, ge=2)
    psi_flag: float = Field(0.2, gt=0.0)
    ks_alpha: float = Field(0.01, gt=0.0, lt=1.0)
    ks_frac: float = Field(0.2, gt=0.0, lt=1.0)
    conf_flag: float = Field(0.1, gt=0.0, lt=1.0)
    #: |log ratio| of the share of units at or above the validation threshold; 0.5 ≈ a 65% change.
    alert_flag: float = Field(0.5, gt=0.0)
    #: Replace the fixed flags with each detector's leave-one-out maximum inside the reference.
    calibrate: bool = False


class DriftConfig(DetectorConfig):
    """Detector thresholds plus the label-side lead-time rule (PR-R4)."""

    #: Relative F1 fall from the validation mean, sustained for ``drop_run`` batches, = broken;
    #: a detector warns only after ``drop_run`` consecutive flags.
    f1_drop: float = Field(0.2, gt=0.0, lt=1.0)
    drop_run: int = Field(2, ge=1)
    #: First batch of a known external event; set, the run writes a before/after table (PR-R5).
    event: int | None = None


class DriftRunConfig(RunConfig):
    drift: DriftConfig = Field(default_factory=DriftConfig)

    @model_validator(mode="after")
    def _one_temporal_regime(self) -> DriftRunConfig:
        regimes = self.split.regimes
        if len(regimes) != 1 or regimes[0].regime != "temporal":
            raise ValueError(
                "drift needs exactly one regime and it must be temporal: the reference is the "
                "validation window and every scored batch must come after it in time (PR-R2)"
            )
        test = regimes[0].test
        event = self.drift.event
        if event is not None and test is not None and not test[0] < event <= test[1]:
            raise ValueError(
                f"drift.event {event} must fall inside the test window {test} after its first "
                "batch, so the test window has a batch on each side of it (PR-R5)"
            )
        return self


class ScoreConfig(Strict):
    """One deployed model scoring one batch, with a label-free health check (D5, PR-R2)."""

    name: str
    dataset: DatasetConfig
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    split: SplitConfig
    model: ModelConfig
    health: DetectorConfig = Field(default_factory=DetectorConfig)
    #: The batch to score; it must fall in the regime's test window, after the reference.
    batch: int
    device: Literal["auto", "cuda", "cpu"] = "auto"

    @model_validator(mode="after")
    def _check(self) -> ScoreConfig:
        regimes = self.split.regimes
        if len(regimes) != 1 or regimes[0].regime != "temporal":
            raise ValueError(
                "score needs exactly one temporal regime: the model trains before the "
                "validation window, and the scored batch comes after it (PR-E4, PR-R2)"
            )
        if self.model.name != "xgb":
            raise ValueError(
                f"score deploys xgb only, got {self.model.name!r}: a GNN scores through the "
                "whole graph, so it has no saved model to reuse from one day to the next"
            )
        test = regimes[0].test
        assert test is not None
        if not test[0] <= self.batch <= test[1]:
            raise ValueError(
                f"batch {self.batch} must fall in the test window {test}, after the "
                "validation window the threshold and the health reference come from (PR-E4)"
            )
        return self

    @property
    def needs_gfp(self) -> bool:
        return self.model.features == "base_gfp"


C = TypeVar("C", bound=BaseModel)


def load_config(path: str | Path, cls: type[C] = RunConfig) -> C:  # type: ignore[assignment]
    """Read and validate a YAML config; ``MULEGRAPH_DEVICE`` overrides ``device``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}: {path}")

    device = os.environ.get("MULEGRAPH_DEVICE")
    if device:
        raw["device"] = device
        log.info("MULEGRAPH_DEVICE=%s overrides device", device)

    return cls.model_validate(raw)
