"""Orchestrator — the only module that imports across subsystems."""

from __future__ import annotations

from pathlib import Path

from mulegraph.config import RunConfig

_PENDING = (
    "pipeline wiring is the integration step of the MVP build; the loader, feature "
    "builder, splits, models and evaluator land first"
)


def run_benchmark(cfg: RunConfig, config_path: Path) -> Path:
    """Load, build features, split, fit across seeds, evaluate, write the table."""
    raise NotImplementedError(_PENDING)


def run_smoke(keep: bool = False) -> Path:
    """End-to-end run on a synthetic graph, under two minutes on CPU."""
    raise NotImplementedError(_PENDING)
