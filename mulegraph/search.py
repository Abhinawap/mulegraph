"""Hyperparameter search phase.

Search runs **once** per (model config, regime, dataset) and the winning
configuration is then refitted across seeds (D2) — the search is not inside the
seed loop.

In the MVP no search runs. The budget is wall-clock and must be identical for
every model on a dataset, and ``W`` is not known until week-one gate 5 measures
the slowest single fit. Running an unequal search now would produce numbers that
cannot honestly be compared, so instead each model uses its fixed trial-0
reference configuration and the run records that no search happened
(``trials_completed = 0``) rather than reporting a nominal 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mulegraph.config import ModelConfig, SearchConfig

log = logging.getLogger("mulegraph")


@dataclass(frozen=True)
class SearchResult:
    """Outcome of the search phase, logged in full so fairness is auditable (PR-M6)."""

    best_params: dict[str, Any]
    best_trial: int
    trials_completed: int
    trial_ceiling: int | None
    wallclock_cap_min: float | None
    wallclock_used_min: float
    trial0_source: str
    enabled: bool

    def as_params(self) -> dict[str, Any]:
        """Flatten for MLflow, keeping "unset" distinguishable from a real value."""
        return {
            "search_enabled": self.enabled,
            "best_trial": self.best_trial,
            "trials_completed": self.trials_completed,
            "trial_ceiling": "unset" if self.trial_ceiling is None else self.trial_ceiling,
            "wallclock_cap_min": (
                "unset" if self.wallclock_cap_min is None else self.wallclock_cap_min
            ),
            "wallclock_used_min": round(self.wallclock_used_min, 3),
            "trial0_source": self.trial0_source,
        }


def run_search(
    model_cls: type,
    model_cfg: ModelConfig,
    search_cfg: SearchConfig,
    **_context: Any,
) -> SearchResult:
    """Choose the configuration to refit across seeds.

    Args:
        model_cls: The model class, for its ``trial0`` reference configuration.
        model_cfg: The config entry, whose ``params`` override trial 0.
        search_cfg: Search settings; ``enabled`` is False for the whole MVP.
        **_context: Dataset, features and split, accepted so the v1a signature
            does not change when trials actually run.

    Returns:
        The parameters to fit with, plus the provenance fields every run logs.
    """
    params, trial0_source = model_cls.trial0()
    params = {**params, **model_cfg.params}

    if not search_cfg.enabled:
        log.info(
            "%s: no search (v1a); using trial-0 reference config %r",
            model_cfg.key,
            trial0_source,
        )
        return SearchResult(
            best_params=params,
            best_trial=0,
            trials_completed=0,
            trial_ceiling=search_cfg.trial_ceiling.get(model_cfg.name),
            wallclock_cap_min=search_cfg.wallclock_cap_minutes,
            wallclock_used_min=0.0,
            trial0_source=trial0_source,
            enabled=False,
        )

    # Unreachable: SearchConfig rejects enabled=True. Kept so the v1a work has an
    # obvious home rather than being scattered through the pipeline.
    raise NotImplementedError(
        "Optuna search is v1a: enqueue trial 0 as the fixed reference config, then run "
        "trials at search_seed under the shared wall-clock cap W until the cap or the "
        "per-model ceiling is hit, scoring on validation PR-AUC with a median pruner"
    )
