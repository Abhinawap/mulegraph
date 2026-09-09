"""Search phase: the MVP records that no search ran rather than faking one."""

from __future__ import annotations

import pytest

from mulegraph.config import ModelConfig, SearchConfig
from mulegraph.search import run_search


class FakeModel:
    @classmethod
    def trial0(cls):
        return {"n_estimators": 1000, "max_depth": 6}, "fake_defaults"


def test_disabled_search_returns_trial0() -> None:
    cfg = ModelConfig(name="xgb", features="base")
    result = run_search(FakeModel, cfg, SearchConfig())
    assert result.best_params == {"n_estimators": 1000, "max_depth": 6}
    assert result.trial0_source == "fake_defaults"
    assert result.best_trial == 0
    assert result.enabled is False


def test_no_search_is_reported_as_zero_trials_not_one() -> None:
    # An honest zero: nothing was searched, so nothing is claimed (D2, PR-M6).
    result = run_search(FakeModel, ModelConfig(name="xgb", features="base"), SearchConfig())
    assert result.trials_completed == 0
    assert result.wallclock_used_min == 0.0


def test_config_params_override_trial0() -> None:
    cfg = ModelConfig(name="xgb", features="base", params={"max_depth": 3})
    result = run_search(FakeModel, cfg, SearchConfig())
    assert result.best_params["max_depth"] == 3
    assert result.best_params["n_estimators"] == 1000


def test_unset_budget_is_distinguishable_from_a_value() -> None:
    params = run_search(
        FakeModel, ModelConfig(name="xgb", features="base"), SearchConfig()
    ).as_params()
    assert params["wallclock_cap_min"] == "unset"
    assert params["trial_ceiling"] == 40


def test_enabling_search_is_refused_at_config_time() -> None:
    with pytest.raises(Exception, match="v1a"):
        SearchConfig(enabled=True)
