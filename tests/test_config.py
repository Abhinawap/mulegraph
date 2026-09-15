"""Config schema: the shipped configs parse, and invalid ones are rejected loudly."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mulegraph.config import RunConfig, load_config

MINIMAL = {
    "dataset": {"name": "synthetic_elliptic"},
    "split": {"regimes": [{"regime": "random"}]},
    "models": [{"name": "xgb", "features": "base"}],
    "mlflow": {"experiment": "t"},
}


def _cfg(**overrides):
    return RunConfig.model_validate({**MINIMAL, **overrides})


@pytest.mark.parametrize("name", ["elliptic_mvp.yaml", "smoke.yaml"])
def test_shipped_configs_validate(repo_root: Path, name: str) -> None:
    cfg = load_config(repo_root / "configs" / name)
    assert cfg.models
    assert cfg.split.regimes


def test_elliptic_mvp_covers_the_mvp_grid(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs" / "elliptic_mvp.yaml")
    assert {m.key for m in cfg.models} == {
        "xgb.base",
        "xgb.base_gfp",
        "sage.base",
        "sage.base_gfp",
        "xgb.raw165",
    }
    assert [r.regime for r in cfg.split.regimes] == ["random", "temporal"]
    assert cfg.seeds == [0, 1, 2, 3, 4]
    assert cfg.total_fits() == 50


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _cfg(seedz=[0])


def test_raw165_is_xgb_only() -> None:
    with pytest.raises(ValidationError, match="raw165"):
        _cfg(models=[{"name": "sage", "features": "raw165"}])


def test_search_enabled_is_v1a() -> None:
    with pytest.raises(ValidationError, match="v1a"):
        _cfg(search={"enabled": True})


def test_temporal_inductive_parses_but_is_the_split_builders_problem() -> None:
    # D1 rejection depends on meta.cross_time_edges, which the config cannot see.
    cfg = _cfg(
        split={
            "regimes": [
                {"regime": "temporal_inductive", "train_end": 34, "val": [35, 37], "test": [38, 49]}
            ]
        }
    )
    assert cfg.split.regimes[0].regime == "temporal_inductive"


def test_temporal_regime_requires_boundaries() -> None:
    with pytest.raises(ValidationError, match="train_end"):
        _cfg(split={"regimes": [{"regime": "temporal"}]})


def test_temporal_boundaries_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="train_end"):
        _cfg(
            split={
                "regimes": [
                    {"regime": "temporal", "train_end": 40, "val": [35, 37], "test": [38, 49]}
                ]
            }
        )


def test_duplicate_regimes_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        _cfg(split={"regimes": [{"regime": "random"}, {"regime": "random"}]})


def test_duplicate_model_keys_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        _cfg(
            models=[
                {"name": "xgb", "features": "base"},
                {"name": "xgb", "features": "base"},
            ]
        )


def test_random_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        _cfg(split={"regimes": [{"regime": "random", "fractions": [0.7, 0.2, 0.2]}]})


def test_bins_must_be_strictly_increasing() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        _cfg(features={"bins": [2, 2, 8]})


def test_sage_gets_a_default_sampler() -> None:
    cfg = _cfg(models=[{"name": "sage", "features": "base"}])
    assert cfg.models[0].sampler is not None
    assert cfg.models[0].sampler.kind == "neighbor"


def test_needs_gfp_reflects_the_model_list() -> None:
    assert not _cfg().needs_gfp
    assert _cfg(models=[{"name": "xgb", "features": "base_gfp"}]).needs_gfp


def test_env_override_applies_and_is_visible(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(MINIMAL))
    monkeypatch.setenv("MULEGRAPH_FEATURE_BACKEND", "igraph")
    monkeypatch.setenv("MULEGRAPH_DEVICE", "cpu")
    cfg = load_config(path)
    assert cfg.features.backend == "igraph"
    assert cfg.device == "cpu"


def test_missing_config_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(tmp_path / "nope.yaml")
