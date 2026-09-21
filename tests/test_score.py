"""The deployed scoring path: threshold from validation, reuse, no labels, drift exit (D5).

``score`` is the one command that runs a model on a batch nobody has labelled yet, so the
integrity rules the benchmark enforces have to hold here too.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from mulegraph import pipeline
from mulegraph.cli import EXIT_DRIFT, app
from mulegraph.config import ScoreConfig
from mulegraph.report.html import TOP_ALERTS, write_score_report

BATCH = 10


def _raw(**over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "score_test",
        "dataset": {
            "name": "synthetic_elliptic",
            "version": "test",
            "synthetic": {"n_nodes": 600, "n_timesteps": 12, "illicit_rate": 0.2, "seed": 0},
        },
        "features": {"bins": [2, 4], "cycle_len": 4, "num_threads": 2},
        "split": {
            "regimes": [{"regime": "temporal", "train_end": 6, "val": [7, 8], "test": [9, 12]}]
        },
        "model": {"name": "xgb", "features": "base_gfp", "params": {"n_estimators": 20}},
        "health": {"calibrate": True},  # as configs/amlworld_score.yaml runs it
        "batch": BATCH,
        "device": "cpu",
    }
    return {**raw, **over}


@pytest.fixture
def score_cfg(tmp_data_dir: Path) -> ScoreConfig:
    return ScoreConfig.model_validate(_raw())


def _outputs(alerts: Path, health: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    return pd.read_csv(alerts), json.loads(health.read_text())


def test_queue_is_ranked_and_cut_at_the_validation_threshold(
    score_cfg: ScoreConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-E4: one threshold call, on validation rows only; every alert sits at or above it."""
    from mulegraph.splits.builder import build_split
    from mulegraph.util import Paths

    seen: list[int] = []
    real = pipeline.choose_threshold

    def spy(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
        seen.append(int(np.asarray(y_val).size))
        return real(y_val, p_val)

    monkeypatch.setattr(pipeline, "choose_threshold", spy)
    alerts, health = _outputs(*pipeline.run_score(score_cfg)[:2])

    paths = Paths.from_env()
    data = pipeline.load_dataset(score_cfg.dataset, paths.data_dir)
    split = build_split(
        data,
        score_cfg.split.regimes[0],
        paths.cache_dir(score_cfg.dataset.name, score_cfg.dataset.version),
    )
    assert seen == [split.val.size]
    assert len(alerts) == health["alerts"] > 0
    assert (alerts["score"] >= health["threshold"]).all()
    assert alerts["score"].is_monotonic_decreasing
    assert alerts["rank"].tolist() == list(range(1, len(alerts) + 1))
    assert (data.batch_id[alerts["unit"]] == BATCH).all()
    assert alerts["top_features"].str.count(";").eq(2).all()
    assert health["status"] in {"drift_flagged", "no_drift_flagged"}
    assert health["units_scored"] == int((data.batch_id == BATCH).sum())


def test_a_second_batch_reuses_the_deployed_model(
    score_cfg: ScoreConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fit per definition: the next day is scored by the same model at the same threshold."""
    fits: list[int] = []
    real = pipeline._fit_predict

    def counting(*args: Any, **kwargs: Any) -> Any:
        fits.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_fit_predict", counting)
    _, first = _outputs(*pipeline.run_score(score_cfg)[:2])
    _, second = _outputs(*pipeline.run_score(ScoreConfig.model_validate(_raw(batch=11)))[:2])

    assert len(fits) == 1
    assert first["threshold"] == second["threshold"]
    assert first["model_fit"] == second["model_fit"]

    # Any change to the definition is a new model, never a silent cache hit.
    changed = _raw(model={"name": "xgb", "features": "base_gfp", "params": {"n_estimators": 21}})
    pipeline.run_score(ScoreConfig.model_validate(changed))
    assert len(fits) == 2


def test_a_longer_reference_changes_the_health_check_and_not_the_model(
    score_cfg: ScoreConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5: the reference window is a health-check setting; the fitted model and threshold stay."""
    fits: list[int] = []
    real = pipeline._fit_predict
    monkeypatch.setattr(
        pipeline, "_fit_predict", lambda *a, **k: (fits.append(1), real(*a, **k))[1]
    )
    _, default = _outputs(*pipeline.run_score(score_cfg)[:2])
    wide = ScoreConfig.model_validate(_raw(reference=[3, 9]))
    _, longer = _outputs(*pipeline.run_score(wide)[:2])

    assert default["reference_batches"] == [7, 8]  # the validation window, when unset
    assert longer["reference_batches"] == [3, 9]
    assert len(fits) == 1
    assert longer["threshold"] == default["threshold"]
    assert longer["alerts"] == default["alerts"]


def test_reference_must_lie_wholly_before_the_scored_batch() -> None:
    for bad in ([7, BATCH], [8, 7], [9, 12]):
        with pytest.raises(ValueError, match="come before batch"):
            ScoreConfig.model_validate(_raw(reference=bad))


def test_labels_of_the_scored_batch_change_nothing(
    score_cfg: ScoreConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-R2: unlabel the scored batch and fit cold in a fresh data directory. The threshold,
    queue and health check come out identical: labels reach neither the fit nor the score."""
    alerts, health = _outputs(*pipeline.run_score(score_cfg)[:2])

    real = pipeline.load_dataset

    def unlabelled(*args: Any, **kwargs: Any) -> Any:
        data = real(*args, **kwargs)
        return dataclasses.replace(data, y=np.where(data.batch_id == BATCH, -1, data.y))

    monkeypatch.setattr(pipeline, "load_dataset", unlabelled)
    # A new labelling is a new definition: the split cache (PR-D4) rightly refuses to mix them.
    monkeypatch.setenv("MULEGRAPH_DATA_DIR", str(tmp_path / "cold"))
    alerts_after, health_after = _outputs(*pipeline.run_score(score_cfg)[:2])

    pd.testing.assert_frame_equal(alerts, alerts_after)
    fit, fit_after = health.pop("model_fit"), health_after.pop("model_fit")
    assert health == health_after
    assert fit["split_hash"] != fit_after["split_hash"], "the second run must be a cold fit"
    assert {k: v for k, v in fit.items() if k != "split_hash"} == {
        k: v for k, v in fit_after.items() if k != "split_hash"
    }


def test_a_shifted_batch_exits_with_the_drift_status(
    tmp_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch far outside the reference is flagged, and the CLI says so with exit status 3."""
    real = pipeline.select_features

    def shifted(data: Any, gfp: Any, name: str) -> Any:
        feats = real(data, gfp, name)
        values = feats.values.copy()
        values[data.batch_id == BATCH] += 100.0
        return dataclasses.replace(feats, values=values)

    monkeypatch.setattr(pipeline, "select_features", shifted)
    config = tmp_path / "score.yaml"
    config.write_text(yaml.safe_dump(_raw()))

    result = CliRunner().invoke(app, ["score", "--config", str(config)])

    assert result.exit_code == EXIT_DRIFT, result.output
    health = json.loads(
        (tmp_path / "report" / "tables" / f"score_test_batch{BATCH}_health.json").read_text()
    )
    assert health["status"] == "drift_flagged"
    assert "psi" in health["flagged"]
    assert (
        "Drift flagged by"
        in (tmp_path / "report" / "tables" / f"score_test_batch{BATCH}_report.html").read_text()
    )


def test_the_report_page_is_self_contained_and_matches_the_health_file(
    score_cfg: ScoreConfig,
) -> None:
    """D5: the page a person opens says what the health file says, and loads nothing else."""
    _, health_path, page_path, _ = pipeline.run_score(score_cfg)
    health, page = json.loads(health_path.read_text()), page_path.read_text()

    assert page_path.name == f"score_test_batch{BATCH}_report.html"
    for d in health["detectors"]:
        assert d["detector"] in page
    assert health["commit"] in page and health["model_fit"]["commit"] in page
    assert ("Drift flagged" in page) == (health["status"] == "drift_flagged")
    assert "<script" not in page
    assert not re.findall(r'(?:src|href)="(?!data:)', page), "the page must not load anything"


def test_the_report_escapes_what_the_data_supplies_and_truncates_the_queue(tmp_path: Path) -> None:
    fit = dict.fromkeys(
        [
            "commit",
            "model",
            "device",
            "xgboost",
            "dataset_version",
            "feature_version",
            "split_hash",
        ],
        "x",
    )
    health = {
        "batch": 3, "flagged": [], "reference_batches": [0, 1], "units_scored": 40, "alerts": 30,
        "threshold": 0.9, "commit": "abc", "model_fit": fit, "note": "a <note>",
        "detectors": [{"detector": "psi", "score": 0.1, "flag_at": 0.2}],
    }  # fmt: skip
    alerts = pd.DataFrame(
        {
            "rank": range(1, 31),
            "unit": range(30),
            "score": 0.99,
            "top_features": "<script>alert(1)</script> +1.0",
        }
    )

    page = write_score_report(tmp_path / "p.html", health, alerts).read_text()

    assert "<script>alert" not in page and "&lt;script&gt;alert" in page
    assert "a &lt;note&gt;" in page
    assert f"Top {TOP_ALERTS} of 30" in page
    assert page.count("&lt;script&gt;alert") == TOP_ALERTS


@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"model": {"name": "sage", "features": "base"}}, "xgb only"),
        ({"batch": 8}, "must fall in the test window"),
        (
            {"split": {"regimes": [{"regime": "random"}]}},
            "exactly one temporal regime",
        ),
    ],
)
def test_config_refuses_what_score_cannot_do_and_says_why(over: dict[str, Any], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        ScoreConfig.model_validate(_raw(**over))
