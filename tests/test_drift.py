"""Detectors are monotone in shift, take no labels, and lead time is arithmetic (PR-R1..R4)."""

from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulegraph import pipeline
from mulegraph.config import DriftRunConfig
from mulegraph.drift.detectors import alert_shift, conf_shift, ks_frac, psi
from mulegraph.drift.monitor import lead_time, score_batches

#: DriftConfig's defaults, spelled out: drift/ never imports the config (coupling rule).
FLAGS = dict(
    detectors=("psi", "ks", "conf", "alert"),
    bins=10,
    psi_flag=0.2,
    ks_alpha=0.01,
    ks_frac_flag=0.2,
    conf_flag=0.1,
    alert_flag=0.2,
)


def _shifted(delta: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(800, 5))
    cur = rng.normal(loc=delta, size=(800, 5))
    return ref, cur


def test_detectors_increase_with_shift() -> None:
    deltas = [0.0, 0.3, 0.8, 2.0]
    psi_max = [psi(*_shifted(d)).max() for d in deltas]
    ks = [ks_frac(*_shifted(d)) for d in deltas]
    conf = [conf_shift(_shifted(d)[0][:, 0], _shifted(d)[1][:, 0])[0] for d in deltas]
    for series in (psi_max, ks, conf):
        assert all(a <= b for a, b in zip(series, series[1:], strict=False)), series
    assert psi_max[0] < 0.2 < psi_max[-1]
    assert ks[0] == 0.0 and ks[-1] == 1.0


def test_psi_sees_a_shift_away_from_a_constant_reference_column() -> None:
    # snapml turns on flush-to-zero for the process; the feature builder imports it before
    # any detector runs, so the detector must work under it regardless of test order.
    pytest.importorskip("snapml")
    ref = np.zeros((100, 1))
    cur = np.ones((100, 1))
    assert psi(ref, cur)[0] > 0.2
    binary_ref = np.repeat([[0.0], [1.0]], 50, axis=0)
    assert psi(binary_ref, np.ones((100, 1)))[0] > 0.2


def test_alert_shift_increases_as_scores_cross_threshold() -> None:
    rng = np.random.default_rng(2)
    ref = rng.uniform(size=500)
    shifts = [0.0, 0.2, 0.4, 0.6]
    scores = [alert_shift(ref, np.clip(ref + s, 0, 1), threshold=0.5) for s in shifts]
    assert all(a <= b for a, b in zip(scores, scores[1:], strict=False)), scores
    assert scores[0] == pytest.approx(0.0, abs=1e-6)
    assert scores[-1] > scores[0]
    with pytest.raises(ValueError, match="non-empty"):
        alert_shift(np.array([]), ref, threshold=0.5)


def test_no_detector_accepts_labels() -> None:
    """PR-R2: the only way a label could reach a detector is through its signature."""
    for fn in (psi, ks_frac, conf_shift, alert_shift, score_batches):
        names = set(inspect.signature(fn).parameters)
        assert not any(n == "y" or "label" in n for n in names), (fn.__name__, names)


def test_lead_time_is_first_drop_minus_first_flag() -> None:
    curve = pd.DataFrame({"time": [9, 10, 11, 12], "f1": [0.8, 0.8, 0.3, 0.2]})
    scores = pd.DataFrame(
        {
            "detector": ["psi", "psi", "ks", "ks"],
            "batch_id": [9, 10, 9, 10],
            "score": [0.5, 0.5, 0.0, 0.0],
            "flagged": [True, True, False, False],
            "threshold": [0.2, 0.2, 0.2, 0.2],
        }
    )
    table = lead_time(curve, scores, ref_f1=0.9, drop=0.2, drop_run=2).set_index("detector")

    assert table.loc["psi", "first_drop"] == 11 and table.loc["psi", "first_flag"] == 9
    assert table.loc["psi", "lead"] == 2
    assert np.isnan(table.loc["ks", "first_flag"]) and np.isnan(table.loc["ks", "lead"])
    assert table["drop_level"].tolist() == pytest.approx([0.72, 0.72])


def test_a_single_dip_is_not_a_drop_but_a_sustained_one_is() -> None:
    curve = pd.DataFrame({"time": [9, 10, 11, 12, 13], "f1": [0.8, 0.5, 0.8, 0.3, 0.2]})
    scores = pd.DataFrame(
        {"detector": ["psi"], "batch_id": [9], "score": [0.5], "flagged": [True]}
    ).assign(threshold=0.2)
    assert lead_time(curve, scores, 0.9, 0.2, drop_run=2)["first_drop"].item() == 12
    assert lead_time(curve, scores, 0.9, 0.2, drop_run=1)["first_drop"].item() == 10


def test_a_single_flag_is_not_a_warning_but_a_sustained_one_is() -> None:
    """Flags need the same ``drop_run`` persistence as the F1 drop (PR-R4)."""
    curve = pd.DataFrame({"time": [9, 10, 11, 12, 13], "f1": [0.8, 0.8, 0.8, 0.3, 0.2]})
    scores = pd.DataFrame(
        {
            "detector": "psi",
            "batch_id": [13, 9, 10, 11, 12],
            "flagged": [True, True, False, True, True],
        }
    ).assign(score=0.5, threshold=0.2)
    assert lead_time(curve, scores, 0.9, 0.2, drop_run=2)["first_flag"].item() == 11
    assert lead_time(curve, scores, 0.9, 0.2, drop_run=1)["first_flag"].item() == 9
    flicker = scores.assign(flagged=[False, True, False, True, False])
    assert np.isnan(lead_time(curve, flicker, 0.9, 0.2, drop_run=2)["lead"].item())


def test_calibrated_thresholds_are_the_reference_noise_floor() -> None:
    """A reference batch scored against the rest is never above the floor; a shifted one is."""
    rng = np.random.default_rng(1)
    batch = np.repeat([1, 2, 3, 10, 11], 400)
    values = rng.normal(size=(2000, 4))
    values[batch == 11] += 3.0
    proba = rng.uniform(size=2000)
    proba[batch == 11] = rng.uniform(0.5, 1.0, size=400)

    table = score_batches(values, proba, batch, [1, 2, 3], calibrate=True, threshold=0.5, **FLAGS)
    flagged = table.set_index(["detector", "batch_id"])["flagged"]
    assert not flagged.loc[("psi", 10)] and not flagged.loc[("conf", 10)]
    assert not flagged.loc[("alert", 10)]
    assert flagged.loc[("psi", 11)] and flagged.loc[("ks", 11)] and flagged.loc[("conf", 11)]
    assert flagged.loc[("alert", 11)]
    floor = table.groupby("detector")["threshold"].first()
    assert floor["psi"] > 0 and floor["conf"] > 0 and floor["ks"] >= 0
    assert floor["alert"] >= 0

    with pytest.raises(ValueError, match="two populated reference batches"):
        score_batches(values, proba, batch, [1], calibrate=True, threshold=0.5, **FLAGS)


@pytest.fixture
def drift_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DriftRunConfig:
    monkeypatch.setenv("MULEGRAPH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MULEGRAPH_REPORT_DIR", str(tmp_path / "report"))
    return DriftRunConfig.model_validate(
        {
            "dataset": {
                "name": "synthetic_elliptic",
                "version": "test",
                "synthetic": {"n_nodes": 600, "n_timesteps": 12, "illicit_rate": 0.2, "seed": 0},
            },
            "features": {"bins": [2, 4], "cycle_len": 4, "num_threads": 2},
            "split": {
                "regimes": [{"regime": "temporal", "train_end": 6, "val": [7, 8], "test": [9, 12]}]
            },
            "models": [{"name": "xgb", "features": "base_gfp", "params": {"n_estimators": 20}}],
            "seeds": [0, 1],
            "mlflow": {
                "experiment": "drift_test",
                "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            },
            "device": "cpu",
        }
    )


def test_drift_refuses_anything_but_one_temporal_regime(drift_config: DriftRunConfig) -> None:
    raw = drift_config.model_dump()
    raw["split"]["regimes"] = [{"regime": "random"}]
    with pytest.raises(ValueError, match="temporal"):
        DriftRunConfig.model_validate(raw)


def test_event_must_leave_a_test_batch_on_each_side(drift_config: DriftRunConfig) -> None:
    raw = drift_config.model_dump()
    for event in (9, 13):
        raw["drift"]["event"] = event
        with pytest.raises(ValueError, match="inside the test window"):
            DriftRunConfig.model_validate(raw)


def test_run_drift_with_an_event_writes_before_and_after_rows(
    drift_config: DriftRunConfig, tmp_path: Path
) -> None:
    raw = drift_config.model_dump()
    raw["drift"]["event"] = 11
    pipeline.run_drift(DriftRunConfig.model_validate(raw), pipeline.SMOKE_CONFIG)

    table = pd.read_csv(tmp_path / "report" / "tables" / "drift_test_event.csv")
    assert len(table) == 2 * 2  # windows x seeds
    assert set(table["window"]) == {"before", "after"}
    assert table["probe_roc_auc"].between(0, 1).all()
    # The probe depends on the window and features only, not the seed of the fitted model.
    assert table.groupby("window")["probe_roc_auc"].nunique().eq(1).all()


def test_run_drift_writes_scores_and_lead_time(
    drift_config: DriftRunConfig, tmp_path: Path
) -> None:
    table = pipeline.run_drift(drift_config, pipeline.SMOKE_CONFIG)

    assert table == tmp_path / "report" / "tables" / "drift_test_lead_time.csv"
    with open(table, newline="") as fh:
        leads = list(csv.DictReader(fh))
    assert len(leads) == 4 * 2  # detectors x seeds
    assert {r["detector"] for r in leads} == {"psi", "ks", "conf", "alert"}

    with open(tmp_path / "report" / "tables" / "drift_test_scores.csv", newline="") as fh:
        scores = list(csv.DictReader(fh))
    assert {r["batch_id"] for r in scores} == {"9", "10", "11", "12"}
    assert len(scores) == 4 * 4 * 2
    assert (tmp_path / "report" / "figures" / "drift_test_drift.png").stat().st_size > 0
    assert not (tmp_path / "report" / "tables" / "drift_test_event.csv").exists()
