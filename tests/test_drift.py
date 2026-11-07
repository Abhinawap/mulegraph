"""Detectors are monotone in shift, take no labels, and lead time is arithmetic (PR-R1..R4)."""

from __future__ import annotations

import csv
import inspect
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mulegraph import pipeline
from mulegraph.config import DriftRunConfig
from mulegraph.data import load_dataset
from mulegraph.drift.detectors import alert_shift, conf_shift, ks_frac, psi
from mulegraph.drift.monitor import lead_time, score_batches
from mulegraph.eval.threshold import choose_threshold
from mulegraph.features.builder import build_features
from mulegraph.features.select import select_features
from mulegraph.util import Paths

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
    conf = [conf_shift(_shifted(d)[0][:, 0], _shifted(d)[1][:, 0]) for d in deltas]
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


def _drift_outputs(
    val: pd.DataFrame,
    test: pd.DataFrame,
    thr: float,
    event: int,
    detector_call: mock._Call,
    cfg: DriftRunConfig,
) -> dict:
    """What a drift run derives from its threshold, recomputed from its scored rows (PR-E4)."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    def f1(rows: pd.DataFrame) -> float:
        return f1_score(rows["y"], rows["proba"] >= thr, zero_division=0.0)

    curve = pd.DataFrame([(int(t), f1(g)) for t, g in test.groupby("time")], columns=["time", "f1"])
    ref_f1 = np.mean([f1(g) for _, g in val.groupby("time")])
    # The detector inputs run_drift passed, checked against the run by the caller.
    scores = score_batches(*detector_call.args, **{**detector_call.kwargs, "threshold": thr})
    sides = {"before": test[test["time"] < event], "after": test[test["time"] >= event]}
    hits = test["proba"] >= thr
    return {
        "ref_f1": ref_f1,
        "test_f1": f1(test),
        "ts_f1": dict(zip(curve["time"], curve["f1"], strict=True)),
        "diag_precision": precision_score(test["y"], hits, zero_division=0.0),
        "diag_recall": recall_score(test["y"], hits),
        "recall": {w: recall_score(r["y"], r["proba"] >= thr) for w, r in sides.items()},
        "scores": scores,
        "lead": lead_time(curve, scores, ref_f1, cfg.drift.f1_drop, cfg.drift.drop_run),
    }


def test_every_drift_use_of_the_threshold_is_its_runs_validation_optimum(
    drift_config: DriftRunConfig, tmp_path: Path
) -> None:
    """PR-E4 end to end: every drift output that depends on the threshold uses the val optimum."""
    import mlflow

    event = 11
    raw = drift_config.model_dump()
    # Calibrated, as configs/elliptic_drift.yaml is: the alert flag level then depends on
    # the threshold too.
    raw["drift"] |= {"event": event, "calibrate": True}
    # On graph seed 9 test units score between the val and the test optimum, and test F1
    # falls below the drop level, so every output can move; the guard in the loop fails if
    # that stops being true.
    raw["dataset"]["synthetic"]["seed"] = 9
    cfg = DriftRunConfig.model_validate(raw)
    with (
        mock.patch.object(pipeline, "score_batches", wraps=pipeline.score_batches) as detect,
        mock.patch.object(pipeline, "plot_drift", wraps=pipeline.plot_drift) as draw,
    ):
        pinned = cfg.model_dump()  # frozen stops assignment, not in-place edits
        pipeline.run_drift(cfg, pipeline.SMOKE_CONFIG)
        assert cfg.model_dump() == pinned, "the run rewrote its config"

    tables = tmp_path / "report" / "tables"
    run_key = ["model", "features", "seed"]
    scores, leads, events = (
        pd.read_csv(tables / f"drift_test_{name}.csv").groupby(run_key)
        for name in ("scores", "lead_time", "event")
    )
    mlflow.set_tracking_uri(cfg.mlflow.resolved_uri())
    client = mlflow.MlflowClient()
    runs = mlflow.search_runs(
        experiment_names=[cfg.mlflow.experiment], filter_string="tags.kind = 'child'"
    )
    assert len(runs) == len(cfg.models) * len(cfg.seeds)
    # run_drift fits and scores in config order, then draws once.
    order = [(m.name, m.features, s) for m in cfg.models for s in cfg.seeds]
    assert detect.call_count == len(order) and draw.call_count == 1
    drawn_curves, drawn_leads = (frame.groupby(run_key) for frame in draw.call_args.args[:2])
    # What every detector call should receive, built without run_drift (PR-R2).
    paths = Paths.from_env()
    data = load_dataset(cfg.dataset, paths.data_dir)
    gfp = build_features(data, cfg.features, paths.cache_dir(cfg.dataset.name, cfg.dataset.version))
    regime = cfg.split.regimes[0]
    window = (data.batch_id >= regime.val[0]) & (data.batch_id <= regime.test[1])
    scored_idx = np.flatnonzero(window)
    d = cfg.drift
    flags = {
        "detectors": d.detectors,
        "bins": d.bins,
        "psi_flag": d.psi_flag,
        "ks_alpha": d.ks_alpha,
        "ks_frac_flag": d.ks_frac,
        "conf_flag": d.conf_flag,
        "alert_flag": d.alert_flag,
        "calibrate": d.calibrate,
    }
    for run_id in runs["run_id"]:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="predictions.parquet"
        )
        frame = pd.read_parquet(path)
        val, test = frame[frame["part"] == "val"], frame[frame["part"] == "test"]
        # The rows everything is re-derived from are their configured windows' and no other.
        assert val["time"].between(*regime.val).all() and test["time"].between(*regime.test).all()
        run = client.get_run(run_id)
        key = (run.data.tags["model"], run.data.tags["features"], int(run.data.params["seed"]))
        # The detectors got this run's features (no label column), batches, reference window,
        # configured flags and model scores. Only labelled scores are stored, so only those are
        # pinned; a shift of every score, as toward a test-searched cut, moves them too.
        call = detect.call_args_list[order.index(key)]
        values, proba, batch, ref = call.args
        np.testing.assert_array_equal(values, select_features(data, gfp, key[1]).values[window])
        np.testing.assert_array_equal(batch, data.batch_id[window])
        np.testing.assert_array_equal(ref, np.arange(regime.val[0], regime.val[1] + 1))
        stored = frame.set_index("idx")["proba"]
        labelled = np.isin(scored_idx, stored.index)
        np.testing.assert_array_equal(proba[labelled], stored.loc[scored_idx[labelled]])
        assert {k: v for k, v in call.kwargs.items() if k != "threshold"} == flags

        thr, val_f1 = choose_threshold(val["y"], val["proba"])
        want = _drift_outputs(val, test, thr, event, call, cfg)
        # Each check below can fail: a threshold searched on test moves every output...
        bad = choose_threshold(test["y"], test["proba"])[0]
        planted = _drift_outputs(val, test, bad, event, call, cfg)
        for name, value in want.items():
            if isinstance(value, pd.DataFrame):
                assert not value.equals(planted[name]), name
            else:
                assert planted[name] != pytest.approx(value), name
        # ...including when F1 first drops and the lead of a detector that holds a flag:
        # measured here, not NaN at both thresholds.
        for column in ("first_drop", "lead"):
            got, moved = want["lead"][column], planted["lead"][column]
            assert got.notna().any() and not got.equals(moved), column

        assert (run.data.metrics["threshold"], run.data.metrics["val_f1"]) == (thr, val_f1)
        # Every detector score, flag and flag level; only the alert rate's depend on it (PR-R1).
        written = scores.get_group(key)[want["scores"].columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(written, want["scores"], check_dtype=False)
        # Lead time: its drop level comes from ref_f1, its drop from the F1 curve (PR-R4).
        assert leads.get_group(key)["ref_f1"].to_numpy() == pytest.approx(want["ref_f1"])
        drawn = drawn_curves.get_group(key)
        assert dict(zip(drawn["time"], drawn["f1"], strict=True)) == pytest.approx(want["ts_f1"])
        # The lead table from those checked inputs and the configured drop rule, as written
        # and as drawn.
        for table in (leads.get_group(key), drawn_leads.get_group(key)):
            written = table[want["lead"].columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(written, want["lead"], check_dtype=False)
        # What the drift child run logs, as a benchmark run's (PR-E5).
        for name in ("test_f1", "diag_precision", "diag_recall"):
            assert run.data.metrics[name] == pytest.approx(want[name]), name
        ts_f1 = {m.step: m.value for m in client.get_metric_history(run_id, "ts_f1")}
        assert ts_f1 == pytest.approx(want["ts_f1"])
        # The event table's recall on each side of the event (PR-R5).
        recall = events.get_group(key).set_index("window")["recall"].to_dict()
        assert recall == pytest.approx(want["recall"])


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
