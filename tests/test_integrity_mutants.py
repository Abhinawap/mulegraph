"""Each integrity guard fails its named tests when its bug is planted (PR-E4, PR-E7)."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = pytest.mark.integrity

PIPELINE = "tests/test_pipeline.py::"
FIT_THRESHOLD = PIPELINE + "test_the_threshold_is_the_validation_optimum_applied_to_every_test_row"
LOGGED_F1 = PIPELINE + "test_every_logged_f1_is_scored_at_its_runs_validation_optimum[tiny_config]"
ROLLING_FIT = PIPELINE + "test_each_rolling_step_is_decided_at_its_own_validation_threshold"
ROLLING_LOGGED = PIPELINE + "test_every_logged_rolling_step_threshold_is_its_own_validation_optimum"
ROLLING_TABLE = PIPELINE + "test_rolling_regime_writes_one_child_run_per_seed"
MIXED_REGIMES = LOGGED_F1.replace("[tiny_config]", "[multi_config]")
DRIFT_E2E = (
    "tests/test_drift.py::test_every_drift_use_of_the_threshold_is_its_runs_validation_optimum"
)
SCORE_E2E = "tests/test_score.py::test_queue_is_ranked_and_cut_at_the_validation_threshold"
REUSE = "tests/test_score.py::test_a_second_batch_reuses_the_deployed_model"
#: The threshold searched on the test rows in scope, which no guarded site may use.
ON_TEST = "choose_threshold(test.y, test.proba)[0]"
#: The same, where the test rows in scope are a Fitted's.
ON_FITTED_TEST = "choose_threshold(fitted.test.y, fitted.test.proba)[0]"
#: The same in ``run_score``, scored by the deployed model on its split's test rows.
ON_SPLIT_TEST = (
    "choose_threshold(data.y[split.test], model.predict_proba(data, feats, split.test))[0]"
)
#: The score health check's flags at a test-searched threshold, by a call the queue test's
#: wrapper on pipeline.score_batches does not count.
FLAGS_AT_TEST = (
    '__import__("mulegraph.drift.monitor").drift.monitor.score_batches(feats.values[idx], proba,'
    " batch, ref_batches, detectors=h.detectors, bins=h.bins, psi_flag=h.psi_flag,"
    " ks_alpha=h.ks_alpha, ks_frac_flag=h.ks_frac, conf_flag=h.conf_flag,"
    f" alert_flag=h.alert_flag, threshold={ON_SPLIT_TEST}, calibrate=h.calibrate)"
    '.query("flagged").detector.tolist()'
)
DETECTOR_ROW = '{"detector": r.detector, "score": r.score, "flag_at": r.threshold}'
ALERT_AT = '__import__("mulegraph.drift.detectors").drift.detectors.alert_shift'
HEALTH_SITE = "alert_flag=h.alert_flag,\n        threshold=threshold,"
LEAD_SITE = "lead = lead_time(curve, table, ref_f1, cfg.drift.f1_drop, cfg.drift.drop_run)"
SCORE_ARGS = """feats.values[scored_idx], proba, batch, ref_batches, detectors=cfg.drift.detectors,
                    bins=cfg.drift.bins, psi_flag=cfg.drift.psi_flag, ks_alpha=cfg.drift.ks_alpha,
                    ks_frac_flag=cfg.drift.ks_frac, conf_flag=cfg.drift.conf_flag,
                    alert_flag=cfg.drift.alert_flag, calibrate=cfg.drift.calibrate"""
#: F1 of ``p >= thr`` as ``choose_threshold`` computes it: sklearn's f1_score can differ by an
#: ulp, which would turn a val_f1 row red with nothing planted.
_SK = '__import__("sklearn.metrics").metrics'
F1_AT = (
    "(lambda _p, _r: 2.0 * _p * _r / (_p + _r))(*(f({y}, {p} >= {thr})"
    f" for f in ({_SK}.precision_score, {_SK}.recall_score)))"
)
#: Plants beside the benchmark loop that collect each run's curve at a test-searched threshold.
BAD_CURVES = (
    (
        "    curves: list[pd.DataFrame] = []\n",
        "    curves: list[pd.DataFrame] = []\n    bad_curves = []\n",
    ),
    (
        "                    done += regime_cfg.fits\n",
        "                    done += regime_cfg.fits\n"
        f"                    bad_curves.append(per_timestep(fitted.test, {ON_FITTED_TEST})"
        ".assign(regime=regime_cfg.regime, model=model_cfg.name,"
        " features=model_cfg.features, seed=seed))\n",
    ),
)
#: Plants beside the drift loop that collect each run's lead table at a test-searched threshold.
BAD_LEADS = (
    (
        "    scores, leads, curves, events = [], [], [], []\n",
        "    scores, leads, curves, events = [], [], [], []\n    bad_leads = []\n",
    ),
    (
        "                curves.append(curve.assign(**tag))\n",
        "                curves.append(curve.assign(**tag))\n"
        f"                _bad = {ON_FITTED_TEST}\n"
        "                _rf = float(per_timestep(fitted.val, _bad)['f1'].mean())\n"
        "                bad_leads.append(lead_time(per_timestep(fitted.test, _bad), table,"
        " _rf, cfg.drift.f1_drop, cfg.drift.drop_run).assign(ref_f1=_rf, **tag))\n",
    ),
)
DIAG_AT_TEST = (
    f"_d = compute_metrics(test.y, test.proba, {ON_TEST}, ['f1'])\n"
    "    diagnostics['diag_precision'], diagnostics['diag_recall'] ="
    " _d['precision'], _d['recall']\n"
    "    curve = per_timestep(test, fitted.test_threshold)"
)


class Mutant(NamedTuple):
    name: str
    file: str
    site: str
    broken: str
    tests: tuple[str, ...]
    #: Further (site, broken) edits the plant needs; only ``site`` counts as the planted use.
    also: tuple[tuple[str, str], ...] = ()


MUTANTS = [
    Mutant(
        "threshold on test inside fit()",
        "mulegraph/pipeline.py",
        "threshold, val_f1 = choose_threshold(val.y, val.proba)",
        "threshold, val_f1 = (choose_threshold(data.y[step.test],"
        " model.predict_proba(data, feats, step.test))[0], choose_threshold(val.y, val.proba)[1])",
        (FIT_THRESHOLD,),
    ),
    Mutant(
        "per-row test threshold searched on test inside fit()",
        "mulegraph/pipeline.py",
        "np.full(test.idx.size, threshold, dtype=np.float64)",
        f"np.full(test.idx.size, {ON_TEST}, dtype=np.float64)",
        (FIT_THRESHOLD,),
    ),
    Mutant(
        "Fitted.threshold searched on test",
        "mulegraph/pipeline.py",
        "threshold=threshold,\n        val_f1=val_f1,",
        f"threshold={ON_TEST},\n        val_f1=val_f1,",
        (FIT_THRESHOLD,),
    ),
    Mutant(
        "Fitted.test_threshold searched on test",
        "mulegraph/pipeline.py",
        "test_threshold=np.concatenate(thresholds),",
        f"test_threshold=np.full(sum(t.idx.size for t in tests), {ON_TEST}),",
        (FIT_THRESHOLD,),
    ),
    Mutant(
        "rolling step threshold searched on test inside fit()",
        "mulegraph/pipeline.py",
        '"ts_threshold": threshold,',
        f'"ts_threshold": {ON_TEST},',
        (ROLLING_FIT,),
    ),
    Mutant(
        "logged threshold searched on test",
        "mulegraph/pipeline.py",
        '"threshold": fitted.threshold,\n                "val_f1": fitted.val_f1,',
        f'"threshold": {ON_TEST},\n                "val_f1": fitted.val_f1,',
        (LOGGED_F1,),
    ),
    Mutant(
        "test F1 at a test-searched threshold",
        "mulegraph/pipeline.py",
        "compute_metrics(test.y, test.proba, fitted.test_threshold, metrics,",
        f"compute_metrics(test.y, test.proba, {ON_TEST}, metrics,",
        (LOGGED_F1,),
    ),
    Mutant(
        "per-timestep curve at a test-searched threshold",
        "mulegraph/pipeline.py",
        "curve = per_timestep(test, fitted.test_threshold)",
        f"curve = per_timestep(test, {ON_TEST})",
        (LOGGED_F1,),
    ),
    Mutant(
        "curves table at a test-searched threshold",
        "mulegraph/pipeline.py",
        "curves.append(\n                        curve.assign(",
        "curves.append(\n                        "
        f"per_timestep(fitted.test, {ON_FITTED_TEST}).assign(",
        (LOGGED_F1,),
    ),
    Mutant(
        "logged rolling ts_threshold altered",
        "mulegraph/pipeline.py",
        "mlflow.log_metrics(step_metrics, step=t)",
        "mlflow.log_metrics({**step_metrics, 'ts_threshold': choose_threshold("
        "test.y[test.time == t], test.proba[test.time == t])[0]}, step=t)",
        (ROLLING_LOGGED,),
    ),
    Mutant(
        "rank metrics pooled on a rolling run",
        "mulegraph/pipeline.py",
        'metrics = [m for m in cfg.eval.metrics if m == "f1"] if fitted.steps else',
        "metrics = cfg.eval.metrics if fitted.steps else",
        (ROLLING_TABLE,),
    ),
    Mutant(
        "logged diag_* at a test-searched threshold (benchmark)",
        "mulegraph/pipeline.py",
        "curve = per_timestep(test, fitted.test_threshold)",
        DIAG_AT_TEST,
        (LOGGED_F1,),
    ),
    Mutant(
        "test rows unioned into validation under random only",
        "mulegraph/pipeline.py",
        "val = _predict(model, data, feats, step.val)",
        "val = _predict(model, data, feats, np.union1d(step.val, step.test)"
        " if step.regime == 'random' else step.val)",
        (MIXED_REGIMES,),
    ),
    Mutant(
        "rolling ts_val_f1 at a test-searched threshold",
        "mulegraph/pipeline.py",
        '"ts_val_f1": val_f1,',
        '"ts_val_f1": ' + F1_AT.format(y="val.y", p="val.proba", thr=ON_TEST) + ",",
        (ROLLING_LOGGED,),
    ),
    Mutant(
        "plot_curves draws test-searched curves",
        "mulegraph/pipeline.py",
        "plot_curves(curve_table,",
        "plot_curves(pd.concat(bad_curves, ignore_index=True),",
        (LOGGED_F1,),
        also=BAD_CURVES,
    ),
    # Drift: every row fails the one end-to-end drift test (#51, #56).
    Mutant(
        "alert-rate detector at a test-searched threshold",
        "mulegraph/pipeline.py",
        "threshold=fitted.threshold,\n                    calibrate",
        f"threshold={ON_FITTED_TEST},\n                    calibrate",
        (DRIFT_E2E,),
    ),
    Mutant(
        "drift ref_f1 at a test-searched threshold",
        "mulegraph/pipeline.py",
        "per_timestep(fitted.val, fitted.threshold)",
        f"per_timestep(fitted.val, {ON_FITTED_TEST})",
        (DRIFT_E2E,),
    ),
    Mutant(
        "event table at a test-searched threshold",
        "mulegraph/pipeline.py",
        "event_windows(fitted.test, fitted.threshold, event)",
        f"event_windows(fitted.test, {ON_FITTED_TEST}, event)",
        (DRIFT_E2E,),
    ),
    Mutant(
        "lead_time reads a test-searched curve",
        "mulegraph/pipeline.py",
        LEAD_SITE,
        LEAD_SITE.replace(
            "lead_time(curve,", f"lead_time(per_timestep(fitted.test, {ON_FITTED_TEST}),"
        ),
        (DRIFT_E2E,),
    ),
    Mutant(
        "lead_time reads a test-searched ref_f1",
        "mulegraph/pipeline.py",
        LEAD_SITE,
        LEAD_SITE.replace(
            "table, ref_f1,",
            f"table, float(per_timestep(fitted.val, {ON_FITTED_TEST})['f1'].mean()),",
        ),
        (DRIFT_E2E,),
    ),
    Mutant(
        "drift child run logged at a test-searched threshold",
        "mulegraph/pipeline.py",
        "cfg, data, feats, split.regime, model_cfg, seed, device, commit, fitted\n"
        "                )",
        "cfg, data, feats, split.regime, model_cfg, seed, device, commit, replace(fitted,"
        f" test_threshold=np.full_like(fitted.test_threshold, {ON_FITTED_TEST}))\n"
        "                )\n"
        "                curve = per_timestep(fitted.test, fitted.threshold)",
        (DRIFT_E2E,),
    ),
    Mutant(
        "plot_drift draws test-searched curves",
        "mulegraph/pipeline.py",
        "curves.append(curve.assign(**tag))",
        f"curves.append(per_timestep(fitted.test, {ON_FITTED_TEST}).assign(**tag))",
        (DRIFT_E2E,),
    ),
    Mutant(
        "drift validation rows include a test batch",
        "mulegraph/pipeline.py",
        "split = build_split(data, regime_cfg, cache_dir)\n",
        "split = build_split(data, regime_cfg, cache_dir)\n"
        "    split = replace(split, val=np.union1d(split.val,"
        " split.test[data.batch_id[split.test] == regime_cfg.test[0]]))\n",
        (DRIFT_E2E,),
    ),
    Mutant(
        "post-hoc drop_level from a test-searched ref_f1",
        "mulegraph/pipeline.py",
        LEAD_SITE,
        LEAD_SITE + "\n                lead['drop_level'] = (1 - cfg.drift.f1_drop)"
        f" * float(per_timestep(fitted.val, {ON_FITTED_TEST})['f1'].mean())",
        (DRIFT_E2E,),
    ),
    Mutant(
        "drift val_f1 at a test-searched threshold",
        "mulegraph/pipeline.py",
        "fitted = fit(data, feats, [split], model_cfg, seed, device)\n",
        "fitted = fit(data, feats, [split], model_cfg, seed, device)\n"
        "                fitted = replace(fitted, val_f1="
        + F1_AT.format(y="fitted.val.y", p="fitted.val.proba", thr=ON_FITTED_TEST)
        + ")\n",
        (DRIFT_E2E,),
    ),
    Mutant(
        "alert rows rewritten at a test-searched threshold",
        "mulegraph/pipeline.py",
        LEAD_SITE,
        "from mulegraph.drift.monitor import score_batches as _sb\n"
        f"                _alt = _sb({SCORE_ARGS}, threshold={ON_FITTED_TEST})\n"
        '                _a = table["detector"] == "alert"\n'
        '                table.loc[_a, ["score", "flagged", "threshold"]] = '
        '_alt.loc[_a, ["score", "flagged", "threshold"]]\n'
        f"                {LEAD_SITE}",
        (DRIFT_E2E,),
    ),
    Mutant(
        "test-derived F1 drop rule",
        "mulegraph/pipeline.py",
        LEAD_SITE,
        LEAD_SITE.replace(
            "cfg.drift.f1_drop,",
            f"1 - (1 - cfg.drift.f1_drop) * {ON_FITTED_TEST} / fitted.threshold,",
        ),
        (DRIFT_E2E,),
    ),
    Mutant(
        "first_drop and lead overwritten at a test-searched threshold",
        "mulegraph/pipeline.py",
        LEAD_SITE,
        f"{LEAD_SITE}\n"
        f"                _l = lead_time(per_timestep(fitted.test, {ON_FITTED_TEST}),"
        " table, ref_f1, cfg.drift.f1_drop, cfg.drift.drop_run)\n"
        '                lead[["first_drop", "lead"]] = _l[["first_drop", "lead"]]',
        (DRIFT_E2E,),
    ),
    Mutant(
        "alert flag level derived from test F1 (PR-R2)",
        "mulegraph/pipeline.py",
        "alert_flag=cfg.drift.alert_flag,",
        'alert_flag=float(curve["f1"].min()) / 100,',
        (DRIFT_E2E,),
    ),
    Mutant(
        "label column in the detector features (PR-R2)",
        "mulegraph/pipeline.py",
        "feats.values[scored_idx],\n                    proba,",
        "np.column_stack([feats.values[scored_idx], data.y[scored_idx]]),\n"
        "                    proba,",
        (DRIFT_E2E,),
    ),
    Mutant(
        "scores shifted so the alert rate cuts at a test-searched threshold",
        "mulegraph/pipeline.py",
        "proba = fitted.model.predict_proba(data, feats, scored_idx)\n",
        "proba = fitted.model.predict_proba(data, feats, scored_idx)\n"
        f"                proba = proba - {ON_FITTED_TEST} + fitted.threshold\n",
        (DRIFT_E2E,),
    ),
    Mutant(
        "logged diag_* at a test-searched threshold (drift)",
        "mulegraph/pipeline.py",
        "curve = per_timestep(test, fitted.test_threshold)",
        DIAG_AT_TEST,
        (DRIFT_E2E,),
    ),
    # If cfg is ever frozen, these plants raise instead of asserting, which the row check
    # rejects: re-plant them as some other rewrite of the drift settings, don't drop them.
    Mutant(
        "run rewrites cfg.drift.f1_drop from test F1",
        "mulegraph/pipeline.py",
        'ref_f1 = float(per_timestep(fitted.val, fitted.threshold)["f1"].mean())\n',
        'ref_f1 = float(per_timestep(fitted.val, fitted.threshold)["f1"].mean())\n'
        "                object.__setattr__(\n"
        '                    cfg.drift, "f1_drop", float(1 - curve["f1"].median())\n'
        "                )\n",
        (DRIFT_E2E,),
    ),
    Mutant(
        "run rewrites cfg.drift flags from test F1",
        "mulegraph/pipeline.py",
        "proba = fitted.model.predict_proba(data, feats, scored_idx)\n",
        "proba = fitted.model.predict_proba(data, feats, scored_idx)\n"
        '                object.__setattr__(cfg.drift, "calibrate", False)\n'
        "                object.__setattr__(\n"
        '                    cfg.drift, "alert_flag", float(curve["f1"].min()) / 100\n'
        "                )\n",
        (DRIFT_E2E,),
    ),
    # From the integrity audit of #53.
    Mutant(
        "benchmark val_f1 at a test-searched threshold",
        "mulegraph/pipeline.py",
        '"val_f1": fitted.val_f1,\n                "fit_seconds"',
        '"val_f1": '
        + F1_AT.format(y="fitted.val.y", p="fitted.val.proba", thr=ON_TEST)
        + ',\n                "fit_seconds"',
        (LOGGED_F1,),
    ),
    Mutant(
        "plot_drift draws a test-searched lead table",
        "mulegraph/pipeline.py",
        "            lead_table,\n            paths.figures_dir()",
        "            pd.concat(bad_leads, ignore_index=True),\n            paths.figures_dir()",
        (DRIFT_E2E,),
        also=BAD_LEADS,
    ),
    Mutant(
        "lead CSV's ref_f1 column at a test-searched threshold",
        "mulegraph/pipeline.py",
        "leads.append(lead.assign(ref_f1=ref_f1, **tag))",
        "leads.append(lead.assign(ref_f1="
        f"float(per_timestep(fitted.val, {ON_FITTED_TEST})['f1'].mean()), **tag))",
        (DRIFT_E2E,),
    ),
    Mutant(
        "logged ts_* metrics at a test-searched threshold",
        "mulegraph/pipeline.py",
        "for row in curve.itertuples(index=False):\n            mlflow.log_metrics(",
        f"for row in per_timestep(test, {ON_TEST}).itertuples(index=False):\n"
        "            mlflow.log_metrics(",
        (LOGGED_F1,),
    ),
    Mutant(
        "labels folded into the detector batch ids (PR-R2)",
        "mulegraph/pipeline.py",
        "batch = data.batch_id[scored_idx]\n",
        "batch = data.batch_id[scored_idx] + (data.y[scored_idx] == 1)\n",
        (DRIFT_E2E,),
    ),
    Mutant(
        "deployed model's cached threshold searched on test",
        "mulegraph/pipeline.py",
        'meta = {"threshold": fitted.threshold,',
        f'meta = {{"threshold": {ON_FITTED_TEST},',
        (SCORE_E2E,),
    ),
    Mutant(
        "score threshold read back as a test-searched one",
        "mulegraph/pipeline.py",
        'threshold = fit_meta["threshold"]',
        f"threshold = {ON_SPLIT_TEST}",
        (SCORE_E2E,),
    ),
    Mutant(
        "health check's alert-rate detector at a test-searched threshold (PR-R2)",
        "mulegraph/pipeline.py",
        HEALTH_SITE,
        HEALTH_SITE.replace("threshold=threshold,", f"threshold={ON_SPLIT_TEST},"),
        (SCORE_E2E,),
    ),
    Mutant(
        "alert queue cut at a test-searched threshold",
        "mulegraph/pipeline.py",
        "order = order[scores[order] >= threshold]",
        f"order = order[scores[order] >= {ON_SPLIT_TEST}]",
        (SCORE_E2E,),
    ),
    Mutant(
        "health file's threshold searched on test",
        "mulegraph/pipeline.py",
        '"threshold": threshold,',
        f'"threshold": {ON_SPLIT_TEST},',
        (SCORE_E2E,),
    ),
    Mutant(
        "report page's threshold searched on test",
        "mulegraph/pipeline.py",
        'page_path = write_score_report(tables_dir / f"{stem}_report.html", report, alerts)',
        'page_path = write_score_report(tables_dir / f"{stem}_report.html",'
        f' {{**report, "threshold": {ON_SPLIT_TEST}}}, alerts)',
        (SCORE_E2E,),
    ),
    Mutant(
        "health file and page's alert-rate score at a test-searched threshold",
        "mulegraph/pipeline.py",
        DETECTOR_ROW,
        DETECTOR_ROW.replace(
            '"score": r.score',
            f'"score": {ALERT_AT}(proba[np.isin(batch, ref_batches)], scores, {ON_SPLIT_TEST})'
            ' if r.detector == "alert" else r.score',
        ),
        (SCORE_E2E,),
    ),
    Mutant(
        "health file and page's alert-rate flag level at a test-searched threshold",
        "mulegraph/pipeline.py",
        DETECTOR_ROW,
        DETECTOR_ROW.replace(
            '"flag_at": r.threshold',
            f'"flag_at": max({ALERT_AT}(proba[np.isin(batch, ref_batches) & (batch != b)],'
            f" proba[batch == b], {ON_SPLIT_TEST}) for b in ref_batches)"
            ' if r.detector == "alert" else r.threshold',
        ),
        (SCORE_E2E,),
    ),
    Mutant(
        "report page's alert count at a test-searched threshold",
        "mulegraph/pipeline.py",
        'page_path = write_score_report(tables_dir / f"{stem}_report.html", report, alerts)',
        'page_path = write_score_report(tables_dir / f"{stem}_report.html",'
        f' {{**report, "alerts": int((scores >= {ON_SPLIT_TEST}).sum())}}, alerts)',
        (SCORE_E2E,),
    ),
    Mutant(
        "report page's queue cut at a test-searched threshold",
        "mulegraph/pipeline.py",
        'page_path = write_score_report(tables_dir / f"{stem}_report.html", report, alerts)',
        # Re-cut from today's scores, not a filter of alerts, so a lower test threshold shows too.
        '_o = np.argsort(-scores, kind="stable")\n'
        f"    _o = _o[scores[_o] >= {ON_SPLIT_TEST}]\n"
        '    _q = pd.DataFrame({"rank": np.arange(1, _o.size + 1), "unit": units[_o],'
        ' "score": scores[_o], "node_id": data.node_ids[units[_o]]})\n'
        '    page_path = write_score_report(tables_dir / f"{stem}_report.html", report, _q)',
        (SCORE_E2E,),
    ),
    # One row per output sink, planted on the line that writes it (#70).
    Mutant(
        "curves CSV written at a test-searched threshold",
        "mulegraph/pipeline.py",
        "curve_table.to_csv(",
        "pd.concat(bad_curves, ignore_index=True).to_csv(",
        (LOGGED_F1,),
        also=BAD_CURVES,
    ),
    Mutant(
        "drift scores CSV written at a test-searched threshold",
        "mulegraph/pipeline.py",
        "pd.concat(scores, ignore_index=True).to_csv(score_path, index=False)",
        "pd.concat(bad_scores, ignore_index=True).to_csv(score_path, index=False)",
        (DRIFT_E2E,),
        also=(
            (
                "    scores, leads, curves, events = [], [], [], []\n",
                "    scores, leads, curves, events = [], [], [], []\n    bad_scores = []\n",
            ),
            (
                "                scores.append(table.assign(**tag))\n",
                "                scores.append(table.assign(**tag))\n"
                # Imported here so the guard's call count on pipeline.score_batches stays honest.
                "                from mulegraph.drift.monitor import score_batches as _sb\n"
                f"                bad_scores.append(_sb({SCORE_ARGS}, threshold={ON_FITTED_TEST})"
                ".assign(**tag))\n",
            ),
        ),
    ),
    Mutant(
        "lead CSV written at a test-searched threshold",
        "mulegraph/pipeline.py",
        "lead_table.to_csv(lead_path, index=False)",
        "pd.concat(bad_leads, ignore_index=True).to_csv(lead_path, index=False)",
        (DRIFT_E2E,),
        also=BAD_LEADS,
    ),
    Mutant(
        "event CSV written at a test-searched threshold",
        "mulegraph/pipeline.py",
        "pd.concat(events, ignore_index=True).to_csv(artifacts[-1], index=False)",
        "pd.concat(bad_events, ignore_index=True).to_csv(artifacts[-1], index=False)",
        (DRIFT_E2E,),
        also=(
            (
                "    scores, leads, curves, events = [], [], [], []\n",
                "    scores, leads, curves, events = [], [], [], []\n    bad_events = []\n",
            ),
            (
                "                    events.append(table.assign(**tag))\n",
                "                    events.append(table.assign(**tag))\n"
                "                    bad_events.append(event_windows(fitted.test,"
                f" {ON_FITTED_TEST}, event).assign(**tag))\n",
            ),
        ),
    ),
    Mutant(
        "deployed model's meta file written with a test-searched threshold",
        "mulegraph/pipeline.py",
        "meta_path.write_text(json.dumps(meta, indent=2)",
        f'meta_path.write_text(json.dumps({{**meta, "threshold": {ON_FITTED_TEST}}}, indent=2)',
        (REUSE,),
    ),
    Mutant(
        "alerts CSV cut at a test-searched threshold",
        "mulegraph/pipeline.py",
        "alerts.to_csv(alerts_path, index=False)",
        '_o = np.argsort(-scores, kind="stable")\n'
        f"    _o = _o[scores[_o] >= {ON_SPLIT_TEST}]\n"
        '    pd.DataFrame({"rank": np.arange(1, _o.size + 1), "unit": units[_o],'
        ' "score": scores[_o]}).to_csv(alerts_path, index=False)',
        (SCORE_E2E,),
    ),
    Mutant(
        "health file written with a test-searched threshold",
        "mulegraph/pipeline.py",
        "health_path.write_text(json.dumps(report, indent=2)",
        f'health_path.write_text(json.dumps({{**report, "threshold": {ON_SPLIT_TEST}}}, indent=2)',
        (SCORE_E2E,),
    ),
    Mutant(
        "CLI drift flags from a test-searched threshold",
        "mulegraph/pipeline.py",
        "    return alerts_path, health_path, page_path, flagged",
        f"    return alerts_path, health_path, page_path, {FLAGS_AT_TEST}",
        (SCORE_E2E,),
    ),
    # From the delta audit of #70: values inside already-planted outputs.
    Mutant(
        "health file's flagged list from a test-searched threshold",
        "mulegraph/pipeline.py",
        '"flagged": flagged,',
        f'"flagged": {FLAGS_AT_TEST},',
        (SCORE_E2E,),
    ),
    Mutant(
        "report page's flags from a test-searched threshold",
        "mulegraph/pipeline.py",
        'page_path = write_score_report(tables_dir / f"{stem}_report.html", report, alerts)',
        'page_path = write_score_report(tables_dir / f"{stem}_report.html",'
        f' {{**report, "flagged": {FLAGS_AT_TEST}}}, alerts)',
        (SCORE_E2E,),
    ),
    Mutant(
        # Written into the file only: the report dict also feeds the page, whose count has its row.
        "health file's alert count at a test-searched threshold",
        "mulegraph/pipeline.py",
        "health_path.write_text(json.dumps(report, indent=2)",
        'health_path.write_text(json.dumps({**report, "alerts":'
        f" int((scores >= {ON_SPLIT_TEST}).sum())}}, indent=2)",
        (SCORE_E2E,),
    ),
    Mutant(
        "deployed model's cached val_f1 at a test-searched threshold",
        "mulegraph/pipeline.py",
        '"val_f1": fitted.val_f1, "commit": commit',
        '"val_f1": '
        + F1_AT.format(y="fitted.val.y", p="fitted.val.proba", thr=ON_FITTED_TEST)
        + ', "commit": commit',
        (REUSE,),
    ),
    Mutant(
        "health file's model_fit threshold searched on test",
        "mulegraph/pipeline.py",
        '"model_fit": fit_meta,',
        f'"model_fit": {{**fit_meta, "threshold": {ON_SPLIT_TEST}}},',
        (REUSE,),
    ),
]

#: Every threshold-bearing line of pipeline.py that no row plants, with why (stripped line: reason).
UNPLANTED = {
    "from mulegraph.eval.threshold import choose_threshold": "import",
    "threshold: float": "field declaration",
    "test_threshold: np.ndarray": "field declaration",
    "tests, thresholds, per_step, seconds = [], [], {}, 0.0": "empty-list initialiser",
    'log.info("deployed model %s, validation threshold %.4f", model_path, fitted.threshold)': (
        "log line"
    ),
    "threshold,": "log line",
}


#: Every output of pipeline.py that no row plants, with why no threshold-derived value reaches
#: it (``function: source`` as ``ast.unparse`` prints it: reason).
UNSUNK = {
    "_log_predictions: frame.to_parquet(path, index=False)": (
        "scored rows only (idx, proba, y, time, part), from which the guards re-derive thresholds"
    ),
    "_log_predictions: mlflow.log_artifact(str(path))": "logs the file written on the line above",
    "run_benchmark: mlflow.log_artifact(str(config_path))": "logs the run's config file as given",
    "run_drift: mlflow.log_artifact(str(config_path))": "logs the run's config file as given",
    "run_drift: mlflow.log_artifact(str(artifact))": (
        "copies into the run the drift tables and figure, each planted where it is written"
    ),
    "run_benchmark: mlflow.log_params({'dataset_version': data.meta.version, 'feature_version':"
    " gfp.feature_version if gfp else 'none', 'seeds': list(cfg.seeds), 'device': device,"
    " 'total_fits': total, 'trials_completed': TRIALS_COMPLETED})": (
        "versions, seeds, device and fit count, all set before any fit"
    ),
    "run_drift: mlflow.log_params({'dataset_version': data.meta.version,"
    " 'seeds': list(cfg.seeds)})": "dataset version and seeds, set before any fit",
    "_log_child: mlflow.log_params({'seed': seed, 'device': device, 'trials_completed':"
    " TRIALS_COMPLETED, 'trial0_source': info.extra.get('trial0_source', 'unknown'), 'sampler':"
    " model_cfg.sampler.kind if model_cfg.sampler else 'none', **{f'model.{k}': v for k, v in"
    " info.extra.get('params', {}).items()}})": (
        "seed, device and the model's fixed reference parameters (D2), set before the fit"
    ),
    "_log_child: mlflow.set_tags({'kind': 'child', 'dataset': data.meta.dataset,"
    " 'dataset_version': data.meta.version, 'regime': regime, 'model': model_cfg.name,"
    " 'features': model_cfg.features, 'feature_version': feats.feature_version, 'split_hash':"
    " fitted.split_hash, 'git_commit': commit})": "identity and provenance tags, no metric",
    "run_benchmark: mlflow.set_tags({'kind': 'parent', 'dataset': data.meta.dataset,"
    " 'git_commit': commit})": "identity and provenance tags, no metric",
    "run_drift: mlflow.set_tags({'kind': 'parent', 'dataset': data.meta.dataset,"
    " 'git_commit': commit})": "identity and provenance tags, no metric",
    "_deployed_model: model.save(model_path)": (
        "the fitted booster; its threshold goes in the meta file beside it"
    ),
    "run_benchmark: write_results_table(cfg.mlflow.experiment, paths.tables_dir(),"
    " tracking_uri=tracking_uri, parent_run_id=parent.info.run_id)": (
        "passes names, paths and the parent run id; the table reads the child runs' metrics,"
        " planted where they are logged"
    ),
    "run_drift: return lead_path": "a path, its file planted where it is written",
    "run_smoke: return (table, *_smoke_score(cfg))": (
        "paths from run_benchmark and run_score, each file planted where it is written"
    ),
}

#: Every way pipeline.py writes an output: a call whose dotted name, imports resolved, starts
#: with a prefix, a method in SINK_METHODS, or a ``run_*`` function's return value (the CLI
#: exits on run_score's). A new kind goes here first, then gets planted or excused.
SINK_METHODS = {"to_csv", "to_parquet", "write_text", "save", "savefig"}
SINK_PREFIXES = ("mulegraph.report.", "mlflow.log_", "mlflow.set_tag")


def _planted_lines(text: str) -> set[int]:
    """Line numbers of pipeline.py ``text`` that some row's site plants a bug on."""
    planted = set()
    for m in MUTANTS:
        # A row that keeps its site verbatim and only inserts code beside it plants nothing there.
        if m.file == "mulegraph/pipeline.py" and m.site in text and m.site not in m.broken:
            start = text[: text.index(m.site)].count("\n") + 1
            planted.update(range(start, start + m.site.count("\n") + 1))
    return planted


def _imports(tree: ast.Module) -> dict[str, str]:
    """Each name pipeline.py binds by import, to the dotted path it binds."""
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound[a.asname or a.name.split(".")[0]] = (
                    a.name if a.asname else a.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            # pipeline.py sits at the package top, so a relative import is from mulegraph.
            base = ".".join(filter(None, ("mulegraph" if node.level else "", node.module)))
            for a in node.names:
                bound[a.asname or a.name] = f"{base}.{a.name}"
    return bound


def _unsunk(text: str) -> list[str]:
    """``function: source`` of each output in pipeline.py ``text`` that no row's site overlaps."""
    tree = ast.parse(text)
    bound, planted = _imports(tree), _planted_lines(text)

    def sink(node: ast.AST | None) -> bool:
        if not isinstance(node, ast.Call):
            return False
        head, dot, rest = ast.unparse(node.func).partition(".")
        return (bound.get(head, head) + dot + rest).startswith(SINK_PREFIXES) or (
            isinstance(node.func, ast.Attribute) and node.func.attr in SINK_METHODS
        )

    owner = {n: f.name for f in tree.body if isinstance(f, ast.FunctionDef) for n in ast.walk(f)}
    outputs = [
        n
        for n in ast.walk(tree)
        if sink(n)
        or (isinstance(n, ast.Return) and owner[n].startswith("run_") and not sink(n.value))
    ]
    # ponytail: planted per output, not per argument, so a new key inside an already-planted
    # log_metrics slips through; pin the logged key set in its guard if that ever matters.
    return [
        f"{owner.get(n, '<module>')}: {ast.unparse(n)}"
        for n in outputs
        if not planted & set(range(n.lineno, n.end_lineno + 1))
    ]


def _threshold_lines(path: Path) -> dict[int, str]:
    """Line number to stripped text of each code line naming a threshold or logging one."""
    with path.open("rb") as fh:
        hits = {
            tok.start[0]
            for tok in tokenize.tokenize(fh.readline)
            if (tok.type == tokenize.NAME and "threshold" in tok.string)
            or (
                tok.type == tokenize.STRING
                and tok.string.strip("\"'") in {"threshold", "ts_threshold"}
            )
        }
    lines = path.read_text().splitlines()
    return {n: lines[n - 1].strip() for n in hits}


def _copy(repo_root: Path, tmp_path: Path) -> Path:
    """The package, its tests and configs, so pytest in the copy imports the copy."""
    root = tmp_path / "repo"
    for name in ("mulegraph", "tests", "configs"):
        shutil.copytree(repo_root / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(repo_root / "pyproject.toml", root)
    return root


def _plant(root: Path, mutant: Mutant) -> None:
    path = root / mutant.file
    text = path.read_text()
    for site, broken in ((mutant.site, mutant.broken), *mutant.also):
        if text.count(site) != 1:
            pytest.fail(f"site moved: re-check this guard ({mutant.name}, {mutant.file})")
        text = text.replace(site, broken)
    path.write_text(text)


def _run_pytest(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # cwd and PYTHONPATH both put the copy ahead of the editable install's .pth entry.
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=root,
        # A wide terminal keeps each failure's reason on its summary line. One thread per row:
        # rows run side by side under xdist, and oversubscribed BLAS threads made them 7x slower.
        env={**os.environ, "PYTHONPATH": str(root), "COLUMNS": "1000", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
    )


def test_every_guard_passes_in_the_unplanted_copy(repo_root: Path, tmp_path: Path) -> None:
    """Control: the copy imports itself and its guards pass, so a row's failure is its bug."""
    root = _copy(repo_root, tmp_path)
    (root / "tests" / "test_which.py").write_text(
        "import mulegraph\n\n\n"
        "def test_which():\n"
        f"    assert mulegraph.__file__ == {str(root / 'mulegraph' / '__init__.py')!r}\n"
    )
    guards = sorted({t for m in MUTANTS for t in m.tests})
    out = _run_pytest(root, "tests/test_which.py", *guards)
    assert out.returncode == 0, out.stdout + out.stderr


def test_a_missing_site_fails_with_site_moved(repo_root: Path, tmp_path: Path) -> None:
    root = _copy(repo_root, tmp_path)
    gone = MUTANTS[0]._replace(site="no such fragment")
    with pytest.raises(pytest.fail.Exception, match="site moved"):
        _plant(root, gone)


def test_every_threshold_use_is_planted_or_excused(repo_root: Path) -> None:
    """PR-E4: a new threshold use in pipeline.py fails here until it gets a row or a reason."""
    path = repo_root / "mulegraph" / "pipeline.py"
    planted = _planted_lines(path.read_text())
    unplanted = [s for n, s in _threshold_lines(path).items() if n not in planted]
    assert sorted(unplanted) == sorted(UNPLANTED), "each needs a MUTANTS row or an UNPLANTED reason"


def test_every_output_sink_is_planted_or_excused(repo_root: Path) -> None:
    """PR-E4: a new output in pipeline.py fails here until it gets a row or an UNSUNK reason."""
    unsunk = _unsunk((repo_root / "mulegraph" / "pipeline.py").read_text())
    # As lists: a copy of an excused output in another function needs its own reason, and
    # one in the same function fails until it gets a row (its key would repeat).
    assert sorted(unsunk) == sorted(UNSUNK), "each needs a MUTANTS row or an UNSUNK reason"


def test_a_new_output_is_seen_by_the_sink_scan(repo_root: Path) -> None:
    """Listed kinds, imported report writers and aliases show, and so does a copied excuse."""
    text = (repo_root / "mulegraph" / "pipeline.py").read_text()
    added = (
        "\nimport mlflow as mf\nfrom .report import figures\n"
        "from mulegraph.report.figures import plot_queue\n\n\n"
        "def _new(frame, path):\n    frame.to_csv('x.csv')\n    plot_queue(frame)\n"
        "    figures.plot_x(frame)\n    mf.set_tag('t', 1)\n    mlflow.log_artifact(str(path))\n"
    )
    assert set(_unsunk(text + added)) - set(_unsunk(text)) == {
        "_new: frame.to_csv('x.csv')",
        "_new: plot_queue(frame)",
        "_new: figures.plot_x(frame)",
        "_new: mf.set_tag('t', 1)",
        "_new: mlflow.log_artifact(str(path))",
    }


#: Rows that recompute an F1, with the honest threshold in their scope.
HONEST = {
    "rolling ts_val_f1 at a test-searched threshold": (ON_TEST, "threshold"),
    "drift val_f1 at a test-searched threshold": (ON_FITTED_TEST, "fitted.threshold"),
    "benchmark val_f1 at a test-searched threshold": (ON_TEST, "fitted.threshold"),
    "deployed model's cached val_f1 at a test-searched threshold": (
        ON_FITTED_TEST,
        "fitted.threshold",
    ),
}


@pytest.mark.parametrize("name", HONEST)
def test_a_recomputed_f1_at_the_honest_threshold_passes(
    name: str, repo_root: Path, tmp_path: Path
) -> None:
    """Control: the row is red for its threshold, not for how it recomputes F1."""
    mutant = next(m for m in MUTANTS if m.name == name)
    root = _copy(repo_root, tmp_path)
    _plant(root, mutant._replace(broken=mutant.broken.replace(*HONEST[name])))
    out = _run_pytest(root, *mutant.tests)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize("mutant", MUTANTS, ids=[m.name for m in MUTANTS])
def test_planting_the_bug_fails_its_guard(mutant: Mutant, repo_root: Path, tmp_path: Path) -> None:
    root = _copy(repo_root, tmp_path)
    _plant(root, mutant)
    out = _run_pytest(root, "--tb=line", *mutant.tests)
    log = out.stdout + out.stderr
    # A NameError or crash would also exit nonzero without proving the guard catches anything.
    for t in mutant.tests:
        pattern = rf"FAILED {re.escape(t)} - (assert|AssertionError)"
        assert re.search(pattern, log), log
    # ...and so would a bare assert inside the planted package, which --tb=line places there.
    assert not re.search(rf"^{re.escape(str(root / 'mulegraph'))}\S*:\d+: ", log, re.M), log
