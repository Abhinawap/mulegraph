"""Orchestrator — the only module that imports across subsystems (PR-E4, PR-O1)."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mulegraph.config import (
    DetectorConfig,
    DriftRunConfig,
    ModelConfig,
    RunConfig,
    ScoreConfig,
    load_config,
)
from mulegraph.data import load_dataset
from mulegraph.drift.monitor import lead_time, score_batches
from mulegraph.eval.curves import CURVE_METRICS, per_timestep
from mulegraph.eval.event import event_windows, probe_auc
from mulegraph.eval.metrics import EXTRA_KEYS, compute_metrics
from mulegraph.eval.threshold import choose_threshold
from mulegraph.features.builder import build_features
from mulegraph.features.select import select_features
from mulegraph.models import get_model
from mulegraph.models.base import BaseModel, FitInfo
from mulegraph.report.figures import plot_curves, plot_drift
from mulegraph.report.html import write_score_report
from mulegraph.report.tables import write_results_table
from mulegraph.splits.builder import build_split, rolling_steps
from mulegraph.types import FeatureMatrix, GraphDataset, Predictions, Split
from mulegraph.util import Paths, git_commit, git_dirty, hash_dict, resolve_device, seed_all

if TYPE_CHECKING:
    from mulegraph.models.xgb import XGBModel

log = logging.getLogger("mulegraph")

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"

#: The MVP fits trial 0 only, so the honest search count is zero, not a nominal one (D2).
TRIALS_COMPLETED = 0

#: Above this, the run is a background job rather than something to sit and watch.
LONG_RUN_FITS = 20


def _ensure_local_store(uri: str) -> None:
    """Create a SQLite tracking file's directory before MLflow opens it."""
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        Path(uri[len(prefix) :]).expanduser().parent.mkdir(parents=True, exist_ok=True)


@dataclass(eq=False)
class Fitted:
    """One (model, seed) fitted per step, thresholded on validation, scored on test (PR-E4)."""

    model: BaseModel  # the last step's, as are info, val and val_f1
    info: FitInfo  # seconds summed over every step
    #: The validation-chosen threshold drift and score apply; on a rolling run, the last step's.
    threshold: float
    val_f1: float
    val: Predictions
    test: Predictions  # every step's test rows, stitched in step order
    #: One entry per test row: the threshold of the step that scored it (PR-E7).
    test_threshold: np.ndarray
    split_hash: str
    #: Rolling only: each step's threshold and fit diagnostics, keyed on its test batch (PR-E7).
    steps: dict[int, dict[str, float]]


def _predict(
    model: BaseModel, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray
) -> Predictions:
    return Predictions(
        idx=idx,
        proba=model.predict_proba(data, feats, idx),
        y=data.y[idx],
        time=data.batch_id[idx],
    )


def _log_predictions(val: Predictions, test: Predictions) -> None:
    """Persist scored rows as an artifact so curves and drift are regenerable (NFR-1)."""
    import mlflow

    frame = pd.concat(
        [
            pd.DataFrame({"idx": p.idx, "proba": p.proba, "y": p.y, "time": p.time, "part": part})
            for part, p in (("val", val), ("test", test))
        ],
        ignore_index=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.parquet"
        frame.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))


def fit(
    data: GraphDataset,
    feats: FeatureMatrix,
    steps: list[Split],
    model_cfg: ModelConfig,
    seed: int,
    device: str,
) -> Fitted:
    """Per step: fit on train, threshold on val (PR-E4), score test; >1 step is rolling (PR-E7)."""
    # Only the last step's model is kept: twelve fitted SAGE models would each hold the full
    # graph on the GPU.
    rolling = len(steps) > 1
    tests, thresholds, per_step, seconds = [], [], {}, 0.0
    for step in steps:
        seed_all(seed)
        model = get_model(
            model_cfg.name,
            model_cfg.params,
            device=device,
            sampler=model_cfg.sampler,
            task=data.task,
        )
        info = model.fit(data, feats, step, seed)

        # PR-E4: the threshold comes from validation scores; test is scored with it, never searched.
        val = _predict(model, data, feats, step.val)
        threshold, val_f1 = choose_threshold(val.y, val.proba)
        test = _predict(model, data, feats, step.test)
        tests.append(test)
        thresholds.append(np.full(test.idx.size, threshold, dtype=np.float64))
        seconds += info.seconds
        if rolling:
            per_step[int(step.params["test"][0])] = {
                "ts_threshold": threshold,
                "ts_val_f1": val_f1,
                **({} if info.val_pr_auc is None else {"ts_val_pr_auc": info.val_pr_auc}),
                **(
                    {}
                    if info.best_iteration is None
                    else {"ts_best_iteration": info.best_iteration}
                ),
            }

    return Fitted(
        model=model,
        info=replace(info, seconds=seconds),
        threshold=threshold,
        val_f1=val_f1,
        val=val,
        test=Predictions(
            idx=np.concatenate([t.idx for t in tests]),
            proba=np.concatenate([t.proba for t in tests]),
            y=np.concatenate([t.y for t in tests]),
            time=np.concatenate([t.time for t in tests]),
        ),
        test_threshold=np.concatenate(thresholds),
        # A rolling run is identified by every step's partition (PR-E7).
        split_hash=hash_dict([s.split_hash for s in steps]) if rolling else steps[0].split_hash,
        steps=per_step,
    )


def _log_child(
    cfg: RunConfig,
    data: GraphDataset,
    feats: FeatureMatrix,
    regime: str,
    model_cfg: ModelConfig,
    seed: int,
    device: str,
    commit: str,
    fitted: Fitted,
) -> pd.DataFrame:
    """Log one MLflow child run and return its per-timestep test curve (PR-E4, PR-O1, PR-E7)."""
    import mlflow

    test, info = fitted.test, fitted.info
    # A rolling test set is scored by one model per batch, so only the thresholded metric is
    # comparable across the stitched rows; rank metrics over incomparable score scales are
    # not logged and live in the per-timestep curve instead (PR-E7).
    metrics = [m for m in cfg.eval.metrics if m == "f1"] if fitted.steps else cfg.eval.metrics
    scored = compute_metrics(test.y, test.proba, fitted.test_threshold, metrics, prefix="test_")
    # Counts and point precision/recall are diagnostics: the results table is metrics.test_* only.
    diagnostics = {f"diag_{key}": scored.pop(f"test_{key}") for key in EXTRA_KEYS}
    curve = per_timestep(test, fitted.test_threshold)

    with mlflow.start_run(run_name=f"{regime}.{model_cfg.key}.s{seed}", nested=True):
        mlflow.set_tags(
            {
                "kind": "child",
                "dataset": data.meta.dataset,
                # Spec §2.3 makes this a tag, not a parent param: nested runs inherit
                # nothing, and the reporter only ever reads child runs.
                "dataset_version": data.meta.version,
                "regime": regime,
                "model": model_cfg.name,
                "features": model_cfg.features,
                "feature_version": feats.feature_version,
                "split_hash": fitted.split_hash,
                "git_commit": commit,
            }
        )
        mlflow.log_params(
            {
                "seed": seed,
                "device": device,
                "trials_completed": TRIALS_COMPLETED,
                "trial0_source": info.extra.get("trial0_source", "unknown"),
                "sampler": model_cfg.sampler.kind if model_cfg.sampler else "none",
                **{f"model.{k}": v for k, v in info.extra.get("params", {}).items()},
            }
        )
        mlflow.log_metrics(
            {
                **scored,
                **diagnostics,
                # val_f1 is a selection diagnostic, never a headline metric (PR-E6). On a rolling
                # run both are the last step's; every step's are logged at step=t (PR-E7).
                "threshold": fitted.threshold,
                "val_f1": fitted.val_f1,
                "fit_seconds": info.seconds,
                **({} if info.val_pr_auc is None else {"val_pr_auc": info.val_pr_auc}),
                **({} if info.best_iteration is None else {"best_iteration": info.best_iteration}),
            }
        )
        for row in curve.itertuples(index=False):
            mlflow.log_metrics(
                {
                    f"ts_{name}": getattr(row, name)
                    for name in CURVE_METRICS
                    if not math.isnan(getattr(row, name))
                },
                step=int(row.time),
            )
        for t, step_metrics in fitted.steps.items():
            mlflow.log_metrics(step_metrics, step=t)
        _log_predictions(fitted.val, test)
    return curve


def _prepare(cfg: RunConfig) -> tuple[Paths, str, str, str]:
    """Point MLflow at the store and stamp the run: ``(paths, tracking_uri, device, commit)``."""
    import mlflow

    paths = Paths.from_env()
    tracking_uri = cfg.mlflow.resolved_uri()
    _ensure_local_store(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    return paths, tracking_uri, resolve_device(cfg.device), _stamp()


def _stamp() -> str:
    """The commit this run's outputs are stamped with, ``-dirty`` on an uncommitted tree."""
    commit = git_commit()
    if git_dirty():
        # The stamp has to carry what the warning says: a log line vanishes, and a
        # clean sha on the table would claim provenance this run does not have (NFR-1).
        commit = f"{commit}-dirty"
        log.warning(
            "working tree has uncommitted changes: this run is not regenerable from a "
            "tagged commit, so its runs are stamped %s (NFR-1)",
            commit,
        )
    return commit


def run_benchmark(cfg: RunConfig, config_path: Path) -> Path:
    """Load, build features, split, fit across seeds, evaluate, write the table."""
    import mlflow

    paths, tracking_uri, device, commit = _prepare(cfg)
    data = load_dataset(cfg.dataset, paths.data_dir)
    cache_dir = paths.cache_dir(cfg.dataset.name, cfg.dataset.version)

    gfp = build_features(data, cfg.features, cache_dir) if cfg.needs_gfp else None
    matrices = {m.features: select_features(data, gfp, m.features) for m in cfg.models}
    # Built before the first fit so an undefined regime fails before hours of work (D1).
    # One partition per regime, or one per test batch when rolling (PR-E7).
    splits: dict[str, list[Split]] = {}
    for r in cfg.split.regimes:
        steps = rolling_steps(r) if r.regime == "temporal_rolling" else [r]
        splits[r.regime] = [build_split(data, step, cache_dir) for step in steps]

    total = cfg.total_fits()
    log.info(
        "%s: %d fits (%d models x %d seeds x %d fits over %d regimes) on %s, tracking to %s",
        cfg.mlflow.experiment,
        total,
        len(cfg.models),
        len(cfg.seeds),
        sum(r.fits for r in cfg.split.regimes),
        len(cfg.split.regimes),
        device,
        tracking_uri,
    )
    if total >= LONG_RUN_FITS:
        log.info(
            "this is a long run: %d fits, each logged as its own MLflow child run as it finishes",
            total,
        )

    done = 0
    curves: list[pd.DataFrame] = []
    with mlflow.start_run(run_name=f"{cfg.mlflow.experiment}.{commit}") as parent:
        mlflow.set_tags({"kind": "parent", "dataset": data.meta.dataset, "git_commit": commit})
        mlflow.log_params(
            {
                "dataset_version": data.meta.version,
                "feature_version": gfp.feature_version if gfp else "none",
                "seeds": list(cfg.seeds),
                "device": device,
                "total_fits": total,
                "trials_completed": TRIALS_COMPLETED,
            }
        )
        mlflow.log_artifact(str(config_path))
        for regime_cfg in cfg.split.regimes:
            if regime_cfg.regime == "temporal_rolling":
                log.info("rolling regime: %d refits per (model, seed)", regime_cfg.fits)
            for model_cfg in cfg.models:
                for seed in cfg.seeds:
                    log.info(
                        "fit %d/%d: %s %s seed %d",
                        done + 1,
                        total,
                        regime_cfg.regime,
                        model_cfg.key,
                        seed,
                    )
                    feats = matrices[model_cfg.features]
                    fitted = fit(data, feats, splits[regime_cfg.regime], model_cfg, seed, device)
                    curve = _log_child(
                        cfg, data, feats, regime_cfg.regime, model_cfg, seed, device, commit, fitted
                    )
                    done += regime_cfg.fits
                    curves.append(
                        curve.assign(
                            regime=regime_cfg.regime,
                            model=model_cfg.name,
                            features=model_cfg.features,
                            seed=seed,
                        )
                    )

    curve_table = pd.concat(curves, ignore_index=True)
    tables_dir = paths.tables_dir()
    tables_dir.mkdir(parents=True, exist_ok=True)
    curve_table.to_csv(tables_dir / f"{cfg.mlflow.experiment}_curves.csv", index=False)
    plot_curves(curve_table, paths.figures_dir() / f"{cfg.mlflow.experiment}_curves.png")

    # Scoped to this run's children: an earlier run of the same config in the same
    # experiment must not be counted as extra seeds (PR-E3).
    return write_results_table(
        cfg.mlflow.experiment,
        paths.tables_dir(),
        tracking_uri=tracking_uri,
        parent_run_id=parent.info.run_id,
    )


def run_drift(cfg: DriftRunConfig, config_path: Path) -> Path:
    """Fit, then score every post-training batch label-free and report lead time (PR-R2, PR-R4)."""
    import mlflow

    paths, tracking_uri, device, commit = _prepare(cfg)
    regime_cfg = cfg.split.regimes[0]
    assert regime_cfg.val is not None and regime_cfg.test is not None
    data = load_dataset(cfg.dataset, paths.data_dir)
    cache_dir = paths.cache_dir(cfg.dataset.name, cfg.dataset.version)
    gfp = build_features(data, cfg.features, cache_dir) if cfg.needs_gfp else None
    matrices = {m.features: select_features(data, gfp, m.features) for m in cfg.models}
    split = build_split(data, regime_cfg, cache_dir)

    ref_batches = np.arange(regime_cfg.val[0], regime_cfg.val[1] + 1)
    # Every unit from the reference window onward, labelled or not: the monitor sees
    # what a deployed model sees, and labels never reach a detector (PR-R2).
    scored_idx = np.flatnonzero(
        (data.batch_id >= regime_cfg.val[0]) & (data.batch_id <= regime_cfg.test[1])
    )
    batch = data.batch_id[scored_idx]

    log.info(
        "%s: %d fits, then %d batches scored against reference batches %s..%s",
        cfg.mlflow.experiment,
        cfg.total_fits(),
        len(np.unique(batch)) - len(ref_batches),
        ref_batches[0],
        ref_batches[-1],
    )
    event = cfg.drift.event
    probes: dict[tuple[str, str], float] = {}
    if event is not None:
        # Labelled test units only; the probe is a separability check, never a detector (PR-R5).
        before = data.batch_id[split.test] < event
        for key in matrices:
            for window, mask in (("before", before), ("after", ~before)):
                idx = split.test[mask]
                probes[key, window] = probe_auc(
                    matrices[key].values[idx], data.y[idx], cfg.seeds[0]
                )
        log.info("event at batch %d: probe ROC-AUC %s", event, probes)
    scores, leads, curves, events = [], [], [], []
    with mlflow.start_run(run_name=f"{cfg.mlflow.experiment}.{commit}"):
        mlflow.set_tags({"kind": "parent", "dataset": data.meta.dataset, "git_commit": commit})
        mlflow.log_params({"dataset_version": data.meta.version, "seeds": list(cfg.seeds)})
        mlflow.log_artifact(str(config_path))
        for model_cfg in cfg.models:
            feats = matrices[model_cfg.features]
            for seed in cfg.seeds:
                log.info("fit %s seed %d", model_cfg.key, seed)
                fitted = fit(data, feats, [split], model_cfg, seed, device)
                curve = _log_child(
                    cfg, data, feats, split.regime, model_cfg, seed, device, commit, fitted
                )
                proba = fitted.model.predict_proba(data, feats, scored_idx)
                table = score_batches(
                    feats.values[scored_idx],
                    proba,
                    batch,
                    ref_batches,
                    detectors=cfg.drift.detectors,
                    bins=cfg.drift.bins,
                    psi_flag=cfg.drift.psi_flag,
                    ks_alpha=cfg.drift.ks_alpha,
                    ks_frac_flag=cfg.drift.ks_frac,
                    conf_flag=cfg.drift.conf_flag,
                    alert_flag=cfg.drift.alert_flag,
                    # The validation-chosen threshold (PR-E4); the alert-rate detector counts
                    # units at or above it and never sees a label (PR-R2).
                    threshold=fitted.threshold,
                    calibrate=cfg.drift.calibrate,
                )
                ref_f1 = float(per_timestep(fitted.val, fitted.threshold)["f1"].mean())
                lead = lead_time(curve, table, ref_f1, cfg.drift.f1_drop, cfg.drift.drop_run)
                tag = {"model": model_cfg.name, "features": model_cfg.features, "seed": seed}
                scores.append(table.assign(**tag))
                leads.append(lead.assign(ref_f1=ref_f1, **tag))
                curves.append(curve.assign(**tag))
                if event is not None:
                    table = event_windows(fitted.test, fitted.threshold, event)
                    table["probe_roc_auc"] = [
                        probes[model_cfg.features, w] for w in table["window"]
                    ]
                    events.append(table.assign(**tag))

        tables_dir = paths.tables_dir()
        tables_dir.mkdir(parents=True, exist_ok=True)
        stem = cfg.mlflow.experiment
        score_path = tables_dir / f"{stem}_scores.csv"
        lead_path = tables_dir / f"{stem}_lead_time.csv"
        pd.concat(scores, ignore_index=True).to_csv(score_path, index=False)
        lead_table = pd.concat(leads, ignore_index=True)
        lead_table.to_csv(lead_path, index=False)
        figure = plot_drift(
            pd.concat(curves, ignore_index=True),
            lead_table,
            paths.figures_dir() / f"{stem}_drift.png",
        )
        artifacts = [score_path, lead_path, figure]
        if events:
            artifacts.append(tables_dir / f"{stem}_event.csv")
            pd.concat(events, ignore_index=True).to_csv(artifacts[-1], index=False)
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
    log.info("wrote %s and %s", score_path, lead_path)
    return lead_path


def _deployed_model(
    cfg: ScoreConfig,
    data: GraphDataset,
    feats: FeatureMatrix,
    split: Split,
    device: str,
    cache_dir: Path,
    commit: str,
) -> tuple[XGBModel, dict[str, Any]]:
    """Fit once and threshold on validation (PR-E4); reuse both until a definition changes."""
    import xgboost

    from mulegraph.models.xgb import XGBModel

    definition = {
        "dataset_version": data.meta.version,
        "feature_version": feats.feature_version,
        "split_hash": split.split_hash,
        "model": cfg.model.key,
        "params": cfg.model.params,
        "seed": 0,
        # GPU and CPU hist grow different trees, and a library upgrade may too.
        "device": device,
        "xgboost": xgboost.__version__,
    }
    model_path = cache_dir / "models" / f"{hash_dict(definition)}.ubj"
    meta_path = model_path.with_suffix(".json")
    if model_path.is_file() and meta_path.is_file():
        log.info("reusing deployed model %s", model_path)
        return XGBModel.load(model_path), json.loads(meta_path.read_text())

    # Gone before the refit, so a crash can never pair a new booster with an old threshold.
    meta_path.unlink(missing_ok=True)
    fitted = fit(data, feats, [split], cfg.model, 0, device)
    model = fitted.model
    assert isinstance(model, XGBModel)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    # The health file is the only record of a score run, so the fit's provenance lives here
    # (NFR-1): the commit that fitted the model, not the one that later scores with it.
    meta = {"threshold": fitted.threshold, "val_f1": fitted.val_f1, "commit": commit, **definition}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("deployed model %s, validation threshold %.4f", model_path, fitted.threshold)
    return model, meta


def run_score(cfg: ScoreConfig) -> tuple[Path, Path, Path, list[str]]:
    """Score one batch: alert queue, label-free health check, HTML page (D5, PR-R2)."""
    paths = Paths.from_env()
    device = resolve_device(cfg.device)
    commit = _stamp()
    regime = cfg.split.regimes[0]
    assert regime.val is not None and regime.train_end is not None
    data = load_dataset(cfg.dataset, paths.data_dir)
    cache_dir = paths.cache_dir(cfg.dataset.name, cfg.dataset.version)
    # Causal by construction: the row for a unit at batch d sees only edges at or before d (PR-F2).
    gfp = build_features(data, cfg.features, cache_dir) if cfg.needs_gfp else None
    feats = select_features(data, gfp, cfg.model.features)
    split = build_split(data, regime, cache_dir)
    model, fit_meta = _deployed_model(cfg, data, feats, split, device, cache_dir, commit)
    threshold = fit_meta["threshold"]

    first, last = cfg.reference or regime.val
    ref_batches = np.arange(first, last + 1)
    if first <= regime.train_end:
        log.warning(
            "reference batches %d..%d reach into training batches (<= %d): the model scored "
            "those in-sample, so the score-based detectors compare against fitted scores",
            first,
            last,
            regime.train_end,
        )
    # Every unit in the reference window and in this batch, labelled or not: the health
    # check sees what the deployed model sees, and no label reaches it (PR-R2).
    idx = np.flatnonzero(np.isin(data.batch_id, ref_batches) | (data.batch_id == cfg.batch))
    batch = data.batch_id[idx]
    today = batch == cfg.batch
    if not today.any():
        raise ValueError(f"batch {cfg.batch} has no {data.task}s in {data.meta.dataset} to score")
    proba = model.predict_proba(data, feats, idx)
    h = cfg.health
    health = score_batches(
        feats.values[idx],
        proba,
        batch,
        ref_batches,
        detectors=h.detectors,
        bins=h.bins,
        psi_flag=h.psi_flag,
        ks_alpha=h.ks_alpha,
        ks_frac_flag=h.ks_frac,
        conf_flag=h.conf_flag,
        alert_flag=h.alert_flag,
        threshold=threshold,
        calibrate=h.calibrate,
    )

    units, scores = idx[today], proba[today]
    order = np.argsort(-scores, kind="stable")
    order = order[scores[order] >= threshold]
    alerted = units[order]
    alerts = pd.DataFrame(
        {"rank": np.arange(1, alerted.size + 1), "unit": alerted, "score": scores[order]}
    )
    if data.task == "edge":
        alerts["src_account"] = data.node_ids[data.src[alerted]]
        alerts["dst_account"] = data.node_ids[data.dst[alerted]]
    else:
        alerts["node_id"] = data.node_ids[alerted]
    if alerted.size:
        # The features that pushed each alert toward illicit, largest first.
        contrib = model.contributions(feats, alerted)
        top = np.argsort(-contrib, axis=1)[:, :3]
        alerts["top_features"] = [
            "; ".join(f"{feats.columns[j]} {contrib[i, j]:+.2f}" for j in row)
            for i, row in enumerate(top)
        ]

    flagged = health.loc[health["flagged"], "detector"].tolist()
    tables_dir = paths.tables_dir()
    tables_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.name}_batch{cfg.batch}"
    alerts_path = tables_dir / f"{stem}_alerts.csv"
    health_path = tables_dir / f"{stem}_health.json"
    alerts.to_csv(alerts_path, index=False)
    report = {
        "batch": cfg.batch,
        "status": "drift_flagged" if flagged else "no_drift_flagged",
        "flagged": flagged,
        "detectors": [
            {"detector": r.detector, "score": r.score, "flag_at": r.threshold}
            for r in health.itertuples(index=False)
        ],
        "reference_batches": [int(ref_batches[0]), int(ref_batches[-1])],
        "units_scored": int(today.sum()),
        "alerts": int(alerted.size),
        "threshold": threshold,
        "commit": commit,
        "model_fit": fit_meta,
        # A clean check is evidence, not an all-clear; every health file says so (D5).
        "note": "no flag is not an all-clear: on Elliptic's t43 shutdown F1 fell to 0.02 "
        "while no detector held a flag for two batches (docs/methods.md §12.4-12.5)",
    }
    health_path.write_text(json.dumps(report, indent=2) + "\n")
    page_path = write_score_report(tables_dir / f"{stem}_report.html", report, alerts)
    log.info(
        "batch %d: %d %ss scored, %d alerts at threshold %.4f, health %s",
        cfg.batch,
        report["units_scored"],
        data.task,
        report["alerts"],
        threshold,
        report["status"],
    )
    return alerts_path, health_path, page_path, flagged


def _smoke_score(cfg: RunConfig) -> tuple[Path, Path, Path]:
    """Score the first test batch with configs/amlworld_score.yaml's health settings (D5)."""
    regime = cfg.split.regimes[0]
    assert regime.val is not None and regime.test is not None
    model = next(m for m in cfg.models if m.name == "xgb" and m.features == "base_gfp")
    score_cfg = ScoreConfig(
        name=cfg.mlflow.experiment,
        dataset=cfg.dataset,
        features=cfg.features,
        split=cfg.split,
        model=model,
        health=DetectorConfig(calibrate=True),
        reference=(0, regime.val[1]),
        batch=regime.test[0],
        device=cfg.device,
    )
    alerts, health, page, _ = run_score(score_cfg)
    return alerts, health, page


def run_smoke(keep: bool = False) -> tuple[Path, Path, Path, Path]:
    """Benchmark then one scored batch on a synthetic graph: results, queue, health, page (D5)."""
    if not SMOKE_CONFIG.is_file():
        raise FileNotFoundError(
            f"smoke config not found at {SMOKE_CONFIG}; `mulegraph smoke` runs from a checkout "
            "of the repository, where configs/smoke.yaml sits beside the package"
        )
    cfg = load_config(SMOKE_CONFIG)
    # Graph cache and run store are throwaway; the results table lands in the usual place.
    tmp = Path(tempfile.mkdtemp(prefix="mulegraph-smoke-"))
    overrides = {
        "MULEGRAPH_DATA_DIR": str(tmp / "data"),
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp / 'mlflow.db'}",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        table = run_benchmark(cfg, SMOKE_CONFIG)
        return (table, *_smoke_score(cfg))
    finally:
        # Restore the environment: an in-process caller after this must not inherit
        # the throwaway data directory and run store.
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if keep:
            log.info("smoke run directory kept at %s", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
