"""Orchestrator — the only module that imports across subsystems (PR-E4, PR-O1)."""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mulegraph.config import DriftRunConfig, ModelConfig, RunConfig, load_config
from mulegraph.data import load_dataset
from mulegraph.drift.monitor import lead_time, score_batches
from mulegraph.eval.curves import CURVE_METRICS, per_timestep
from mulegraph.eval.metrics import EXTRA_KEYS, compute_metrics
from mulegraph.eval.threshold import choose_threshold
from mulegraph.features.builder import build_features
from mulegraph.features.select import select_features
from mulegraph.models import get_model
from mulegraph.models.base import BaseModel
from mulegraph.report.figures import plot_curves, plot_drift
from mulegraph.report.tables import write_results_table
from mulegraph.splits.builder import build_split
from mulegraph.types import FeatureMatrix, GraphDataset, Predictions, Split
from mulegraph.util import Paths, git_commit, git_dirty, resolve_device, seed_all

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
    """One fitted (regime, model, seed) with its validation threshold and scored predictions."""

    model: BaseModel
    threshold: float
    val: Predictions
    test: Predictions
    curve: pd.DataFrame  # per-timestep metrics on test (PR-E5)


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


def _fit_one(
    cfg: RunConfig,
    data: GraphDataset,
    feats: FeatureMatrix,
    split: Split,
    model_cfg: ModelConfig,
    seed: int,
    device: str,
    commit: str,
) -> Fitted:
    """Fit one (regime, model, seed), threshold on validation, score test, log a child run."""
    import mlflow

    seed_all(seed)
    model = get_model(
        model_cfg.name,
        model_cfg.params,
        device=device,
        sampler=model_cfg.sampler,
        task=data.task,
    )
    info = model.fit(data, feats, split, seed)

    # PR-E4: the threshold comes from validation scores; test is scored with it, never searched.
    val = _predict(model, data, feats, split.val)
    threshold, val_f1 = choose_threshold(val.y, val.proba)
    test = _predict(model, data, feats, split.test)
    scored = compute_metrics(test.y, test.proba, threshold, cfg.eval.metrics, prefix="test_")
    # Counts and point precision/recall are diagnostics: the results table is metrics.test_* only.
    diagnostics = {f"diag_{key}": scored.pop(f"test_{key}") for key in EXTRA_KEYS}
    curve = per_timestep(test, threshold)

    with mlflow.start_run(run_name=f"{split.regime}.{model_cfg.key}.s{seed}", nested=True):
        mlflow.set_tags(
            {
                "kind": "child",
                "dataset": data.meta.dataset,
                # Spec §2.3 makes this a tag, not a parent param: nested runs inherit
                # nothing, and the reporter only ever reads child runs.
                "dataset_version": data.meta.version,
                "regime": split.regime,
                "model": model_cfg.name,
                "features": model_cfg.features,
                "feature_version": feats.feature_version,
                "split_hash": split.split_hash,
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
                # val_f1 is a selection diagnostic, never a headline metric (PR-E6).
                "threshold": threshold,
                "val_f1": val_f1,
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
        _log_predictions(val, test)
    return Fitted(model, threshold, val, test, curve)


def _prepare(cfg: RunConfig) -> tuple[Paths, str, str, str]:
    """Point MLflow at the store and stamp the run: ``(paths, tracking_uri, device, commit)``."""
    import mlflow

    paths = Paths.from_env()
    tracking_uri = cfg.mlflow.resolved_uri()
    _ensure_local_store(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    device = resolve_device(cfg.device)
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
    return paths, tracking_uri, device, commit


def run_benchmark(cfg: RunConfig, config_path: Path) -> Path:
    """Load, build features, split, fit across seeds, evaluate, write the table."""
    import mlflow

    paths, tracking_uri, device, commit = _prepare(cfg)
    data = load_dataset(cfg.dataset, paths.data_dir)
    cache_dir = paths.cache_dir(cfg.dataset.name, cfg.dataset.version)

    gfp = build_features(data, cfg.features, cache_dir) if cfg.needs_gfp else None
    matrices = {m.features: select_features(data, gfp, m.features) for m in cfg.models}
    # Built before the first fit so an undefined regime fails before hours of work (D1).
    splits = {r.regime: build_split(data, r, cache_dir) for r in cfg.split.regimes}

    total = cfg.total_fits()
    log.info(
        "%s: %d fits (%d models x %d regimes x %d seeds) on %s, tracking to %s",
        cfg.mlflow.experiment,
        total,
        len(cfg.models),
        len(cfg.split.regimes),
        len(cfg.seeds),
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
            split = splits[regime_cfg.regime]
            for model_cfg in cfg.models:
                for seed in cfg.seeds:
                    done += 1
                    log.info(
                        "fit %d/%d: %s %s seed %d",
                        done,
                        total,
                        regime_cfg.regime,
                        model_cfg.key,
                        seed,
                    )
                    fitted = _fit_one(
                        cfg,
                        data,
                        matrices[model_cfg.features],
                        split,
                        model_cfg,
                        seed,
                        device,
                        commit,
                    )
                    curves.append(
                        fitted.curve.assign(
                            regime=split.regime,
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
    scores, leads, curves = [], [], []
    with mlflow.start_run(run_name=f"{cfg.mlflow.experiment}.{commit}"):
        mlflow.set_tags({"kind": "parent", "dataset": data.meta.dataset, "git_commit": commit})
        mlflow.log_params({"dataset_version": data.meta.version, "seeds": list(cfg.seeds)})
        mlflow.log_artifact(str(config_path))
        for model_cfg in cfg.models:
            feats = matrices[model_cfg.features]
            for seed in cfg.seeds:
                log.info("fit %s seed %d", model_cfg.key, seed)
                fitted = _fit_one(cfg, data, feats, split, model_cfg, seed, device, commit)
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
                    calibrate=cfg.drift.calibrate,
                )
                ref_f1 = float(per_timestep(fitted.val, fitted.threshold)["f1"].mean())
                lead = lead_time(fitted.curve, table, ref_f1, cfg.drift.f1_drop, cfg.drift.drop_run)
                tag = {"model": model_cfg.name, "features": model_cfg.features, "seed": seed}
                scores.append(table.assign(**tag))
                leads.append(lead.assign(ref_f1=ref_f1, **tag))
                curves.append(fitted.curve.assign(**tag))

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
        for artifact in (score_path, lead_path, figure):
            mlflow.log_artifact(str(artifact))
    log.info("wrote %s and %s", score_path, lead_path)
    return lead_path


def run_smoke(keep: bool = False) -> Path:
    """End-to-end run on a synthetic graph, about 10 s."""
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
        return run_benchmark(cfg, SMOKE_CONFIG)
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
