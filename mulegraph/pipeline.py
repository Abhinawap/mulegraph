"""Orchestrator — the only module that imports across subsystems (PR-E4, PR-O1)."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from mulegraph.config import ModelConfig, RunConfig, load_config
from mulegraph.data import load_dataset
from mulegraph.eval.metrics import EXTRA_KEYS, compute_metrics
from mulegraph.eval.threshold import choose_threshold
from mulegraph.features.builder import build_features
from mulegraph.features.select import select_features
from mulegraph.models import get_model
from mulegraph.report.tables import write_results_table
from mulegraph.splits.builder import build_split
from mulegraph.types import FeatureMatrix, GraphDataset, Split
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


def _fit_one(
    cfg: RunConfig,
    data: GraphDataset,
    feats: FeatureMatrix,
    split: Split,
    model_cfg: ModelConfig,
    seed: int,
    device: str,
    commit: str,
) -> None:
    """Fit one (regime, model, seed), threshold on validation, score test, log a child run."""
    import mlflow

    seed_all(seed)
    model = get_model(model_cfg.name, model_cfg.params, device=device, sampler=model_cfg.sampler)
    info = model.fit(data, feats, split, seed)

    # PR-E4: the threshold comes from validation scores; test is scored with it, never searched.
    p_val = model.predict_proba(data, feats, split.val)
    threshold, val_f1 = choose_threshold(data.y[split.val], p_val)
    scored = compute_metrics(
        data.y[split.test],
        model.predict_proba(data, feats, split.test),
        threshold,
        cfg.eval.metrics,
        prefix="test_",
    )
    # Counts and point precision/recall are diagnostics: the results table is metrics.test_* only.
    diagnostics = {f"diag_{key}": scored.pop(f"test_{key}") for key in EXTRA_KEYS}

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


def run_benchmark(cfg: RunConfig, config_path: Path) -> Path:
    """Load, build features, split, fit across seeds, evaluate, write the table."""
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
                    _fit_one(
                        cfg,
                        data,
                        matrices[model_cfg.features],
                        split,
                        model_cfg,
                        seed,
                        device,
                        commit,
                    )

    # Scoped to this run's children: an earlier run of the same config in the same
    # experiment must not be counted as extra seeds (PR-E3).
    return write_results_table(
        cfg.mlflow.experiment,
        paths.tables_dir(),
        tracking_uri=tracking_uri,
        parent_run_id=parent.info.run_id,
    )


def run_smoke(keep: bool = False) -> Path:
    """End-to-end run on a synthetic graph, under two minutes on CPU."""
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
