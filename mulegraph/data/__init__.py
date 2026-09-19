"""Deterministic dataset loading, cached by ``(dataset, version)`` (PR-D4)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from mulegraph.config import DatasetConfig
from mulegraph.types import DatasetMeta, GraphDataset

log = logging.getLogger("mulegraph")

CACHE_FILENAME = "graph.pt"


def _load_elliptic(cfg: DatasetConfig, raw_dir: Path) -> GraphDataset:
    # Lazy, so the synthetic path never pays for pyarrow.
    from mulegraph.data.elliptic import load_elliptic_raw

    return load_elliptic_raw(raw_dir, cfg.version)


def _load_synthetic(cfg: DatasetConfig, raw_dir: Path) -> GraphDataset:
    from mulegraph.data.synthetic import make_synthetic_elliptic

    params = (cfg.synthetic.model_dump() if cfg.synthetic is not None else {}) | {
        "version": cfg.version
    }
    return make_synthetic_elliptic(**params)


def _load_amlworld(cfg: DatasetConfig, raw_dir: Path) -> GraphDataset:
    from mulegraph.data.amlworld import load_amlworld_raw

    return load_amlworld_raw(raw_dir, cfg.version, max_days=cfg.max_days)


LOADERS: dict[str, Callable[[DatasetConfig, Path], GraphDataset]] = {
    "elliptic_pp": _load_elliptic,
    "synthetic_elliptic": _load_synthetic,
    "amlworld": _load_amlworld,
}


def _save_cache(path: Path, data: GraphDataset) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(data)
    payload["meta"] = asdict(data.meta)
    torch.save(payload, path)


def _load_cache(path: Path) -> GraphDataset:
    import torch

    payload = torch.load(path, weights_only=False)
    meta = DatasetMeta(**payload.pop("meta"))
    # torch.save round-trips tuples as lists; feature_blocks must stay tuples.
    object.__setattr__(
        meta, "feature_blocks", {k: tuple(v) for k, v in meta.feature_blocks.items()}
    )
    return GraphDataset(meta=meta, **payload)


def load_dataset(cfg: DatasetConfig, data_dir: Path) -> GraphDataset:
    """Load a dataset from ``data_dir/cache``, or build it from ``data_dir/raw`` and cache it."""
    # A truncated AMLworld load (max_days) must not be served as the full graph (NFR-1).
    from mulegraph.data.amlworld import cache_version

    version = cache_version(cfg.version, cfg.max_days)
    cache_path = data_dir / "cache" / cfg.name / version / CACHE_FILENAME
    if cache_path.is_file():
        log.info("loading cached graph from %s", cache_path)
        return _load_cache(cache_path)

    raw_dir = data_dir / "raw" / cfg.name / cfg.version
    data = LOADERS[cfg.name](cfg, raw_dir)
    log.info(
        "loaded %s %s: %d nodes, %d edges, %d features, %d timesteps, labels %s",
        data.meta.dataset,
        data.meta.version,
        data.num_nodes,
        data.num_edges,
        data.unit_features.shape[1],
        data.meta.num_timesteps,
        data.meta.label_counts,
    )
    _save_cache(cache_path, data)
    log.info("cached graph to %s", cache_path)
    return data
