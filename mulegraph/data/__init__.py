"""Dataset loading and caching.

Loaders are deterministic and cache to disk (PR-D4). The cache is keyed by
``(dataset, version)``; anything derived from a *definition* — features, splits —
is keyed by a content hash of that definition instead, so a definition change can
never silently reuse a stale cache.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mulegraph.config import DatasetConfig
from mulegraph.types import DatasetMeta, GraphDataset

log = logging.getLogger("mulegraph")

CACHE_FILENAME = "graph.pt"


def _load_elliptic(cfg: DatasetConfig, raw_dir: Path) -> GraphDataset:
    # Imported lazily so the synthetic path never pays for pyarrow.
    from mulegraph.data.elliptic import load_elliptic_raw

    return load_elliptic_raw(raw_dir, cfg.version)


def _load_synthetic(cfg: DatasetConfig, raw_dir: Path) -> GraphDataset:
    from mulegraph.data.synthetic import make_synthetic_elliptic

    params = (cfg.synthetic.model_dump() if cfg.synthetic is not None else {}) | {
        "version": cfg.version
    }
    return make_synthetic_elliptic(**params)


LOADERS: dict[str, Callable[[DatasetConfig, Path], GraphDataset]] = {
    "elliptic_pp": _load_elliptic,
    "synthetic_elliptic": _load_synthetic,
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


def load_dataset(
    cfg: DatasetConfig, data_dir: Path, *, force: bool = False, use_cache: bool = True
) -> GraphDataset:
    """Load a dataset, from cache when available.

    Args:
        cfg: Which dataset and version.
        data_dir: Root holding ``raw/`` and ``cache/`` (``$MULEGRAPH_DATA_DIR``).
        force: Rebuild from raw even if a cache exists.
        use_cache: Set False to skip reading and writing the cache entirely.

    Raises:
        FileNotFoundError: If the raw files are absent; the message names the
            expected path, because there is no auto-download in the MVP.
    """
    if cfg.name not in LOADERS:
        raise ValueError(f"Unknown dataset {cfg.name!r}; known: {sorted(LOADERS)}")

    cache_path = data_dir / "cache" / cfg.name / cfg.version / CACHE_FILENAME
    if use_cache and not force and cache_path.is_file():
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
        data.x.shape[1],
        data.meta.num_timesteps,
        data.meta.label_counts,
    )
    if use_cache:
        _save_cache(cache_path, data)
        log.info("cached graph to %s", cache_path)
    return data


def subsample(data: GraphDataset, node_idx: np.ndarray) -> GraphDataset:
    """Induced subgraph on ``node_idx``, renumbering nodes to 0..k-1.

    Used to build small fixtures; edges are kept only when both endpoints survive.
    """
    node_idx = np.unique(np.asarray(node_idx, dtype=np.int64))
    remap = np.full(data.num_nodes, -1, dtype=np.int64)
    remap[node_idx] = np.arange(node_idx.size, dtype=np.int64)
    keep = (remap[data.src] >= 0) & (remap[data.dst] >= 0)
    return GraphDataset(
        x=data.x[node_idx],
        edge_index=np.stack([remap[data.src[keep]], remap[data.dst[keep]]]).astype(np.int64),
        edge_attr=None if data.edge_attr is None else data.edge_attr[keep],
        node_time=data.node_time[node_idx],
        edge_time=data.edge_time[keep],
        batch_id=data.batch_id[node_idx],
        y=data.y[node_idx],
        node_ids=data.node_ids[node_idx],
        task=data.task,
        meta=data.meta,
    )
