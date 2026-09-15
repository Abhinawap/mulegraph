"""Split construction: labelled nodes only, leakage assertions on every build (PR-E1, D1)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from mulegraph.config import RegimeConfig
from mulegraph.types import LABEL_ILLICIT, GraphDataset, Split
from mulegraph.util import hash_arrays, hash_dict, write_json

log = logging.getLogger("mulegraph")

TEMPORAL_REGIMES = ("temporal", "temporal_inductive")


class RegimeNotSupportedError(ValueError):
    """A regime that is undefined on this dataset, not merely unimplemented."""


def split_hash(train: np.ndarray, val: np.ndarray, test: np.ndarray, params: dict[str, Any]) -> str:
    """Content hash of a partition plus the definition that produced it (PR-O1)."""
    return hash_arrays({"train": train, "val": val, "test": test}, extra=params)


def _params(data: GraphDataset, cfg: RegimeConfig) -> dict[str, Any]:
    """The regime-defining fields; ``raw_sha256`` separates revisions sharing a version string."""
    params: dict[str, Any] = {
        "dataset": data.meta.dataset,
        "version": data.meta.version,
        "regime": cfg.regime,
        "raw_sha256": data.meta.raw_sha256,
    }
    if cfg.regime == "random":
        params |= {"fractions": list(cfg.fractions), "seed": cfg.seed}
    else:
        params |= {
            "train_end": cfg.train_end,
            "val": list(cfg.val) if cfg.val else None,
            "test": list(cfg.test) if cfg.test else None,
        }
    return params


def _random_split(data: GraphDataset, cfg: RegimeConfig) -> tuple[np.ndarray, ...]:
    """Stratified partition of the labelled nodes at the configured fractions."""
    idx = data.labelled_idx
    y = data.y[idx]
    train_frac, val_frac, test_frac = cfg.fractions

    train, rest = train_test_split(
        idx, train_size=train_frac, stratify=y, random_state=cfg.seed, shuffle=True
    )
    # Second cut is relative to what is left, so the three shares match the config.
    val_share = val_frac / (val_frac + test_frac)
    val, test = train_test_split(
        rest, train_size=val_share, stratify=data.y[rest], random_state=cfg.seed, shuffle=True
    )
    return train, val, test


def _temporal_split(data: GraphDataset, cfg: RegimeConfig) -> tuple[np.ndarray, ...]:
    """Chronological partition; both val and test bounds are inclusive."""
    assert cfg.train_end is not None and cfg.val is not None and cfg.test is not None
    idx = data.labelled_idx
    time = data.node_time[idx]
    train = idx[time <= cfg.train_end]
    val = idx[(time >= cfg.val[0]) & (time <= cfg.val[1])]
    test = idx[(time >= cfg.test[0]) & (time <= cfg.test[1])]
    return train, val, test


def check_leakage(
    data: GraphDataset, regime: str, train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> None:
    """Raise on overlap, empty/unlabelled/positive-free parts, joining edges, or time disorder."""
    parts = {"train": train, "val": val, "test": test}

    for (a, left), (b, right) in (
        (("train", train), ("val", val)),
        (("train", train), ("test", test)),
        (("val", val), ("test", test)),
    ):
        overlap = np.intersect1d(left, right)
        if overlap.size:
            raise ValueError(
                f"{regime}: {a} and {b} share {overlap.size} nodes, e.g. {overlap[:5].tolist()}"
            )

    for name, part in parts.items():
        if part.size == 0:
            raise ValueError(f"{regime}: {name} split is empty")
        unlabelled = part[data.y[part] < 0]
        if unlabelled.size:
            raise ValueError(
                f"{regime}: {name} contains {unlabelled.size} unlabelled nodes, "
                f"e.g. {unlabelled[:5].tolist()}"
            )
        n_pos = int((data.y[part] == LABEL_ILLICIT).sum())
        if n_pos == 0:
            raise ValueError(
                f"{regime}: {name} contains no positives out of {part.size} nodes; "
                "PR-AUC is undefined and threshold selection is meaningless"
            )

    if regime in TEMPORAL_REGIMES and data.meta.cross_time_edges:
        # PR-E1: no test id in any training neighbourhood. Implied by time order when
        # timestep components are disconnected, so only checked where an edge can join them.
        in_train = np.zeros(data.num_nodes, dtype=bool)
        in_test = np.zeros(data.num_nodes, dtype=bool)
        in_train[train] = True
        in_test[test] = True
        joins = (in_train[data.src] & in_test[data.dst]) | (in_test[data.src] & in_train[data.dst])
        n_joins = int(joins.sum())
        if n_joins:
            raise ValueError(
                f"{regime}: {n_joins} edges join a training node to a test node, so test "
                "nodes appear in training neighbourhoods and message passing would leak "
                "them into the fitted model (PR-E1)"
            )

    if regime in TEMPORAL_REGIMES:
        for earlier, later in (("train", "val"), ("val", "test")):
            hi = int(data.node_time[parts[earlier]].max())
            lo = int(data.node_time[parts[later]].min())
            if hi >= lo:
                raise ValueError(
                    f"{regime}: max(node_time[{earlier}]) = {hi} is not < "
                    f"min(node_time[{later}]) = {lo}; the split is not chronological"
                )


def definition_hash(params: dict[str, Any]) -> str:
    """Cache key over the split *definition*, so a perturbed partition is caught (PR-D4)."""
    return hash_dict(params)


def _cache_paths(cache_dir: Path, digest: str) -> dict[str, Path]:
    root = cache_dir / "splits" / digest
    return {name: root / f"{name}.npy" for name in ("train", "val", "test")} | {
        "params": root / "params.json"
    }


def build_split(data: GraphDataset, cfg: RegimeConfig, cache_dir: Path) -> Split:
    """Build, leak-check and cache one regime's partition; a cache hit checks determinism."""
    if cfg.regime == "temporal_inductive":
        if data.meta.cross_time_edges is False:
            raise RegimeNotSupportedError(
                f"temporal_inductive is rejected on {data.meta.dataset}: meta.cross_time_edges "
                "is False, so its temporal split is already inductive by construction and a "
                "separate inductive regime is undefined (D1)."
            )
        raise NotImplementedError("temporal_inductive is v1b (AMLworld)")

    if cfg.regime == "random":
        train, val, test = _random_split(data, cfg)
    elif cfg.regime == "temporal":
        train, val, test = _temporal_split(data, cfg)
    else:
        raise RegimeNotSupportedError(f"Unknown regime {cfg.regime!r}")

    # Split.__post_init__ requires sorted, duplicate-free int64 indices.
    train, val, test = (np.sort(p).astype(np.int64) for p in (train, val, test))
    check_leakage(data, cfg.regime, train, val, test)

    params = _params(data, cfg)
    digest = split_hash(train, val, test, params)
    _sync_cache(cache_dir, definition_hash(params), digest, params, train, val, test)

    log.info(
        "%s split %s: train %d, val %d, test %d labelled nodes",
        cfg.regime,
        digest,
        train.size,
        val.size,
        test.size,
    )
    return Split(
        regime=cfg.regime, train=train, val=val, test=test, split_hash=digest, params=params
    )


def _sync_cache(
    cache_dir: Path,
    key: str,
    digest: str,
    params: dict[str, Any],
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> None:
    paths = _cache_paths(cache_dir, key)
    fresh = {"train": train, "val": val, "test": test}
    if all(paths[name].is_file() for name in fresh):
        for name, array in fresh.items():
            cached = np.load(paths[name])
            if not np.array_equal(cached, array):
                raise ValueError(
                    f"cached {name} split at {paths[name]} has {cached.size} indices and differs "
                    f"from the freshly built {array.size}; the split builder is not deterministic "
                    "for this definition (PR-D4) — delete the cache only if you know why"
                )
        return
    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    for name, array in fresh.items():
        np.save(paths[name], array)
    write_json(
        paths["params"],
        params
        | {
            "split_hash": digest,
            "definition_hash": key,
            "sizes": {k: int(v.size) for k, v in fresh.items()},
        },
    )
