"""A synthetic Elliptic-shaped graph.

Two jobs, both of which the real dataset cannot do:

* **Smoke test.** CI cannot download Elliptic++, so ``mulegraph smoke`` runs the
  whole pipeline on a graph generated here.
* **Feature causality (PR-F2).** On Elliptic the causality property holds
  trivially — timesteps are disconnected components, so there is nothing for a
  feature to peek at. Proving the feature builder is causal needs a graph whose
  edges genuinely span timesteps, which ``cross_time_edges=True`` produces.

The planted signal is deliberately learnable: illicit nodes get both a shifted
mean on some local features and a fan-in star, so a smoke run exercises the
local-feature and graph-feature paths without either being degenerate.
"""

from __future__ import annotations

import numpy as np

from mulegraph.types import (
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    DatasetMeta,
    GraphDataset,
)

#: Mirrors Elliptic++ so ``base``/``raw165`` selection behaves identically here.
N_LOCAL = 93
N_AGG1HOP = 72

#: Local features whose mean is shifted for illicit nodes.
SIGNAL_COLUMNS = 10
SIGNAL_SHIFT = 1.2

#: In-edges planted on an illicit node's fan-in star.
STAR_MIN_SOURCES = 3
STAR_MAX_SOURCES = 6


def make_synthetic_elliptic(
    n_nodes: int = 2000,
    n_timesteps: int = 49,
    n_features: int = N_LOCAL + N_AGG1HOP,
    illicit_rate: float = 0.10,
    unknown_rate: float = 0.5,
    edges_per_node: float = 1.2,
    cross_time_edges: bool = False,
    cross_time_fraction: float = 0.3,
    max_time_gap: int = 3,
    seed: int = 0,
    version: str = "synthetic",
) -> GraphDataset:
    """Build a deterministic Elliptic-shaped ``GraphDataset``.

    Args:
        n_nodes: Total nodes, spread evenly over ``n_timesteps``.
        illicit_rate: Fraction of *labelled* nodes that are illicit.
        unknown_rate: Fraction of all nodes left unlabelled (``y == -1``). These
            stay in the graph for message passing but never enter a loss or metric.
        cross_time_edges: When True, ``cross_time_fraction`` of edges run forward
            in time by up to ``max_time_gap`` timesteps, and ``meta.cross_time_edges``
            is set accordingly. An edge's time is that of its *later* endpoint, so
            a node never has an incident edge predating its own appearance.
        max_time_gap: Largest forward jump for a cross-time edge.

    Returns:
        A ``GraphDataset`` with ``feature_blocks`` matching Elliptic's
        ``{"local": (0, 93), "agg1hop": (93, 165)}``.
    """
    if n_features < N_LOCAL:
        raise ValueError(f"n_features must be at least {N_LOCAL}, got {n_features}")
    rng = np.random.default_rng(seed)

    # --- nodes: evenly spread over timesteps 1..T (Elliptic is 1-indexed) ---
    node_time = np.sort(rng.integers(1, n_timesteps + 1, size=n_nodes)).astype(np.int64)

    # --- labels ---
    y = np.full(n_nodes, LABEL_UNKNOWN, dtype=np.int64)
    labelled = rng.random(n_nodes) >= unknown_rate
    n_labelled = int(labelled.sum())
    label_draw = rng.random(n_labelled) < illicit_rate
    y[labelled] = np.where(label_draw, LABEL_ILLICIT, LABEL_LICIT)
    illicit = np.flatnonzero(y == LABEL_ILLICIT)

    # --- features: standard normal, with a shifted mean on illicit nodes ---
    x = rng.standard_normal((n_nodes, n_features)).astype(np.float32)
    x[illicit, :SIGNAL_COLUMNS] += SIGNAL_SHIFT

    # --- background edges, within a timestep ---
    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    for t in range(1, n_timesteps + 1):
        members = np.flatnonzero(node_time == t)
        if members.size < 2:
            continue
        n_edges = int(round(members.size * edges_per_node))
        if n_edges == 0:
            continue
        src_list.append(rng.choice(members, size=n_edges))
        dst_list.append(rng.choice(members, size=n_edges))

    # --- planted fan-in stars: several senders converge on each illicit node ---
    for node in illicit:
        t = node_time[node]
        peers = np.flatnonzero(node_time == t)
        peers = peers[peers != node]
        if peers.size == 0:
            continue
        k = int(rng.integers(STAR_MIN_SOURCES, STAR_MAX_SOURCES + 1))
        senders = rng.choice(peers, size=min(k, peers.size), replace=False)
        src_list.append(senders)
        dst_list.append(np.full(senders.size, node))

    src = np.concatenate(src_list).astype(np.int64) if src_list else np.zeros(0, dtype=np.int64)
    dst = np.concatenate(dst_list).astype(np.int64) if dst_list else np.zeros(0, dtype=np.int64)

    # --- optionally redirect some edges forward in time ---
    if cross_time_edges and src.size:
        n_cross = int(round(src.size * cross_time_fraction))
        chosen = rng.choice(src.size, size=n_cross, replace=False)
        for i in chosen:
            gap = int(rng.integers(1, max_time_gap + 1))
            target_t = node_time[src[i]] + gap
            if target_t > n_timesteps:
                continue
            candidates = np.flatnonzero(node_time == target_t)
            if candidates.size:
                dst[i] = rng.choice(candidates)

    # Drop self-loops, which carry no pattern information and confuse fan counts.
    keep = src != dst
    src, dst = src[keep], dst[keep]
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]

    # An edge exists once both endpoints do: its time is the later endpoint's.
    edge_time = np.maximum(node_time[src], node_time[dst]).astype(np.int64)
    order = np.argsort(edge_time, kind="stable")
    src, dst, edge_time = src[order], dst[order], edge_time[order]

    observed_cross = bool(np.any(node_time[src] != node_time[dst]))
    label_counts = {
        "illicit": int((y == LABEL_ILLICIT).sum()),
        "licit": int((y == LABEL_LICIT).sum()),
        "unknown": int((y == LABEL_UNKNOWN).sum()),
    }
    feature_names = [f"Local_feature_{i + 1}" for i in range(N_LOCAL)] + [
        f"Aggregate_feature_{i + 1}" for i in range(n_features - N_LOCAL)
    ]
    meta = DatasetMeta(
        dataset="synthetic_elliptic",
        version=f"{version}-{seed}",
        cross_time_edges=observed_cross,
        source_url="generated by mulegraph.data.synthetic",
        feature_blocks={"local": (0, N_LOCAL), "agg1hop": (N_LOCAL, n_features)},
        feature_names=feature_names,
        dropped_columns=[],
        num_timesteps=n_timesteps,
        label_counts=label_counts,
        raw_sha256="",
        task="node",
    )
    return GraphDataset(
        x=x,
        edge_index=np.stack([src, dst]).astype(np.int64),
        edge_attr=None,
        node_time=node_time,
        edge_time=edge_time,
        batch_id=node_time.copy(),
        y=y,
        node_ids=np.arange(n_nodes, dtype=np.int64),
        task="node",
        meta=meta,
    )
