"""Build, version and cache causal node-level graph features (PR-F1, PR-F2, PR-F3).

The cache key ``feature_version`` hashes the full *definition*, including the drive pattern.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging
from pathlib import Path

import numpy as np

from mulegraph.config import FeaturesConfig
from mulegraph.features.aggregate import empty_node_features, node_agg_v1, node_columns
from mulegraph.features.gfp import (
    DUMMY_COLUMN,
    RAW_WIDTH,
    GfpDriver,
    GfpLayout,
    make_gfp_params,
    probe_layout,
)
from mulegraph.types import DatasetMeta, FeatureMatrix, GraphDataset
from mulegraph.util import Timer, hash_dict, write_json

log = logging.getLogger("mulegraph")

#: The causal drive pattern; in the hash so changing it invalidates every cache.
DRIVE = "transform_only"

#: lc-cycle cost is superlinear in density, so a slow timestep means revisit ``cycle_len``.
SLOW_TIMESTEP_SECONDS = 60.0

COLUMN_PREFIX = "gfp_"


def snapml_version() -> str:
    try:
        return importlib_metadata.version("snapml")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - env without snapml
        return "unknown"


def feature_definition(
    cfg: FeaturesConfig, meta: DatasetMeta, layout: GfpLayout
) -> dict[str, object]:
    """The complete, hashable description of what the feature columns mean."""
    return {
        "backend": "gfp",  # constant kept so existing feature versions still hash the same (PR-F3)
        "snapml_version": snapml_version(),
        "families": list(cfg.families),
        "bins": list(cfg.bins),
        "window": cfg.window,
        "cycle_len": cfg.cycle_len,
        "vertex_stats": cfg.vertex_stats,
        "vertex_stats_feats": list(make_gfp_params(cfg)["vertex_stats_feats"]),  # type: ignore[arg-type]
        "aggregation": cfg.aggregation,
        "drive": DRIVE,
        "edge_columns": list(layout.columns),
        "node_columns": node_columns(layout),
        "dataset": meta.dataset,
        "dataset_version": meta.version,
        # Rows are edges on an edge task, aggregated nodes otherwise: different matrices.
        "unit": meta.task,
        # Same version string over different raw files must not share a cache (NFR-1).
        "raw_sha256": meta.raw_sha256,
    }


def feature_version(cfg: FeaturesConfig, meta: DatasetMeta, layout: GfpLayout) -> str:
    """sha256 prefix over the feature definition, logged with every run (PR-F3)."""
    return hash_dict(feature_definition(cfg, meta, layout))


def _batch(edge_ids: np.ndarray, src: np.ndarray, dst: np.ndarray, t: int) -> np.ndarray:
    """One timestep's edges as snapml input rows; ``edge_ids`` must be globally unique."""
    batch = np.empty((edge_ids.size, RAW_WIDTH), dtype=np.float64)
    batch[:, 0] = edge_ids
    batch[:, 1] = src
    batch[:, 2] = dst
    batch[:, 3] = t
    batch[:, DUMMY_COLUMN] = 1.0
    return batch


def _read_cache(path: Path, columns: list[str], time: np.ndarray) -> FeatureMatrix | None:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = list(table.column_names)
    if names[:2] != ["id", "time"]:
        raise ValueError(f"feature cache {path} does not start with id, time: {names[:2]}")
    if names[2:] != columns:
        raise ValueError(
            f"feature cache {path} holds columns {names[2:]} but this feature version "
            f"defines {columns}; delete the cache or fix the definition"
        )
    values = np.column_stack(
        [table.column(name).to_numpy(zero_copy_only=False) for name in columns]
    ).astype(np.float32)
    cached_time = table.column("time").to_numpy(zero_copy_only=False).astype(np.int64)
    if not np.array_equal(cached_time, time):
        raise ValueError(f"feature cache {path} was built for different unit timestamps")
    return FeatureMatrix(
        values=values,
        columns=columns,
        time=time,
        feature_version=path.stem,
        name="gfp",
        blocks={"gfp": (0, len(columns))},
    )


def _write_cache(
    path: Path,
    values: np.ndarray,
    ids: np.ndarray,
    time: np.ndarray,
    columns: list[str],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrays = [pa.array(ids), pa.array(time)]
    arrays.extend(pa.array(values[:, i]) for i in range(values.shape[1]))
    table = pa.Table.from_arrays(arrays, names=["id", "time", *columns])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def build_features(
    data: GraphDataset,
    cfg: FeaturesConfig,
    cache_dir: Path,
    *,
    force: bool = False,
) -> FeatureMatrix:
    """Causal graph features in unit order (per edge on an edge task), cached under
    ``<cache_dir>/features/``."""
    layout = probe_layout(cfg)
    fv = feature_version(cfg, data.meta, layout)
    edge_task = data.task == "edge"
    raw_columns = list(layout.columns) if edge_task else node_columns(layout)
    columns = [COLUMN_PREFIX + name for name in raw_columns]
    ids = np.arange(data.num_edges, dtype=np.int64) if edge_task else data.node_ids
    parquet_path = Path(cache_dir) / "features" / f"{fv}.parquet"
    json_path = parquet_path.with_suffix(".json")

    if parquet_path.is_file() and not force:
        log.info("features %s: cache hit at %s", fv, parquet_path)
        cached = _read_cache(parquet_path, columns, data.unit_time)
        if cached is not None:
            return cached

    times = np.unique(data.edge_time)
    log.info(
        "features %s: computing %d columns for %d %ss over %d timesteps (%d edges)",
        fv,
        len(columns),
        data.num_units,
        data.task,
        times.size,
        data.num_edges,
    )
    driver = GfpDriver(make_gfp_params(cfg), layout)
    values = (
        np.zeros((data.num_edges, layout.width), dtype=np.float32)
        if edge_task
        else empty_node_features(data.num_nodes, layout)
    )
    src_all, dst_all = data.src, data.dst
    per_timestep: dict[str, float] = {}

    with Timer() as total:
        for t in times.tolist():
            mask = np.flatnonzero(data.edge_time == t)
            with Timer() as step:
                edge_feats = driver.step(_batch(mask, src_all[mask], dst_all[mask], t))
                if edge_task:
                    values[mask] = edge_feats
                else:
                    node_agg_v1(
                        values, edge_feats, src_all[mask], dst_all[mask], data.node_time, t, layout
                    )
            per_timestep[str(t)] = round(step.seconds, 3)
            log.info("gfp t=%d edges=%d secs=%.2f", t, mask.size, step.seconds)
            if step.seconds > SLOW_TIMESTEP_SECONDS:
                log.warning(
                    "gfp t=%d took %.1fs for %d edges; lc-cycle cost is superlinear in density, "
                    "so consider lowering features.cycle_len before it eats the search budget",
                    t,
                    step.seconds,
                    mask.size,
                )

    log.info("features %s: done in %.1fs", fv, total.seconds)
    _write_cache(parquet_path, values, ids, data.unit_time, columns)
    write_json(
        json_path,
        {
            "feature_version": fv,
            "definition": feature_definition(cfg, data.meta, layout),
            "edge_columns": list(layout.columns),
            "node_columns": node_columns(layout),
            "matrix_columns": columns,
            "snapml_version": snapml_version(),
            "seconds_per_timestep": per_timestep,
            "seconds_total": round(total.seconds, 3),
        },
    )
    return FeatureMatrix(
        values=values,
        columns=columns,
        time=data.unit_time,
        feature_version=fv,
        name="gfp",
        blocks={"gfp": (0, len(columns))},
    )
