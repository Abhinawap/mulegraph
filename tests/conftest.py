"""Shared fixtures.

Tests never touch the user's real data directory or MLflow store: ``tmp_data_dir``
redirects both through the environment.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from mulegraph.config import FeaturesConfig
from mulegraph.data.synthetic import make_synthetic_elliptic
from mulegraph.types import GraphDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def synthetic_ds() -> GraphDataset:
    """Small Elliptic-shaped graph with no cross-timestep edges (as Elliptic is)."""
    return make_synthetic_elliptic(n_nodes=600, n_timesteps=12, seed=0)


@pytest.fixture
def synthetic_cross_time_ds() -> GraphDataset:
    """Graph whose edges span timesteps — the only setting where causality is testable."""
    return make_synthetic_elliptic(
        n_nodes=600, n_timesteps=12, cross_time_edges=True, cross_time_fraction=0.4, seed=0
    )


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    """Cheap feature config: coarse bins and a short cycle bound keep tests quick."""
    return FeaturesConfig(bins=[2, 4, 8], cycle_len=6, num_threads=2)


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point data, report and MLflow roots at a temporary directory."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MULEGRAPH_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MULEGRAPH_REPORT_DIR", str(tmp_path / "report"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    yield data_dir


@pytest.fixture
def elliptic_raw_dir() -> Path:
    """The real Elliptic++ raw directory, skipping the test when it is absent."""
    root = Path(os.environ.get("MULEGRAPH_DATA_DIR", "./data"))
    raw = root / "raw" / "elliptic_pp" / "2023.1"
    if not (raw / "txs_features.csv").is_file():
        pytest.skip(f"Elliptic++ raw files not present at {raw}; see README")
    return raw
