"""SAGE edge head: protocol shape, train-only loss, temporal sampling (PR-M5, PR-F2, PR-E1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch_geometric.loader as pyg_loader

from mulegraph.config import SamplerConfig
from mulegraph.models import get_model
from mulegraph.models.sage_edge import SAGEEdgeModel
from mulegraph.splits.builder import build_split
from mulegraph.types import FeatureMatrix, GraphDataset, Split
from tests.test_splits import TEMPORAL, make_edge_ds

PARAMS = {"epochs": 3, "patience": 2, "hidden": 8, "node_dim": 4}
SAMPLER = SamplerConfig(kind="neighbor", fanout=[5, 5], batch_size=64)


def make_feats(data: GraphDataset) -> FeatureMatrix:
    return FeatureMatrix(
        values=data.edge_attr,
        columns=list(data.meta.feature_names),
        time=data.edge_time,
        feature_version="test",
        name="base",
    )


@pytest.fixture(params=["cpu", "cuda"])
def fitted(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[SAGEEdgeModel, GraphDataset, FeatureMatrix, Split]:
    # CPU hides device bugs: batch.to("cpu") is a no-op (the Kaggle P100 run found one).
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    data = make_edge_ds()
    feats = make_feats(data)
    split = build_split(data, TEMPORAL, tmp_path)
    model = get_model("sage", PARAMS, device=request.param, sampler=SAMPLER, task="edge")
    assert isinstance(model, SAGEEdgeModel)
    model.fit(data, feats, split, seed=0)
    return model, data, feats, split


def test_predict_proba_shape_and_range(fitted) -> None:
    model, data, feats, split = fitted
    proba = model.predict_proba(data, feats, split.test)
    assert proba.shape == (split.test.size,) and proba.dtype == np.float32
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_loss_sees_train_edges_only_and_sampling_is_temporal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR-E4/PR-F2: the labelled loader covers split.train exactly; every loader is time-aware."""
    data = make_edge_ds()
    feats = make_feats(data)
    split = build_split(data, TEMPORAL, tmp_path)
    calls: list[dict[str, Any]] = []
    real = pyg_loader.LinkNeighborLoader

    def spy(graph, **kwargs):
        calls.append(kwargs)
        return real(graph, **kwargs)

    monkeypatch.setattr(pyg_loader, "LinkNeighborLoader", spy)
    SAGEEdgeModel(PARAMS, sampler=SAMPLER).fit(data, feats, split, seed=0)

    assert calls and all(c["time_attr"] == "edge_time" for c in calls)
    labelled = [c for c in calls if c["edge_label"] is not None]
    assert len(labelled) == 1
    seeds = labelled[0]["edge_label_index"].numpy()
    assert np.array_equal(seeds, data.edge_index[:, split.train])
    assert np.array_equal(labelled[0]["edge_label"].numpy(), data.y[split.train])


def test_full_batch_sampler_is_refused() -> None:
    with pytest.raises(ValueError, match="neighbor"):
        SAGEEdgeModel(sampler=SamplerConfig(kind="full_batch"))


def test_trial0_is_the_logged_reference() -> None:
    params, source = SAGEEdgeModel.trial0()
    assert source == "sage_edge_default_2x64" and params["layers"] == 2
