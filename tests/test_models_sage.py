"""GraphSAGE tests (PR-M3, PR-M5).

Everything runs on the 600-node synthetic graph, on CPU, with a small model, so
the file is a fast guard rather than a benchmark. The split and feature matrix
are built here from ``GraphDataset`` alone: this module must not depend on the
feature or split subsystems, so a break in either cannot masquerade as a model
failure.

The load-bearing test is ``test_unlabelled_nodes_never_enter_the_loss``: an
unlabelled node reaching ``BCEWithLogitsLoss`` would be trained as a confident
negative and nothing downstream would notice.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mulegraph.config import SamplerConfig
from mulegraph.models import get_model
from mulegraph.models.base import BaseModel
from mulegraph.models.sage import SAGEModel
from mulegraph.types import FeatureMatrix, GraphDataset, Split

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

HAS_SAMPLER = importlib.util.find_spec("pyg_lib") is not None or (
    importlib.util.find_spec("torch_sparse") is not None
)

#: Small enough that a full fit is seconds, large enough to learn the planted signal.
TEST_PARAMS = {"epochs": 15, "patience": 5, "hidden": 16, "lr": 0.01}


def make_feats(data: GraphDataset) -> FeatureMatrix:
    """A ``base``-like matrix: the local block only, in node order."""
    start, stop = data.meta.feature_blocks["local"]
    return FeatureMatrix(
        values=np.ascontiguousarray(data.x[:, start:stop]),
        columns=data.meta.feature_names[start:stop],
        time=data.node_time.copy(),
        feature_version="test",
        name="base",
    )


def make_split(data: GraphDataset, seed: int = 0) -> Split:
    """A 60/20/20 random partition of the labelled nodes."""
    idx = data.labelled_idx
    order = np.random.default_rng(seed).permutation(idx.size)
    n_train, n_val = int(0.6 * idx.size), int(0.2 * idx.size)
    parts = np.split(idx[order], [n_train, n_train + n_val])
    return Split(
        regime="random",
        train=np.sort(parts[0]),
        val=np.sort(parts[1]),
        test=np.sort(parts[2]),
        split_hash="test",
    )


def fit_model(data: GraphDataset, sampler_kind: str = "full_batch", seed: int = 0, **overrides):
    feats, split = make_feats(data), make_split(data)
    model = SAGEModel(
        params={**TEST_PARAMS, **overrides},
        device="cpu",
        sampler=SamplerConfig(kind=sampler_kind, batch_size=64),
    )
    info = model.fit(data, feats, split, seed=seed)
    return model, feats, split, info


def test_registry_builds_a_sage_model() -> None:
    model = get_model("sage", params={"hidden": 16}, device="cpu", sampler=SamplerConfig())
    assert isinstance(model, SAGEModel)
    assert isinstance(model, BaseModel)  # the shared fit/predict interface (PR-M5)
    assert model.name == "sage"
    assert model.params["hidden"] == 16
    assert model.params["layers"] == 2  # trial-0 defaults fill the rest


def test_fit_reports_timing_and_validation_ap(synthetic_ds: GraphDataset) -> None:
    _, _, _, info = fit_model(synthetic_ds)
    assert info.seconds > 0
    assert info.val_pr_auc is not None and np.isfinite(info.val_pr_auc)
    assert 0.0 <= info.val_pr_auc <= 1.0
    assert 1 <= info.best_iteration <= TEST_PARAMS["epochs"]
    assert info.extra["sampler"] == "full_batch"
    assert info.extra["undirected"] is True
    assert info.extra["pos_weight"] > 1.0  # illicit is the minority class


def test_predict_proba_shape_dtype_and_range(synthetic_ds: GraphDataset) -> None:
    model, feats, split, _ = fit_model(synthetic_ds)
    proba = model.predict_proba(synthetic_ds, feats, split.test)
    assert proba.shape == (split.test.size,)
    assert proba.dtype == np.float32
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert np.isfinite(proba).all()


def test_embed_returns_hidden_representation(synthetic_ds: GraphDataset) -> None:
    model, feats, split, _ = fit_model(synthetic_ds)
    emb = model.embed(synthetic_ds, feats, split.test)
    assert emb.shape == (split.test.size, TEST_PARAMS["hidden"])
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()


def test_predict_before_fit_raises(synthetic_ds: GraphDataset) -> None:
    model = SAGEModel(params=TEST_PARAMS, device="cpu")
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(synthetic_ds, make_feats(synthetic_ds), np.array([0], dtype=np.int64))


@pytest.mark.skipif(not HAS_SAMPLER, reason="neighbour sampling needs pyg-lib or torch-sparse")
def test_neighbor_sampling_fits(synthetic_ds: GraphDataset) -> None:
    model, feats, split, info = fit_model(synthetic_ds, sampler_kind="neighbor")
    assert info.extra["sampler"] == "neighbor"
    assert info.extra["fanout"] == [15, 10]
    proba = model.predict_proba(synthetic_ds, feats, split.test)
    assert proba.shape == (split.test.size,) and np.isfinite(proba).all()


def test_neighbor_without_sampler_backend_refuses(
    synthetic_ds: GraphDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing pyg-lib must name the fix, not silently fall back to full batch."""
    import torch_geometric.typing as pyg_typing

    monkeypatch.setattr(pyg_typing, "WITH_PYG_LIB", False)
    monkeypatch.setattr(pyg_typing, "WITH_TORCH_SPARSE", False)
    with pytest.raises(RuntimeError, match="full_batch"):
        fit_model(synthetic_ds, sampler_kind="neighbor")


def test_planted_signal_is_learnable(synthetic_ds: GraphDataset) -> None:
    """Test PR-AUC must clearly beat the positive base rate.

    The synthetic generator shifts local features on illicit nodes, so a model
    that has learned nothing scores at the base rate; a wide margin here is what
    makes the other tests' numbers meaningful.
    """
    from sklearn.metrics import average_precision_score

    model, feats, split, _ = fit_model(synthetic_ds, epochs=60, patience=20)
    proba = model.predict_proba(synthetic_ds, feats, split.test)
    y_test = synthetic_ds.y[split.test]
    base_rate = float((y_test == 1).mean())
    ap = float(average_precision_score(y_test, proba))
    assert ap > base_rate + 0.2, f"PR-AUC {ap:.3f} vs base rate {base_rate:.3f}"


def test_unlabelled_nodes_never_enter_the_loss(synthetic_ds: GraphDataset) -> None:
    split = make_split(synthetic_ds)
    unlabelled = int(np.flatnonzero(synthetic_ds.y < 0)[0])
    poisoned = Split(
        regime="random",
        train=np.sort(np.append(split.train, unlabelled)),
        val=split.val,
        test=split.test,
        split_hash="test",
    )
    model = SAGEModel(params=TEST_PARAMS, device="cpu", sampler=SamplerConfig(kind="full_batch"))
    with pytest.raises(ValueError, match="unlabelled"):
        model.fit(synthetic_ds, make_feats(synthetic_ds), poisoned, seed=0)


def test_seed_controls_the_fit(synthetic_ds: GraphDataset) -> None:
    """Same seed reproduces on CPU; a different seed moves the model."""
    model_a, feats, split, info_a = fit_model(synthetic_ds, seed=7)
    model_b, _, _, info_b = fit_model(synthetic_ds, seed=7)
    model_c, _, _, _ = fit_model(synthetic_ds, seed=8)

    proba_a = model_a.predict_proba(synthetic_ds, feats, split.test)
    proba_b = model_b.predict_proba(synthetic_ds, feats, split.test)
    proba_c = model_c.predict_proba(synthetic_ds, feats, split.test)
    np.testing.assert_allclose(proba_a, proba_b, rtol=0, atol=1e-6)
    assert info_a.val_pr_auc == pytest.approx(info_b.val_pr_auc, abs=1e-9)
    assert not np.allclose(proba_a, proba_c, atol=1e-6)


def test_trial0_is_the_logged_reference() -> None:
    params, source = SAGEModel.trial0()
    assert source == "sage_default_2x64"
    assert params["hidden"] == 64 and params["layers"] == 2
    params["hidden"] = 999  # trial0 must hand back a copy, not the module constant
    assert SAGEModel.trial0()[0]["hidden"] == 64


def test_search_space_samples_with_a_fixed_trial() -> None:
    optuna = pytest.importorskip("optuna")
    trial = optuna.trial.FixedTrial({"hidden": 128, "dropout": 0.3, "lr": 3e-4, "layers": 3})
    space = SAGEModel.search_space(trial)
    assert space == {"hidden": 128, "dropout": 0.3, "lr": 3e-4, "layers": 3}
    model = SAGEModel(params={**TEST_PARAMS, **space}, device="cpu")
    assert model.params["layers"] == 3


def test_three_layer_model_fits(synthetic_ds: GraphDataset) -> None:
    model, feats, split, info = fit_model(synthetic_ds, epochs=5, patience=5, layers=3)
    emb = model.embed(synthetic_ds, feats, split.test)
    assert emb.shape == (split.test.size, TEST_PARAMS["hidden"])
    assert info.val_pr_auc is not None


def test_save_writes_a_loadable_state_dict(synthetic_ds: GraphDataset, tmp_path) -> None:
    import torch

    model, _, _, _ = fit_model(synthetic_ds, epochs=3, patience=3)
    target = model.save(tmp_path / "sage")
    assert target.is_file()
    state = torch.load(target, map_location="cpu", weights_only=True)
    assert "head.weight" in state
    assert state["head.weight"].shape == (1, TEST_PARAMS["hidden"])
