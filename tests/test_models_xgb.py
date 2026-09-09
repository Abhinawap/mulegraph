"""XGBoost model tests (PR-M1, PR-M2, PR-M5).

The feature matrix and split are built here rather than imported from the feature
and split modules: this stream must be testable before those land, and a model
test that depends on them would be testing three components at once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from optuna.trial import FixedTrial

from mulegraph.eval.metrics import compute_metrics
from mulegraph.models.base import FitInfo
from mulegraph.models.xgb import XGBModel
from mulegraph.types import FeatureMatrix, GraphDataset, Split

# Trial 0's 1000-tree ceiling is a wall-clock budget question, not a correctness
# one; tests cap it so the suite stays fast.
FAST_PARAMS = {"n_estimators": 60, "early_stopping_rounds": 10, "max_depth": 4}


def make_feats(data: GraphDataset) -> FeatureMatrix:
    """The dataset's own node features, in node order."""
    return FeatureMatrix(
        values=data.x,
        columns=list(data.meta.feature_names),
        time=data.node_time,
        feature_version="test",
        name="base",
    )


def make_split(data: GraphDataset) -> Split:
    """A 60/20/20 positional split over the labelled nodes."""
    idx = data.labelled_idx
    n = idx.size
    train, val, test = idx[: int(0.6 * n)], idx[int(0.6 * n) : int(0.8 * n)], idx[int(0.8 * n) :]
    for name, part in (("train", train), ("val", val), ("test", test)):
        assert (data.y[part] == 1).sum() > 0, f"fixture gives {name} no positives"
    return Split(regime="random", train=train, val=val, test=test, split_hash="test")


Fitted = tuple[XGBModel, GraphDataset, FeatureMatrix, Split, FitInfo]


@pytest.fixture
def fitted(synthetic_ds: GraphDataset) -> Fitted:
    feats = make_feats(synthetic_ds)
    split = make_split(synthetic_ds)
    model = XGBModel(params=FAST_PARAMS, device="cpu")
    info = model.fit(synthetic_ds, feats, split, seed=0)
    return model, synthetic_ds, feats, split, info


def test_fit_reports_a_real_measurement(fitted: Fitted) -> None:
    _, _, _, _, info = fitted
    assert isinstance(info, FitInfo)
    assert info.seconds > 0  # feeds the wall-clock budget arithmetic (D2)
    assert info.best_iteration is not None and info.best_iteration >= 0
    assert info.val_pr_auc is not None and 0.0 <= info.val_pr_auc <= 1.0
    # scale_pos_weight is computed on train only, so it must exceed 1 on an
    # imbalanced fixture and be reported for audit.
    assert info.extra["scale_pos_weight"] > 1.0


def test_predict_proba_shape_dtype_and_range(fitted: Fitted) -> None:
    model, data, feats, split, _ = fitted
    proba = model.predict_proba(data, feats, split.test)
    assert proba.shape == (split.test.size,)
    assert proba.dtype == np.float32
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_planted_signal_is_learnable(fitted: Fitted) -> None:
    """A model that cannot beat the base rate here would make every later gap meaningless."""
    model, data, feats, split, _ = fitted
    proba = model.predict_proba(data, feats, split.test)
    y_test = data.y[split.test]
    positive_rate = float((y_test == 1).mean())
    metrics = compute_metrics(y_test, proba, threshold=0.5)
    assert metrics["pr_auc"] > positive_rate * 2.0
    assert metrics["roc_auc"] > 0.75


def test_embed_is_none(fitted: Fitted) -> None:
    model, data, feats, split, _ = fitted
    assert model.embed(data, feats, split.test) is None


def test_save_writes_a_loadable_model(fitted: Fitted, tmp_path: Path) -> None:
    model, data, feats, split, _ = fitted
    path = model.save(tmp_path / "artifact")
    assert path.is_file() and path.name == "model.json" and path.stat().st_size > 0

    from xgboost import XGBClassifier

    reloaded = XGBClassifier()
    reloaded.load_model(path)
    np.testing.assert_allclose(
        reloaded.predict_proba(feats.values[split.test])[:, 1],
        model.predict_proba(data, feats, split.test),
        rtol=1e-5,
        atol=1e-6,
    )


def test_same_seed_is_deterministic(synthetic_ds: GraphDataset) -> None:
    feats, split = make_feats(synthetic_ds), make_split(synthetic_ds)
    runs = []
    for _ in range(2):
        model = XGBModel(params=FAST_PARAMS, device="cpu")
        model.fit(synthetic_ds, feats, split, seed=3)
        runs.append(model.predict_proba(synthetic_ds, feats, split.test))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_trial0_is_the_fixed_reference_config() -> None:
    params, source = XGBModel.trial0()
    assert source == "xgboost_defaults"
    # Only the early-stopping ceiling is set; nothing here is a tuned value (D2).
    assert params == {"n_estimators": 1000, "early_stopping_rounds": 50}


def test_search_space_works_with_a_fixed_trial() -> None:
    chosen = {
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
    }
    assert XGBModel.search_space(FixedTrial(chosen)) == chosen


def test_search_space_matches_the_spec_bounds() -> None:
    """Spec §2.5: depth 3-10, lr log-uniform 1e-2..0.3, subsample/colsample 0.5..1, mcw 1-10."""
    trial = FixedTrial(
        {
            "max_depth": 3,
            "learning_rate": 1e-2,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "min_child_weight": 1,
        }
    )
    XGBModel.search_space(trial)
    dists = trial.distributions
    assert (dists["max_depth"].low, dists["max_depth"].high) == (3, 10)
    assert (dists["min_child_weight"].low, dists["min_child_weight"].high) == (1, 10)
    assert dists["learning_rate"].log is True
    assert (dists["learning_rate"].low, dists["learning_rate"].high) == (1e-2, 0.3)
    for name in ("subsample", "colsample_bytree"):
        assert (dists[name].low, dists[name].high) == (0.5, 1.0)


def test_fit_rejects_an_unlabelled_node_in_train(synthetic_ds: GraphDataset) -> None:
    """check_train_labelled must fire before a -1 label is learned as a negative."""
    feats, split = make_feats(synthetic_ds), make_split(synthetic_ds)
    unlabelled = int(np.flatnonzero(synthetic_ds.y < 0)[0])
    leaky = Split(
        regime="random",
        train=np.union1d(split.train, np.array([unlabelled], dtype=np.int64)).astype(np.int64),
        val=split.val,
        test=split.test,
        split_hash="leaky",
    )
    model = XGBModel(params=FAST_PARAMS, device="cpu")
    with pytest.raises(ValueError, match="unlabelled nodes"):
        model.fit(synthetic_ds, feats, leaky, seed=0)


def test_predict_before_fit_is_an_error(synthetic_ds: GraphDataset, tmp_path: Path) -> None:
    feats, split = make_feats(synthetic_ds), make_split(synthetic_ds)
    model = XGBModel(params=FAST_PARAMS, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict_proba(synthetic_ds, feats, split.test)
    with pytest.raises(RuntimeError, match="before fit"):
        model.save(tmp_path / "artifact")


def test_registry_constructor_signature_matches(synthetic_ds: GraphDataset) -> None:
    """models.get_model calls XGBModel(params=..., device=...); keep that exact shape."""
    from mulegraph.models import get_model

    model = get_model("xgb", params=FAST_PARAMS, device="cpu")
    assert isinstance(model, XGBModel)
    assert model.name == "xgb"
