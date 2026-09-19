"""XGBoost model tests (PR-M1, PR-M2, PR-M5).

The feature matrix and split are built here rather than imported from the feature
and split modules: this stream must be testable before those land, and a model
test that depends on them would be testing three components at once.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_predict_before_fit_is_an_error(synthetic_ds: GraphDataset) -> None:
    feats, split = make_feats(synthetic_ds), make_split(synthetic_ds)
    model = XGBModel(params=FAST_PARAMS, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict_proba(synthetic_ds, feats, split.test)


def test_registry_constructor_signature_matches(synthetic_ds: GraphDataset) -> None:
    """models.get_model calls XGBModel(params=..., device=...); keep that exact shape."""
    from mulegraph.models import get_model

    model = get_model("xgb", params=FAST_PARAMS, device="cpu")
    assert isinstance(model, XGBModel)
    assert model.name == "xgb"


def test_trial0_is_applied_under_config_overrides() -> None:
    """D2: trial 0 is the base every run starts from; config params win over it."""
    from mulegraph.models import get_model

    assert get_model("xgb", params={}).params["early_stopping_rounds"] == 50
    model = get_model("xgb", params={"n_estimators": 20})
    assert model.params == {"n_estimators": 20, "early_stopping_rounds": 50}
