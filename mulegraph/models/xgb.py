"""XGBoost challenger (PR-M1, PR-M2, PR-M5).

The tabular baseline the whole project is a test of: the question is whether a
graph *model* beats this once graph *features* are available to both. It is
therefore given every advantage a practitioner would give it — class weighting
from the training ratio, early stopping on validation PR-AUC — and none that a
practitioner could not: nothing here sees ``split.test``.

The feature matrix arrives already selected (``base`` / ``base_gfp`` /
``raw165``); this module never decides which columns a row means, so PR-M7's
"base excludes pre-aggregated neighbour features" is enforced once, upstream,
rather than re-argued per model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from xgboost import XGBClassifier

from mulegraph.models.base import FitInfo, check_train_labelled
from mulegraph.types import FeatureMatrix, GraphDataset, Split
from mulegraph.util import Timer, log

if TYPE_CHECKING:  # pragma: no cover - typing only
    import optuna

#: -1 means "every core". Overridable through ``ModelConfig.params`` on a shared
#: machine, where the wall-clock budget W (D2) would otherwise be unfair.
DEFAULT_N_JOBS = -1


class XGBModel:
    """Gradient-boosted trees over a node-aligned feature matrix."""

    def __init__(self, params: dict[str, Any], device: str = "cpu") -> None:
        self.name = "xgb"
        self.params = dict(params)
        self.device = device
        self.clf: XGBClassifier | None = None

    # ----------------------------------------------------------------- #
    # Reference configuration and search space
    # ----------------------------------------------------------------- #

    @classmethod
    def trial0(cls) -> tuple[dict[str, Any], str]:
        """The fixed reference configuration (D2): xgboost's own defaults.

        Only two keys are set, and neither is a tuned value. ``n_estimators`` is a
        *ceiling* under ``early_stopping_rounds`` — boosting stops when validation
        PR-AUC stops improving, so 1000 buys headroom rather than 1000 trees.
        Everything else (depth, learning rate, subsampling) is deliberately left
        at the library default: trial 0 must be a configuration a reader can
        reproduce without knowing anything about this dataset, otherwise "trial 0
        is a fixed reference config" quietly becomes "trial 0 is our tuning".

        ``scale_pos_weight`` is not listed here because it is not free: it is
        computed from the training class ratio in ``fit``.
        """
        return {"n_estimators": 1000, "early_stopping_rounds": 50}, "xgboost_defaults"

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        """Sample the spec's XGBoost search space (§2.5).

        Unused in the MVP — the search phase is v1a (D2, PR-M6) — but defined now
        so v1a adds a search loop instead of reworking the model.
        """
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

    # ----------------------------------------------------------------- #
    # Fit / predict
    # ----------------------------------------------------------------- #

    def fit(self, data: GraphDataset, feats: FeatureMatrix, split: Split, seed: int) -> FitInfo:
        """Fit on ``split.train``, early-stopping on ``split.val``."""
        check_train_labelled(data, split)
        x = feats.values
        if x.shape[0] != data.num_nodes:
            raise ValueError(
                f"feature matrix has {x.shape[0]} rows but the graph has {data.num_nodes} nodes; "
                "FeatureMatrix rows are in node order"
            )
        y = data.y
        y_train = y[split.train]
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"training split has {n_pos} illicit and {n_neg} licit nodes; "
                "both classes are required to fit and to weight"
            )
        if int((y[split.val] == 1).sum()) == 0:
            raise ValueError(
                "validation split contains no illicit nodes, so validation PR-AUC is "
                "undefined and early stopping would be meaningless"
            )

        # Weight the positive class by the training imbalance. Computed on TRAIN
        # ONLY: the val and test ratios are properties of the future.
        scale_pos_weight = n_neg / n_pos

        kwargs: dict[str, Any] = {
            "tree_method": "hist",
            "n_jobs": DEFAULT_N_JOBS,
            **self.params,
            # Fixed by the evaluation protocol, so they win over any override:
            # early stopping is on validation PR-AUC (§2.5), the class weight comes
            # from the data, and the seed comes from the seed loop.
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "device": self.device,
            "random_state": seed,
        }
        # xgboost >= 2 takes early_stopping_rounds on the constructor, not on fit,
        # and selects the accelerator with device= rather than a gpu_* tree_method.
        clf = XGBClassifier(**kwargs)

        with Timer() as timer:
            clf.fit(
                x[split.train],
                y_train,
                eval_set=[(x[split.val], y[split.val])],
                verbose=False,
            )
        self.clf = clf

        best_iteration = getattr(clf, "best_iteration", None)
        val_pr_auc = getattr(clf, "best_score", None)
        log.info(
            "xgb fit: %d train (%d illicit, spw=%.1f) in %.1fs, best_iteration=%s, val aucpr=%s",
            split.train.size,
            n_pos,
            scale_pos_weight,
            timer.seconds,
            best_iteration,
            None if val_pr_auc is None else f"{val_pr_auc:.4f}",
        )
        return FitInfo(
            seconds=timer.seconds,
            best_iteration=None if best_iteration is None else int(best_iteration),
            val_pr_auc=None if val_pr_auc is None else float(val_pr_auc),
            extra={"scale_pos_weight": scale_pos_weight, "n_features": int(x.shape[1])},
        )

    def predict_proba(
        self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray
    ) -> np.ndarray:
        """P(illicit) for ``idx``, as float32 in [0, 1]."""
        if self.clf is None:
            raise RuntimeError("XGBModel.predict_proba called before fit")
        return self.clf.predict_proba(feats.values[idx])[:, 1].astype(np.float32)

    def embed(self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray | None:
        """None: a tree ensemble has no representation space to drift in (v2, PR-R1)."""
        return None

    def save(self, path: Path) -> Path:
        """Write the booster as JSON, which is version-portable unlike a pickle."""
        if self.clf is None:
            raise RuntimeError("XGBModel.save called before fit")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "model.json"
        self.clf.save_model(target)
        return target
