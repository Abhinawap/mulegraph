"""XGBoost challenger on an already-selected feature matrix (PR-M1, PR-M2, PR-M5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from xgboost import DMatrix, XGBClassifier

from mulegraph.models.base import FitInfo, check_train_labelled
from mulegraph.types import FeatureMatrix, GraphDataset, Split
from mulegraph.util import Timer, log

#: -1 = every core; overridable via ``ModelConfig.params`` on a shared machine (D2 fairness).
DEFAULT_N_JOBS = -1


class XGBModel:
    """Gradient-boosted trees over a node-aligned feature matrix."""

    def __init__(self, params: dict[str, Any], device: str = "cpu") -> None:
        self.name = "xgb"
        # D2: every run starts from the reference config; config params override it.
        self.params = {**self.trial0()[0], **params}
        self.device = device
        self.clf: XGBClassifier | None = None

    @classmethod
    def trial0(cls) -> tuple[dict[str, Any], str]:
        """Library defaults (D2); ``n_estimators`` is a ceiling under early stopping."""
        return {"n_estimators": 1000, "early_stopping_rounds": 50}, "xgboost_defaults"

    def fit(self, data: GraphDataset, feats: FeatureMatrix, split: Split, seed: int) -> FitInfo:
        """Fit on ``split.train``, early-stopping on validation PR-AUC."""
        check_train_labelled(data, split)
        x = feats.values
        if x.shape[0] != data.num_units:
            raise ValueError(
                f"feature matrix has {x.shape[0]} rows but the graph has {data.num_units} "
                f"{data.task}s; FeatureMatrix rows are in unit order"
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

        # Computed on TRAIN ONLY: the val and test ratios are properties of the future.
        scale_pos_weight = n_neg / n_pos

        kwargs: dict[str, Any] = {
            "tree_method": "hist",
            "n_jobs": DEFAULT_N_JOBS,
            **self.params,
            # Fixed by the protocol, so they win over any override.
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "device": self.device,
            "random_state": seed,
        }
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
            extra={
                "scale_pos_weight": scale_pos_weight,
                "n_features": int(x.shape[1]),
                "trial0_source": self.trial0()[1],
                "params": dict(self.params),
            },
        )

    def predict_proba(
        self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray
    ) -> np.ndarray:
        """P(illicit) for ``idx`` as float32 in [0, 1]."""
        if self.clf is None:
            raise RuntimeError("XGBModel.predict_proba called before fit")
        return self.clf.predict_proba(feats.values[idx])[:, 1].astype(np.float32)

    def contributions(self, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray:
        """Per-feature log-odds contributions (TreeSHAP) for ``idx``, bias column dropped (D5)."""
        if self.clf is None:
            raise RuntimeError("XGBModel.contributions called before fit")
        # Only the trees predict_proba uses: early stopping leaves 50 more in the booster.
        best = getattr(self.clf, "best_iteration", None)
        trees = (0, 0) if best is None else (0, best + 1)
        matrix = DMatrix(feats.values[idx])
        return self.clf.get_booster().predict(matrix, pred_contribs=True, iteration_range=trees)[
            :, :-1
        ]

    def save(self, path: Path) -> None:
        """Write the fitted booster so a later ``score`` reuses it (D5)."""
        if self.clf is None:
            raise RuntimeError("XGBModel.save called before fit")
        self.clf.save_model(path)

    @classmethod
    def load(cls, path: Path) -> XGBModel:
        """Read a booster written by ``save``."""
        model = cls({})
        model.clf = XGBClassifier()
        model.clf.load_model(path)
        return model
