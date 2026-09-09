"""GraphSAGE node classifier (PR-M3).

The graph *model* half of the benchmark's central comparison: XGBoost on
hand-built graph features versus a GNN that learns its own aggregation. Keeping
this model behind the same ``fit``/``predict_proba`` protocol as XGBoost (PR-M5)
is what makes the two rows comparable rather than merely adjacent in a table.

Three decisions are worth stating because they are ours, not the spec's:

* **Features come from ``feats.values``, never ``data.x``.** Feature selection
  happens upstream, so ``base`` and ``base_gfp`` differ only in the matrix handed
  to the model. Reading ``data.x`` here would silently give the GNN the
  pre-aggregated neighbour block and destroy that comparison (PR-M7).
* **Unlabelled nodes stay in the graph but never enter the loss.** Message
  passing over the ``y == -1`` majority is the point of a GNN on Elliptic; a -1
  reaching ``BCEWithLogitsLoss`` would be learned as a confident negative.
* **Inference is always full-batch, even when training samples.** The graph is
  small (203k nodes), full-batch inference is exact, and a sampled estimate would
  put sampling noise into the reported metric and into the early-stopping signal.

The spec fixes only "two ``SAGEConv`` layers, 64 hidden, class-weighted BCE"
(spec 2.5). Everything else in :meth:`SAGEModel.trial0` — dropout, learning rate,
weight decay, epoch cap, patience, symmetrisation — is our choice with no
published source, so trial 0 is logged verbatim with every run and its name is
``sage_default_2x64`` rather than a citation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mulegraph.config import SamplerConfig
from mulegraph.models.base import FitInfo, check_train_labelled
from mulegraph.types import FeatureMatrix, GraphDataset, Split
from mulegraph.util import Timer, log, seed_all

if TYPE_CHECKING:  # pragma: no cover - typing only
    import optuna
    import torch

#: Our reference configuration. Only ``layers``/``hidden`` come from the spec.
TRIAL0: dict[str, Any] = {
    "hidden": 64,
    "layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 200,
    "patience": 20,
    "undirected": True,
}
TRIAL0_SOURCE = "sage_default_2x64"


@dataclass
class _Graph:
    """Torch views of one (dataset, feature matrix) pair, built once per fit.

    Rebuilding these every epoch would dominate the epoch cost and, on a 7 GB
    host, churn memory for no reason.
    """

    x: torch.Tensor  # float32 [N, K] on the model's device
    edge_index: torch.Tensor  # int64 [2, E'] on the model's device, symmetrised
    y: torch.Tensor  # float32 [N] on the model's device; -1 entries never used
    key: tuple[int, int, int, int]


def _build_net(in_dim: int, hidden: int, layers: int, dropout: float) -> torch.nn.Module:
    """``SAGEConv -> ReLU -> Dropout`` stacked ``layers`` deep, then a linear head."""
    import torch
    from torch_geometric.nn import SAGEConv

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [in_dim] + [hidden] * layers
            self.convs = torch.nn.ModuleList(SAGEConv(dims[i], dims[i + 1]) for i in range(layers))
            self.dropout = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(hidden, 1)

        def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            for i, conv in enumerate(self.convs):
                x = torch.relu(conv(x, edge_index))
                if i < len(self.convs) - 1:  # no dropout on the representation itself
                    x = self.dropout(x)
            return x

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            return self.head(self.embed(x, edge_index)).squeeze(-1)

    return Net()


class SAGEModel:
    """Two- or three-layer GraphSAGE with class-weighted BCE and early stopping."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        device: str = "cpu",
        sampler: SamplerConfig | None = None,
    ) -> None:
        self.name = "sage"
        self.params: dict[str, Any] = {**TRIAL0, **(params or {})}
        self.device = device
        self.sampler = sampler or SamplerConfig()
        if self.params["layers"] not in (2, 3):
            raise ValueError(f"sage layers must be 2 or 3, got {self.params['layers']}")
        self._net: torch.nn.Module | None = None
        self._graph: _Graph | None = None

    # ----------------------------------------------------------------- setup #

    def _prepare(self, data: GraphDataset, feats: FeatureMatrix) -> _Graph:
        """Convert one (dataset, features) pair to device tensors, cached by identity.

        The pipeline hands the same two objects to ``fit`` and then to every
        ``predict_proba`` call, so identity is a sound cache key here; a miss only
        costs a rebuild, never a wrong answer.
        """
        import torch

        if feats.values.shape[0] != data.num_nodes:
            raise ValueError(
                f"feature matrix has {feats.values.shape[0]} rows but the graph has "
                f"{data.num_nodes} nodes; features must be in node order"
            )
        key = (id(data), id(feats), data.num_nodes, feats.num_features)
        if self._graph is not None and self._graph.key == key:
            return self._graph

        edge_index = torch.from_numpy(np.ascontiguousarray(data.edge_index))
        if self.params["undirected"]:
            # Elliptic's transaction graph is directed, but node classification on
            # it is conventionally run on the symmetrised graph so a node sees its
            # payers as well as its payees. The dataset is untouched; the choice is
            # recorded in FitInfo.extra so it is auditable.
            from torch_geometric.utils import to_undirected

            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
        graph = _Graph(
            x=torch.from_numpy(np.ascontiguousarray(feats.values)).to(self.device),
            edge_index=edge_index.to(self.device),
            y=torch.from_numpy(data.y.astype(np.float32)).to(self.device),
            key=key,
        )
        self._graph = graph
        return graph

    def _fanout(self) -> list[int]:
        """One fan-out per message-passing layer, extending or truncating the config."""
        layers = int(self.params["layers"])
        fanout = list(self.sampler.fanout)
        if len(fanout) < layers:
            fanout = fanout + [fanout[-1]] * (layers - len(fanout))
            log.info("sampler.fanout extended to %s to match layers=%d", fanout, layers)
        elif len(fanout) > layers:
            fanout = fanout[:layers]
            log.info("sampler.fanout truncated to %s to match layers=%d", fanout, layers)
        return fanout

    def _make_loader(self, graph: _Graph, train_idx: np.ndarray, num_nodes: int) -> Any:
        import torch
        import torch_geometric.typing as pyg_typing
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader

        if not (pyg_typing.WITH_PYG_LIB or pyg_typing.WITH_TORCH_SPARSE):
            raise RuntimeError(
                "sampler.kind 'neighbor' needs pyg-lib or torch-sparse, neither of which is "
                "importable here; set sampler: {kind: full_batch} in the config (exact, and "
                "affordable on a graph this size) or install pyg-lib for your torch build"
            )
        # The loader samples on CPU and each batch is moved to the device; keeping
        # the sampling graph on CPU is also what lets an 8 GB card hold the model.
        cpu = Data(
            x=graph.x.cpu(),
            edge_index=graph.edge_index.cpu(),
            y=graph.y.cpu(),
            num_nodes=num_nodes,
        )
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(train_idx)] = True
        return NeighborLoader(
            cpu,
            num_neighbors=self._fanout(),
            input_nodes=mask,
            batch_size=self.sampler.batch_size,
            shuffle=True,
            num_workers=0,  # workers would fork a copy of the graph; host RAM is 7 GB
        )

    # ------------------------------------------------------------------- fit #

    def fit(self, data: GraphDataset, feats: FeatureMatrix, split: Split, seed: int) -> FitInfo:
        """Train on ``split.train``, early-stopping on validation PR-AUC.

        ``split.test`` is never touched: not for early stopping, not for
        normalisation, not for the threshold (PR-E4).
        """
        import torch
        from sklearn.metrics import average_precision_score

        check_train_labelled(data, split)
        # seed_all covers python/numpy/torch/cuda. CUDA scatter reductions inside
        # SAGEConv are not bitwise deterministic, so runs at the same seed can
        # differ slightly on GPU. That is left alone deliberately: the across-seed
        # t-interval (PR-E3) is what reports run-to-run variation, and forcing
        # torch.use_deterministic_algorithms would trade a large slowdown for a
        # number the interval already covers.
        seed_all(seed)

        graph = self._prepare(data, feats)
        train_idx = np.asarray(split.train, dtype=np.int64)
        val_idx = np.asarray(split.val, dtype=np.int64)
        y_train = data.y[train_idx]
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"split.train has {n_pos} illicit and {n_neg} licit nodes; "
                "class-weighted BCE needs both classes present"
            )
        pos_weight = n_neg / n_pos

        self._net = _build_net(
            feats.num_features,
            int(self.params["hidden"]),
            int(self.params["layers"]),
            float(self.params["dropout"]),
        ).to(self.device)
        opt = torch.optim.Adam(
            self._net.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
        )

        loader = None
        if self.sampler.kind == "neighbor":
            loader = self._make_loader(graph, train_idx, data.num_nodes)
        train_t = torch.from_numpy(train_idx).to(self.device)

        y_val = data.y[val_idx]
        epochs = int(self.params["epochs"])
        patience = int(self.params["patience"])
        best_ap, best_epoch, best_state, stale, epoch = -1.0, 0, None, 0, 0
        log.info(
            "sage fit: %d train / %d val nodes, sampler=%s, up to %d epochs on %s",
            train_idx.size,
            val_idx.size,
            self.sampler.kind,
            epochs,
            self.device,
        )

        with Timer() as timer:
            for epoch in range(1, epochs + 1):
                self._net.train()
                if loader is None:
                    opt.zero_grad()
                    logits = self._net(graph.x, graph.edge_index)[train_t]
                    loss = criterion(logits, graph.y[train_t])
                    loss.backward()
                    opt.step()
                else:
                    for batch in loader:
                        batch = batch.to(self.device)
                        opt.zero_grad()
                        # Only the seed nodes (the first batch_size rows) carry a
                        # loss: the sampled neighbourhood is context, and it also
                        # contains unlabelled and validation nodes.
                        n_seed = batch.batch_size
                        logits = self._net(batch.x, batch.edge_index)[:n_seed]
                        loss = criterion(logits, batch.y[:n_seed])
                        loss.backward()
                        opt.step()

                proba = self._infer(graph)[val_idx]
                ap = float(average_precision_score(y_val, proba))
                if ap > best_ap:
                    best_ap, best_epoch, stale = ap, epoch, 0
                    best_state = copy.deepcopy(
                        {k: v.detach().cpu() for k, v in self._net.state_dict().items()}
                    )
                else:
                    stale += 1
                    if stale >= patience:
                        log.info("sage early stop at epoch %d (best %d)", epoch, best_epoch)
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        log.info(
            "sage fit done in %.1fs: best val PR-AUC %.4f at epoch %d",
            timer.seconds,
            best_ap,
            best_epoch,
        )
        return FitInfo(
            seconds=timer.seconds,
            best_iteration=best_epoch,
            val_pr_auc=best_ap,
            extra={
                "sampler": self.sampler.kind,
                "fanout": self._fanout() if self.sampler.kind == "neighbor" else None,
                "batch_size": self.sampler.batch_size,
                "undirected": bool(self.params["undirected"]),
                "epochs_run": epoch,
                "pos_weight": pos_weight,
                "n_train_pos": n_pos,
                "n_train_neg": n_neg,
                "device": self.device,
                "trial0_source": TRIAL0_SOURCE,
                "params": dict(self.params),
            },
        )

    # ------------------------------------------------------------- inference #

    def _infer(self, graph: _Graph, embeddings: bool = False) -> np.ndarray:
        """Full-batch forward pass over the whole graph, in eval mode."""
        import torch

        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            if embeddings:
                out = self._net.embed(graph.x, graph.edge_index)
                return out.detach().cpu().numpy().astype(np.float32)
            logits = self._net(graph.x, graph.edge_index)
            return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    def _require_fitted(self) -> None:
        if self._net is None:
            raise RuntimeError("sage model is not fitted; call fit() before predict_proba/embed")

    def predict_proba(
        self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray
    ) -> np.ndarray:
        """P(illicit) for ``idx``, as float32 in [0, 1]."""
        self._require_fitted()
        graph = self._prepare(data, feats)
        proba = self._infer(graph)[np.asarray(idx, dtype=np.int64)]
        return np.clip(proba, 0.0, 1.0).astype(np.float32)

    def embed(self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray:
        """Hidden activations for ``idx``, taken before the linear head.

        These are what the v2 embedding-drift detector compares between batches,
        so they must be the representation, not the score.
        """
        self._require_fitted()
        graph = self._prepare(data, feats)
        return self._infer(graph, embeddings=True)[np.asarray(idx, dtype=np.int64)]

    # ------------------------------------------------------------ plumbing #

    def save(self, path: Path) -> Path:
        """Write the fitted weights under ``path``; returns the file written."""
        import torch

        self._require_fitted()
        assert self._net is not None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "model.pt"
        torch.save({k: v.detach().cpu() for k, v in self._net.state_dict().items()}, target)
        return target

    @classmethod
    def trial0(cls) -> tuple[dict[str, Any], str]:
        return dict(TRIAL0), TRIAL0_SOURCE

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        """Optuna space for v1a (D2, PR-M6); nothing in the MVP calls it."""
        return {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "layers": trial.suggest_categorical("layers", [2, 3]),
        }
