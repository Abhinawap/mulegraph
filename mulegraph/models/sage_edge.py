"""GraphSAGE edge classifier for account graphs (PR-M3, PR-M5).

Accounts carry no features, so nodes start from a learned embedding; a seed edge is scored from
``cat[h_src, h_dst, z(edge features)]``. Neighbourhoods are sampled with ``time_attr`` so an edge
at time t never sees an edge after t (PR-F2). Edge features are z-scored on train rows (PR-E1).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from mulegraph.config import SamplerConfig
from mulegraph.models.base import FitInfo, check_train_labelled
from mulegraph.types import FeatureMatrix, GraphDataset, Split
from mulegraph.util import Timer, log, seed_all

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

TRIAL0: dict[str, Any] = {
    "hidden": 64,
    "layers": 2,
    "node_dim": 32,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "epochs": 200,
    "patience": 20,
}
TRIAL0_SOURCE = "sage_edge_default_2x64"


def _build_net(
    num_nodes: int, node_dim: int, edge_dim: int, hidden: int, layers: int, dropout: float
) -> torch.nn.Module:
    """Embedding -> ``SAGEConv`` x ``layers`` -> MLP head over (src, dst, edge features)."""
    import torch
    from torch_geometric.nn import SAGEConv

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(num_nodes, node_dim)
            dims = [node_dim] + [hidden] * layers
            self.convs = torch.nn.ModuleList(SAGEConv(dims[i], dims[i + 1]) for i in range(layers))
            self.dropout = torch.nn.Dropout(dropout)
            self.mix = torch.nn.Linear(2 * hidden + edge_dim, hidden)
            self.out = torch.nn.Linear(hidden, 1)

        def forward(
            self,
            n_id: torch.Tensor,
            edge_index: torch.Tensor,
            seeds: torch.Tensor,
            edge_x: torch.Tensor,
        ) -> torch.Tensor:
            h = self.emb(n_id)
            for conv in self.convs:
                h = self.dropout(torch.relu(conv(h, edge_index)))
            mixed = torch.relu(self.mix(torch.cat([h[seeds[0]], h[seeds[1]], edge_x], dim=-1)))
            return self.out(mixed).squeeze(-1)

    return Net()


class SAGEEdgeModel:
    """GraphSAGE over accounts with an edge head, class-weighted BCE and early stopping."""

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
        if self.sampler.kind != "neighbor":
            raise ValueError(
                "the edge model samples seed-edge neighbourhoods with LinkNeighborLoader; "
                "sampler.kind must be 'neighbor' (full_batch would materialise every account)"
            )
        if len(self.sampler.fanout) != int(self.params["layers"]):
            raise ValueError(
                f"sampler.fanout {self.sampler.fanout} has {len(self.sampler.fanout)} entries "
                f"but the model has {self.params['layers']} layers; give one fan-out per layer"
            )
        self._net: torch.nn.Module | None = None
        self._graph: Any = None
        self._graph_key: tuple[int, int] | None = None
        self._scale: tuple[np.ndarray, np.ndarray] | None = None

    def _prepare(self, data: GraphDataset, feats: FeatureMatrix) -> Any:
        """CPU ``Data`` with both edge directions, each carrying its original ``edge_time``."""
        import torch
        import torch_geometric.typing as pyg_typing
        from torch_geometric.data import Data

        if not pyg_typing.WITH_PYG_LIB:
            raise RuntimeError(
                "temporal neighbour sampling needs pyg-lib, which is not importable here; "
                "install pyg-lib for your torch build"
            )
        if feats.values.shape[0] != data.num_units:
            raise ValueError(
                f"feature matrix has {feats.values.shape[0]} rows but the dataset has "
                f"{data.num_units} {data.task}s; FeatureMatrix rows are in unit order"
            )
        key = (id(data), data.num_edges)
        if self._graph is not None and self._graph_key == key:
            return self._graph
        ei = torch.from_numpy(np.ascontiguousarray(data.edge_index))
        t = torch.from_numpy(np.ascontiguousarray(data.edge_time))
        self._graph = Data(
            num_nodes=data.num_nodes,
            edge_index=torch.cat([ei, ei.flip(0)], dim=1),
            edge_time=torch.cat([t, t]),
        )
        self._graph_key = key
        return self._graph

    def _loader(self, data: GraphDataset, idx: np.ndarray, labels: bool, shuffle: bool) -> Any:
        import torch
        from torch_geometric.loader import LinkNeighborLoader

        return LinkNeighborLoader(
            self._graph,
            num_neighbors=list(self.sampler.fanout),
            edge_label_index=torch.from_numpy(np.ascontiguousarray(data.edge_index[:, idx])),
            edge_label=torch.from_numpy(data.y[idx].astype(np.float32)) if labels else None,
            edge_label_time=torch.from_numpy(np.ascontiguousarray(data.edge_time[idx])),
            time_attr="edge_time",  # PR-F2: only edges at or before the seed's time are sampled
            batch_size=self.sampler.batch_size,
            shuffle=shuffle,
            num_workers=0,  # workers would fork a copy of the graph; host RAM is 7 GB
        )

    def _edge_x(self, feats: FeatureMatrix, rows: np.ndarray) -> torch.Tensor:
        import torch

        assert self._scale is not None
        mean, std = self._scale
        return torch.from_numpy(((feats.values[rows] - mean) / std).astype(np.float32)).to(
            self.device
        )

    def _forward(self, batch: Any, feats: FeatureMatrix, idx: np.ndarray):
        assert self._net is not None
        rows = idx[batch.input_id.numpy()]
        batch = batch.to(self.device)
        return self._net(
            batch.n_id, batch.edge_index, batch.edge_label_index, self._edge_x(feats, rows)
        )

    def fit(self, data: GraphDataset, feats: FeatureMatrix, split: Split, seed: int) -> FitInfo:
        """Train on ``split.train``, early-stopping on validation PR-AUC; test never touched."""
        import torch
        from sklearn.metrics import average_precision_score

        check_train_labelled(data, split)
        seed_all(seed)
        train_idx = np.asarray(split.train, dtype=np.int64)
        val_idx = np.asarray(split.val, dtype=np.int64)

        # Preprocessing is fitted on train rows only; val and test never shape it (PR-E1).
        train_x = feats.values[train_idx].astype(np.float64)
        # ponytail: z-score only, as in sage.py; decide on log1p before the AMLworld run (D2)
        std = train_x.std(axis=0)
        std[std == 0] = 1.0
        self._scale = (train_x.mean(axis=0), std)
        self._prepare(data, feats)

        y_train = data.y[train_idx]
        n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"split.train has {n_pos} illicit and {n_neg} licit edges; "
                "class-weighted BCE needs both classes present"
            )
        pos_weight = n_neg / n_pos

        self._net = _build_net(
            data.num_nodes,
            int(self.params["node_dim"]),
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
        loader = self._loader(data, train_idx, labels=True, shuffle=True)
        y_val = data.y[val_idx]
        epochs, patience = int(self.params["epochs"]), int(self.params["patience"])
        best_ap, best_epoch, best_state, stale, epoch = -1.0, 0, None, 0, 0
        log.info(
            "sage-edge fit: %d train / %d val edges, %d accounts, fanout=%s, up to %d epochs on %s",
            train_idx.size,
            val_idx.size,
            data.num_nodes,
            list(self.sampler.fanout),
            epochs,
            self.device,
        )

        with Timer() as timer:
            for epoch in range(1, epochs + 1):
                self._net.train()
                for batch in loader:
                    opt.zero_grad()
                    logits = self._forward(batch, feats, train_idx)
                    loss = criterion(logits, batch.edge_label.to(self.device))
                    loss.backward()
                    opt.step()

                ap = float(average_precision_score(y_val, self._infer(data, feats, val_idx)))
                if ap > best_ap:
                    best_ap, best_epoch, stale = ap, epoch, 0
                    best_state = copy.deepcopy(
                        {k: v.detach().cpu() for k, v in self._net.state_dict().items()}
                    )
                else:
                    stale += 1
                    if stale >= patience:
                        log.info("sage-edge early stop at epoch %d (best %d)", epoch, best_epoch)
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        log.info(
            "sage-edge fit done in %.1fs: best val PR-AUC %.4f at epoch %d",
            timer.seconds,
            best_ap,
            best_epoch,
        )
        return FitInfo(
            seconds=timer.seconds,
            best_iteration=best_epoch,
            val_pr_auc=best_ap,
            extra={"trial0_source": TRIAL0_SOURCE, "params": dict(self.params)},
        )

    def _infer(self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray) -> np.ndarray:
        """Mini-batched eval-mode pass over ``idx``, in ``idx`` order."""
        import torch

        assert self._net is not None
        self._net.eval()
        out = np.empty(idx.size, dtype=np.float32)
        with torch.no_grad():
            for batch in self._loader(data, idx, labels=False, shuffle=False):
                value = torch.sigmoid(self._forward(batch, feats, idx))
                out[batch.input_id.numpy()] = value.detach().cpu().numpy()
        return out

    def _require_fitted(self) -> None:
        if self._net is None:
            raise RuntimeError("sage-edge model is not fitted; call fit() before predict_proba")

    def predict_proba(
        self, data: GraphDataset, feats: FeatureMatrix, idx: np.ndarray
    ) -> np.ndarray:
        """P(illicit) for the edges ``idx`` as float32 in [0, 1]."""
        self._require_fitted()
        self._prepare(data, feats)
        return self._infer(data, feats, np.asarray(idx, dtype=np.int64))

    @classmethod
    def trial0(cls) -> tuple[dict[str, Any], str]:
        return dict(TRIAL0), TRIAL0_SOURCE
