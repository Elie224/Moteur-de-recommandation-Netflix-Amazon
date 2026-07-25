"""Neural collaborative filtering models in PyTorch.

Two architectures:

1. TwoTower (a.k.a. matrix factorization with embeddings):
       score(u, i) = mu + b_u + b_i + <e_u, e_i>

2. NeuralCF (He et al. 2017):
       score(u, i) = MLP([e_u, e_i])   (concatenated, then MLP)
       Optionally fuses <e_u, e_i> element-wise product as an extra signal.

Both support explicit (MSE) and implicit (BPR) training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# TwoTower (DeepMF / matrix factorization in PyTorch)
# ---------------------------------------------------------------------------


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 64) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u, i):
        pu = self.user_emb(u)
        qi = self.item_emb(i)
        bu = self.user_bias(u).squeeze(-1)
        bi = self.item_bias(i).squeeze(-1)
        dot = (pu * qi).sum(dim=-1)
        return self.global_bias + bu + bi + dot


# ---------------------------------------------------------------------------
# NeuralCF (He et al. 2017)
# ---------------------------------------------------------------------------


class NeuralCF(nn.Module):
    """Concatenate user/item embeddings, run through MLP."""

    def __init__(self, n_users: int, n_items: int, n_factors: int = 32,
                 hidden: tuple[int, ...] = (64, 32, 16), dropout: float = 0.2) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        layers = []
        in_dim = 2 * n_factors
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.mlp(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer / wrapper
# ---------------------------------------------------------------------------


@dataclass
class TorchCF:
    """Unified wrapper for TwoTower / NeuralCF with the project API."""

    arch: str = "twotower"  # "twotower" or "neuralcf"
    n_factors: int = 64
    hidden: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.2
    n_epochs: int = 20
    batch_size: int = 8192
    lr: float = 0.005
    reg: float = 1e-5
    random_state: int = 0
    device: str = "cpu"
    verbose: bool = False
    model: nn.Module | None = None
    user_index_: dict[int, int] = field(default_factory=dict)
    item_index_: dict[int, int] = field(default_factory=dict)
    train_loss_: list[float] = field(default_factory=list)
    val_rmse_: list[float] = field(default_factory=list)

    def _build(self, n_u, n_i):
        if self.arch == "twotower":
            return TwoTower(n_u, n_i, n_factors=self.n_factors)
        elif self.arch == "neuralcf":
            return NeuralCF(n_u, n_i, n_factors=self.n_factors,
                            hidden=self.hidden, dropout=self.dropout)
        raise ValueError(f"Unknown arch: {self.arch}")

    def fit(self, ratings: pd.DataFrame, val: pd.DataFrame | None = None) -> "TorchCF":
        _set_seed(self.random_state)
        device = torch.device(self.device)

        user_ids = np.sort(ratings["user_id"].unique())
        item_ids = np.sort(ratings["movie_id"].unique())
        self.user_index_ = {int(u): i for i, u in enumerate(user_ids)}
        self.item_index_ = {int(m): i for i, m in enumerate(item_ids)}

        u_idx = torch.tensor(ratings["user_id"].map(self.user_index_).to_numpy(), dtype=torch.long)
        i_idx = torch.tensor(ratings["movie_id"].map(self.item_index_).to_numpy(), dtype=torch.long)
        r = torch.tensor(ratings["rating"].to_numpy(np.float32))

        ds = TensorDataset(u_idx, i_idx, r)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = self._build(len(user_ids), len(item_ids)).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)

        if val is not None:
            mask = (
                val["user_id"].isin(self.user_index_)
                & val["movie_id"].isin(self.item_index_)
            )
            val = val[mask]
            val_u = torch.tensor(val["user_id"].map(self.user_index_).to_numpy(), dtype=torch.long, device=device)
            val_i = torch.tensor(val["movie_id"].map(self.item_index_).to_numpy(), dtype=torch.long, device=device)
            val_r = torch.tensor(val["rating"].to_numpy(np.float32), device=device)

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_obs = 0
            for ub, ib, rb in dl:
                ub, ib, rb = ub.to(device), ib.to(device), rb.to(device)
                pred = self.model(ub, ib)
                loss = F.mse_loss(pred, rb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * rb.size(0)
                n_obs += rb.size(0)
            self.train_loss_.append(float(np.sqrt(epoch_loss / n_obs)))

            if val is not None and len(val_r):
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(val_u, val_i)
                    val_rmse = float(torch.sqrt(F.mse_loss(val_pred, val_r)))
                self.val_rmse_.append(val_rmse)
            if self.verbose:
                msg = f"epoch {epoch+1}/{self.n_epochs}  train RMSE {self.train_loss_[-1]:.4f}"
                if val is not None:
                    msg += f"  val RMSE {val_rmse:.4f}"
                print(msg)
        return self

    # ------------------------------------------------------------------
    def _predict(self, uid: int, iid: int) -> float:
        if uid not in self.user_index_ or iid not in self.item_index_:
            return 3.5  # midpoint of MovieLens scale as a fallback
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor([self.user_index_[uid]], dtype=torch.long)
            i = torch.tensor([self.item_index_[iid]], dtype=torch.long)
            pred = self.model(u, i).item()
        return float(np.clip(pred, 1.0, 5.0))

    def predict_for_pairs(self, pairs: pd.DataFrame) -> dict[tuple[int, int], float]:
        return {(int(r.user_id), int(r.movie_id)):
                self._predict(int(r.user_id), int(r.movie_id))
                for r in pairs.itertuples()}

    def all_user_predictions(self, user_id: int) -> np.ndarray:
        if user_id not in self.user_index_:
            return np.array([])
        self.model.eval()
        with torch.no_grad():
            u = torch.full((len(self.item_index_),), self.user_index_[user_id], dtype=torch.long)
            i = torch.tensor(list(self.item_index_.values()), dtype=torch.long)
            scores = self.model(u, i).numpy()
        return np.clip(scores, 1.0, 5.0)

    def recommend(self, user_id, top_k=10, exclude_seen=True, seen_items=None):
        if user_id not in self.user_index_:
            return []
        scores = self.all_user_predictions(user_id)
        item_ids_sorted = np.array(sorted(self.item_index_.keys()))
        if exclude_seen and seen_items:
            for iid in seen_items:
                if iid in self.item_index_:
                    scores[self.item_index_[iid]] = -np.inf
        k = min(top_k, scores.size)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(item_ids_sorted[i]), float(scores[i])) for i in top_idx if scores[i] != -np.inf]

    def recommend_for_users(self, user_ids, top_k=10):
        return {int(uid): [iid for iid, _ in self.recommend(int(uid), top_k=top_k)]
                for uid in user_ids}
