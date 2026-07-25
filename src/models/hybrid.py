"""LightFM-style hybrid model implemented from scratch in PyTorch."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class HybridCF(nn.Module):
    def __init__(self, n_users: int, n_items: int,
                 n_user_features: int, n_item_features: int,
                 n_factors: int = 32) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        # +1 row reserved as "padding" feature (index 0 is unused for real features)
        self.user_feat_emb = nn.Embedding(n_user_features + 1, n_factors, padding_idx=0)
        self.item_feat_emb = nn.Embedding(n_item_features + 1, n_factors, padding_idx=0)
        self.user_feat_bias = nn.Embedding(n_user_features + 1, 1, padding_idx=0)
        self.item_feat_bias = nn.Embedding(n_item_features + 1, 1, padding_idx=0)
        for emb in [self.user_emb, self.item_emb,
                    self.user_feat_emb, self.item_feat_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        for emb in [self.user_bias, self.item_bias,
                    self.user_feat_bias, self.item_feat_bias]:
            nn.init.zeros_(emb.weight)

    def forward(self, u, i, uf, if_):
        u_id = self.user_emb(u)
        i_id = self.item_emb(i)
        u_b = self.user_bias(u).squeeze(-1)
        i_b = self.item_bias(i).squeeze(-1)
        # Mask padding (0): zeros out contributions
        u_mask = (uf != 0).float().unsqueeze(-1)
        i_mask = (if_ != 0).float().unsqueeze(-1)
        u_feat = (self.user_feat_emb(uf) * u_mask).sum(dim=1)
        i_feat = (self.item_feat_emb(if_) * i_mask).sum(dim=1)
        u_b_feat = (self.user_feat_bias(uf).squeeze(-1) * (uf != 0).float()).sum(dim=1)
        i_b_feat = (self.item_feat_bias(if_).squeeze(-1) * (if_ != 0).float()).sum(dim=1)
        u_repr = u_id + u_feat
        i_repr = i_id + i_feat
        return (u_repr * i_repr).sum(dim=-1) + u_b + i_b + u_b_feat + i_b_feat


@dataclass
class LightFMModel:
    n_factors: int = 32
    n_epochs: int = 15
    batch_size: int = 4096
    lr: float = 0.005
    reg: float = 1e-5
    loss: str = "mse"  # "mse" or "bpr"
    n_neg: int = 4
    random_state: int = 0
    verbose: bool = False

    user_index_: dict[int, int] = field(default_factory=dict)
    item_index_: dict[int, int] = field(default_factory=dict)
    user_feat_index_: dict[tuple[str, str], int] = field(default_factory=dict)
    item_feat_index_: dict[str, int] = field(default_factory=dict)
    user_features_mat_: torch.Tensor | None = None
    item_features_mat_: torch.Tensor | None = None
    model: HybridCF | None = None
    train_loss_: list[float] = field(default_factory=list)

    def build_user_features(self, users: pd.DataFrame) -> None:
        feats = []
        feat_keys = []
        for _, row in users.iterrows():
            keys = [
                ("gender", str(row["gender"])),
                ("age", str(row["age"])),
                ("occupation", str(row["occupation"])),
            ]
            feats.append(keys)
            feat_keys.extend(keys)
        uniq = sorted(set(feat_keys))
        # Reserve 0 for padding
        self.user_feat_index_ = {k: i + 1 for i, k in enumerate(uniq)}
        max_n = max(len(f) for f in feats) if feats else 1
        mat = np.zeros((len(feats), max_n), dtype=np.int64)
        for i, keys in enumerate(feats):
            for j, k in enumerate(keys):
                mat[i, j] = self.user_feat_index_[k]
        self.user_features_mat_ = torch.tensor(mat)

    def build_item_features(self, movies: pd.DataFrame) -> None:
        feats = []
        feat_keys = []
        for _, row in movies.iterrows():
            keys = [("genre", g) for g in (row["genres_list"] or [])]
            try:
                year = int(row["year"]) if row["year"] == row["year"] else None
            except (TypeError, ValueError):
                year = None
            if year:
                decade = (year // 10) * 10
                keys.append(("decade", str(decade)))
            feats.append(keys)
            feat_keys.extend(keys)
        uniq = sorted(set(feat_keys))
        self.item_feat_index_ = {k: i + 1 for i, k in enumerate(uniq)}
        max_n = max(len(f) for f in feats) if feats else 1
        mat = np.zeros((len(feats), max_n), dtype=np.int64)
        for i, keys in enumerate(feats):
            for j, k in enumerate(keys):
                mat[i, j] = self.item_feat_index_[k]
        self.item_features_mat_ = torch.tensor(mat)

    def fit(self, ratings: pd.DataFrame) -> "LightFMModel":
        _set_seed(self.random_state)
        device = torch.device("cpu")
        user_ids = np.sort(ratings["user_id"].unique())
        item_ids = np.sort(ratings["movie_id"].unique())
        self.user_index_ = {int(u): i for i, u in enumerate(user_ids)}
        self.item_index_ = {int(m): i for i, m in enumerate(item_ids)}

        if self.user_features_mat_ is None or self.item_features_mat_ is None:
            raise ValueError("Call build_user_features() and build_item_features() first.")

        u_idx = torch.tensor(ratings["user_id"].map(self.user_index_).to_numpy(), dtype=torch.long)
        i_idx = torch.tensor(ratings["movie_id"].map(self.item_index_).to_numpy(), dtype=torch.long)
        uf_mat = self.user_features_mat_[u_idx]
        if_mat = self.item_features_mat_[i_idx]
        n_user_feat_total = len(self.user_feat_index_) + 1
        n_item_feat_total = len(self.item_feat_index_) + 1

        self.model = HybridCF(len(user_ids), len(item_ids),
                              n_user_feat_total, n_item_feat_total,
                              n_factors=self.n_factors).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        n = len(ratings)

        if self.loss == "mse":
            r = torch.tensor(ratings["rating"].to_numpy(np.float32))
            for epoch in range(self.n_epochs):
                perm = torch.randperm(n)
                ep_loss = 0.0
                for start in range(0, n, self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    ub, ib, rb = u_idx[idx], i_idx[idx], r[idx]
                    ufb, ifb = uf_mat[idx], if_mat[idx]
                    pred = self.model(ub, ib, ufb, ifb)
                    loss = F.mse_loss(pred, rb)
                    opt.zero_grad(); loss.backward(); opt.step()
                    ep_loss += loss.item() * rb.size(0)
                self.train_loss_.append(float(np.sqrt(ep_loss / n)))
                if self.verbose:
                    print(f"epoch {epoch+1}/{self.n_epochs}  RMSE {self.train_loss_[-1]:.4f}")
        elif self.loss == "bpr":
            for epoch in range(self.n_epochs):
                perm = torch.randperm(n)
                ep_loss = 0.0; n_batches = 0
                for start in range(0, n, self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    ub, ib = u_idx[idx], i_idx[idx]
                    ufb, ifb = uf_mat[idx], if_mat[idx]
                    neg_ib = torch.randint(0, len(item_ids), (ib.size(0) * self.n_neg,))
                    neg_ub = ub.repeat_interleave(self.n_neg)
                    neg_ufb = ufb.repeat_interleave(self.n_neg, 0)
                    neg_ifb = if_mat[neg_ib]
                    pos = self.model(ub, ib, ufb, ifb)
                    neg = self.model(neg_ub, neg_ib, neg_ufb, neg_ifb)
                    loss = -F.logsigmoid(pos - neg).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                    ep_loss += loss.item(); n_batches += 1
                self.train_loss_.append(ep_loss / max(n_batches, 1))
                if self.verbose:
                    print(f"epoch {epoch+1}/{self.n_epochs}  BPR {self.train_loss_[-1]:.4f}")
        return self

    def _predict(self, uid, iid):
        if uid not in self.user_index_ or iid not in self.item_index_:
            return 3.5
        u = torch.tensor([self.user_index_[uid]], dtype=torch.long)
        i = torch.tensor([self.item_index_[iid]], dtype=torch.long)
        ufb = self.user_features_mat_[self.user_index_[uid]].unsqueeze(0)
        ifb = self.item_features_mat_[self.item_index_[iid]].unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(u, i, ufb, ifb).item()
        return float(np.clip(pred, 1.0, 5.0)) if self.loss == "mse" else float(pred)

    def predict_for_pairs(self, pairs):
        return {(int(r.user_id), int(r.movie_id)):
                self._predict(int(r.user_id), int(r.movie_id))
                for r in pairs.itertuples()}

    def all_user_predictions(self, uid):
        if uid not in self.user_index_:
            return np.array([])
        u = torch.full((len(self.item_index_),), self.user_index_[uid], dtype=torch.long)
        i = torch.tensor(list(self.item_index_.values()), dtype=torch.long)
        ufb = self.user_features_mat_[self.user_index_[uid]].unsqueeze(0).expand(len(i), -1)
        ifb = self.item_features_mat_[i]
        self.model.eval()
        with torch.no_grad():
            return self.model(u, i, ufb, ifb).numpy()

    def recommend(self, uid, top_k=10, exclude_seen=True, seen_items=None):
        if uid not in self.user_index_:
            return []
        scores = self.all_user_predictions(uid)
        item_ids_sorted = np.array(sorted(self.item_index_.keys()))
        if exclude_seen and seen_items:
            for iid in seen_items:
                if iid in self.item_index_:
                    scores[self.item_index_[iid]] = -np.inf
        k = min(top_k, scores.size)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(item_ids_sorted[i]), float(scores[i]))
                for i in top_idx if scores[i] != -np.inf]

    def recommend_for_users(self, user_ids, top_k=10):
        return {int(uid): [iid for iid, _ in self.recommend(int(uid), top_k=top_k)]
                for uid in user_ids}

    def recommend_for_new_user(self, user_features, top_k=10, seen_items=None):
        if not self.user_feat_index_:
            raise ValueError("build_user_features() not called")
        feat_ids = [self.user_feat_index_[k] for k in user_features if k in self.user_feat_index_]
        if not feat_ids:
            return []
        max_n = self.user_features_mat_.shape[1]
        uf = torch.tensor((feat_ids + [0] * max_n)[:max_n], dtype=torch.long).unsqueeze(0)
        n_i = len(self.item_index_)
        ufb = uf.expand(n_i, -1)
        ifb = self.item_features_mat_
        i = torch.tensor(list(self.item_index_.values()), dtype=torch.long)
        u_dummy = torch.zeros(n_i, dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(u_dummy, i, ufb, ifb).numpy()
        if self.loss == "mse":
            scores = np.clip(scores, 1.0, 5.0)
        if seen_items:
            for iid in seen_items:
                if iid in self.item_index_:
                    scores[self.item_index_[iid]] = -np.inf
        k = min(top_k, scores.size)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        item_ids_sorted = np.array(sorted(self.item_index_.keys()))
        return [(int(item_ids_sorted[i]), float(scores[i]))
                for i in top_idx if scores[i] != -np.inf]
