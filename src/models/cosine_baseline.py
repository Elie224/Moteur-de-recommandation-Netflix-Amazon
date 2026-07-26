"""Item-item collaborative filtering using cosine similarity (mean-centred).

Approach
--------
1. Build the user x item rating matrix from the training set (sparse).
2. Mean-centre each row (per user) to remove user bias.
3. Compute item-item cosine similarity on the centred matrix.
4. To recommend to user ``u``:
       score(i) = sum_{j in seen(u) and liked(u)} sim(i, j) * (rating(u, j) - mean(u))
   Items already seen are excluded.  Return the top-K.
5. Rating prediction (KNN-style with means):
       r_hat(u, i) = mean(u) + sum_j sim(i, j) * (rating(u, j) - mean(u))
                              / sum_j |sim(i, j)|
   Clipped to [1, 5].

Cold users are skipped (empty recommendation list).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ItemItemCosine:
    like_threshold: float = 4.0
    top_neighbours: Optional[int] = 200
    rating_clip: tuple[float, float] = (1.0, 5.0)

    user_index_: dict[int, int] = field(default_factory=dict)
    item_index_: dict[int, int] = field(default_factory=dict)
    user_ids_: np.ndarray = field(default_factory=lambda: np.array([]))
    item_ids_: np.ndarray = field(default_factory=lambda: np.array([]))
    user_means_: np.ndarray = field(default_factory=lambda: np.array([]))
    item_similarity_: Optional[sparse.csr_matrix] = None
    R_centred_: Optional[sparse.csr_matrix] = None
    user_seen_: dict[int, set[int]] = field(default_factory=dict)

    def fit(self, ratings: pd.DataFrame) -> "ItemItemCosine":
        user_ids = np.sort(ratings["user_id"].unique())
        item_ids = np.sort(ratings["movie_id"].unique())
        self.user_index_ = {int(u): i for i, u in enumerate(user_ids)}
        self.item_index_ = {int(m): i for i, m in enumerate(item_ids)}
        self.user_ids_ = np.array([int(u) for u in user_ids])
        self.item_ids_ = np.array([int(m) for m in item_ids])

        u_idx = ratings["user_id"].map(self.user_index_).to_numpy()
        i_idx = ratings["movie_id"].map(self.item_index_).to_numpy()
        r = ratings["rating"].to_numpy(dtype=np.float32)

        R = sparse.csr_matrix((r, (u_idx, i_idx)), shape=(len(user_ids), len(item_ids)))

        user_sums = np.asarray(R.sum(axis=1)).ravel()
        user_counts = np.diff(R.indptr)
        means = np.divide(
            user_sums, user_counts,
            out=np.zeros_like(user_sums, dtype=np.float32),
            where=user_counts > 0,
        )
        self.user_means_ = means

        centred = r - means[u_idx]
        R_c = sparse.csr_matrix((centred, (u_idx, i_idx)), shape=R.shape)
        self.R_centred_ = R_c

        sim = cosine_similarity(R_c.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        # Optional: keep only the top-N neighbours per item to make scoring
        # O(top_N * n_users) instead of O(n_items) and cut memory use.
        if self.top_neighbours is not None and self.top_neighbours > 0:
            k = min(self.top_neighbours, sim.shape[0] - 1)
            top_idx = np.argpartition(-sim, k, axis=1)[:, :k]
            mask = np.zeros_like(sim, dtype=bool)
            rows = np.repeat(np.arange(sim.shape[0]), k)
            mask[rows, top_idx.ravel()] = True
            sim = np.where(mask & (sim > 0), sim, 0.0).astype(np.float32)
        self.item_similarity_ = sparse.csr_matrix(sim)

        # Store per-user seen items (python ints for downstream set ops)
        self.user_seen_ = (
            ratings.groupby("user_id")["movie_id"]
            .apply(lambda s: {int(x) for x in s.tolist()})
            .to_dict()
        )
        return self

    def _user_row(self, user_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = self.user_index_[user_id]
        row = self.R_centred_.getrow(u)
        return row.indices, row.data, np.abs(row.data)

    def _score_user_ranking(self, user_id: int) -> np.ndarray:
        cols, data, _ = self._user_row(user_id)
        if cols.size == 0:
            return np.zeros(self.R_centred_.shape[1], dtype=np.float32)
        keep = data > 0
        if not np.any(keep):
            top = np.argsort(-data)[:5]
            keep_idx = top
        else:
            keep_idx = np.where(keep)[0]
        sim_block = self.item_similarity_[cols[keep_idx]].toarray()
        return (sim_block.T @ data[keep_idx]).astype(np.float32)

    def _score_user_rating(self, user_id: int) -> tuple[np.ndarray, np.ndarray]:
        cols, data, _ = self._user_row(user_id)
        if cols.size == 0:
            z = np.zeros(self.R_centred_.shape[1], dtype=np.float32)
            return z, z
        sim_block = self.item_similarity_[cols].toarray()
        weighted_sum = sim_block.T @ data
        abs_sum = np.abs(sim_block).sum(axis=0).astype(np.float32)
        return weighted_sum, abs_sum

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        if user_id not in self.user_index_:
            return []  # cold user -> empty list

        scores = self._score_user_ranking(user_id)
        if exclude_seen:
            if seen_items is None:
                seen_items = self.user_seen_.get(int(user_id), set())
            seen_idx = {self.item_index_[i] for i in seen_items if i in self.item_index_}
            for idx in seen_idx:
                scores[idx] = -np.inf

        k = min(top_k, scores.size)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            (int(self.item_ids_[i]), float(scores[i]))
            for i in top_idx
            if scores[i] != -np.inf
        ]

    def recommend_for_users(
        self, user_ids: list[int], top_k: int = 10
    ) -> dict[int, list[int]]:
        return {
            int(uid): [iid for iid, _ in self.recommend(int(uid), top_k=top_k)]
            for uid in user_ids
            if int(uid) in self.user_index_
        }

    def predict_for_pairs(self, pairs: pd.DataFrame) -> dict[tuple[int, int], float]:
        lo, hi = self.rating_clip
        preds: dict[tuple[int, int], float] = {}
        for uid, group in pairs.groupby("user_id"):
            uid = int(uid)
            if uid not in self.user_index_:
                continue
            u = self.user_index_[uid]
            user_mean = float(self.user_means_[u])
            weighted_sum, abs_sum = self._score_user_rating(uid)
            for _, row in group.iterrows():
                iid = int(row["movie_id"])
                if iid not in self.item_index_:
                    continue
                i = self.item_index_[iid]
                denom = abs_sum[i]
                pred = user_mean if denom == 0 else user_mean + weighted_sum[i] / denom
                preds[(uid, iid)] = float(np.clip(pred, lo, hi))
        return preds
