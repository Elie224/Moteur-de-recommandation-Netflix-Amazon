"""Item-item cosine similarity recommender.

This is the V1-safe port of the legacy ``ItemItemCosine`` model. It applies
mean-centred cosine similarity on the user x item matrix and ranks items by
their weighted alignment with the user's signal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from ..base import BaseRecommender, Recommendation, RecommendationContext


class ItemItemCosineRecommender(BaseRecommender):
    name = "item_cosine"

    def __init__(self, top_neighbours: int = 200, min_weight: float = 0.0) -> None:
        self.top_neighbours = top_neighbours
        self.min_weight = min_weight
        self._user_ids: list[int] = []
        self._item_ids: list[int] = []
        self._user_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._sim: sparse.csr_matrix | None = None
        self._R: sparse.csr_matrix | None = None
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: Any, items: Any) -> "ItemItemCosineRecommender":
        # Build (user, item, weight) tuples from the interaction frame.
        grouped: dict[int, dict[int, float]] = {}
        for row in interactions.weighted_rows():
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            w = float(row["weight"])
            if uid not in grouped:
                grouped[uid] = {}
            grouped[uid][iid] = grouped[uid].get(iid, 0.0) + w

        user_ids = sorted(grouped.keys())
        item_ids_set: set[int] = set()
        for uid in user_ids:
            item_ids_set.update(grouped[uid].keys())
        item_ids = sorted(item_ids_set)

        self._user_ids = user_ids
        self._item_ids = item_ids
        self._user_index = {u: i for i, u in enumerate(user_ids)}
        self._item_index = {i: j for j, i in enumerate(item_ids)}

        if not user_ids or not item_ids:
            self._R = sparse.csr_matrix((len(user_ids), len(item_ids)), dtype=np.float32)
            self._sim = sparse.csr_matrix((len(item_ids), len(item_ids)), dtype=np.float32)
            self._seen = {uid: set(item_map) for uid, item_map in grouped.items()}
            return self

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for uid, items_map in grouped.items():
            u = self._user_index[uid]
            for iid, w in items_map.items():
                i = self._item_index[iid]
                rows.append(u)
                cols.append(i)
                data.append(w)

        R = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )

        # Mean-centre per user so user bias is removed.
        user_sums = np.asarray(R.sum(axis=1)).ravel()
        user_counts = np.diff(R.indptr)
        means = np.divide(
            user_sums,
            user_counts,
            out=np.zeros_like(user_sums, dtype=np.float32),
            where=user_counts > 0,
        )
        centred = np.asarray(R.tocoo().data) - np.repeat(means, user_counts)
        R_c = sparse.csr_matrix(
            (centred.astype(np.float32), (R.tocoo().row, R.tocoo().col)),
            shape=R.shape,
        )

        sim = cosine_similarity(R_c.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(sim, 0.0)

        if self.top_neighbours and self.top_neighbours > 0 and sim.shape[0] > 1:
            k = min(self.top_neighbours, sim.shape[0] - 1)
            top_idx = np.argpartition(-sim, k, axis=1)[:, :k]
            mask = np.zeros_like(sim, dtype=bool)
            rows_n = np.repeat(np.arange(sim.shape[0]), k)
            mask[rows_n, top_idx.ravel()] = True
            sim = np.where(mask & (sim > 0), sim, 0.0).astype(np.float32)

        self._R = R_c
        self._sim = sparse.csr_matrix(sim)
        self._seen = {uid: set(m.keys()) for uid, m in grouped.items()}
        return self

    def recommend(self, context: RecommendationContext) -> list[Recommendation]:
        if self._R is None or self._sim is None:
            return []
        if context.user_id not in self._user_index:
            return []
        u = self._user_index[context.user_id]
        row = self._R.getrow(u)
        cols = row.indices
        data = row.data
        if cols.size == 0:
            return []
        sim_block = self._sim[cols].toarray()
        scores = (sim_block.T @ data).astype(np.float32)
        excluded_ids = context.seen_item_ids if context.excluded_item_ids is None else context.excluded_item_ids
        for iid in excluded_ids:
            if iid in self._item_index:
                scores[self._item_index[iid]] = -np.inf
        order = np.argsort(-scores)
        out: list[Recommendation] = []
        for i in order:
            score = float(scores[i])
            if score == -np.inf or score <= 0:
                continue
            iid = self._item_ids[i]
            out.append(
                Recommendation(
                    item_id=iid,
                    score=score,
                    reason="Similar to items you engaged with",
                    components={"item_cosine": score},
                )
            )
            if len(out) >= context.top_k:
                break
        return out
