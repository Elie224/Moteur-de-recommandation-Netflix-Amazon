"""Popularity baseline.

Recommends the most-rated (or highest-mean-rated) items globally. Useful as
a sanity-check floor: any decent CF model should beat this.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


class PopularityBaseline:
    """Rank items by rating count (or by weighted score)."""

    def __init__(self, min_ratings: int = 50, score: str = "count") -> None:
        assert score in {"count", "mean", "bayesian"}
        self.min_ratings = min_ratings
        self.score = score
        self.item_rank_: list[tuple[int, float]] = []
        # user_id -> set of items seen during fit
        self.user_seen_: dict[int, set[int]] = {}

    def fit(self, ratings: pd.DataFrame) -> "PopularityBaseline":
        agg = ratings.groupby("movie_id")["rating"].agg(["count", "mean"])
        agg = agg[agg["count"] >= self.min_ratings]
        if self.score == "count":
            order = agg["count"]
        elif self.score == "mean":
            order = agg["mean"]
        else:  # bayesian: shrinkage toward global mean
            C = agg["count"].sum() / len(agg)
            m = ratings["rating"].mean()
            order = (C * m + agg["count"] * agg["mean"]) / (C + agg["count"])
        order = order.sort_values(ascending=False)
        self.item_rank_ = list(zip(order.index.tolist(), order.tolist()))
        self.user_seen_ = (
            ratings.groupby("user_id")["movie_id"].apply(set).to_dict()
        )
        # Convert numpy int64 keys to plain python ints
        self.user_seen_ = {int(k): {int(x) for x in v} for k, v in self.user_seen_.items()}
        return self

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        seen = set(exclude_seen) if exclude_seen else set()
        out = []
        for iid, sc in self.item_rank_:
            if int(iid) in seen:
                continue
            out.append((int(iid), float(sc)))
            if len(out) >= top_k:
                break
        return out

    def recommend_for_users(
        self, user_ids: list[int], top_k: int = 10
    ) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for uid in user_ids:
            seen = self.user_seen_.get(int(uid), set())
            out[int(uid)] = [iid for iid, _ in self.recommend(int(uid), top_k=top_k, exclude_seen=seen)]
        return out
