"""Popularity baseline recommender (domain-agnostic)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..base import BaseRecommender, Recommendation, RecommendationContext


class PopularityRecommender(BaseRecommender):
    """Rank items globally by their weighted interaction count.

    Works for both movies and products. Used as the cold-start fallback
    and as the percentile floor any other model has to beat.
    """

    name = "popularity"

    def __init__(self, min_interactions: int = 5) -> None:
        self.min_interactions = min_interactions
        self._rank: list[tuple[int, float]] = []
        self._seen_by_type: dict[str, set[int]] = defaultdict(set)

    def fit(self, interactions: Any, items: Any) -> "PopularityRecommender":
        scores: dict[int, float] = defaultdict(float)
        for row in items.iter_rows():
            scores[int(row["catalog_item_id"])] += float(row.get("popularity_score") or 0.0)
        for row in interactions.weighted_rows():
            iid = int(row["item_id"])
            scores[iid] += float(row["weight"])
        filtered = [(iid, score) for iid, score in scores.items() if score > 0]
        filtered.sort(key=lambda x: x[1], reverse=True)
        self._rank = filtered
        return self

    def recommend(self, context: RecommendationContext) -> list[Recommendation]:
        out: list[Recommendation] = []
        for iid, score in self._rank:
            if context.is_excluded(iid):
                continue
            score = float(score)
            if score <= 0:
                continue
            out.append(
                Recommendation(
                    item_id=iid,
                    score=score,
                    reason="Globally popular item",
                    components={"popularity": score},
                )
            )
            if len(out) >= context.top_k:
                break
        return out

    def supports_cold_start(self) -> bool:
        return True
