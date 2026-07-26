"""Configurable hybrid recommender that combines multiple sub-models."""
from __future__ import annotations

from typing import Any, Mapping

from ..base import BaseRecommender, Recommendation, RecommendationContext


DEFAULT_WEIGHTS: Mapping[str, float] = {
    "item_cosine": 0.35,
    "content_based": 0.30,
    "recent_activity": 0.20,
    "popularity": 0.15,
}


class HybridRecommender(BaseRecommender):
    name = "hybrid"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self._weights: dict[str, float] = dict(weights or DEFAULT_WEIGHTS)
        self._sub: dict[str, BaseRecommender] = {}

    def add(self, model: BaseRecommender) -> "HybridRecommender":
        self._sub[model.name] = model
        return self

    def fit(self, interactions: Any, items: Any) -> "HybridRecommender":
        for m in self._sub.values():
            m.fit(interactions, items)
        return self

    def recommend(self, context: RecommendationContext) -> list[Recommendation]:
        all_scores: dict[int, dict[str, float]] = {}
        reasons: dict[int, str] = {}
        for name, model in self._sub.items():
            weight = self._weights.get(name, 0.0)
            if weight <= 0:
                continue
            for rec in model.recommend(context):
                bucket = all_scores.setdefault(rec.item_id, {})
                bucket[name] = rec.score * weight
                if rec.reason and rec.item_id not in reasons:
                    reasons[rec.item_id] = rec.reason
        out: list[Recommendation] = []
        for iid, breakdown in all_scores.items():
            if iid in context.seen_item_ids:
                continue
            total = sum(breakdown.values())
            out.append(
                Recommendation(
                    item_id=iid,
                    score=total,
                    reason=reasons.get(iid, "Multiple signals agree"),
                    components=breakdown,
                )
            )
        out.sort(key=lambda r: r.score, reverse=True)
        return out[: context.top_k]
