"""Recommender that reacts to the most recent user interactions.

Falls back to popularity if the user has no recent activity. Designed to
update as soon as a new interaction is recorded.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..base import BaseRecommender, Recommendation, RecommendationContext


class RecentActivityRecommender(BaseRecommender):
    name = "recent_activity"

    def __init__(self, half_life_hours: float = 24.0, max_recent: int = 20) -> None:
        self.half_life_hours = half_life_hours
        self.max_recent = max_recent
        self._tag_neighbours: dict[str, dict[int, float]] = {}
        self._item_tags: dict[int, list[str]] = {}
        self._popularity: list[tuple[int, float]] = []

    def fit(self, interactions: Any, items: Any) -> "RecentActivityRecommender":
        # Build a simple tag-level co-occurrence map from item metadata.
        for row in items.iter_rows():
            iid = int(row["catalog_item_id"])
            tags = [str(tag).strip().lower() for tag in row.get("tags") or [] if str(tag).strip()]
            self._item_tags[iid] = tags

        # Popularity fallback (used when a user has no signal).
        pop: dict[int, float] = defaultdict(float)
        for row in items.iter_rows():
            pop[int(row["catalog_item_id"])] += float(row.get("popularity_score") or 0.0)
        for row in interactions.weighted_rows():
            pop[int(row["item_id"])] += float(row["weight"])
        self._popularity = sorted(pop.items(), key=lambda x: x[1], reverse=True)

        # Build a basic neighbour map: tag -> item -> weight
        tag_to_items: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for row in interactions.weighted_rows():
            iid = int(row["item_id"])
            w = float(row["weight"])
            for tag in self._item_tags.get(iid, []):
                tag_to_items[tag][iid] += w
        self._tag_neighbours = {tag: dict(items_) for tag, items_ in tag_to_items.items()}
        return self

    def recommend(self, context: RecommendationContext) -> list[Recommendation]:
        seed_item_ids = list(context.recent_item_ids)
        for item_id in context.favorite_item_ids:
            if item_id not in seed_item_ids:
                seed_item_ids.append(item_id)
        if not seed_item_ids:
            return self._popularity_fallback(context)
        scores: dict[int, float] = defaultdict(float)
        for iid in seed_item_ids[: self.max_recent]:
            for tag in self._item_tags.get(iid, []):
                for cand, w in self._tag_neighbours.get(tag, {}).items():
                    if context.is_excluded(cand):
                        continue
                    if cand == iid:
                        continue
                    scores[cand] += float(w)
        out: list[Recommendation] = []
        for iid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[: context.top_k]:
            out.append(
                Recommendation(
                    item_id=iid,
                    score=score,
                    reason="Related to items you interacted with recently",
                    components={"recent_activity": score},
                )
            )
        if not out:
            return self._popularity_fallback(context)
        return out

    def _popularity_fallback(self, context: RecommendationContext) -> list[Recommendation]:
        out: list[Recommendation] = []
        for iid, score in self._popularity:
            if context.is_excluded(iid):
                continue
            out.append(
                Recommendation(
                    item_id=iid,
                    score=float(score),
                    reason="Popular among current users",
                    components={"recent_activity": float(score)},
                )
            )
            if len(out) >= context.top_k:
                break
        return out

    def supports_cold_start(self) -> bool:
        return True
