"""Content-based recommender.

Uses the catalog item metadata (text blob + categorical tags) to build a
TF-IDF vector per item and serves nearest-neighbour rankings at query time.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..base import BaseRecommender, Recommendation, RecommendationContext


def _item_text(row: Mapping[str, Any]) -> str:
    parts = []
    title = row.get("title") or ""
    if title:
        parts.append(str(title))
    desc = row.get("description") or ""
    if desc:
        parts.append(str(desc))
    cat = row.get("category") or ""
    if cat:
        parts.append(str(cat).replace("|", " "))
    genres = row.get("genres") or []
    if isinstance(genres, (list, tuple)):
        parts.extend(str(g) for g in genres)
    return " ".join(parts).lower()


class ContentBasedRecommender(BaseRecommender):
    name = "content_based"

    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._item_ids: list[int] = []
        self._item_index: dict[int, int] = {}

    def fit(self, interactions: Any, items: Any) -> "ContentBasedRecommender":
        text_by_id: dict[int, str] = {}
        for row in items.iter_rows():
            iid = int(row["catalog_item_id"])
            text_by_id[iid] = _item_text(row)
        self._item_ids = sorted(text_by_id.keys())
        self._item_index = {iid: i for i, iid in enumerate(self._item_ids)}
        if not self._item_ids:
            self._vectorizer = None
            self._matrix = None
            return self
        corpus = [text_by_id[iid] for iid in self._item_ids]
        if not any(text.strip() for text in corpus):
            corpus = [f"item-{item_id}" for item_id in self._item_ids]
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )
        self._matrix = self._vectorizer.fit_transform(corpus)
        return self

    def _profile_vector(self, interactions: Any, user_id: int) -> np.ndarray | None:
        rows = list(interactions.rows_for_user(user_id))
        if not rows:
            return None
        weights = []
        indices = []
        for row in rows:
            iid = int(row["item_id"])
            if iid not in self._item_index:
                continue
            indices.append(self._item_index[iid])
            weights.append(float(row["weight"]))
        if not indices:
            return None
        rows_array = np.zeros(len(indices), dtype=np.int32)
        cols_array = np.array(indices, dtype=np.int32)
        data_array = np.array(weights, dtype=np.float32)
        profile = np.asarray(
            (self._matrix[cols_array].multiply(data_array[:, None])).sum(axis=0)
        ).ravel()
        if float(profile.sum()) <= 0:
            return None
        return profile

    def recommend(self, context: RecommendationContext) -> list[Recommendation]:
        if self._matrix is None:
            return []
        interactions = context.extra.get("interactions")
        profile = self._profile_vector(interactions, context.user_id) if interactions is not None else None
        if profile is None:
            return []
        sims = linear_kernel(profile.reshape(1, -1), self._matrix).ravel()
        order = np.argsort(-sims)
        out: list[Recommendation] = []
        for i in order:
            score = float(sims[i])
            if score <= 0:
                continue
            iid = self._item_ids[i]
            if iid in context.seen_item_ids:
                continue
            out.append(
                Recommendation(
                    item_id=iid,
                    score=score,
                    reason="Matches your recent interests",
                    components={"content_similarity": score},
                )
            )
            if len(out) >= context.top_k:
                break
        return out

    def supports_new_items(self) -> bool:
        return True
