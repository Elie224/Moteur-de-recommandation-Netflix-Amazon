"""Bridge between SQLAlchemy and the recommendation engine."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from recommendation_engine import (
    BaseRecommender,
    InteractionFrame,
    ItemFrame,
    Recommendation,
    RecommendationContext,
    get_default_registry,
)

from ..config import get_settings
from ..models import CatalogItem, Interaction
from .recommendation_model_service import recommendation_model_service


class DBInteractionFrame(InteractionFrame):
    """Loads interactions from SQL and projects them into the engine frame."""

    def __init__(self, db: Session, weights: Mapping[str, float], item_type: str | None = None):
        self.db = db
        self._weights = dict(weights)
        self.item_type = item_type
        self._cache: list[dict[str, Any]] | None = None
        self._rows_by_user: dict[int, list[dict[str, Any]]] = {}

    def user_ids(self) -> Iterable[int]:
        return {row["user_id"] for row in self.weighted_rows()}

    def item_ids(self) -> Iterable[int]:
        return {row["item_id"] for row in self.weighted_rows()}

    def rows_for_user(self, user_id: int) -> Iterable[Mapping[str, Any]]:
        if user_id not in self._rows_by_user:
            self._rows_by_user[user_id] = [
                row for row in self.weighted_rows() if row["user_id"] == user_id
            ]
        return iter(self._rows_by_user[user_id])

    def weighted_rows(self) -> list[dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        stmt = select(
            Interaction.user_id,
            Interaction.catalog_item_id,
            Interaction.event_type,
            Interaction.event_value,
            CatalogItem.item_type,
        )
        stmt = stmt.join(CatalogItem, CatalogItem.id == Interaction.catalog_item_id).where(
            CatalogItem.is_active.is_(True)
        )
        if self.item_type:
            stmt = stmt.where(CatalogItem.item_type == self.item_type)
        rows = self.db.execute(stmt).all()
        out: list[dict[str, Any]] = []
        for r in rows:
            weight = float(r.event_value) if r.event_value is not None else self._weights.get(r.event_type, 0.0)
            out.append({
                "user_id": int(r.user_id),
                "item_id": int(r.catalog_item_id),
                "item_type": r.item_type,
                "weight": weight,
                "event_type": r.event_type,
            })
        self._cache = out
        self._rows_by_user = {}
        return out


class DBItemFrame(ItemFrame):
    """Loads catalog items from SQL into the engine frame."""

    def __init__(self, db: Session, item_type: str | None = None):
        self.db = db
        self.item_type = item_type
        self._items: list[dict[str, Any]] | None = None
        self._by_id: dict[int, dict[str, Any]] = {}

    def item_ids(self) -> Iterable[int]:
        return [row["catalog_item_id"] for row in self.iter_rows()]

    def metadata(self, item_id: int) -> Mapping[str, Any]:
        self.iter_rows()
        return self._by_id.get(item_id, {})

    def iter_rows(self) -> list[dict[str, Any]]:
        if self._items is not None:
            return self._items
        stmt = select(
            CatalogItem.id,
            CatalogItem.item_type,
            CatalogItem.title,
            CatalogItem.description,
            CatalogItem.category,
            CatalogItem.tags,
            CatalogItem.language,
            CatalogItem.country,
            CatalogItem.popularity_score,
        )
        stmt = stmt.where(CatalogItem.is_active.is_(True))
        if self.item_type:
            stmt = stmt.where(CatalogItem.item_type == self.item_type)
        rows = self.db.execute(stmt).all()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "catalog_item_id": int(r.id),
                "item_type": r.item_type,
                "title": r.title or "",
                "description": r.description or "",
                "category": r.category or "",
                "tags": list(r.tags or []),
                "language": r.language or "",
                "country": r.country or "",
                "popularity_score": float(r.popularity_score or 0.0),
            })
        self._items = out
        self._by_id = {row["catalog_item_id"]: row for row in out}
        return out


def build_recommender_registry(db: Session) -> dict[str, BaseRecommender]:
    settings = get_settings()
    from recommendation_engine.models import ContentBasedRecommender
    from recommendation_engine.models import ItemItemCosineRecommender
    from recommendation_engine.models import PopularityRecommender
    from recommendation_engine.models import RecentActivityRecommender

    registry = get_default_registry()
    registry.register(PopularityRecommender(min_interactions=settings.popularity_min_interactions))
    registry.register(ItemItemCosineRecommender())
    registry.register(ContentBasedRecommender())
    registry.register(
        RecentActivityRecommender(
            half_life_hours=settings.recent_activity_half_life_hours,
            max_recent=settings.recent_activity_window,
        )
    )
    return registry.as_dict()


def fit_recommender_for_domain(db: Session, item_type: str) -> BaseRecommender:
    """Fit a hybrid recommender combining all sub-models for the given domain."""
    from recommendation_engine.models import HybridRecommender

    registry = build_recommender_registry(db)
    interactions = DBInteractionFrame(db, weights=_event_weights(), item_type=item_type)
    items = DBItemFrame(db, item_type=item_type)
    hybrid = HybridRecommender()
    for name in ("popularity", "item_cosine", "content_based", "recent_activity"):
        if name in registry:
            hybrid.add(registry[name])
    hybrid.fit(interactions, items)
    return hybrid


def build_recommendation_context(db: Session, user_id: int, item_type: str, top_k: int, preferences: Mapping[str, Any]) -> RecommendationContext:
    from .interaction_service import (
        disliked_items_for,
        excluded_items_for,
        favorite_items_for,
        recent_items_for,
        seen_items_for,
    )

    favorite_item_ids = favorite_items_for(db, user_id, item_type)

    return RecommendationContext(
        user_id=user_id,
        item_type=item_type,
        top_k=top_k,
        seen_item_ids=seen_items_for(db, user_id, item_type),
        excluded_item_ids=excluded_items_for(db, user_id, item_type),
        favorite_item_ids=favorite_item_ids,
        recent_item_ids=recent_items_for(db, user_id, item_type=item_type),
        preferences=dict(preferences),
        extra={
            "interactions": DBInteractionFrame(db, _event_weights(), item_type=item_type),
            "disliked_item_ids": disliked_items_for(db, user_id, item_type),
        },
    )


def _event_weights() -> dict[str, float]:
    from .interaction_service import EVENT_WEIGHTS
    return EVENT_WEIGHTS
