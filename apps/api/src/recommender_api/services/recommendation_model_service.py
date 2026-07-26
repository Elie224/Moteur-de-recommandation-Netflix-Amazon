"""In-process cache for fitted recommendation models."""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic

from sqlalchemy.orm import Session

from recommendation_engine import BaseRecommender

from ..config import get_settings


@dataclass(slots=True)
class _CachedModel:
    model: BaseRecommender
    created_at: float


class RecommendationModelService:
    """Keep one fitted model per domain for a bounded period.

    The cache is process-local by design. It removes repeated fitting from
    normal recommendation requests while keeping an explicit invalidation hook
    for catalog and interaction changes. A shared cache can replace this
    service later without changing the API route contract.
    """

    def __init__(self, ttl_seconds: float | None = None) -> None:
        self._ttl_override = ttl_seconds
        self._models: dict[str, _CachedModel] = {}
        self._lock = RLock()

    @property
    def ttl_seconds(self) -> float:
        if self._ttl_override is not None:
            return self._ttl_override
        return get_settings().recommendation_model_cache_ttl_seconds

    def get_model(self, db: Session, item_type: str) -> BaseRecommender:
        now = monotonic()
        with self._lock:
            cached = self._models.get(item_type)
            if cached is not None and now - cached.created_at < self.ttl_seconds:
                return cached.model
            return self._fit_and_store(db, item_type, now)

    def refresh_model(self, db: Session, item_type: str) -> BaseRecommender:
        with self._lock:
            return self._fit_and_store(db, item_type, monotonic())

    def invalidate(self, item_type: str | None = None) -> None:
        with self._lock:
            if item_type is None:
                self._models.clear()
            else:
                self._models.pop(item_type, None)

    def _fit_and_store(self, db: Session, item_type: str, created_at: float) -> BaseRecommender:
        from .recommendation_service import fit_recommender_for_domain

        model = fit_recommender_for_domain(db, item_type)
        self._models[item_type] = _CachedModel(model=model, created_at=created_at)
        return model


recommendation_model_service = RecommendationModelService()
