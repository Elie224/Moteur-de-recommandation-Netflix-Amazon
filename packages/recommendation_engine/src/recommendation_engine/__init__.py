"""Generic recommendation engine for the RecoSphere platform.

Every recommender in the engine implements the same :class:`BaseRecommender`
interface so the API layer can route between domains (movies, products, ...)
without coupling to a specific algorithm.
"""

from .base import (
    BaseRecommender,
    InteractionFrame,
    ItemFrame,
    Recommendation,
    RecommendationContext,
)
from .registry import ModelRegistry, get_default_registry

__all__ = [
    "BaseRecommender",
    "InteractionFrame",
    "ItemFrame",
    "Recommendation",
    "RecommendationContext",
    "ModelRegistry",
    "get_default_registry",
]
