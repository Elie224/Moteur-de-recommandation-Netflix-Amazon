"""Common interface for all recommenders in the engine."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Recommendation:
    """A single ranked suggestion returned by a recommender."""

    item_id: int
    score: float
    reason: str = ""
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass
class RecommendationContext:
    """Everything a recommender might need when scoring items."""

    user_id: int
    item_type: str  # "movie" or "product"
    top_k: int = 20
    seen_item_ids: set[int] = field(default_factory=set)
    excluded_item_ids: set[int] | None = None
    favorite_item_ids: set[int] = field(default_factory=set)
    recent_item_ids: list[int] = field(default_factory=list)
    preferences: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def is_excluded(self, item_id: int) -> bool:
        """Return whether an item must be omitted from recommendations.

        Older callers only provide ``seen_item_ids``; those contexts retain
        the previous exclusion behavior. API contexts can provide an empty
        ``excluded_item_ids`` to keep views and impressions as soft signals.
        """
        excluded = self.seen_item_ids if self.excluded_item_ids is None else self.excluded_item_ids
        return item_id in excluded


class BaseRecommender(ABC):
    """Contract that every recommender in the engine must implement."""

    name: str = "base"

    @abstractmethod
    def fit(self, interactions, items):  # type: ignore[no-untyped-def]
        """Train the model from a unified interaction/item frame."""

    @abstractmethod
    def recommend(self, context: RecommendationContext) -> list[Recommendation]:
        """Return the top ``top_k`` recommendations for the given context."""

    def supports_cold_start(self) -> bool:
        return False

    def supports_new_items(self) -> bool:
        return False


class InteractionFrame(ABC):  # pragma: no cover - structural typing only
    """Minimal interface over the interactions table."""

    @abstractmethod
    def user_ids(self) -> Iterable[int]: ...

    @abstractmethod
    def item_ids(self) -> Iterable[int]: ...

    @abstractmethod
    def rows_for_user(self, user_id: int) -> Iterable[Mapping[str, Any]]: ...

    @abstractmethod
    def weighted_rows(self) -> Iterable[Mapping[str, Any]]: ...


class ItemFrame(ABC):  # pragma: no cover - structural typing only
    """Minimal interface over the unified catalog."""

    @abstractmethod
    def item_ids(self) -> Iterable[int]: ...

    @abstractmethod
    def metadata(self, item_id: int) -> Mapping[str, Any]: ...

    @abstractmethod
    def iter_rows(self) -> Iterable[Mapping[str, Any]]: ...
