"""User interaction tracking."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import CatalogItem, Favorite, Interaction, UserPreference


EVENT_WEIGHTS: dict[str, float] = {
    "impression": 0.05,
    "view": 0.20,
    "click": 0.50,
    "search_click": 0.75,
    "trailer_play": 1.50,
    "favorite": 3.00,
    "like": 4.00,
    "rating": 1.00,
    "provider_click": 2.00,
    "purchase_redirect": 5.00,
    "dislike": -4.00,
}

VALID_EVENT_TYPES = set(EVENT_WEIGHTS)
STRONG_POSITIVE_EVENTS = frozenset({"like", "favorite", "rating", "purchase_redirect"})
STRONG_NEGATIVE_EVENTS = frozenset({"dislike"})
STRONG_EVENTS = STRONG_POSITIVE_EVENTS | STRONG_NEGATIVE_EVENTS


def weight_for(event_type: str) -> float:
    return EVENT_WEIGHTS.get(event_type, 0.0)


def record_interaction(db: Session, *, user, catalog_item_id: int, event_type: str, event_value=None, source_page=None, recommendation_id=None, session_id=None) -> Interaction:
    interaction = Interaction(
        user_id=user.id,
        catalog_item_id=catalog_item_id,
        event_type=event_type,
        event_value=event_value,
        source_page=source_page,
        recommendation_id=recommendation_id,
        session_id=session_id,
    )
    db.add(interaction)
    db.flush()
    return interaction


def recent_items_for(
    db: Session,
    user_id: int,
    limit: int = 20,
    item_type: str | None = None,
) -> list[int]:
    stmt = (
        select(Interaction.catalog_item_id)
        .join(CatalogItem, CatalogItem.id == Interaction.catalog_item_id)
        .where(Interaction.user_id == user_id)
        .where(Interaction.event_type != "impression")
        .where(CatalogItem.is_active.is_(True))
        .order_by(Interaction.created_at.desc())
        .limit(limit)
    )
    if item_type:
        stmt = stmt.where(CatalogItem.item_type == item_type)
    return [row[0] for row in db.execute(stmt).all()]


def _latest_strong_events_for(
    db: Session,
    user_id: int,
    item_type: str | None = None,
) -> dict[int, str]:
    stmt = (
        select(Interaction.catalog_item_id, Interaction.event_type)
        .join(CatalogItem, CatalogItem.id == Interaction.catalog_item_id)
        .where(
            Interaction.user_id == user_id,
            Interaction.event_type.in_(STRONG_EVENTS),
            CatalogItem.is_active.is_(True),
        )
        .order_by(Interaction.created_at.desc(), Interaction.id.desc())
    )
    if item_type:
        stmt = stmt.where(CatalogItem.item_type == item_type)

    latest: dict[int, str] = {}
    for item_id, event_type in db.execute(stmt).all():
        latest.setdefault(int(item_id), event_type)
    return latest


def favorite_items_for(
    db: Session,
    user_id: int,
    item_type: str | None = None,
) -> set[int]:
    stmt = (
        select(Favorite.catalog_item_id)
        .join(CatalogItem, CatalogItem.id == Favorite.catalog_item_id)
        .where(
            Favorite.user_id == user_id,
            CatalogItem.is_active.is_(True),
        )
    )
    if item_type:
        stmt = stmt.where(CatalogItem.item_type == item_type)
    return {int(item_id) for item_id in db.execute(stmt).scalars()}


def disliked_items_for(
    db: Session,
    user_id: int,
    item_type: str | None = None,
) -> set[int]:
    latest = _latest_strong_events_for(db, user_id, item_type)
    return {item_id for item_id, event_type in latest.items() if event_type in STRONG_NEGATIVE_EVENTS}


def seen_items_for(
    db: Session,
    user_id: int,
    item_type: str | None = None,
) -> set[int]:
    """Return hard exclusions, not every item merely viewed or impressed."""
    latest = _latest_strong_events_for(db, user_id, item_type)
    seen = set(latest)
    seen.update(favorite_items_for(db, user_id, item_type))
    return seen


def list_interactions_for(db: Session, user_id: int, limit: int = 100) -> list[Interaction]:
    stmt = (
        select(Interaction)
        .where(Interaction.user_id == user_id)
        .order_by(Interaction.created_at.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())


def save_preferences(db: Session, user, preferences: dict[str, Any]) -> None:
    for key, value in preferences.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            stored = ",".join(str(v) for v in value)
        else:
            stored = str(value)
        existing = db.execute(
            select(UserPreference).where(
                UserPreference.user_id == user.id,
                UserPreference.preference_key == key,
            )
        ).scalar_one_or_none()
        if existing is None:
            db.add(UserPreference(user_id=user.id, preference_key=key, preference_value=stored))
        else:
            existing.preference_value = stored
            existing.updated_at = datetime.now(timezone.utc)


def load_preferences(db: Session, user_id: int) -> dict[str, Any]:
    stmt = select(UserPreference).where(UserPreference.user_id == user_id)
    out: dict[str, Any] = {}
    for row in db.execute(stmt).scalars().all():
        value = row.preference_value
        if "," in value:
            out[row.preference_key] = [v for v in value.split(",") if v]
        else:
            out[row.preference_key] = value
    return out
