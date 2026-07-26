"""User interaction tracking."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Interaction, UserPreference


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


def recent_items_for(db: Session, user_id: int, limit: int = 20) -> list[int]:
    stmt = (
        select(Interaction.catalog_item_id)
        .where(Interaction.user_id == user_id)
        .order_by(Interaction.created_at.desc())
        .limit(limit)
    )
    return [row[0] for row in db.execute(stmt).all()]


def seen_items_for(db: Session, user_id: int) -> set[int]:
    stmt = select(Interaction.catalog_item_id).where(Interaction.user_id == user_id).distinct()
    return {row[0] for row in db.execute(stmt).all()}


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
