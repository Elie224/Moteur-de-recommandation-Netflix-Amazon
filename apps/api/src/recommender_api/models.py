"""SQLAlchemy ORM models for the unified catalog.

The catalog is split into three layers:
- ``CatalogItem`` is the normalized entry each source maps to.
- ``Movie`` and ``Product`` hold the domain-specific metadata.
- ``Interaction`` and ``Favorite`` keep the user signal.

User accounts live in ``User`` and preferences in ``UserPreference``.
``SyncRun`` records the latest state of each external integration.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

from .database import Base


ID_TYPE = BigInteger().with_variant(Integer, "sqlite")
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(150))
    preferred_language: Mapped[str] = mapped_column(String(10), default="fr")
    country: Mapped[str] = mapped_column(String(10), default="FR")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    interactions: Mapped[list["Interaction"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    favorites: Mapped[list["Favorite"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    preferences: Mapped[list["UserPreference"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class CatalogItem(Base):
    __tablename__ = "catalog_items"
    __table_args__ = (UniqueConstraint("external_source", "external_id", name="uq_catalog_source_external"),)

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    item_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    external_source: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    image_url: Mapped[str | None] = mapped_column(Text)
    detail_url: Mapped[str | None] = mapped_column(Text)

    category: Mapped[str | None] = mapped_column(String(255), index=True)
    language: Mapped[str | None] = mapped_column(String(20))
    country: Mapped[str | None] = mapped_column(String(20))

    popularity_score: Mapped[float] = mapped_column(Float, default=0.0)
    average_rating: Mapped[float | None] = mapped_column(Float)
    rating_count: Mapped[int] = mapped_column(Integer, default=0)

    tags: Mapped[list[Any]] = mapped_column(JSON_TYPE, default=list)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    synchronized_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    movie: Mapped["Movie | None"] = relationship(back_populates="catalog_item", cascade="all, delete-orphan", uselist=False)
    product: Mapped["Product | None"] = relationship(back_populates="catalog_item", cascade="all, delete-orphan", uselist=False)
    interactions: Mapped[list["Interaction"]] = relationship(back_populates="catalog_item", cascade="all, delete-orphan")


class Movie(Base):
    __tablename__ = "movies"

    catalog_item_id: Mapped[int] = mapped_column(
        ID_TYPE, ForeignKey("catalog_items.id", ondelete="CASCADE"), primary_key=True
    )
    release_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    runtime_minutes: Mapped[int | None] = mapped_column(Integer)
    original_title: Mapped[str | None] = mapped_column(String(500))
    original_language: Mapped[str | None] = mapped_column(String(20))
    trailer_url: Mapped[str | None] = mapped_column(Text)
    genres: Mapped[list[Any]] = mapped_column(JSON_TYPE, default=list)
    cast_members: Mapped[list[Any]] = mapped_column(JSON_TYPE, default=list)
    directors: Mapped[list[Any]] = mapped_column(JSON_TYPE, default=list)
    watch_providers: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)

    catalog_item: Mapped[CatalogItem] = relationship(back_populates="movie")


class Product(Base):
    __tablename__ = "products"

    catalog_item_id: Mapped[int] = mapped_column(
        ID_TYPE, ForeignKey("catalog_items.id", ondelete="CASCADE"), primary_key=True
    )
    price_amount: Mapped[float | None] = mapped_column(Numeric(12, 2))
    price_currency: Mapped[str | None] = mapped_column(String(10))
    condition: Mapped[str | None] = mapped_column(String(100))
    brand: Mapped[str | None] = mapped_column(String(255), index=True)
    seller_name: Mapped[str | None] = mapped_column(String(255))
    availability: Mapped[str | None] = mapped_column(String(100), index=True)
    shipping_cost: Mapped[float | None] = mapped_column(Numeric(12, 2))
    marketplace: Mapped[str | None] = mapped_column(String(50), index=True)
    product_url: Mapped[str | None] = mapped_column(Text)
    condition_description: Mapped[str | None] = mapped_column(Text)
    seller_feedback_percentage: Mapped[float | None] = mapped_column(Numeric(5, 2))
    seller_feedback_score: Mapped[int | None] = mapped_column(BigInteger)
    shipping_currency: Mapped[str | None] = mapped_column(String(10))
    additional_images: Mapped[list[Any]] = mapped_column(JSON_TYPE, default=list)
    item_end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    catalog_item: Mapped[CatalogItem] = relationship(back_populates="product")


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    user_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    catalog_item_id: Mapped[int] = mapped_column(
        ID_TYPE, ForeignKey("catalog_items.id", ondelete="CASCADE"), index=True
    )

    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    event_value: Mapped[float | None] = mapped_column(Float)
    source_page: Mapped[str | None] = mapped_column(String(100))
    recommendation_id: Mapped[int | None] = mapped_column(BigInteger)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    user: Mapped[User] = relationship(back_populates="interactions")
    catalog_item: Mapped[CatalogItem] = relationship(back_populates="interactions")


class Favorite(Base):
    __tablename__ = "favorites"
    __table_args__ = (UniqueConstraint("user_id", "catalog_item_id", name="uq_favorite_user_item"),)

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    user_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    catalog_item_id: Mapped[int] = mapped_column(
        ID_TYPE, ForeignKey("catalog_items.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="favorites")
    catalog_item: Mapped[CatalogItem] = relationship()


class UserPreference(Base):
    __tablename__ = "user_preferences"
    __table_args__ = (UniqueConstraint("user_id", "preference_key", name="uq_preference_user_key"),)

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    user_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    preference_key: Mapped[str] = mapped_column(String(100), nullable=False)
    preference_value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship(back_populates="preferences")


class RecommendationBatch(Base):
    __tablename__ = "recommendation_batches"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    user_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    item_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    context: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    results: Mapped[list["RecommendationResult"]] = relationship(back_populates="batch", cascade="all, delete-orphan")


class RecommendationResult(Base):
    __tablename__ = "recommendation_results"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ID_TYPE, ForeignKey("recommendation_batches.id", ondelete="CASCADE"), index=True
    )
    catalog_item_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("catalog_items.id", ondelete="CASCADE"))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    components: Mapped[dict[str, float]] = mapped_column(JSON_TYPE, default=dict)

    batch: Mapped[RecommendationBatch] = relationship(back_populates="results")
    catalog_item: Mapped[CatalogItem] = relationship()


class SyncRun(Base):
    __tablename__ = "sync_runs"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    source: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    items_processed: Mapped[int] = mapped_column(Integer, default=0)
    items_created: Mapped[int] = mapped_column(Integer, default=0)
    items_updated: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[list[Any]] = mapped_column(JSON_TYPE, default=list)


class SourceLink(Base):
    __tablename__ = "source_links"
    __table_args__ = (UniqueConstraint("source", "external_id", name="uq_mapping_source_external"),)

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True)
    catalog_item_id: Mapped[int] = mapped_column(
        ID_TYPE, ForeignKey("catalog_items.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
