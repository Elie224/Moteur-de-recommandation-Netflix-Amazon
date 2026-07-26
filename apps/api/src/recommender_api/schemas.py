"""Pydantic schemas for the public API surface."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------------------------------------------------------------- auth ---

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=150)
    preferred_language: str | None = Field(default="fr", max_length=10)
    country: str | None = Field(default="FR", max_length=10)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    display_name: str | None
    preferred_language: str
    country: str
    is_admin: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic


# -------------------------------------------------------- preferences ---

class UserPreferenceIn(BaseModel):
    preference_key: str = Field(max_length=100)
    preference_value: str


class UserPreferenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    preference_key: str
    preference_value: str


class OnboardingIn(BaseModel):
    favorite_movie_ids: list[int] = Field(default_factory=list)
    preferred_genres: list[str] = Field(default_factory=list)
    preferred_product_categories: list[str] = Field(default_factory=list)
    budget_currency: str | None = Field(default="EUR", max_length=10)
    budget_max: float | None = None


# ------------------------------------------------------------ catalog ---

class CatalogItemBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    item_type: Literal["movie", "product"]
    title: str
    description: str | None
    image_url: str | None
    detail_url: str | None
    category: str | None
    language: str | None
    country: str | None
    popularity_score: float
    average_rating: float | None
    rating_count: int
    is_active: bool
    published_at: datetime | None
    price_amount: float | None = None
    price_currency: str | None = None
    condition: str | None = None
    brand: str | None = None
    seller_name: str | None = None
    availability: str | None = None
    shipping_cost: float | None = None
    marketplace: str | None = None
    product_url: str | None = None


class MovieOut(CatalogItemBase):
    release_date: datetime | None
    runtime_minutes: int | None
    original_title: str | None
    original_language: str | None
    trailer_url: str | None
    genres: list[str]
    cast_members: list[str]
    directors: list[str]
    watch_providers: dict[str, Any]


class ProductOut(CatalogItemBase):
    price_amount: float | None
    price_currency: str | None
    condition: str | None
    brand: str | None
    seller_name: str | None
    availability: str | None
    shipping_cost: float | None
    marketplace: str | None
    product_url: str | None
    condition_description: str | None = None
    seller_feedback_percentage: float | None = None
    seller_feedback_score: int | None = None
    shipping_currency: str | None = None
    additional_images: list[str] = Field(default_factory=list)
    item_end_date: datetime | None = None


# -------------------------------------------------------- interactions ---

class InteractionCreate(BaseModel):
    catalog_item_id: int
    event_type: str = Field(min_length=1, max_length=50)
    event_value: float | None = None
    source_page: str | None = Field(default=None, max_length=100)
    recommendation_id: int | None = None
    session_id: str | None = Field(default=None, max_length=64)


class InteractionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    catalog_item_id: int
    event_type: str
    event_value: float | None
    source_page: str | None
    recommendation_id: int | None
    session_id: str | None
    created_at: datetime


# ---------------------------------------------------- recommendations ---

class RecommendationOut(BaseModel):
    catalog_item_id: int
    title: str
    image_url: str | None
    score: float
    reason: str | None
    components: dict[str, float]

    item_type: Literal["movie", "product"]
    detail: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    user_id: int
    item_type: Literal["movie", "product"]
    model_version: str
    top_k: int
    recommendations: list[RecommendationOut]


# ----------------------------------------------------------- favorites ---

class FavoriteToggleResponse(BaseModel):
    catalog_item_id: int
    is_favorite: bool


# ------------------------------------------------------------- admin ---

class SyncRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    items_processed: int
    items_created: int
    items_updated: int
    error_message: str | None


class AdminMetrics(BaseModel):
    users: int
    movies: int
    products: int
    interactions: int
    favorites: int
    last_sync: dict[str, SyncRunOut]


class EbaySyncRequest(BaseModel):
    categories: list[Literal["laptops", "smartphones", "headphones"]] = Field(default_factory=list)
    max_pages: int = Field(default=1, ge=1, le=20)
