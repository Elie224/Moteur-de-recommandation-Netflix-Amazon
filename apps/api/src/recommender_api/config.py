"""Runtime configuration for the RecoSphere API.

Reads from environment variables (or a ``.env`` file) so the same image
works in dev, CI, and production. ``RECOSPHERE_DATABASE_URL`` defaults to a
local PostgreSQL database; SQLite is reserved for isolated tests.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATABASE_URL = "postgresql+psycopg://recosphere:recosphere@localhost:5432/recosphere"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="RECOSPHERE_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "RecoSphere API"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    auto_create_schema: bool = Field(default=False)
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173")

    database_url: str = Field(default=DEFAULT_DATABASE_URL)

    jwt_secret: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_ttl_minutes: int = Field(default=60 * 24 * 7)

    ai_provider: str = Field(default="anthropic")
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-3-5-haiku-latest")

    tmdb_access_token: str | None = Field(default=None)
    tmdb_base_url: str = Field(default="https://api.themoviedb.org/3")
    tmdb_image_base_url: str = Field(default="https://image.tmdb.org/t/p")
    tmdb_image_size: str = Field(default="w500")
    tmdb_language: str = Field(default="fr-FR")
    tmdb_region: str = Field(default="FR")
    tmdb_sync_max_pages: int = Field(default=5, ge=1, le=50)
    tmdb_request_timeout: float = Field(default=20.0, gt=0)
    tmdb_dry_run: bool = Field(default=True)

    ebay_environment: Literal["sandbox", "production"] = "sandbox"
    ebay_client_id: str | None = Field(default=None)
    ebay_client_secret: str | None = Field(default=None)
    ebay_marketplace_id: str = Field(default="EBAY_FR")
    ebay_language: str = Field(default="fr-FR")
    ebay_sandbox_api_base_url: str = Field(default="https://api.sandbox.ebay.com")
    ebay_production_api_base_url: str = Field(default="https://api.ebay.com")
    ebay_sandbox_auth_url: str = Field(default="https://api.sandbox.ebay.com/identity/v1/oauth2/token")
    ebay_production_auth_url: str = Field(default="https://api.ebay.com/identity/v1/oauth2/token")
    ebay_request_timeout: float = Field(default=20.0, gt=0)
    ebay_sync_limit: int = Field(default=50, ge=1, le=200)
    ebay_sync_max_pages: int = Field(default=3, ge=1, le=20)
    ebay_concurrency: int = Field(default=5, ge=1, le=20)
    ebay_marketplace: str = Field(default="EBAY_FR")
    ebay_dry_run: bool = Field(default=True)

    @property
    def ebay_api_base_url(self) -> str:
        return self.ebay_production_api_base_url if self.ebay_environment == "production" else self.ebay_sandbox_api_base_url

    @property
    def ebay_auth_url(self) -> str:
        return self.ebay_production_auth_url if self.ebay_environment == "production" else self.ebay_sandbox_auth_url

    recommendation_default_top_k: int = Field(default=20)
    recommendation_max_top_k: int = Field(default=100)
    recommendation_model_cache_ttl_seconds: float = Field(default=600.0, ge=0)
    popularity_min_interactions: int = Field(default=3)
    recent_activity_window: int = Field(default=20)
    recent_activity_half_life_hours: float = Field(default=24.0)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
