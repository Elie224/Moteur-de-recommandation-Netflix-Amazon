"""Application service for external catalog synchronizations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Iterable

from sqlalchemy.orm import Session

from ..config import Settings
from ..integrations.tmdb.client import TMDBClient
from ..integrations.tmdb.mapper import map_tmdb_movie
from ..models import SyncRun
from .catalog_service import attach_movie, upsert_catalog_item


@dataclass(slots=True)
class SyncSummary:
    source: str
    status: str
    received: int
    created: int
    updated: int
    failed: int
    sync_run_id: int


class TMDBSyncService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    async def sync(
        self,
        *,
        collections: Iterable[str] = ("trending", "popular", "upcoming"),
        max_pages: int | None = None,
    ) -> SyncSummary:
        page_limit = min(max_pages or self.settings.tmdb_sync_max_pages, self.settings.tmdb_sync_max_pages)
        async with TMDBClient(self.settings) as client:
            sync_run = SyncRun(source="tmdb", status="running")
            self.db.add(sync_run)
            self.db.commit()
            self.db.refresh(sync_run)
            try:
                summaries = await self._collect_summaries(client, collections, page_limit)
            except Exception as exc:
                self.db.rollback()
                failed_run = self.db.get(SyncRun, sync_run.id)
                if failed_run is not None:
                    failed_run.status = "failed"
                    failed_run.finished_at = datetime.now(timezone.utc)
                    failed_run.error_message = str(exc)
                    failed_run.error_details = [{"type": type(exc).__name__, "message": str(exc)}]
                    self.db.commit()
                raise
            errors: list[dict[str, str]] = []
            created = 0
            updated = 0
            for movie_summary in summaries:
                try:
                    details = await client.movie_details(movie_summary.id)
                    normalized = map_tmdb_movie(
                        details,
                        image_base_url=self.settings.tmdb_image_base_url,
                        image_size=self.settings.tmdb_image_size,
                    )
                    catalog_item, was_created = upsert_catalog_item(
                        self.db,
                        item_type="movie",
                        external_source=normalized.source,
                        external_id=normalized.external_id,
                        title=normalized.title,
                        description=normalized.description,
                        image_url=normalized.image_url,
                        detail_url=f"https://www.themoviedb.org/movie/{normalized.external_id}",
                        category=normalized.category,
                        language=normalized.language,
                        country=self.settings.tmdb_region,
                        popularity_score=normalized.popularity_score,
                        average_rating=normalized.average_rating,
                        rating_count=normalized.rating_count,
                        tags=normalized.genres,
                        extra_metadata={**normalized.raw_metadata, "backdrop_url": normalized.backdrop_url},
                        published_at=_as_datetime(normalized.published_at),
                    )
                    attach_movie(
                        self.db,
                        catalog_item.id,
                        release_date=_as_datetime(normalized.release_date),
                        runtime_minutes=normalized.runtime_minutes,
                        original_title=normalized.original_title,
                        original_language=normalized.language,
                        trailer_url=normalized.trailer_url,
                        genres=normalized.genres,
                    )
                    self.db.commit()
                    created += int(was_created)
                    updated += int(not was_created)
                except Exception as exc:  # isolate one bad upstream item
                    self.db.rollback()
                    errors.append({"external_id": str(movie_summary.id), "type": type(exc).__name__, "message": str(exc)})

            sync_run = self.db.get(SyncRun, sync_run.id)
            assert sync_run is not None
            sync_run.status = "completed"
            sync_run.finished_at = datetime.now(timezone.utc)
            sync_run.items_processed = len(summaries)
            sync_run.items_created = created
            sync_run.items_updated = updated
            sync_run.error_details = errors
            sync_run.error_message = f"{len(errors)} item(s) failed" if errors else None
            self.db.commit()
            return SyncSummary("tmdb", "completed", len(summaries), created, updated, len(errors), sync_run.id)

    async def _collect_summaries(self, client: TMDBClient, collections: Iterable[str], page_limit: int):
        seen: set[int] = set()
        summaries = []
        for collection in collections:
            for page in range(1, page_limit + 1):
                if collection == "trending":
                    response = await client.trending_movies(page=page)
                elif collection == "popular":
                    response = await client.popular_movies(page=page)
                elif collection == "upcoming":
                    response = await client.upcoming_movies(page=page)
                else:
                    raise ValueError(f"Unsupported TMDB collection: {collection}")
                for movie in response.results:
                    if movie.id not in seen:
                        seen.add(movie.id)
                        summaries.append(movie)
                if page >= response.total_pages:
                    break
        return summaries


def _as_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time.min)
