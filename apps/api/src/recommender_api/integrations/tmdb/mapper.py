"""Pure TMDB-to-catalog normalization functions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from .schemas import TMDBMovieDetails, TMDBVideo


@dataclass(slots=True)
class NormalizedMovie:
    source: str
    external_id: str
    title: str
    description: str | None
    image_url: str | None
    backdrop_url: str | None
    category: str
    language: str | None
    popularity_score: float
    average_rating: float | None
    rating_count: int
    published_at: date | None
    original_title: str | None
    release_date: date | None
    runtime_minutes: int | None
    genres: list[str]
    trailer_url: str | None
    raw_metadata: dict[str, Any]


def build_image_url(path: str | None, *, base_url: str, size: str) -> str | None:
    if not path:
        return None
    return f"{base_url.rstrip('/')}/{size.strip('/')}{path}"


def select_trailer(movie: TMDBMovieDetails) -> str | None:
    videos = movie.videos.results if movie.videos else []
    candidates = [video for video in videos if video.site.lower() == "youtube" and video.type in {"Trailer", "Teaser"}]
    if not candidates:
        return None
    official = next((video for video in candidates if video.official), candidates[0])
    return f"https://www.youtube.com/watch?v={official.key}"


def map_tmdb_movie(
    movie: TMDBMovieDetails,
    *,
    image_base_url: str,
    image_size: str,
) -> NormalizedMovie:
    return NormalizedMovie(
        source="tmdb",
        external_id=str(movie.id),
        title=movie.title,
        description=movie.overview,
        image_url=build_image_url(movie.poster_path, base_url=image_base_url, size=image_size),
        backdrop_url=build_image_url(movie.backdrop_path, base_url=image_base_url, size="original"),
        category="movie",
        language=movie.original_language,
        popularity_score=movie.popularity,
        average_rating=movie.vote_average,
        rating_count=movie.vote_count,
        published_at=movie.release_date,
        original_title=movie.original_title,
        release_date=movie.release_date,
        runtime_minutes=movie.runtime,
        genres=[genre.name for genre in movie.genres],
        trailer_url=select_trailer(movie),
        raw_metadata=movie.model_dump(mode="json"),
    )
