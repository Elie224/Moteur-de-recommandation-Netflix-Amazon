"""Offline tests for the TMDB integration."""
from __future__ import annotations

import asyncio
import sys
from datetime import date
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "apps" / "api" / "src"), str(ROOT / "packages" / "recommendation_engine" / "src")]

from recommender_api.config import Settings
from recommender_api.integrations.tmdb.client import TMDBClient
from recommender_api.integrations.tmdb.exceptions import TMDBConfigurationError
from recommender_api.integrations.tmdb.mapper import map_tmdb_movie, select_trailer
from recommender_api.integrations.tmdb.schemas import TMDBMovieDetails


def settings() -> Settings:
    return Settings(tmdb_access_token="test-token", tmdb_base_url="https://tmdb.test")


def test_missing_token_is_clear():
    with pytest.raises(TMDBConfigurationError, match="RECOSPHERE_TMDB_ACCESS_TOKEN"):
        TMDBClient(Settings(tmdb_access_token=None))


def test_client_validates_trending_payload_without_network():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trending/movie/day"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={
                "page": 1,
                "results": [{"id": 550, "title": "Fight Club", "release_date": "1999-10-15", "genre_ids": [18]}],
                "total_pages": 1,
                "total_results": 1,
            },
        )

    async def run():
        async with TMDBClient(settings(), transport=httpx.MockTransport(handler)) as client:
            response = await client.trending_movies()
            assert response.results[0].id == 550
            assert response.results[0].release_date == date(1999, 10, 15)

    asyncio.run(run())


def test_mapper_prefers_official_trailer_and_builds_urls():
    movie = TMDBMovieDetails.model_validate(
        {
            "id": 1,
            "title": "Dune",
            "poster_path": "/poster.jpg",
            "backdrop_path": "/backdrop.jpg",
            "genres": [{"id": 878, "name": "Science-Fiction"}],
            "videos": {
                "results": [
                    {"key": "teaser", "site": "YouTube", "type": "Teaser"},
                    {"key": "trailer", "site": "YouTube", "type": "Trailer", "official": True},
                ]
            },
        }
    )
    normalized = map_tmdb_movie(movie, image_base_url="https://image.test/t/p", image_size="w500")
    assert normalized.image_url == "https://image.test/t/p/w500/poster.jpg"
    assert normalized.trailer_url == "https://www.youtube.com/watch?v=trailer"
    assert normalized.genres == ["Science-Fiction"]
    assert select_trailer(movie) == normalized.trailer_url
