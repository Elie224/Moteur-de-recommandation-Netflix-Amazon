"""Small asynchronous TMDB HTTP client."""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import ValidationError

from ...config import Settings
from .exceptions import TMDBConfigurationError, TMDBRateLimitError, TMDBRequestError
from .schemas import TMDBMovieDetails, TMDBMovieListResponse


class TMDBClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not settings.tmdb_access_token:
            raise TMDBConfigurationError("RECOSPHERE_TMDB_ACCESS_TOKEN is not configured.")
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.tmdb_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.tmdb_access_token}",
                "Accept": "application/json",
            },
            timeout=settings.tmdb_request_timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "TMDBClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> dict[str, Any]:
        request_params = {"language": self.settings.tmdb_language, **(params or {})}
        for attempt in range(retries):
            try:
                response = await self._client.get(path, params=request_params)
                if response.status_code == 429:
                    if attempt == retries - 1:
                        raise TMDBRateLimitError("TMDB rate limit reached.")
                    try:
                        delay = max(0, int(response.headers.get("Retry-After", "2")))
                    except ValueError:
                        delay = 2
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TMDBRequestError("TMDB returned a non-object JSON response.")
                return payload
            except TMDBRateLimitError:
                raise
            except httpx.HTTPStatusError as exc:
                raise TMDBRequestError(f"TMDB returned HTTP {exc.response.status_code}.") from exc
            except httpx.RequestError as exc:
                if attempt == retries - 1:
                    raise TMDBRequestError(f"Unable to contact TMDB: {exc}") from exc
                await asyncio.sleep(2**attempt)
            except ValueError as exc:
                raise TMDBRequestError("TMDB returned invalid JSON.") from exc
        raise TMDBRequestError("Unexpected TMDB request failure.")

    async def trending_movies(self, *, time_window: str = "day", page: int = 1) -> TMDBMovieListResponse:
        payload = await self._get(f"/trending/movie/{time_window}", params={"page": page})
        try:
            return TMDBMovieListResponse.model_validate(payload)
        except ValidationError as exc:
            raise TMDBRequestError("TMDB returned an invalid trending movie payload.") from exc

    async def popular_movies(self, *, page: int = 1) -> TMDBMovieListResponse:
        payload = await self._get("/movie/popular", params={"page": page, "region": self.settings.tmdb_region})
        try:
            return TMDBMovieListResponse.model_validate(payload)
        except ValidationError as exc:
            raise TMDBRequestError("TMDB returned an invalid popular movie payload.") from exc

    async def upcoming_movies(self, *, page: int = 1) -> TMDBMovieListResponse:
        payload = await self._get("/movie/upcoming", params={"page": page, "region": self.settings.tmdb_region})
        try:
            return TMDBMovieListResponse.model_validate(payload)
        except ValidationError as exc:
            raise TMDBRequestError("TMDB returned an invalid upcoming movie payload.") from exc

    async def movie_details(self, movie_id: int) -> TMDBMovieDetails:
        payload = await self._get(
            f"/movie/{movie_id}",
            params={"append_to_response": "videos", "include_video_language": "fr,en,null"},
        )
        try:
            return TMDBMovieDetails.model_validate(payload)
        except ValidationError as exc:
            raise TMDBRequestError(f"TMDB returned invalid details for movie {movie_id}.") from exc
