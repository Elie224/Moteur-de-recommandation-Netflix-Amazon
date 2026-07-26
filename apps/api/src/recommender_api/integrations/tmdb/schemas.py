"""Validated TMDB response models used by the integration."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TMDBGenre(BaseModel):
    id: int
    name: str


class TMDBMovieSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    title: str
    original_title: str | None = None
    overview: str | None = None
    poster_path: str | None = None
    backdrop_path: str | None = None
    release_date: date | None = None
    genre_ids: list[int] = Field(default_factory=list)
    original_language: str | None = None
    popularity: float = 0
    vote_average: float = 0
    vote_count: int = 0
    adult: bool = False

    @field_validator("release_date", mode="before")
    @classmethod
    def empty_release_date_is_none(cls, value):
        return None if value == "" else value


class TMDBMovieListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page: int
    results: list[TMDBMovieSummary]
    total_pages: int
    total_results: int


class TMDBVideo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    site: str
    type: str
    official: bool = False
    iso_639_1: str | None = None


class TMDBVideoResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[TMDBVideo] = Field(default_factory=list)


class TMDBMovieDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    title: str
    original_title: str | None = None
    overview: str | None = None
    poster_path: str | None = None
    backdrop_path: str | None = None
    release_date: date | None = None
    runtime: int | None = None
    original_language: str | None = None
    popularity: float = 0
    vote_average: float = 0
    vote_count: int = 0
    genres: list[TMDBGenre] = Field(default_factory=list)
    videos: TMDBVideoResponse | None = None

    @field_validator("release_date", mode="before")
    @classmethod
    def empty_release_date_is_none(cls, value):
        return None if value == "" else value
