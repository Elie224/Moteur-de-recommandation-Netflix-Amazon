"""Data package."""
from .loaders import (
    MovieLens1M,
    load_movielens_1m,
    load_movies,
    load_ratings,
    load_users,
    temporal_split,
)

__all__ = [
    "MovieLens1M",
    "load_movielens_1m",
    "load_movies",
    "load_ratings",
    "load_users",
    "temporal_split",
]
