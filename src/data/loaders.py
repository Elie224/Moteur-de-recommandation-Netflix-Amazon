"""Loaders for MovieLens 1M ratings, users, and movies.

Files layout after extraction:
- ml-1m/ratings.dat   :: UserID::MovieID::Rating::Timestamp (TAB separated in source, "::")
- ml-1m/users.dat     :: UserID::Gender::Age::Occupation::Zip-code
- ml-1m/movies.dat    :: MovieID::Title::Genres
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLS = ["user_id", "gender", "age", "occupation", "zip_code"]
MOVIES_COLS = ["movie_id", "title", "genres"]


@dataclass(frozen=True)
class MovieLens1M:
    ratings: pd.DataFrame
    users: pd.DataFrame
    movies: pd.DataFrame

    @property
    def n_ratings(self) -> int:
        return len(self.ratings)

    @property
    def n_users(self) -> int:
        return self.ratings["user_id"].nunique()

    @property
    def n_items(self) -> int:
        return self.ratings["movie_id"].nunique()

    @property
    def sparsity(self) -> float:
        return 1.0 - self.n_ratings / (self.n_users * self.n_items)


def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=RATINGS_COLS,
        encoding="latin-1",
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_users(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=USERS_COLS,
        encoding="latin-1",
    )


def load_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=MOVIES_COLS,
        encoding="latin-1",
    )
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$").astype("Int64")
    df["genres_list"] = df["genres"].str.split("|")
    return df


def load_movielens_1m(root: Path) -> MovieLens1M:
    """Load the full MovieLens 1M dataset from an extracted ``ml-1m`` folder."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run: python src/data/download_data.py"
        )
    ratings = load_ratings(root / "ratings.dat")
    users = load_users(root / "users.dat")
    movies = load_movies(root / "movies.dat")
    return MovieLens1M(ratings=ratings, users=users, movies=movies)


def temporal_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings by timestamp (last ``test_ratio`` of timestamps go to test).

    This mimics the production setting: predict the future from the past.
    """
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(ratings) * (1 - test_ratio))
    train = ratings.iloc[:cut].copy()
    test = ratings.iloc[cut:].copy()
    # Drop users that appear only in test (cold users - kept on purpose but
    # we will filter when evaluating classic CF methods that need history).
    return train, test


if __name__ == "__main__":
    from .download_data import download_movielens_1m

    root = download_movielens_1m()
    ml = load_movielens_1m(root)
    print(f"Ratings : {ml.n_ratings:,}")
    print(f"Users   : {ml.n_users:,}")
    print(f"Items   : {ml.n_items:,}")
    print(f"Sparsity: {ml.sparsity:.4%}")
