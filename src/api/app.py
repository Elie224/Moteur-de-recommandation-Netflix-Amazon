"""FastAPI app exposing recommendation endpoints.

Endpoints
---------
GET  /                          -> service info
GET  /health                    -> health check
GET  /models                    -> list available models
GET  /movies/{movie_id}         -> movie metadata
GET  /movies?query=...          -> search movies
GET  /users/{user_id}/recommend -> top-K for an existing user
GET  /predict                   -> predict rating for (user, item)
POST /cold-start/recommend      -> recommend for a brand-new user (hybrid only)
GET  /stats                     -> dataset stats

Run with:
    .venv/Scripts/uvicorn src.api.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "data" / "processed" / "artifacts"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MovieInfo(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    genres: str = ""


class Recommendation(BaseModel):
    movie_id: int
    title: str
    score: float
    year: Optional[int] = None
    genres: str = ""


class RecommendationResponse(BaseModel):
    user_id: int
    model: str
    top_k: int
    recommendations: list[Recommendation]
    excluded_seen: int


class PredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    model: str
    predicted_rating: float


class ColdStartFeatures(BaseModel):
    gender: str = Field(..., description="M or F")
    age: int = Field(..., ge=1, le=99)
    occupation: int = Field(..., ge=0, le=20)


class ColdStartResponse(BaseModel):
    model: str
    top_k: int
    recommendations: list[Recommendation]


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------


class ModelStore:
    """Loads trained artifacts once at startup."""

    def __init__(self, art_dir: Path):
        self.art_dir = art_dir
        self.cosine = None
        self.svd = None
        self.lightfm = None
        self.movies: pd.DataFrame | None = None
        self.meta: dict | None = None
        self.lightfm_state: dict | None = None

    def load(self) -> None:
        with open(self.art_dir / "cosine.pkl", "rb") as f:
            self.cosine = pickle.load(f)
        with open(self.art_dir / "svd.pkl", "rb") as f:
            self.svd = pickle.load(f)
        self.lightfm_state = torch.load(self.art_dir / "lightfm.pt",
                                         map_location="cpu", weights_only=False)
        self.movies = pd.read_csv(self.art_dir / "movies.csv")
        with open(self.art_dir / "meta.json") as f:
            self.meta = json.load(f)

    def movie_title(self, movie_id: int) -> tuple[str, Optional[int], str]:
        if self.movies is None:
            return (str(movie_id), None, "")
        row = self.movies[self.movies["movie_id"] == movie_id]
        if row.empty:
            return (f"unknown ({movie_id})", None, "")
        return str(row.iloc[0]["title"]), int(row.iloc[0]["year"]) if row.iloc[0]["year"] == row.iloc[0]["year"] else None, str(row.iloc[0]["genres_str"])

    def get_model(self, name: str):
        name = name.lower()
        if name in ("cosine", "itemitemcosine"):
            return self.cosine
        if name in ("svd", "surrecomendsvd", "surprise_svd"):
            return self.svd
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'. Available: cosine, svd")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


store = ModelStore(ART)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading production models...")
    store.load()
    print(f"[startup] Loaded. Movies: {len(store.movies)}")
    yield
    print("[shutdown] Done.")


app = FastAPI(
    title="Moteur de recommandation",
    description="API pour servir des recommandations de films (MovieLens 1M).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {
        "service": "moteur-recommandation",
        "version": "1.0.0",
        "models": ["cosine", "svd"],
        "endpoints": [
            "GET /health",
            "GET /models",
            "GET /movies/{id}",
            "GET /movies?query=...",
            "GET /users/{user_id}/recommend",
            "GET /predict",
            "POST /cold-start/recommend",
            "GET /stats",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {
        "available": ["cosine", "svd"],
        "details": {
            "cosine": "Item-item cosine with mean centering (Module 1)",
            "svd": "Surprise SVD with biases, 50 factors (Module 2)",
        },
    }


@app.get("/stats")
def stats() -> dict:
    return {
        "n_users": store.meta["n_users"],
        "n_items": store.meta["n_items"],
        "n_ratings": store.meta["n_ratings"],
        "n_user_features": len(store.meta["user_feat_vocab"]),
        "n_item_features": len(store.meta["item_feat_vocab"]),
    }


@app.get("/movies/{movie_id}", response_model=MovieInfo)
def get_movie(movie_id: int) -> MovieInfo:
    title, year, genres = store.movie_title(movie_id)
    return MovieInfo(movie_id=movie_id, title=title, year=year, genres=genres)


@app.get("/movies")
def search_movies(query: str = Query(..., min_length=1), limit: int = 10) -> list[MovieInfo]:
    q = query.lower()
    matches = store.movies[store.movies["title"].str.lower().str.contains(q, na=False)].head(limit)
    return [
        MovieInfo(
            movie_id=int(r.movie_id),
            title=str(r.title),
            year=int(r.year) if r.year == r.year else None,
            genres=str(r.genres_str),
        )
        for r in matches.itertuples()
    ]


@app.get("/users/{user_id}/recommend", response_model=RecommendationResponse)
def recommend_for_user(user_id: int,
                       model: str = Query("cosine"),
                       top_k: int = Query(10, ge=1, le=100),
                       exclude_seen: bool = Query(True)) -> RecommendationResponse:
    m = store.get_model(model)
    recs = m.recommend(user_id, top_k=top_k, exclude_seen=exclude_seen)
    out = []
    for iid, score in recs:
        title, year, genres = store.movie_title(iid)
        out.append(Recommendation(movie_id=iid, title=title, score=float(score),
                                  year=year, genres=genres))
    n_excluded = 0
    if exclude_seen and hasattr(m, "user_seen_"):
        n_excluded = len(m.user_seen_.get(user_id, set()))
    return RecommendationResponse(
        user_id=user_id, model=model.lower(),
        top_k=top_k, recommendations=out, excluded_seen=n_excluded,
    )


@app.get("/predict", response_model=PredictionResponse)
def predict(user_id: int, movie_id: int,
            model: str = Query("svd")) -> PredictionResponse:
    m = store.get_model(model)
    if hasattr(m, "_predict"):
        rating = m._predict(user_id, movie_id)
    else:
        pairs = pd.DataFrame({"user_id": [user_id], "movie_id": [movie_id]})
        rating = list(m.predict_for_pairs(pairs).values())[0]
    return PredictionResponse(user_id=user_id, movie_id=movie_id,
                              model=model.lower(),
                              predicted_rating=float(rating))


@app.post("/cold-start/recommend", response_model=ColdStartResponse)
def cold_start_recommend(features: ColdStartFeatures,
                         top_k: int = Query(10, ge=1, le=50)) -> ColdStartResponse:
    if store.lightfm_state is None:
        raise HTTPException(status_code=503, detail="LightFM model not loaded.")
    # Lazy build of the LightFMModel from state
    lfm = _build_lightfm_from_state(store)
    feats = [
        ("gender", str(features.gender)),
        ("age", str(features.age)),
        ("occupation", str(features.occupation)),
    ]
    recs = lfm.recommend_for_new_user(feats, top_k=top_k)
    out = []
    for iid, score in recs:
        title, year, genres = store.movie_title(iid)
        out.append(Recommendation(movie_id=iid, title=title, score=float(score),
                                  year=year, genres=genres))
    return ColdStartResponse(model="lightfm_hybrid", top_k=top_k, recommendations=out)


def _build_lightfm_from_state(store: ModelStore):
    from src.models.hybrid import LightFMModel, HybridCF
    import numpy as np
    import torch as _torch
    state = store.lightfm_state
    lfm = LightFMModel(n_factors=state["n_factors"])
    # Use the stored vocabs (NOT rebuild) so indices match the trained embeddings
    lfm.user_feat_index_ = {(k.split("=", 1)[0], k.split("=", 1)[1]): v
                             for k, v in store.meta["user_feat_vocab"].items()}
    lfm.item_feat_index_ = {(k.split("=", 1)[0], k.split("=", 1)[1]): v
                             for k, v in store.meta["item_feat_vocab"].items()}
    ml = _load_ml_for_features(store)
    # Rebuild ID mappings from RATINGS (matches what the trained model saw)
    # ml.movies may include movies with no ratings -> would mismatch dimensions
    lfm.user_index_ = {int(u): i for i, u in enumerate(np.sort(ml.ratings["user_id"].unique()))}
    lfm.item_index_ = {int(m): i for i, m in enumerate(np.sort(ml.ratings["movie_id"].unique()))}
    # Build feature matrices using the STORED vocab indices, ordered by ratings
    rating_user_ids = np.sort(ml.ratings["user_id"].unique())
    rating_item_ids = np.sort(ml.ratings["movie_id"].unique())
    users_df = ml.users.set_index("user_id").loc[rating_user_ids].reset_index()
    movies_df = ml.movies.set_index("movie_id").loc[rating_item_ids].reset_index()
    user_keys = []
    for _, row in users_df.iterrows():
        user_keys.append([
            ("gender", str(row["gender"])),
            ("age", str(row["age"])),
            ("occupation", str(row["occupation"])),
        ])
    max_u = max(len(ks) for ks in user_keys)
    u_mat = np.zeros((len(rating_user_ids), max_u), dtype=np.int64)
    for i, ks in enumerate(user_keys):
        for j, k in enumerate(ks):
            if k in lfm.user_feat_index_:
                u_mat[i, j] = lfm.user_feat_index_[k]
    lfm.user_features_mat_ = _torch.tensor(u_mat)
    item_keys = []
    for _, row in movies_df.iterrows():
        ks = [("genre", g) for g in (row["genres_list"] or [])]
        try:
            year = int(row["year"]) if row["year"] == row["year"] else None
        except (TypeError, ValueError):
            year = None
        if year:
            decade = (year // 10) * 10
            ks.append(("decade", str(decade)))
        item_keys.append(ks)
    max_i = max(len(ks) for ks in item_keys) if item_keys else 1
    i_mat = np.zeros((len(rating_item_ids), max_i), dtype=np.int64)
    for i, ks in enumerate(item_keys):
        for j, k in enumerate(ks):
            if k in lfm.item_feat_index_:
                i_mat[i, j] = lfm.item_feat_index_[k]
    lfm.item_features_mat_ = _torch.tensor(i_mat)
    # Re-load the model with right dims
    lfm.model = HybridCF(
        n_users=len(lfm.user_index_), n_items=len(lfm.item_index_),
        n_user_features=state["n_user_feats"], n_item_features=state["n_item_feats"],
        n_factors=state["n_factors"],
    )
    lfm.model.load_state_dict(state["model_state"])
    lfm.model.eval()
    return lfm


def _load_ml_for_features(store: ModelStore):
    """Re-load MovieLens to rebuild feature tables (cheap, ~1s)."""
    from src.data.loaders import load_movielens_1m
    return load_movielens_1m(ROOT / "data" / "raw" / "ml-1m")
