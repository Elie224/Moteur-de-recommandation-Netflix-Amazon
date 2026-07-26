"""FastAPI app exposing recommendation endpoints.

Loads self-contained artifacts (no dependency on raw data files).
"""
from __future__ import annotations

import json
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "data" / "processed" / "artifacts"


class MovieInfo(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: str = ""


class Recommendation(BaseModel):
    movie_id: int
    title: str
    score: float
    year: int | None = None
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


class ModelStore:
    """Loads standardized, self-contained artifacts once at startup."""

    def __init__(self, art_dir: Path):
        self.art_dir = art_dir
        self.cosine = None
        self.svd = None
        self.lightfm = None
        self.lightfm_state: dict | None = None
        self.user_features_mat: np.ndarray | None = None
        self.item_features_mat: np.ndarray | None = None
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.movies: pd.DataFrame | None = None
        self.meta: dict | None = None
        self.manifest: dict | None = None

    def load(self) -> None:
        with open(self.art_dir / "item_item_cosine.pkl", "rb") as f:
            self.cosine = pickle.load(f)
        with open(self.art_dir / "surprise_svd.pkl", "rb") as f:
            self.svd = pickle.load(f)
        self.lightfm_state = torch.load(
            self.art_dir / "hybrid_lightfm_state.pt",
            map_location="cpu", weights_only=False,
        )
        self.user_features_mat = np.load(self.art_dir / "user_features.npy")
        self.item_features_mat = np.load(self.art_dir / "item_features.npy")
        self.user_index = {int(k): int(v) for k, v in json.load(open(self.art_dir / "user_index.json")).items()}
        self.item_index = {int(k): int(v) for k, v in json.load(open(self.art_dir / "item_index.json")).items()}
        self.movies = pd.read_csv(self.art_dir / "movies.csv")
        self.meta = json.load(open(self.art_dir / "metadata.json"))
        self.manifest = json.load(open(self.art_dir / "manifest.json"))

    def movie_title(self, movie_id: int) -> tuple[str, int | None, str]:
        if self.movies is None:
            return (str(movie_id), None, "")
        row = self.movies[self.movies["movie_id"] == movie_id]
        if row.empty:
            return (f"unknown ({movie_id})", None, "")
        y = row.iloc[0]["year"]
        return (
            str(row.iloc[0]["title"]),
            int(y) if y == y else None,
            str(row.iloc[0]["genres_str"]),
        )

    def get_model(self, name: str):
        n = name.lower()
        if n in ("cosine", "itemitemcosine", "item_item_cosine"):
            return self.cosine
        if n in ("svd", "surrecomendsvd", "surprise_svd"):
            return self.svd
        raise HTTPException(status_code=404,
                            detail=f"Unknown model '{name}'. Available: cosine, svd")


store = ModelStore(ART)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading production artifacts...")
    store.load()
    print(f"[startup] Loaded. Movies: {len(store.movies)}, manifest commit: {store.manifest.get('git_commit', 'n/a')}")
    yield
    print("[shutdown] Done.")


app = FastAPI(
    title="Moteur de recommandation",
    description="Recommandations de films (MovieLens 1M) - cosine, SVD, hybride LightFM (cold start).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {
        "service": "moteur-recommandation",
        "version": store.manifest.get("trained_at", "unknown") if store.manifest else "unknown",
        "git_commit": store.manifest.get("git_commit") if store.manifest else None,
        "models": ["cosine", "svd"],
        "endpoints": [
            "GET /health", "GET /models", "GET /stats", "GET /manifest",
            "GET /movies/{id}", "GET /movies?query=...",
            "GET /users/{user_id}/recommend",
            "GET /predict", "POST /cold-start/recommend",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/manifest")
def get_manifest() -> dict:
    return store.manifest or {}


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
        "n_user_features": store.meta["n_user_features"],
        "n_item_features": store.meta["n_item_features"],
    }


@app.get("/movies/{movie_id}", response_model=MovieInfo)
def get_movie(movie_id: int) -> MovieInfo:
    title, year, genres = store.movie_title(movie_id)
    return MovieInfo(movie_id=movie_id, title=title, year=year, genres=genres)


@app.get("/movies")
def search_movies(query: str = Query(..., min_length=1, max_length=200),
                  limit: int = Query(10, ge=1, le=100)) -> list[MovieInfo]:
    q = query.lower()
    matches = store.movies[
        store.movies["title"].str.lower().str.contains(q, na=False, regex=False)
    ].head(limit)
    return [
        MovieInfo(
            movie_id=int(r.movie_id),
            title=str(r.title),
            year=int(r.year) if r.year == r.year else None,
            genres=str(r.genres_str),
        )
        for r in matches.itertuples()
    ]


def _model_knows_user(model, user_id: int) -> bool | None:
    if hasattr(model, "user_index_"):
        return user_id in model.user_index_
    if hasattr(model, "trainset_"):
        try:
            model.trainset_.to_inner_uid(int(user_id))
            return True
        except ValueError:
            return False
    return None


@app.get("/users/{user_id}/recommend", response_model=RecommendationResponse)
def recommend_for_user(user_id: int,
                       model: str = Query("cosine"),
                       top_k: int = Query(10, ge=1, le=100),
                       exclude_seen: bool = Query(True)) -> RecommendationResponse:
    m = store.get_model(model)
    known = _model_knows_user(m, user_id)
    if known is False:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} unknown. Use /cold-start/recommend instead.",
        )
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
    pairs = pd.DataFrame({"user_id": [user_id], "movie_id": [movie_id]})
    predictions = m.predict_for_pairs(pairs)
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} or movie {movie_id} unknown to model {model}.",
        )
    rating = next(iter(predictions.values()))
    return PredictionResponse(user_id=user_id, movie_id=movie_id,
                              model=model.lower(),
                              predicted_rating=float(rating))


def _build_lightfm_from_state() -> "LightFMModel":
    """Build the LightFM model from self-contained artifacts (no raw data needed)."""
    from src.models.hybrid import LightFMModel, HybridCF
    state = store.lightfm_state
    user_vocab = {tuple(k.split("=", 1)): int(v)
                  for k, v in json.load(open(ART / "user_feat_vocab.json")).items()}
    item_vocab = {tuple(k.split("=", 1)): int(v)
                  for k, v in json.load(open(ART / "item_feat_vocab.json")).items()}
    lfm = LightFMModel(n_factors=state["n_factors"])
    lfm.user_feat_index_ = user_vocab
    lfm.item_feat_index_ = item_vocab
    lfm.user_index_ = store.user_index
    lfm.item_index_ = store.item_index
    lfm.user_features_mat_ = torch.tensor(store.user_features_mat)
    lfm.item_features_mat_ = torch.tensor(store.item_features_mat)
    lfm.model = HybridCF(
        n_users=len(store.user_index), n_items=len(store.item_index),
        n_user_features=state["n_user_feats"], n_item_features=state["n_item_feats"],
        n_factors=state["n_factors"],
    )
    lfm.model.load_state_dict(state["model_state"])
    lfm.model.eval()
    return lfm


@app.post("/cold-start/recommend", response_model=ColdStartResponse)
def cold_start_recommend(features: ColdStartFeatures,
                         top_k: int = Query(10, ge=1, le=50)) -> ColdStartResponse:
    if store.lightfm_state is None:
        raise HTTPException(status_code=503, detail="LightFM model not loaded.")
    lfm = _build_lightfm_from_state()
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
    return ColdStartResponse(model="hybrid_lightfm", top_k=top_k, recommendations=out)
