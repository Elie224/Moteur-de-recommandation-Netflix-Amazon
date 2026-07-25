"""Tests for the FastAPI recommendation API.

Uses FastAPI TestClient. Requires production models in
data/processed/artifacts/ (run scripts/train_production_models.py first).
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SURPRISE_DATA_FOLDER", str(Path.cwd() / ".surprise_data"))

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert "cosine" in r.json()["available"]
    assert "svd" in r.json()["available"]


def test_stats(client):
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["n_users"] > 1000
    assert data["n_items"] > 1000
    assert data["n_ratings"] > 100000


def test_movie_info(client):
    r = client.get("/movies/1")
    assert r.status_code == 200
    data = r.json()
    assert data["movie_id"] == 1
    assert "title" in data


def test_movie_search(client):
    r = client.get("/movies", params={"query": "toy story", "limit": 5})
    assert r.status_code == 200
    results = r.json()
    assert isinstance(results, list)
    assert len(results) >= 1
    assert any("toy story" in m["title"].lower() for m in results)


def test_recommend_cosine(client):
    r = client.get("/users/1/recommend", params={"model": "cosine", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == 1
    assert data["model"] == "cosine"
    assert data["top_k"] == 5
    assert len(data["recommendations"]) == 5
    for rec in data["recommendations"]:
        assert "title" in rec
        assert "score" in rec


def test_recommend_svd(client):
    r = client.get("/users/1/recommend", params={"model": "svd", "top_k": 10})
    assert r.status_code == 200
    data = r.json()
    assert len(data["recommendations"]) == 10


def test_recommend_unknown_model(client):
    r = client.get("/users/1/recommend", params={"model": "nope"})
    assert r.status_code == 404


def test_predict_svd(client):
    r = client.get("/predict", params={"user_id": 1, "movie_id": 1, "model": "svd"})
    assert r.status_code == 200
    data = r.json()
    assert 1.0 <= data["predicted_rating"] <= 5.0


def test_cold_start(client):
    payload = {"gender": "F", "age": 25, "occupation": 10}
    r = client.post("/cold-start/recommend?top_k=5", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "lightfm_hybrid"
    assert len(data["recommendations"]) == 5
