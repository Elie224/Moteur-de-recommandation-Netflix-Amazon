"""Integration tests for the modular RecoSphere API."""
import os
import sys
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "apps" / "api" / "src"),
    str(ROOT / "packages" / "recommendation_engine" / "src"),
]
TEST_DB = ROOT / ".pytest_cache" / "recosphere-api-test.db"
TEST_DB.parent.mkdir(parents=True, exist_ok=True)
TEST_DB.unlink(missing_ok=True)
os.environ["RECOSPHERE_DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
os.environ["RECOSPHERE_JWT_SECRET"] = "test-only-secret"
os.environ["RECOSPHERE_AUTO_CREATE_SCHEMA"] = "true"

from recommender_api.database import SessionLocal
from recommender_api.main import app
from recommender_api.models import CatalogItem, Movie, Product, User
from recommender_api.services.interaction_service import disliked_items_for, seen_items_for
from recommender_api.services.recommendation_service import build_recommendation_context


def seed_catalog():
    with SessionLocal.begin() as db:
        movie_1 = CatalogItem(item_type="movie", external_source="test", external_id="movie-1", title="Premier film", category="Science-fiction", tags=["espace"], popularity_score=20)
        movie_2 = CatalogItem(item_type="movie", external_source="test", external_id="movie-2", title="Deuxieme film", category="Science-fiction", tags=["espace"], popularity_score=10)
        product = CatalogItem(item_type="product", external_source="test", external_id="product-1", title="Casque audio", category="Audio", tags=["audio"], popularity_score=15)
        inactive_movie = CatalogItem(item_type="movie", external_source="test", external_id="movie-inactive", title="Film archive", is_active=False, popularity_score=100)
        db.add_all([movie_1, movie_2, product, inactive_movie])
        db.flush()
        db.add_all([
            Movie(catalog_item_id=movie_1.id, genres=["Science-fiction"]),
            Movie(catalog_item_id=movie_2.id, genres=["Science-fiction"]),
            Product(catalog_item_id=product.id, price_amount=99.9, price_currency="EUR"),
        ])


def register(client, email="user@example.com"):
    response = client.post("/api/v1/auth/register", json={"email": email, "password": "mot-de-passe-solide"})
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_user_catalog_and_recommendation_flow():
    with TestClient(app) as client:
        seed_catalog()
        headers = register(client)
        assert client.get("/health").json() == {"status": "ok"}
        movies = client.get("/api/v1/catalog", params={"item_type": "movie"}).json()
        assert len(movies) == 2
        assert client.get("/api/v1/catalog/4").status_code == 404
        assert client.post("/api/v1/interactions", headers=headers, json={"catalog_item_id": 1, "event_type": "view"}).status_code == 201
        favorite = client.post("/api/v1/favorites/2", headers=headers)
        assert favorite.json()["is_favorite"] is True
        assert len(client.get("/api/v1/favorites", headers=headers).json()) == 1
        response = client.get("/api/v1/recommendations/movie", headers=headers, params={"top_k": 5})
        assert response.status_code == 200
        recommendations = response.json()["recommendations"]
        assert [item["catalog_item_id"] for item in recommendations] == [1]
        assert all(item["item_type"] == "movie" for item in recommendations)
        assert all(item["catalog_item_id"] != 4 for item in recommendations)

        products = client.get("/api/v1/recommendations/product", headers=headers).json()["recommendations"]
        assert [item["catalog_item_id"] for item in products] == [3]
        assert all(item["item_type"] == "product" for item in products)

        with SessionLocal() as db:
            product_row = db.get(Product, 3)
            assert isinstance(product_row.price_amount, Decimal)


def test_strong_interaction_replaces_previous_preference_and_favorite_context():
    with TestClient(app) as client:
        headers = register(client, "signals@example.com")
        assert client.post(
            "/api/v1/interactions",
            headers=headers,
            json={"catalog_item_id": 1, "event_type": "view"},
        ).status_code == 201

        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.email == "signals@example.com"))
            assert user is not None
            assert 1 not in seen_items_for(db, user.id, "movie")

        assert client.post(
            "/api/v1/interactions",
            headers=headers,
            json={"catalog_item_id": 1, "event_type": "dislike"},
        ).status_code == 201
        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.email == "signals@example.com"))
            assert user is not None
            assert 1 in seen_items_for(db, user.id, "movie")
            assert 1 in disliked_items_for(db, user.id, "movie")

        assert client.post(
            "/api/v1/interactions",
            headers=headers,
            json={"catalog_item_id": 1, "event_type": "like"},
        ).status_code == 201
        assert client.post("/api/v1/favorites/2", headers=headers).json()["is_favorite"] is True

        with SessionLocal() as db:
            user = db.scalar(select(User).where(User.email == "signals@example.com"))
            assert user is not None
            assert 1 in seen_items_for(db, user.id, "movie")
            assert 1 not in disliked_items_for(db, user.id, "movie")
            context = build_recommendation_context(db, user.id, "movie", 5, {})
            assert context.favorite_item_ids == {2}
            assert 2 in context.seen_item_ids


def test_auth_validation_and_admin_protection():
    with TestClient(app) as client:
        duplicate = client.post("/api/v1/auth/register", json={"email": "user@example.com", "password": "mot-de-passe-solide"})
        assert duplicate.status_code == 409
        login = client.post("/api/v1/auth/login", json={"email": "user@example.com", "password": "mot-de-passe-solide"})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        assert client.get("/api/v1/auth/me", headers=headers).status_code == 200
        assert client.get("/api/v1/admin/metrics", headers=headers).status_code == 403
        assert client.post("/api/v1/admin/sync/ebay", headers=headers).status_code == 403
        invalid = client.post("/api/v1/interactions", headers=headers, json={"catalog_item_id": 1, "event_type": "unknown"})
        assert invalid.status_code == 422
