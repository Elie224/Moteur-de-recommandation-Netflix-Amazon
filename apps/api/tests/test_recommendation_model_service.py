import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "apps" / "api" / "src"),
    str(ROOT / "packages" / "recommendation_engine" / "src"),
]

from recommender_api.services import recommendation_service
from recommender_api.services.recommendation_model_service import RecommendationModelService


def test_model_service_caches_per_domain_and_invalidates(monkeypatch):
    calls: list[str] = []

    def fake_fit(_db, item_type):
        calls.append(item_type)
        return object()

    monkeypatch.setattr(recommendation_service, "fit_recommender_for_domain", fake_fit)
    service = RecommendationModelService(ttl_seconds=600)

    first = service.get_model(object(), "movie")
    assert service.get_model(object(), "movie") is first
    assert calls == ["movie"]

    product = service.get_model(object(), "product")
    assert product is not first
    assert calls == ["movie", "product"]

    service.invalidate("movie")
    assert service.get_model(object(), "movie") is not first
    assert calls == ["movie", "product", "movie"]

    service.invalidate()
    service.get_model(object(), "product")
    assert calls == ["movie", "product", "movie", "product"]
