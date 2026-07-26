"""Offline tests for eBay OAuth, Browse client and mapper."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "apps" / "api" / "src"), str(ROOT / "packages" / "recommendation_engine" / "src")]

from recommender_api.config import Settings
from recommender_api.integrations.ebay.auth import EbayTokenManager
from recommender_api.integrations.ebay.client import EbayClient
from recommender_api.integrations.ebay.exceptions import EbayConfigurationError
from recommender_api.integrations.ebay.exceptions import EbayRequestError
from recommender_api.integrations.ebay.mapper import map_ebay_item
from recommender_api.integrations.ebay.schemas import EbayItemDetails


def test_token_manager_caches_oauth_token():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.method == "POST"
        assert request.url.path.endswith("/identity/v1/oauth2/token")
        assert request.content == b"grant_type=client_credentials&scope=https%3A%2F%2Fapi.ebay.com%2Foauth%2Fapi_scope"
        return httpx.Response(200, json={"access_token": "sandbox-token", "expires_in": 3600, "token_type": "Application Access Token"})

    async def run():
        settings = Settings(ebay_client_id="client", ebay_client_secret="secret")
        manager = EbayTokenManager(settings, transport=httpx.MockTransport(handler))
        assert await manager.get_access_token() == "sandbox-token"
        assert await manager.get_access_token() == "sandbox-token"
        assert calls == 1

    asyncio.run(run())


def test_missing_ebay_credentials_are_clear():
    async def run():
        with pytest.raises(EbayConfigurationError):
            await EbayTokenManager(Settings()).get_access_token()

    asyncio.run(run())


def test_browse_search_is_validated_without_network():
    class FakeTokenManager:
        async def get_access_token(self):
            return "token"

        async def invalidate(self):
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/buy/browse/v1/item_summary/search")
        assert request.headers["Authorization"] == "Bearer token"
        assert request.url.params["q"] == "smartphone"
        return httpx.Response(200, json={"total": 1, "limit": 1, "offset": 0, "itemSummaries": [{"itemId": "v1|123|0", "title": "Téléphone", "price": {"value": "199.99", "currency": "EUR"}}]})

    async def run():
        settings = Settings(ebay_client_id="client", ebay_client_secret="secret")
        async with EbayClient(settings, token_manager=FakeTokenManager(), transport=httpx.MockTransport(handler)) as client:
            response = await client.search_items(query="smartphone", limit=1)
            assert response.item_summaries[0].item_id == "v1|123|0"
            assert response.item_summaries[0].price.value == Decimal("199.99")

    asyncio.run(run())


def test_mapper_extracts_brand_shipping_and_deactivates_expired_item():
    item = EbayItemDetails.model_validate({
        "itemId": "v1|123|0",
        "title": "Casque",
        "image": {"imageUrl": "https://img.test/main.jpg"},
        "additionalImages": [{"imageUrl": "https://img.test/second.jpg"}],
        "price": {"value": "49.90", "currency": "EUR"},
        "condition": "USED",
        "localizedAspects": [{"name": "Brand", "values": ["Acme"]}],
        "shippingOptions": [{"shippingCost": {"value": "4.99", "currency": "EUR"}}],
        "itemEndDate": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "seller": {"username": "seller", "feedbackPercentage": "98.7", "feedbackScore": 120},
    })
    normalized = map_ebay_item(item, internal_category="headphones", marketplace="EBAY_FR")
    assert normalized.price == Decimal("49.90")
    assert normalized.shipping_cost == Decimal("4.99")
    assert normalized.brand == "Acme"
    assert normalized.is_active is False
    assert normalized.additional_images == ["https://img.test/second.jpg"]


def test_client_refreshes_token_after_401():
    class TokenManager:
        invalidations = 0

        async def get_access_token(self):
            return "fresh" if self.invalidations else "stale"

        async def invalidate(self):
            self.invalidations += 1

    manager = TokenManager()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["Authorization"] == "Bearer stale":
            return httpx.Response(401, json={"errors": [{"message": "expired"}]})
        return httpx.Response(200, json={"itemId": "v1|1|0", "title": "Phone"})

    async def run():
        async with EbayClient(Settings(ebay_client_id="id", ebay_client_secret="secret"), token_manager=manager, transport=httpx.MockTransport(handler)) as client:
            assert (await client.get_item("v1|1|0")).title == "Phone"
        assert manager.invalidations == 1

    asyncio.run(run())


@pytest.mark.parametrize("first_status", [429, 500])
def test_client_retries_transient_errors(first_status, monkeypatch):
    calls = 0

    class TokenManager:
        async def get_access_token(self): return "token"
        async def invalidate(self): return None

    async def no_sleep(_): return None
    monkeypatch.setattr("recommender_api.integrations.ebay.client.asyncio.sleep", no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(first_status, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json={"total": 0, "itemSummaries": []})

    async def run():
        async with EbayClient(Settings(ebay_client_id="id", ebay_client_secret="secret"), token_manager=TokenManager(), transport=httpx.MockTransport(handler)) as client:
            await client.search_items(query="phone")
        assert calls == 2

    asyncio.run(run())


def test_client_rejects_invalid_json_without_retry():
    class TokenManager:
        async def get_access_token(self): return "token"
        async def invalidate(self): return None

    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not-json"))
        async with EbayClient(Settings(ebay_client_id="id", ebay_client_secret="secret"), token_manager=TokenManager(), transport=transport) as client:
            with pytest.raises(EbayRequestError):
                await client.search_items(query="phone")

    asyncio.run(run())
