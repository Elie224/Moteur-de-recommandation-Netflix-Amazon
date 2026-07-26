"""Asynchronous eBay Browse API client with OAuth and retry handling."""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import ValidationError

from ...config import Settings
from .auth import EbayTokenManager
from .exceptions import EbayAuthenticationError, EbayRateLimitError, EbayRequestError, EbayValidationError
from .schemas import EbayItemDetails, EbaySearchResponse


class EbayClient:
    def __init__(
        self,
        settings: Settings,
        *,
        token_manager: EbayTokenManager | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.token_manager = token_manager or EbayTokenManager(settings, transport=transport)
        self._client = httpx.AsyncClient(
            base_url=settings.ebay_api_base_url.rstrip("/"),
            timeout=settings.ebay_request_timeout,
            headers={
                "Accept": "application/json",
                "Accept-Language": settings.ebay_language,
                "Content-Language": settings.ebay_language,
                "X-EBAY-C-MARKETPLACE-ID": settings.ebay_marketplace_id,
            },
            transport=transport,
        )

    async def __aenter__(self) -> "EbayClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, *, params: dict[str, Any] | None = None, retries: int = 3, refresh_token_on_401: bool = True) -> dict[str, Any]:
        for attempt in range(retries):
            token = await self.token_manager.get_access_token()
            try:
                response = await self._client.get(path, params=params, headers={"Authorization": f"Bearer {token}"})
            except httpx.RequestError as exc:
                if attempt == retries - 1:
                    raise EbayRequestError(f"Impossible de contacter eBay : {exc}") from exc
                await asyncio.sleep(2**attempt)
                continue
            if response.status_code == 401 and refresh_token_on_401:
                await self.token_manager.invalidate()
                return await self._get(path, params=params, retries=1, refresh_token_on_401=False)
            if response.status_code == 429:
                if attempt == retries - 1:
                    raise EbayRateLimitError("La limite d’appels eBay a été atteinte.")
                try:
                    delay = max(0, int(response.headers.get("Retry-After", "2")))
                except ValueError:
                    delay = 2
                await asyncio.sleep(delay)
                continue
            if 500 <= response.status_code < 600:
                if attempt == retries - 1:
                    raise EbayRequestError(f"eBay est indisponible : HTTP {response.status_code}.")
                await asyncio.sleep(2**attempt)
                continue
            if not response.is_success:
                raise EbayAuthenticationError(self._format_error(response)) if response.status_code == 401 else EbayRequestError(self._format_error(response))
            try:
                payload = response.json()
            except ValueError as exc:
                raise EbayRequestError("eBay a retourné un JSON invalide.") from exc
            if not isinstance(payload, dict):
                raise EbayRequestError("eBay a retourné une réponse JSON invalide.")
            return payload
        raise EbayRequestError("Échec inattendu de la requête eBay.")

    @staticmethod
    def _format_error(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return f"Erreur eBay HTTP {response.status_code}."
        errors = payload.get("errors", []) if isinstance(payload, dict) else []
        if errors:
            error = errors[0]
            return f"Erreur eBay {error.get('errorId', '')}: {error.get('message', 'Requête invalide')}"
        return f"Erreur eBay HTTP {response.status_code}."

    async def search_items(self, *, query: str, limit: int = 50, offset: int = 0, category_ids: list[str] | None = None) -> EbaySearchResponse:
        params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "offset": offset,
            "fieldgroups": "EXTENDED",
            "filter": "buyingOptions:{FIXED_PRICE}" if self.settings.ebay_environment == "sandbox" else "buyingOptions:{FIXED_PRICE},deliveryCountry:FR",
        }
        if category_ids:
            params["category_ids"] = ",".join(category_ids)
        payload = await self._get("/buy/browse/v1/item_summary/search", params=params)
        try:
            return EbaySearchResponse.model_validate(payload)
        except ValidationError as exc:
            raise EbayValidationError("Réponse de recherche eBay invalide.") from exc

    async def get_item(self, item_id: str) -> EbayItemDetails:
        payload = await self._get(f"/buy/browse/v1/item/{item_id}")
        try:
            return EbayItemDetails.model_validate(payload)
        except ValidationError as exc:
            raise EbayValidationError(f"Détails eBay invalides pour {item_id}.") from exc
