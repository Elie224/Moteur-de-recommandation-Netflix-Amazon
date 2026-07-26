"""Application OAuth token management for eBay Browse API."""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timedelta, timezone

import httpx
from pydantic import ValidationError

from ...config import Settings
from .exceptions import EbayAuthenticationError, EbayConfigurationError
from .schemas import EbayTokenResponse


class EbayTokenManager:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport
        self._access_token: str | None = None
        self._expires_at: datetime | None = None
        self._lock = asyncio.Lock()

    def _credentials(self) -> tuple[str, str]:
        if not self.settings.ebay_client_id or not self.settings.ebay_client_secret:
            raise EbayConfigurationError("Les identifiants eBay ne sont pas configurés.")
        return self.settings.ebay_client_id, self.settings.ebay_client_secret

    def _token_is_valid(self) -> bool:
        return bool(
            self._access_token
            and self._expires_at
            and datetime.now(timezone.utc) + timedelta(seconds=60) < self._expires_at
        )

    @property
    def expires_in_seconds(self) -> int:
        if not self._expires_at:
            return 0
        return max(0, int((self._expires_at - datetime.now(timezone.utc)).total_seconds()))

    async def get_access_token(self) -> str:
        if self._token_is_valid():
            return self._access_token  # type: ignore[return-value]
        async with self._lock:
            if self._token_is_valid():
                return self._access_token  # type: ignore[return-value]
            return await self._refresh_token()

    async def invalidate(self) -> None:
        async with self._lock:
            self._access_token = None
            self._expires_at = None

    async def _refresh_token(self) -> str:
        client_id, client_secret = self._credentials()
        credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        try:
            async with httpx.AsyncClient(timeout=self.settings.ebay_request_timeout, transport=self.transport) as client:
                response = await client.post(
                    self.settings.ebay_auth_url,
                    headers={
                        "Authorization": f"Basic {credentials}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise EbayAuthenticationError(f"eBay OAuth a répondu HTTP {exc.response.status_code}.") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise EbayAuthenticationError(f"Impossible de contacter OAuth eBay : {exc}") from exc
        try:
            token = EbayTokenResponse.model_validate(payload)
        except ValidationError as exc:
            raise EbayAuthenticationError("Réponse OAuth eBay invalide.") from exc
        self._access_token = token.access_token
        self._expires_at = datetime.now(timezone.utc) + timedelta(seconds=token.expires_in)
        return token.access_token
