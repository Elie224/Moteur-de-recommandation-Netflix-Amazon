"""Run a non-secret eBay Sandbox OAuth and Browse smoke test."""
from __future__ import annotations

import asyncio
import sys

from ..config import get_settings
from ..integrations.ebay.auth import EbayTokenManager
from ..integrations.ebay.client import EbayClient
from ..integrations.ebay.exceptions import EbayError


async def run() -> None:
    settings = get_settings()
    manager = EbayTokenManager(settings)
    await manager.get_access_token()
    print("eBay OAuth authentication succeeded")
    print(f"expires_in={manager.expires_in_seconds}")
    print(f"environment={settings.ebay_environment}")
    async with EbayClient(settings, token_manager=manager) as client:
        response = await client.search_items(query="phone", limit=5, offset=0)
    print(f"browse_status=ok received={len(response.item_summaries)} total={response.total}")


def main() -> None:
    try:
        asyncio.run(run())
    except EbayError as exc:
        print(f"eBay smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
