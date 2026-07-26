"""eBay synchronization service using the shared PostgreSQL catalog."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..config import Settings
from ..integrations.ebay.client import EbayClient
from ..integrations.ebay.constants import EBAY_SYNC_QUERIES, EbaySyncQuery
from ..integrations.ebay.mapper import map_ebay_item
from ..models import CatalogItem, Product, SyncRun
from .catalog_service import attach_product, upsert_catalog_item


@dataclass(slots=True)
class EbaySyncSummary:
    source: str
    status: str
    received: int
    unique_items: int
    created: int
    updated: int
    deactivated: int
    failed: int
    sync_run_id: int


class EbaySyncService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    async def sync(self, queries: Iterable[EbaySyncQuery] = EBAY_SYNC_QUERIES, *, max_pages: int | None = None) -> EbaySyncSummary:
        async with EbayClient(self.settings) as client:
            sync_run = SyncRun(source="ebay", status="running")
            self.db.add(sync_run)
            self.db.commit()
            self.db.refresh(sync_run)
            received = created = updated = failed = 0
            seen: set[str] = set()
            errors: list[dict[str, str]] = []
            try:
                for query_config in queries:
                    page_limit = min(max_pages or query_config.max_pages, self.settings.ebay_sync_max_pages)
                    for page in range(page_limit):
                        response = await client.search_items(
                            query=query_config.query,
                            limit=self.settings.ebay_sync_limit,
                            offset=page * self.settings.ebay_sync_limit,
                            category_ids=list(query_config.ebay_category_ids),
                        )
                        received += len(response.item_summaries)
                        summaries = [item for item in response.item_summaries if item.item_id not in seen]
                        seen.update(item.item_id for item in summaries)
                        results = await self._fetch_details(client, [item.item_id for item in summaries])
                        for item_id, detail, error in results:
                            if error:
                                failed += 1
                                errors.append({"external_id": item_id, "stage": "details", "type": type(error).__name__, "message": str(error)})
                                continue
                            try:
                                normalized = map_ebay_item(
                                    detail,
                                    internal_category=query_config.internal_category,
                                    marketplace=self.settings.ebay_marketplace_id,
                                )
                                item, was_created = upsert_catalog_item(
                                    self.db,
                                    item_type="product",
                                    external_source=normalized.source,
                                    external_id=normalized.external_id,
                                    title=normalized.title,
                                    description=normalized.description,
                                    image_url=normalized.image_url,
                                    detail_url=normalized.product_url,
                                    category=normalized.category,
                                    country="FR",
                                    popularity_score=0.0,
                                    average_rating=None,
                                    rating_count=0,
                                    tags=[normalized.category, normalized.brand] if normalized.brand else [normalized.category],
                                    extra_metadata=normalized.metadata,
                                    is_active=normalized.is_active,
                                )
                                attach_product(
                                    self.db,
                                    item.id,
                                    price_amount=normalized.price,
                                    price_currency=normalized.currency,
                                    condition=normalized.condition,
                                    condition_description=normalized.condition_description,
                                    brand=normalized.brand,
                                    seller_name=normalized.seller_name,
                                    seller_feedback_percentage=normalized.seller_feedback_percentage,
                                    seller_feedback_score=normalized.seller_feedback_score,
                                    availability=normalized.availability,
                                    shipping_cost=normalized.shipping_cost,
                                    shipping_currency=normalized.shipping_currency,
                                    marketplace=normalized.marketplace,
                                    product_url=normalized.product_url,
                                    additional_images=normalized.additional_images,
                                    item_end_date=normalized.item_end_date,
                                )
                                self.db.commit()
                                created += int(was_created)
                                updated += int(not was_created)
                            except Exception as exc:
                                self.db.rollback()
                                failed += 1
                                errors.append({"external_id": item_id, "stage": "upsert", "type": type(exc).__name__, "message": str(exc)})
                        if not response.next:
                            break
                deactivated = self._deactivate_expired_items()
                sync_run = self.db.get(SyncRun, sync_run.id)
                assert sync_run is not None
                sync_run.status = "completed"
                sync_run.finished_at = datetime.now(timezone.utc)
                sync_run.items_processed = len(seen)
                sync_run.items_created = created
                sync_run.items_updated = updated
                sync_run.error_details = errors[:100]
                sync_run.error_message = f"{failed} item(s) failed" if failed else None
                self.db.commit()
                return EbaySyncSummary("ebay", "completed", received, len(seen), created, updated, deactivated, failed, sync_run.id)
            except Exception as exc:
                self.db.rollback()
                failed_run = self.db.get(SyncRun, sync_run.id)
                if failed_run is not None:
                    failed_run.status = "failed"
                    failed_run.finished_at = datetime.now(timezone.utc)
                    failed_run.error_message = str(exc)
                    failed_run.error_details = [{"type": type(exc).__name__, "message": str(exc)}]
                    self.db.commit()
                raise

    async def _fetch_details(self, client: EbayClient, item_ids: list[str]):
        semaphore = asyncio.Semaphore(self.settings.ebay_concurrency)

        async def fetch(item_id: str):
            async with semaphore:
                try:
                    return item_id, await client.get_item(item_id), None
                except Exception as exc:
                    return item_id, None, exc

        return await asyncio.gather(*(fetch(item_id) for item_id in item_ids))

    def _deactivate_expired_items(self) -> int:
        now = datetime.now(timezone.utc)
        stmt = (
            update(CatalogItem)
            .where(CatalogItem.external_source == "ebay", CatalogItem.is_active.is_(True))
            .where(CatalogItem.id.in_(select(Product.catalog_item_id).where(Product.item_end_date <= now)))
            .values(is_active=False)
        )
        result = self.db.execute(stmt)
        self.db.commit()
        return int(result.rowcount or 0)
