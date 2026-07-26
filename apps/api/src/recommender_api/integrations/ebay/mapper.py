"""Pure eBay-to-product normalization functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .schemas import EbayItemDetails


@dataclass(slots=True)
class NormalizedProduct:
    source: str
    external_id: str
    title: str
    description: str | None
    image_url: str | None
    additional_images: list[str]
    category: str
    brand: str | None
    price: Decimal | None
    currency: str | None
    condition: str | None
    condition_description: str | None
    seller_name: str | None
    seller_feedback_percentage: float | None
    seller_feedback_score: int | None
    availability: str | None
    shipping_cost: Decimal | None
    shipping_currency: str | None
    marketplace: str
    product_url: str | None
    item_end_date: datetime | None
    is_active: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_aspect(item: EbayItemDetails, names: set[str]) -> str | None:
    normalized_names = {name.casefold() for name in names}
    for aspect in item.localized_aspects:
        if aspect.name.casefold() in normalized_names and aspect.values:
            return aspect.values[0]
    return None


def extract_brand(item: EbayItemDetails) -> str | None:
    return extract_aspect(item, {"Brand", "Marque"})


def select_shipping_cost(item: EbayItemDetails) -> tuple[Decimal | None, str | None]:
    for option in item.shipping_options:
        if option.shipping_cost:
            return option.shipping_cost.value, option.shipping_cost.currency
    return None, None


def extract_availability(item: EbayItemDetails) -> str | None:
    if not item.estimated_availabilities:
        return None
    return item.estimated_availabilities[0].get("estimatedAvailabilityStatus")


def map_ebay_item(item: EbayItemDetails, *, internal_category: str, marketplace: str) -> NormalizedProduct:
    shipping_cost, shipping_currency = select_shipping_cost(item)
    additional_images = [image.image_url for image in item.additional_images if image.image_url]
    availability = extract_availability(item)
    expired = bool(item.item_end_date and item.item_end_date <= datetime.now(timezone.utc))
    available = availability not in {"OUT_OF_STOCK", "UNAVAILABLE"}
    feedback = None
    if item.seller and item.seller.feedback_percentage:
        try:
            feedback = float(item.seller.feedback_percentage)
        except ValueError:
            feedback = None
    return NormalizedProduct(
        source="ebay",
        external_id=item.item_id,
        title=item.title,
        description=item.short_description or item.description,
        image_url=item.image.image_url if item.image else None,
        additional_images=additional_images,
        category=internal_category,
        brand=extract_brand(item),
        price=item.price.value if item.price else None,
        currency=item.price.currency if item.price else None,
        condition=item.condition,
        condition_description=item.condition_description,
        seller_name=item.seller.username if item.seller else None,
        seller_feedback_percentage=feedback,
        seller_feedback_score=item.seller.feedback_score if item.seller else None,
        availability=availability,
        shipping_cost=shipping_cost,
        shipping_currency=shipping_currency,
        marketplace=marketplace,
        product_url=item.item_web_url,
        item_end_date=item.item_end_date,
        is_active=not expired and available,
        metadata=item.model_dump(mode="json", by_alias=True),
    )
