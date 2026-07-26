"""Tolerant Pydantic schemas for eBay Browse responses."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class EbayModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class EbayAmount(EbayModel):
    value: Decimal
    currency: str


class EbayImage(EbayModel):
    image_url: str | None = Field(default=None, alias="imageUrl")


class EbaySeller(EbayModel):
    username: str | None = None
    feedback_percentage: str | None = Field(default=None, alias="feedbackPercentage")
    feedback_score: int | None = Field(default=None, alias="feedbackScore")
    seller_account_type: str | None = Field(default=None, alias="sellerAccountType")


class EbayShippingOption(EbayModel):
    shipping_cost_type: str | None = Field(default=None, alias="shippingCostType")
    shipping_cost: EbayAmount | None = Field(default=None, alias="shippingCost")
    min_estimated_delivery_date: str | None = Field(default=None, alias="minEstimatedDeliveryDate")
    max_estimated_delivery_date: str | None = Field(default=None, alias="maxEstimatedDeliveryDate")


class EbayItemSummary(EbayModel):
    item_id: str = Field(alias="itemId")
    title: str
    short_description: str | None = Field(default=None, alias="shortDescription")
    item_web_url: str | None = Field(default=None, alias="itemWebUrl")
    image: EbayImage | None = None
    thumbnail_images: list[EbayImage] = Field(default_factory=list, alias="thumbnailImages")
    price: EbayAmount | None = None
    condition: str | None = None
    condition_id: str | None = Field(default=None, alias="conditionId")
    seller: EbaySeller | None = None
    shipping_options: list[EbayShippingOption] = Field(default_factory=list, alias="shippingOptions")
    buying_options: list[str] = Field(default_factory=list, alias="buyingOptions")
    categories: list[dict] = Field(default_factory=list)
    item_group_id: str | None = Field(default=None, alias="itemGroupId")
    item_end_date: datetime | None = Field(default=None, alias="itemEndDate")
    item_origin_date: datetime | None = Field(default=None, alias="itemOriginDate")
    legacy_item_id: str | None = Field(default=None, alias="legacyItemId")


class EbaySearchResponse(EbayModel):
    href: str | None = None
    total: int = 0
    limit: int = 0
    offset: int = 0
    next: str | None = None
    item_summaries: list[EbayItemSummary] = Field(default_factory=list, alias="itemSummaries")


class EbayAspect(EbayModel):
    name: str
    values: list[str] = Field(default_factory=list)


class EbayItemDetails(EbayModel):
    item_id: str = Field(alias="itemId")
    title: str
    short_description: str | None = Field(default=None, alias="shortDescription")
    description: str | None = None
    price: EbayAmount | None = None
    image: EbayImage | None = None
    additional_images: list[EbayImage] = Field(default_factory=list, alias="additionalImages")
    condition: str | None = None
    condition_description: str | None = Field(default=None, alias="conditionDescription")
    seller: EbaySeller | None = None
    shipping_options: list[EbayShippingOption] = Field(default_factory=list, alias="shippingOptions")
    localized_aspects: list[EbayAspect] = Field(default_factory=list, alias="localizedAspects")
    item_web_url: str | None = Field(default=None, alias="itemWebUrl")
    estimated_availabilities: list[dict] = Field(default_factory=list, alias="estimatedAvailabilities")
    item_end_date: datetime | None = Field(default=None, alias="itemEndDate")
    category_path: str | None = Field(default=None, alias="categoryPath")
    buying_options: list[str] = Field(default_factory=list, alias="buyingOptions")


class EbayTokenResponse(EbayModel):
    access_token: str
    expires_in: int = Field(gt=0)
    token_type: str
