"""Bounded eBay synchronization queries for the first product release."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EbaySyncQuery:
    key: str
    query: str
    internal_category: str
    ebay_category_ids: tuple[str, ...] = ()
    max_pages: int = 2


EBAY_SYNC_QUERIES = (
    EbaySyncQuery("laptops", "ordinateur portable", "laptop"),
    EbaySyncQuery("smartphones", "smartphone", "smartphone"),
    EbaySyncQuery("headphones", "casque bluetooth", "headphones"),
)
