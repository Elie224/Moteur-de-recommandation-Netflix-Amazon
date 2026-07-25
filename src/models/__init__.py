"""Model package."""
from .cosine_baseline import ItemItemCosine
from .popularity import PopularityBaseline

__all__ = ["ItemItemCosine", "PopularityBaseline"]
