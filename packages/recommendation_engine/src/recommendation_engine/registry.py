"""Lightweight registry for the recommender implementations."""
from __future__ import annotations

from threading import Lock
from typing import Mapping

from .base import BaseRecommender


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, BaseRecommender] = {}
        self._lock = Lock()

    def register(self, model: BaseRecommender) -> None:
        with self._lock:
            self._models[model.name] = model

    def get(self, name: str) -> BaseRecommender:
        if name not in self._models:
            raise KeyError(
                f"Unknown recommender {name!r}. Available: {sorted(self._models)}"
            )
        return self._models[name]

    def names(self) -> list[str]:
        return sorted(self._models)

    def as_dict(self) -> Mapping[str, BaseRecommender]:
        return dict(self._models)


_DEFAULT = ModelRegistry()


def get_default_registry() -> ModelRegistry:
    return _DEFAULT
