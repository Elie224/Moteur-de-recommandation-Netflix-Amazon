"""Catalog read/write service."""
from __future__ import annotations

from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models import CatalogItem, Favorite, Movie, Product


def get_catalog_item(db: Session, item_id: int) -> CatalogItem | None:
    stmt = (
        select(CatalogItem)
        .options(selectinload(CatalogItem.movie), selectinload(CatalogItem.product))
        .where(CatalogItem.id == item_id)
    )
    return db.execute(stmt).scalar_one_or_none()


def list_catalog(
    db: Session,
    item_type: str | None = None,
    q: str | None = None,
    category: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> Sequence[CatalogItem]:
    stmt = select(CatalogItem).options(selectinload(CatalogItem.movie), selectinload(CatalogItem.product))
    stmt = stmt.where(CatalogItem.is_active.is_(True))
    if item_type:
        stmt = stmt.where(CatalogItem.item_type == item_type)
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(CatalogItem.title.ilike(like))
    if category:
        stmt = stmt.where(CatalogItem.category == category)
    stmt = stmt.order_by(CatalogItem.popularity_score.desc(), CatalogItem.id.asc())
    stmt = stmt.limit(limit).offset(offset)
    return db.execute(stmt).scalars().all()


def list_categories(db: Session, item_type: str | None = None) -> list[str]:
    stmt = select(CatalogItem.category).where(
        CatalogItem.category.is_not(None),
        CatalogItem.is_active.is_(True),
    )
    if item_type:
        stmt = stmt.where(CatalogItem.item_type == item_type)
    rows = db.execute(stmt.distinct()).all()
    return sorted(r[0] for r in rows if r[0])


def upsert_catalog_item(db: Session, *, item_type: str, external_source: str, external_id: str, title: str, description=None, image_url=None, detail_url=None, category=None, language=None, country=None, popularity_score: float = 0.0, average_rating=None, rating_count: int = 0, tags=None, extra_metadata=None, is_active: bool = True, published_at=None) -> tuple[CatalogItem, bool]:
    stmt = select(CatalogItem).where(
        CatalogItem.external_source == external_source,
        CatalogItem.external_id == external_id,
    )
    item = db.execute(stmt).scalar_one_or_none()
    created = False
    if item is None:
        item = CatalogItem(item_type=item_type, external_source=external_source, external_id=external_id, title=title)
        db.add(item)
        created = True
    item.title = title
    item.description = description
    item.image_url = image_url
    item.detail_url = detail_url
    item.category = category
    item.language = language
    item.country = country
    item.popularity_score = popularity_score
    item.average_rating = average_rating
    item.rating_count = rating_count
    item.tags = list(tags or [])
    item.extra_metadata = dict(extra_metadata or {})
    item.is_active = is_active
    if published_at is not None:
        item.published_at = published_at
    db.flush()
    return item, created


def attach_movie(db: Session, catalog_item_id: int, **kwargs) -> Movie:
    movie = db.get(Movie, catalog_item_id)
    if movie is None:
        movie = Movie(catalog_item_id=catalog_item_id)
        db.add(movie)
    for key, value in kwargs.items():
        if key in {"genres", "cast_members", "directors"} and value is not None:
            setattr(movie, key, list(value))
        elif key == "watch_providers" and value is not None:
            setattr(movie, key, dict(value))
        else:
            setattr(movie, key, value)
    db.flush()
    return movie


def attach_product(db: Session, catalog_item_id: int, **kwargs) -> Product:
    product = db.get(Product, catalog_item_id)
    if product is None:
        product = Product(catalog_item_id=catalog_item_id)
        db.add(product)
    for key, value in kwargs.items():
        setattr(product, key, value)
    db.flush()
    return product


def list_user_favorites(db: Session, user_id: int) -> list[Favorite]:
    stmt = select(Favorite).where(Favorite.user_id == user_id).order_by(Favorite.created_at.desc())
    return list(db.execute(stmt).scalars().all())


def toggle_favorite(db: Session, user_id: int, catalog_item_id: int) -> bool:
    stmt = select(Favorite).where(Favorite.user_id == user_id, Favorite.catalog_item_id == catalog_item_id)
    favorite = db.execute(stmt).scalar_one_or_none()
    if favorite is not None:
        db.delete(favorite)
        return False
    db.add(Favorite(user_id=user_id, catalog_item_id=catalog_item_id))
    return True
