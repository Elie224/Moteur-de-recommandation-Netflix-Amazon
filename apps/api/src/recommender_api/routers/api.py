"""HTTP routes for the RecoSphere V1 API."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from ..config import get_settings
from ..deps import get_current_admin, get_current_user, get_db
from ..integrations.tmdb.exceptions import TMDBConfigurationError, TMDBError, TMDBRateLimitError
from ..integrations.ebay.exceptions import EbayAuthenticationError, EbayConfigurationError, EbayError, EbayRateLimitError
from ..integrations.ebay.constants import EBAY_SYNC_QUERIES
from ..models import (
    CatalogItem,
    Favorite,
    Interaction,
    Movie,
    Product,
    RecommendationBatch,
    RecommendationResult,
    SyncRun,
    User,
)
from ..schemas import (
    AdminMetrics,
    CatalogItemBase,
    FavoriteToggleResponse,
    EbaySyncRequest,
    InteractionCreate,
    InteractionOut,
    MovieOut,
    OnboardingIn,
    ProductOut,
    RecommendationOut,
    RecommendationResponse,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserPublic,
)
from ..security import create_access_token
from ..services import auth_service, catalog_service, interaction_service, recommendation_service
from ..services.sync_service import TMDBSyncService
from ..services.ebay_sync_service import EbaySyncService


router = APIRouter()
Db = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentAdmin = Annotated[User, Depends(get_current_admin)]


def _token_response(user: User) -> TokenResponse:
    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(user.id, {"admin": user.is_admin}),
        expires_in=settings.jwt_access_ttl_minutes * 60,
        user=UserPublic.model_validate(user),
    )


def _catalog_payload(item: CatalogItem) -> dict[str, Any]:
    payload = {
        "id": item.id,
        "item_type": item.item_type,
        "title": item.title,
        "description": item.description,
        "image_url": item.image_url,
        "detail_url": item.detail_url,
        "category": item.category,
        "language": item.language,
        "country": item.country,
        "popularity_score": item.popularity_score,
        "average_rating": item.average_rating,
        "rating_count": item.rating_count,
        "is_active": item.is_active,
        "published_at": item.published_at,
    }
    if item.item_type == "movie":
        movie = item.movie
        payload.update(
            release_date=movie.release_date if movie else None,
            runtime_minutes=movie.runtime_minutes if movie else None,
            original_title=movie.original_title if movie else None,
            original_language=movie.original_language if movie else None,
            trailer_url=movie.trailer_url if movie else None,
            genres=list(movie.genres or []) if movie else [],
            cast_members=list(movie.cast_members or []) if movie else [],
            directors=list(movie.directors or []) if movie else [],
            watch_providers=dict(movie.watch_providers or {}) if movie else {},
        )
    else:
        product = item.product
        payload.update(
            price_amount=float(product.price_amount) if product and product.price_amount is not None else None,
            price_currency=product.price_currency if product else None,
            condition=product.condition if product else None,
            brand=product.brand if product else None,
            seller_name=product.seller_name if product else None,
            availability=product.availability if product else None,
            shipping_cost=float(product.shipping_cost) if product and product.shipping_cost is not None else None,
            marketplace=product.marketplace if product else None,
            product_url=product.product_url if product else None,
            condition_description=product.condition_description if product else None,
            seller_feedback_percentage=float(product.seller_feedback_percentage) if product and product.seller_feedback_percentage is not None else None,
            seller_feedback_score=product.seller_feedback_score if product else None,
            shipping_currency=product.shipping_currency if product else None,
            additional_images=list(product.additional_images or []) if product else [],
            item_end_date=product.item_end_date if product else None,
        )
    return payload


@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Db) -> TokenResponse:
    if auth_service.get_user_by_email(db, payload.email):
        raise HTTPException(status_code=409, detail="An account already exists for this email")
    try:
        user = auth_service.create_user(
            db,
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
            preferred_language=payload.preferred_language or "fr",
            country=payload.country or "FR",
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account already exists for this email") from exc
    return _token_response(user)


@router.post("/auth/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Db) -> TokenResponse:
    user = auth_service.authenticate(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return _token_response(user)


@router.get("/auth/me", response_model=UserPublic)
def me(current_user: CurrentUser) -> User:
    return current_user


@router.get("/catalog", response_model=list[CatalogItemBase])
def catalog(
    db: Db,
    item_type: Literal["movie", "product"] | None = None,
    q: str | None = None,
    category: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    return [
        CatalogItemBase.model_validate(_catalog_payload(item)).model_dump()
        for item in catalog_service.list_catalog(db, item_type, q, category, limit, offset)
    ]


@router.get("/catalog/categories", response_model=list[str])
def categories(db: Db, item_type: Literal["movie", "product"] | None = None) -> list[str]:
    return catalog_service.list_categories(db, item_type)


@router.get("/catalog/{item_id}", response_model=MovieOut | ProductOut)
def catalog_detail(item_id: int, db: Db) -> MovieOut | ProductOut:
    item = catalog_service.get_catalog_item(db, item_id)
    if item is None or not item.is_active:
        raise HTTPException(status_code=404, detail="Catalog item not found")
    payload = _catalog_payload(item)
    return MovieOut.model_validate(payload) if item.item_type == "movie" else ProductOut.model_validate(payload)


@router.post("/interactions", response_model=InteractionOut, status_code=status.HTTP_201_CREATED)
def create_interaction(payload: InteractionCreate, db: Db, current_user: CurrentUser) -> Interaction:
    if payload.event_type not in interaction_service.VALID_EVENT_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported interaction type")
    if db.get(CatalogItem, payload.catalog_item_id) is None:
        raise HTTPException(status_code=404, detail="Catalog item not found")
    interaction = interaction_service.record_interaction(
        db,
        user=current_user,
        **payload.model_dump(),
    )
    db.commit()
    return interaction


@router.get("/interactions", response_model=list[InteractionOut])
def interactions(db: Db, current_user: CurrentUser, limit: int = Query(100, ge=1, le=500)) -> list[Interaction]:
    return interaction_service.list_interactions_for(db, current_user.id, limit)


@router.post("/favorites/{item_id}", response_model=FavoriteToggleResponse)
def toggle_favorite(item_id: int, db: Db, current_user: CurrentUser) -> FavoriteToggleResponse:
    if db.get(CatalogItem, item_id) is None:
        raise HTTPException(status_code=404, detail="Catalog item not found")
    is_favorite = catalog_service.toggle_favorite(db, current_user.id, item_id)
    db.commit()
    return FavoriteToggleResponse(catalog_item_id=item_id, is_favorite=is_favorite)


@router.get("/favorites", response_model=list[CatalogItemBase])
def favorites(db: Db, current_user: CurrentUser) -> list[CatalogItemBase]:
    stmt = (
        select(CatalogItem)
        .join(Favorite, Favorite.catalog_item_id == CatalogItem.id)
        .options(selectinload(CatalogItem.movie), selectinload(CatalogItem.product))
        .where(Favorite.user_id == current_user.id)
        .order_by(Favorite.created_at.desc())
    )
    return [CatalogItemBase.model_validate(_catalog_payload(item)) for item in db.execute(stmt).scalars().all()]


@router.post("/onboarding", status_code=status.HTTP_204_NO_CONTENT)
def onboarding(payload: OnboardingIn, db: Db, current_user: CurrentUser) -> None:
    interaction_service.save_preferences(db, current_user, payload.model_dump(exclude={"favorite_movie_ids"}))
    for item_id in set(payload.favorite_movie_ids):
        item = db.get(CatalogItem, item_id)
        if item is None or item.item_type != "movie":
            continue
        exists = db.scalar(
            select(Favorite.id).where(Favorite.user_id == current_user.id, Favorite.catalog_item_id == item_id)
        )
        if exists is None:
            db.add(Favorite(user_id=current_user.id, catalog_item_id=item_id))
    db.commit()


@router.get("/recommendations/{item_type}", response_model=RecommendationResponse)
def recommendations(
    item_type: Literal["movie", "product"],
    db: Db,
    current_user: CurrentUser,
    top_k: int | None = Query(default=None, ge=1),
) -> RecommendationResponse:
    settings = get_settings()
    requested_top_k = min(top_k or settings.recommendation_default_top_k, settings.recommendation_max_top_k)
    preferences = interaction_service.load_preferences(db, current_user.id)
    recommender = recommendation_service.fit_recommender_for_domain(db, item_type)
    context = recommendation_service.build_recommendation_context(
        db, current_user.id, item_type, requested_top_k, preferences
    )
    ranked = recommender.recommend(context)
    item_ids = [recommendation.item_id for recommendation in ranked]
    if item_ids:
        stmt = (
            select(CatalogItem)
            .options(selectinload(CatalogItem.movie), selectinload(CatalogItem.product))
            .where(CatalogItem.id.in_(item_ids), CatalogItem.item_type == item_type, CatalogItem.is_active.is_(True))
        )
        item_map = {item.id: item for item in db.execute(stmt).scalars().all()}
    else:
        item_map = {}

    batch = RecommendationBatch(
        user_id=current_user.id,
        item_type=item_type,
        model_version="hybrid-v1",
        context={"top_k": requested_top_k, "preferences": preferences},
    )
    db.add(batch)
    db.flush()
    output: list[RecommendationOut] = []
    for recommendation in ranked:
        item = item_map.get(recommendation.item_id)
        if item is None:
            continue
        rank = len(output) + 1
        db.add(
            RecommendationResult(
                batch_id=batch.id,
                catalog_item_id=item.id,
                rank=rank,
                score=recommendation.score,
                reason=recommendation.reason,
                components=dict(recommendation.components),
            )
        )
        detail = _catalog_payload(item)
        output.append(
            RecommendationOut(
                catalog_item_id=item.id,
                title=item.title,
                image_url=item.image_url,
                score=recommendation.score,
                reason=recommendation.reason,
                components=dict(recommendation.components),
                item_type=item_type,
                detail=detail,
            )
        )
    db.commit()
    return RecommendationResponse(
        user_id=current_user.id,
        item_type=item_type,
        model_version="hybrid-v1",
        top_k=requested_top_k,
        recommendations=output,
    )


@router.get("/admin/metrics", response_model=AdminMetrics)
def admin_metrics(db: Db, _: CurrentAdmin) -> AdminMetrics:
    def count(model: type[Any]) -> int:
        return int(db.scalar(select(func.count()).select_from(model)) or 0)

    latest_runs: dict[str, SyncRun] = {}
    for sync_run in db.execute(select(SyncRun).order_by(SyncRun.started_at.desc())).scalars():
        latest_runs.setdefault(sync_run.source, sync_run)
    return AdminMetrics(
        users=count(User),
        movies=count(Movie),
        products=count(Product),
        interactions=count(Interaction),
        favorites=count(Favorite),
        last_sync=latest_runs,
    )


@router.post("/admin/sync/tmdb")
async def sync_tmdb(db: Db, _: CurrentAdmin) -> dict[str, Any]:
    try:
        result = await TMDBSyncService(db, get_settings()).sync(
            collections=("trending", "popular", "upcoming"),
            max_pages=2,
        )
    except TMDBConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TMDBRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except TMDBError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "source": result.source,
        "status": result.status,
        "received": result.received,
        "created": result.created,
        "updated": result.updated,
        "failed": result.failed,
        "sync_run_id": result.sync_run_id,
    }


@router.post("/admin/sync/ebay")
async def sync_ebay(db: Db, _: CurrentAdmin, payload: EbaySyncRequest | None = None) -> dict[str, Any]:
    try:
        request = payload or EbaySyncRequest()
        selected = tuple(query for query in EBAY_SYNC_QUERIES if not request.categories or query.key in request.categories)
        result = await EbaySyncService(db, get_settings()).sync(selected, max_pages=request.max_pages)
    except EbayConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EbayRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except EbayAuthenticationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except EbayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "source": result.source,
        "status": result.status,
        "received": result.received,
        "unique_items": result.unique_items,
        "created": result.created,
        "updated": result.updated,
        "deactivated": result.deactivated,
        "failed": result.failed,
        "sync_run_id": result.sync_run_id,
    }
