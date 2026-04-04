from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
FRONTEND_DIST_ROOT = PROJECT_ROOT / "frontend" / "dist"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.cache import CacheService
from tenderhack.dataset_layout import (
    resolve_fasttext_model_path,
    resolve_preprocessed_db_path,
    resolve_raw_ste_path,
    resolve_search_db_path,
    resolve_synonyms_path,
)
from tenderhack.descriptions import CatalogDescriptionService
from tenderhack.offers import OfferLookupService
from tenderhack.online_state import OnlineStateService
from tenderhack.personalization import PersonalizationService
from tenderhack.personalization_runtime import PersonalizationRuntimeService
from tenderhack.search import SearchService
from tenderhack.text import normalize_text, tokenize, unique_preserve_order

from .runtime_store import RuntimeStateStore


@dataclass
class AppSettings:
    search_db_path: Path = resolve_search_db_path(PROJECT_ROOT)
    preprocessed_db_path: Path = resolve_preprocessed_db_path(PROJECT_ROOT)
    synonyms_path: Path = resolve_synonyms_path(PROJECT_ROOT)
    fasttext_model_path: Path = resolve_fasttext_model_path(PROJECT_ROOT)
    personalization_model_path: Path = PROJECT_ROOT / "artifacts" / "personalization_model.cbm"
    raw_ste_catalog_path: Path = resolve_raw_ste_path(PROJECT_ROOT)
    redis_url: Optional[str] = "memory://"
    semantic_backend: str = "auto"
    login_cache_ttl_seconds: int = 1800
    search_cache_ttl_seconds: int = 120
    suggestions_cache_ttl_seconds: int = 300
    user_profile_cache_ttl_seconds: int = 1800
    offer_lookup_cache_ttl_seconds: int = 1800
    session_state_ttl_seconds: int = 86400

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            search_db_path=Path(os.getenv("TENDERHACK_SEARCH_DB", str(cls.search_db_path))),
            preprocessed_db_path=Path(os.getenv("TENDERHACK_PREPROCESSED_DB", str(cls.preprocessed_db_path))),
            synonyms_path=Path(os.getenv("TENDERHACK_SYNONYMS_PATH", str(cls.synonyms_path))),
            fasttext_model_path=Path(os.getenv("TENDERHACK_FASTTEXT_MODEL_PATH", str(cls.fasttext_model_path))),
            personalization_model_path=Path(
                os.getenv("TENDERHACK_PERSONALIZATION_MODEL_PATH", str(cls.personalization_model_path))
            ),
            raw_ste_catalog_path=Path(os.getenv("TENDERHACK_RAW_STE_CATALOG_PATH", str(cls.raw_ste_catalog_path))),
            redis_url=os.getenv("TENDERHACK_REDIS_URL") or cls.redis_url,
            semantic_backend=os.getenv("TENDERHACK_SEMANTIC_BACKEND", cls.semantic_backend),
            login_cache_ttl_seconds=int(os.getenv("TENDERHACK_LOGIN_CACHE_TTL_SECONDS", cls.login_cache_ttl_seconds)),
            search_cache_ttl_seconds=int(
                os.getenv("TENDERHACK_SEARCH_CACHE_TTL_SECONDS", cls.search_cache_ttl_seconds)
            ),
            suggestions_cache_ttl_seconds=int(
                os.getenv("TENDERHACK_SUGGESTIONS_CACHE_TTL_SECONDS", cls.suggestions_cache_ttl_seconds)
            ),
            user_profile_cache_ttl_seconds=int(
                os.getenv("TENDERHACK_USER_PROFILE_CACHE_TTL_SECONDS", cls.user_profile_cache_ttl_seconds)
            ),
            offer_lookup_cache_ttl_seconds=int(
                os.getenv("TENDERHACK_OFFER_LOOKUP_CACHE_TTL_SECONDS", cls.offer_lookup_cache_ttl_seconds)
            ),
            session_state_ttl_seconds=int(
                os.getenv("TENDERHACK_SESSION_STATE_TTL_SECONDS", cls.session_state_ttl_seconds)
            ),
        )


def build_asset_report(settings: AppSettings) -> Dict[str, object]:
    required = {
        "search_db": settings.search_db_path,
        "preprocessed_db": settings.preprocessed_db_path,
        "synonyms": settings.synonyms_path,
    }
    optional = {
        "fasttext_model": settings.fasttext_model_path,
        "personalization_model": settings.personalization_model_path,
        "raw_ste_catalog": settings.raw_ste_catalog_path,
        "frontend_dist": FRONTEND_DIST_ROOT,
    }
    required_report = {
        name: {
            "path": str(path),
            "exists": Path(path).exists(),
        }
        for name, path in required.items()
    }
    optional_report = {
        name: {
            "path": str(path),
            "exists": Path(path).exists(),
        }
        for name, path in optional.items()
    }
    return {
        "ready": all(item["exists"] for item in required_report.values()),
        "required": required_report,
        "optional": optional_report,
    }


class LoginRequest(BaseModel):
    inn: str = Field(min_length=1)


class UserPayload(BaseModel):
    id: str
    inn: str
    region: str
    viewedCategories: List[str] = Field(default_factory=list)


class SearchUserContext(BaseModel):
    id: Optional[str] = None
    inn: Optional[str] = None
    region: Optional[str] = None
    viewedCategories: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    userContext: Optional[SearchUserContext] = None
    viewedCategories: List[str] = Field(default_factory=list)
    bouncedCategories: List[str] = Field(default_factory=list)
    topK: int = Field(default=20, ge=1, le=50)


class ProductPayload(BaseModel):
    id: str
    name: str
    category: str
    price: float
    supplierInn: str
    descriptionPreview: Optional[str] = None
    reasonToShow: Optional[str] = None


class SearchResponsePayload(BaseModel):
    items: List[ProductPayload]
    totalCount: int
    correctedQuery: Optional[str] = None


class EventRequest(BaseModel):
    userId: Optional[str] = None
    inn: Optional[str] = None
    region: Optional[str] = None
    eventType: str = Field(min_length=1)
    steId: Optional[str] = None
    category: Optional[str] = None
    query: Optional[str] = None
    durationMs: Optional[int] = Field(default=None, ge=0)
    metadata: Dict[str, object] = Field(default_factory=dict)


class SessionStatePayload(BaseModel):
    userId: str
    sessionVersion: int = 0
    clickedSteIds: List[str] = Field(default_factory=list)
    cartSteIds: List[str] = Field(default_factory=list)
    recentCategories: List[str] = Field(default_factory=list)
    viewedCategories: List[str] = Field(default_factory=list)
    bouncedCategories: List[str] = Field(default_factory=list)
    eventCount: int = 0


class ItemPayload(BaseModel):
    id: str
    name: str
    category: str
    normalizedCategory: str
    attributeKeys: List[str] = Field(default_factory=list)
    attributeCount: int = 0
    keyTokens: List[str] = Field(default_factory=list)
    price: float = 0.0
    supplierInn: str = "не указан"
    supplierRegion: str = ""
    offerCount: int = 0


class OfferPayload(BaseModel):
    offerId: str
    steId: str
    supplierInn: str
    supplierRegion: str
    unitPrice: float
    offerScore: float
    explanation: List[str] = Field(default_factory=list)


class OffersResponsePayload(BaseModel):
    itemId: str
    offers: List[OfferPayload] = Field(default_factory=list)


class CartAddRequest(BaseModel):
    userId: str = Field(min_length=1)
    steId: str = Field(min_length=1)
    quantity: int = Field(default=1, ge=1, le=999)


class CartItemPayload(BaseModel):
    steId: str
    name: str
    category: str
    quantity: int
    price: float
    supplierInn: str
    addedAt: str


class CartResponsePayload(BaseModel):
    userId: str
    items: List[CartItemPayload] = Field(default_factory=list)
    totalItems: int = 0
    totalAmount: float = 0.0


class ProcurementCreateRequest(BaseModel):
    userId: str = Field(min_length=1)
    procurementType: str = Field(default="direct_purchase", min_length=1)


class ProcurementPayload(BaseModel):
    procurementId: str
    procurementType: str
    status: str
    itemCount: int
    totalAmount: float
    createdAt: str


class TenderHackApiService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._validate_required_paths()
        self.cache_service = CacheService(url=settings.redis_url, prefix="tenderhack")
        self.description_service = CatalogDescriptionService(raw_catalog_path=settings.raw_ste_catalog_path)
        self.online_state_service = OnlineStateService(
            cache_service=self.cache_service,
            session_ttl_seconds=settings.session_state_ttl_seconds,
        )
        self.runtime_state_store = RuntimeStateStore()
        self.search_service = SearchService(
            search_db_path=settings.search_db_path,
            synonyms_path=settings.synonyms_path,
            semantic_backend=settings.semantic_backend,
            fasttext_model_path=settings.fasttext_model_path,
        )
        self.personalization_service = PersonalizationService(db_path=settings.preprocessed_db_path)
        self.personalization_runtime_service = PersonalizationRuntimeService(
            db_path=settings.preprocessed_db_path,
            model_path=settings.personalization_model_path,
            cache_service=self.cache_service,
            base_profile_ttl_seconds=settings.user_profile_cache_ttl_seconds,
        )
        self.offer_lookup_service = OfferLookupService(
            db_path=settings.preprocessed_db_path,
            cache_service=self.cache_service,
            lookup_ttl_seconds=settings.offer_lookup_cache_ttl_seconds,
        )

    def close(self) -> None:
        self.search_service.close()
        self.personalization_service.close()
        self.personalization_runtime_service.close()
        self.offer_lookup_service.close()
        self.description_service.close()
        self.cache_service.close()

    def _validate_required_paths(self) -> None:
        missing = [
            str(path)
            for path in [self.settings.search_db_path, self.settings.preprocessed_db_path, self.settings.synonyms_path]
            if not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError("Missing required search assets:\n" + "\n".join(f"- {path}" for path in missing))

    @staticmethod
    def _model_dump(model: BaseModel) -> dict:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    @staticmethod
    def _resolve_user_id(
        user_context: Optional[SearchUserContext] = None,
        fallback_user_id: Optional[str] = None,
        customer_inn: Optional[str] = None,
    ) -> str:
        if fallback_user_id:
            return str(fallback_user_id)
        if user_context and user_context.id:
            return str(user_context.id)
        if customer_inn:
            return f"user-{customer_inn}"
        if user_context and user_context.inn:
            return f"user-{user_context.inn}"
        return "anonymous"

    @staticmethod
    def _empty_profile(customer_region: Optional[str] = None) -> Dict[str, object]:
        return {
            "customer_inn": "",
            "customer_region": customer_region or "",
            "top_categories": [],
            "top_ste": [],
            "regional_categories": [],
            "category_affinity": {},
            "ste_affinity": {},
            "regional_affinity": {},
        }

    @staticmethod
    def _normalize_runtime_event_type(event_type: str, duration_ms: Optional[int]) -> str:
        normalized = str(event_type or "").strip().lower()
        mapping = {
            "search_result_click": "click",
            "item_click": "click",
            "click": "click",
            "item_open": "open",
            "open": "open",
            "view": "open",
            "bounce": "fast_bounce",
            "fast_bounce": "fast_bounce",
            "cart": "cart_add",
            "cart_add": "cart_add",
            "cart_remove": "cart_remove",
            "purchase": "purchase",
            "query_select": "query_select",
        }
        if normalized == "item_close":
            return "fast_bounce" if duration_ms and duration_ms < 3000 else "close"
        return mapping.get(normalized, normalized or "view")

    @staticmethod
    def _normalize_online_event_type(event_type: str) -> str:
        normalized = str(event_type or "").strip().lower()
        mapping = {
            "search_result_click": "search_result_click",
            "item_click": "item_click",
            "click": "item_click",
            "item_open": "item_open",
            "open": "item_open",
            "view": "item_open",
            "bounce": "bounce",
            "fast_bounce": "bounce",
            "item_close": "item_close",
            "cart": "cart_add",
            "cart_add": "cart_add",
            "cart_remove": "cart_remove",
            "purchase": "purchase",
            "query_select": "item_click",
        }
        return mapping.get(normalized, "item_click")

    def _build_combined_session_state(
        self,
        *,
        user_id: str,
        customer_inn: Optional[str] = None,
        customer_region: Optional[str] = None,
        viewed_categories: Optional[List[str]] = None,
        bounced_categories: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        online_state = self.online_state_service.get_session_state(
            user_id=user_id,
            customer_inn=customer_inn,
            customer_region=customer_region,
        )
        runtime_snapshot = self.runtime_state_store.get_session_snapshot(user_id)

        combined_recent_categories = unique_preserve_order(
            [normalize_text(str(value)) for value in online_state.get("recent_categories", []) if value]
            + list(runtime_snapshot.recent_categories)
            + [normalize_text(str(value)) for value in (viewed_categories or []) if value]
        )
        combined_clicked = unique_preserve_order(
            [str(value) for value in online_state.get("clicked_ste_ids", []) if value]
            + list(runtime_snapshot.clicked_ste_ids)
        )
        combined_cart = unique_preserve_order(
            [str(value) for value in online_state.get("cart_ste_ids", []) if value]
            + list(runtime_snapshot.cart_ste_ids)
        )
        combined_bounced = unique_preserve_order(
            [normalize_text(str(value)) for value in online_state.get("bounced_categories", []) if value]
            + list(runtime_snapshot.bounced_categories)
            + [normalize_text(str(value)) for value in (bounced_categories or []) if value]
        )
        combined_viewed = unique_preserve_order(
            list(runtime_snapshot.viewed_categories)
            + [str(value) for value in (viewed_categories or []) if value]
        )

        return {
            "user_id": user_id,
            "customer_inn": customer_inn or str(online_state.get("customer_inn") or ""),
            "customer_region": customer_region or str(online_state.get("customer_region") or ""),
            "version": int(online_state.get("version", 0) or 0),
            "event_count": len(runtime_snapshot.events),
            "recent_categories": combined_recent_categories,
            "clicked_ste_ids": combined_clicked,
            "cart_ste_ids": combined_cart,
            "bounced_categories": combined_bounced,
            "viewed_categories": combined_viewed,
        }

    def _session_payload(self, combined_session: Dict[str, object]) -> SessionStatePayload:
        return SessionStatePayload(
            userId=str(combined_session.get("user_id") or "anonymous"),
            sessionVersion=int(combined_session.get("version", 0) or 0),
            clickedSteIds=[str(value) for value in combined_session.get("clicked_ste_ids", []) if value],
            cartSteIds=[str(value) for value in combined_session.get("cart_ste_ids", []) if value],
            recentCategories=[str(value) for value in combined_session.get("recent_categories", []) if value],
            viewedCategories=[str(value) for value in combined_session.get("viewed_categories", []) if value],
            bouncedCategories=[str(value) for value in combined_session.get("bounced_categories", []) if value],
            eventCount=int(combined_session.get("event_count", 0) or 0),
        )

    def login(self, inn: str) -> UserPayload:
        cache_key = self.cache_service.build_key("login", data={"inn": inn})
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, dict):
            return UserPayload(**cached_payload)

        profile = self.personalization_service.build_customer_profile(customer_inn=inn)
        viewed_categories = [item["category"] for item in profile.get("top_categories", [])[:5]]
        payload = UserPayload(
            id=f"user-{inn}",
            inn=inn,
            region=str(profile.get("customer_region") or ""),
            viewedCategories=viewed_categories,
        )
        self.cache_service.set_json(cache_key, self._model_dump(payload), ttl_seconds=self.settings.login_cache_ttl_seconds)
        return payload

    def search(self, payload: SearchRequest) -> SearchResponsePayload:
        user_context = payload.userContext or SearchUserContext()
        user_id = self._resolve_user_id(user_context, customer_inn=user_context.inn)
        combined_session = self._build_combined_session_state(
            user_id=user_id,
            customer_inn=user_context.inn,
            customer_region=user_context.region,
            viewed_categories=payload.viewedCategories + user_context.viewedCategories,
            bounced_categories=payload.bouncedCategories,
        )
        cache_key = self.cache_service.build_key(
            "search",
            data=self._search_cache_data(payload, combined_session=combined_session),
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, dict):
            return SearchResponsePayload(**cached_payload)

        raw_payload = self.search_service.search(query=payload.query, top_k=max(payload.topK * 5, 60))
        results = list(raw_payload["results"])
        session_categories = unique_preserve_order(
            [str(value) for value in user_context.viewedCategories if value]
            + [str(value) for value in payload.viewedCategories if value]
            + [str(value) for value in combined_session.get("viewed_categories", []) if value]
        )

        has_personal_context = bool(
            user_context.inn
            or user_context.region
            or session_categories
            or combined_session["clicked_ste_ids"]
            or combined_session["cart_ste_ids"]
            or combined_session["recent_categories"]
        )

        if has_personal_context:
            personalization_query = (
                raw_payload["query"].get("corrected_query")
                or raw_payload["query"].get("normalized_query")
                or payload.query
            )
            results = self.personalization_runtime_service.rerank_candidates(
                query=str(personalization_query),
                candidates=results,
                user_id=user_id,
                customer_inn=user_context.inn,
                customer_region=user_context.region,
                session_categories=session_categories,
                session_state=combined_session,
            )
        else:
            for item in results:
                item["final_score"] = float(item.get("search_score", 0.0))
                item["top_reason_codes"] = []
                item["reasons"] = ["оставлено выше за счёт базовой текстовой релевантности"]

        results.sort(
            key=lambda item: (
                float(item.get("final_score", item.get("search_score", 0.0))),
                float(item.get("personalization_score", 0.0)),
                float(item.get("search_score", 0.0)),
            ),
            reverse=True,
        )

        ste_ids = [str(item["ste_id"]) for item in results[: payload.topK * 2]]
        offer_lookup = self.offer_lookup_service.get_offer_lookup(ste_ids)
        description_lookup = self.description_service.get_previews(
            ste_ids,
            fallback_by_ste_id={
                str(item["ste_id"]): {"attribute_keys": str(item.get("attribute_keys") or "")}
                for item in results[: payload.topK * 2]
            },
        )

        products: List[ProductPayload] = []
        for item in results:
            if len(products) >= payload.topK:
                break
            ste_id = str(item["ste_id"])
            offer = offer_lookup.get(ste_id, {})
            products.append(
                ProductPayload(
                    id=ste_id,
                    name=str(item.get("clean_name") or item.get("normalized_name") or ste_id),
                    category=str(item.get("category") or ""),
                    price=round(float(offer.get("price", 0.0) or 0.0), 2),
                    supplierInn=str(offer.get("supplier_inn") or "не указан"),
                    descriptionPreview=description_lookup.get(ste_id),
                    reasonToShow=self._map_reason_to_show(
                        reason_codes=[str(code) for code in item.get("top_reason_codes", [])],
                        category=str(item.get("category") or ""),
                        session_categories=list(combined_session.get("recent_categories", [])),
                        is_bounced="SESSION_BOUNCE_PENALTY" in set(item.get("top_reason_codes", [])),
                    ),
                )
            )

        corrected_query = raw_payload["query"].get("corrected_query") or None
        normalized_query = raw_payload["query"].get("normalized_query") or None
        if corrected_query == normalized_query:
            corrected_query = None

        response_payload = SearchResponsePayload(
            items=products,
            totalCount=len(results),
            correctedQuery=corrected_query,
        )
        self.cache_service.set_json(
            cache_key,
            self._model_dump(response_payload),
            ttl_seconds=self.settings.search_cache_ttl_seconds,
        )
        return response_payload

    def record_event(self, payload: EventRequest) -> SessionStatePayload:
        user_id = self._resolve_user_id(fallback_user_id=payload.userId, customer_inn=payload.inn)
        runtime_event_type = self._normalize_runtime_event_type(payload.eventType, payload.durationMs)
        self.runtime_state_store.track_event(
            user_id=user_id,
            event_type=runtime_event_type,
            ste_id=payload.steId,
            category=payload.category,
            query=payload.query,
            metadata=payload.metadata,
        )

        online_event_type = self._normalize_online_event_type(payload.eventType)
        duration_ms = payload.durationMs
        if online_event_type == "bounce" and not duration_ms:
            duration_ms = 500
        self.online_state_service.record_event(
            user_id=user_id,
            customer_inn=payload.inn,
            customer_region=payload.region,
            event_type=online_event_type,
            ste_id=payload.steId,
            category=payload.category,
            duration_ms=duration_ms,
        )

        combined_session = self._build_combined_session_state(
            user_id=user_id,
            customer_inn=payload.inn,
            customer_region=payload.region,
        )
        return self._session_payload(combined_session)

    def get_item(self, ste_id: str) -> ItemPayload:
        item = self.search_service.get_ste_item(ste_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"STE {ste_id} not found")

        offer_lookup = self.offer_lookup_service.get_offer_lookup([ste_id]).get(ste_id, {})
        attribute_keys = [
            part.strip()
            for part in str(item.get("attribute_keys") or "").replace(";", "|").split("|")
            if part.strip()
        ]
        key_tokens = [token for token in str(item.get("key_tokens") or "").split() if token]
        return ItemPayload(
            id=str(item["ste_id"]),
            name=str(item.get("clean_name") or item.get("normalized_name") or ste_id),
            category=str(item.get("category") or ""),
            normalizedCategory=str(item.get("normalized_category") or ""),
            attributeKeys=attribute_keys,
            attributeCount=int(item.get("attribute_count") or 0),
            keyTokens=key_tokens,
            price=round(float(offer_lookup.get("price", 0.0) or 0.0), 2),
            supplierInn=str(offer_lookup.get("supplier_inn") or "не указан"),
            supplierRegion=str(offer_lookup.get("supplier_region") or ""),
            offerCount=int(offer_lookup.get("offer_count") or 0),
        )

    def get_offers(
        self,
        ste_id: str,
        *,
        user_id: Optional[str] = None,
        customer_inn: Optional[str] = None,
        customer_region: Optional[str] = None,
    ) -> OffersResponsePayload:
        item = self.get_item(ste_id)
        offers = self.offer_lookup_service.get_offer_candidates(ste_id=ste_id, category=item.category, limit=20)
        if not offers:
            return OffersResponsePayload(itemId=ste_id, offers=[])

        profile = (
            self.personalization_service.build_customer_profile(
                customer_inn=customer_inn,
                customer_region=customer_region,
            )
            if customer_inn
            else self._empty_profile(customer_region=customer_region)
        )
        combined_session = self._build_combined_session_state(
            user_id=user_id or "anonymous",
            customer_inn=customer_inn,
            customer_region=customer_region,
        )
        reranked = self.personalization_service.rerank_offers(
            offers,
            profile,
            session_state=combined_session,
        )
        return OffersResponsePayload(
            itemId=ste_id,
            offers=[
                OfferPayload(
                    offerId=str(offer["offer_id"]),
                    steId=str(offer["ste_id"]),
                    supplierInn=str(offer.get("supplier_inn") or "не указан"),
                    supplierRegion=str(offer.get("supplier_region") or ""),
                    unitPrice=round(float(offer.get("unit_price", 0.0) or 0.0), 2),
                    offerScore=round(float(offer.get("final_offer_score", offer.get("offer_score", 0.0)) or 0.0), 4),
                    explanation=[str(value) for value in offer.get("offer_explanation", [])],
                )
                for offer in reranked
            ],
        )

    def add_to_cart(self, payload: CartAddRequest) -> CartResponsePayload:
        item = self.get_item(payload.steId)
        self.runtime_state_store.add_to_cart(
            user_id=payload.userId,
            ste_id=item.id,
            name=item.name,
            category=item.category,
            quantity=payload.quantity,
            price=item.price,
            supplier_inn=item.supplierInn,
        )
        self.record_event(
            EventRequest(
                userId=payload.userId,
                eventType="cart_add",
                steId=item.id,
                category=item.category,
                metadata={"quantity": payload.quantity},
            )
        )
        return self.get_cart(payload.userId)

    def get_cart(self, user_id: str) -> CartResponsePayload:
        items = self.runtime_state_store.get_cart(user_id)
        total_amount = sum(float(item.price) * int(item.quantity) for item in items)
        total_items = sum(int(item.quantity) for item in items)
        return CartResponsePayload(
            userId=user_id,
            items=[CartItemPayload(**item.to_mapping()) for item in items],
            totalItems=total_items,
            totalAmount=round(total_amount, 2),
        )

    def create_procurement(self, payload: ProcurementCreateRequest) -> ProcurementPayload:
        cart = self.get_cart(payload.userId)
        if not cart.items:
            raise HTTPException(status_code=400, detail="Cart is empty")

        for item in cart.items:
            self.record_event(
                EventRequest(
                    userId=payload.userId,
                    eventType="purchase",
                    steId=item.steId,
                    category=item.category,
                )
            )

        procurement_id = f"proc-{payload.userId}-{int(datetime.now(timezone.utc).timestamp())}"
        created_at = datetime.now(timezone.utc).isoformat()
        self.runtime_state_store.clear_cart(payload.userId)
        return ProcurementPayload(
            procurementId=procurement_id,
            procurementType=payload.procurementType,
            status="created",
            itemCount=cart.totalItems,
            totalAmount=cart.totalAmount,
            createdAt=created_at,
        )

    def suggestions(self, query: str, top_k: int = 5) -> List[str]:
        cache_key = self.cache_service.build_key("suggestions", data={"query": query, "top_k": top_k})
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, list):
            return [str(item) for item in cached_payload]

        payload = self.search_service.search(query=query, top_k=max(top_k * 3, 12))
        suggestions: List[str] = []
        query_payload = payload["query"]
        corrected_query = query_payload.get("corrected_query")
        normalized_query = query_payload.get("normalized_query")
        if corrected_query and corrected_query != normalized_query:
            suggestions.append(corrected_query)
        suggestions.extend(self._build_abstract_suggestions(query=query, query_payload=query_payload, results=payload["results"]))
        result = unique_preserve_order(suggestions)[:top_k]
        self.cache_service.set_json(cache_key, result, ttl_seconds=self.settings.suggestions_cache_ttl_seconds)
        return result

    @staticmethod
    def _abstract_name_phrase(name: str, query: str) -> str:
        query_tokens = tokenize(query)
        name_tokens = tokenize(name)
        if not name_tokens:
            return ""

        limit = 2 if len(query_tokens) <= 1 else 3
        phrase_tokens: List[str] = []
        for token in name_tokens:
            if token.isdigit() or any(char.isdigit() for char in token):
                break
            phrase_tokens.append(token)
            if len(phrase_tokens) >= limit:
                break
        if len(phrase_tokens) < 2:
            return ""
        return " ".join(phrase_tokens)

    @staticmethod
    def _compact_category_phrase(category: str) -> str:
        category_tokens = [token for token in tokenize(category) if not token.isdigit()]
        if not category_tokens:
            return ""
        return " ".join(category_tokens[:5])

    @classmethod
    def _build_abstract_suggestions(cls, *, query: str, query_payload: dict, results: List[dict]) -> List[str]:
        query_norm = normalize_text(query)
        expanded_tokens = [str(token) for token in query_payload.get("expanded_tokens", []) if token]
        corrected_query = str(query_payload.get("corrected_query") or "")
        query_tokens = unique_preserve_order(tokenize(query) + tokenize(corrected_query) + expanded_tokens)
        suggestions: List[str] = []

        for synonym_rule in query_payload.get("applied_synonyms", []):
            for target in synonym_rule.get("targets", []):
                candidate = normalize_text(str(target))
                if not candidate or candidate == query_norm:
                    continue
                suggestions.append(candidate)

        for item in results:
            name_phrase = cls._abstract_name_phrase(str(item.get("clean_name") or ""), query)
            category_phrase = cls._compact_category_phrase(str(item.get("category") or ""))
            for candidate in [name_phrase, category_phrase]:
                candidate_norm = normalize_text(candidate)
                if not candidate_norm or candidate_norm == query_norm:
                    continue
                if query_tokens and not any(token in candidate_norm.split() for token in query_tokens):
                    continue
                suggestions.append(candidate)

        return unique_preserve_order(suggestions)

    @staticmethod
    def _search_cache_data(
        payload: SearchRequest,
        combined_session: Optional[Dict[str, object]] = None,
        server_session: Optional[Dict[str, object]] = None,
    ) -> dict:
        user_context = payload.userContext or SearchUserContext()
        active_session = combined_session or server_session or {}
        return {
            "query": payload.query,
            "user_id": user_context.id,
            "user_inn": user_context.inn,
            "user_region": user_context.region,
            "user_viewed_categories": unique_preserve_order([str(value) for value in user_context.viewedCategories if value]),
            "viewed_categories": unique_preserve_order([str(value) for value in payload.viewedCategories if value]),
            "bounced_categories": unique_preserve_order(
                [normalize_text(str(value)) for value in payload.bouncedCategories if value]
            ),
            "top_k": int(payload.topK),
            "session_version": int(active_session.get("version", 0) or 0),
            "session_recent_categories": unique_preserve_order(
                [normalize_text(str(value)) for value in active_session.get("recent_categories", []) if value]
            ),
            "session_clicked_ste_ids": unique_preserve_order(
                [str(value) for value in active_session.get("clicked_ste_ids", []) if value]
            ),
            "session_cart_ste_ids": unique_preserve_order(
                [str(value) for value in active_session.get("cart_ste_ids", []) if value]
            ),
            "session_bounced_categories": unique_preserve_order(
                [normalize_text(str(value)) for value in active_session.get("bounced_categories", []) if value]
            ),
        }

    @staticmethod
    def _map_reason_to_show(
        reason_codes: List[str],
        category: str,
        session_categories: List[str],
        is_bounced: bool,
    ) -> Optional[str]:
        if is_bounced:
            return None

        codes = {str(code) for code in reason_codes}
        session_category_set = {normalize_text(value) for value in session_categories if value}
        category_norm = normalize_text(category)

        if codes & {"SESSION_CART_BOOST", "SESSION_CATEGORY_CONTINUATION"}:
            return "Продолжить подбор в этой категории"
        if codes & {"USER_CATEGORY_AFFINITY", "USER_REPEAT_BUY", "RECENT_SIMILAR_PURCHASE", "SUPPLIER_AFFINITY"}:
            return "На основе ваших закупок"
        if codes & {"SESSION_CLICK_BOOST", "SESSION_CLICK", "SESSION_CART"}:
            return "На основе ваших закупок"
        if codes & {"REGIONAL_POPULARITY"}:
            return "Популярно в вашем регионе"
        if codes & {"SIMILAR_CUSTOMER_POPULARITY"}:
            return "Популярно у похожих заказчиков"
        if codes & {"SESSION_CATEGORY_BOOST"}:
            return "Продолжить подбор в этой категории"
        if category_norm and category_norm in session_category_set:
            return "Продолжить подбор в этой категории"
        return None


def create_app(settings: Optional[AppSettings] = None) -> FastAPI:
    active_settings = settings or AppSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        asset_report = build_asset_report(active_settings)
        startup_error: Optional[str] = None
        service: Optional[TenderHackApiService] = None
        if asset_report["ready"]:
            try:
                service = TenderHackApiService(active_settings)
            except Exception as exc:  # pragma: no cover
                startup_error = str(exc)
        else:
            missing = [
                name
                for name, payload in asset_report["required"].items()
                if not bool(payload["exists"])
            ]
            startup_error = "Missing required search assets: " + ", ".join(missing)

        app.state.service = service
        app.state.asset_report = asset_report
        app.state.startup_error = startup_error
        yield
        if service is not None:
            service.close()

    app = FastAPI(title="TenderHack Search API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_service(request: Request) -> TenderHackApiService:
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(
                status_code=503,
                detail=getattr(request.app.state, "startup_error", "Service unavailable"),
            )
        return service

    @app.get("/api/health")
    async def health(request: Request) -> Dict[str, object]:
        service = getattr(request.app.state, "service", None)
        return {
            "status": "ok" if service is not None else "degraded",
            "service": "ready" if service is not None else "unavailable",
            "assets": getattr(request.app.state, "asset_report", build_asset_report(active_settings)),
            "error": getattr(request.app.state, "startup_error", None),
        }

    @app.post("/api/auth/login", response_model=UserPayload)
    async def login(payload: LoginRequest, request: Request) -> UserPayload:
        return require_service(request).login(payload.inn)

    @app.post("/api/search", response_model=SearchResponsePayload)
    async def search(payload: SearchRequest, request: Request) -> SearchResponsePayload:
        return require_service(request).search(payload)

    @app.post("/api/event", response_model=SessionStatePayload)
    async def record_event(payload: EventRequest, request: Request) -> SessionStatePayload:
        return require_service(request).record_event(payload)

    @app.get("/api/search/suggestions", response_model=List[str])
    async def suggestions(
        request: Request,
        q: str = Query(min_length=1),
        top_k: int = Query(default=5, ge=1, le=10),
    ) -> List[str]:
        return require_service(request).suggestions(query=q, top_k=top_k)

    @app.get("/api/items/{ste_id}", response_model=ItemPayload)
    async def get_item(ste_id: str, request: Request) -> ItemPayload:
        return require_service(request).get_item(ste_id)

    @app.get("/api/items/{ste_id}/offers", response_model=OffersResponsePayload)
    async def get_offers(
        ste_id: str,
        request: Request,
        user_id: Optional[str] = Query(default=None),
        customer_inn: Optional[str] = Query(default=None),
        customer_region: Optional[str] = Query(default=None),
    ) -> OffersResponsePayload:
        return require_service(request).get_offers(
            ste_id,
            user_id=user_id,
            customer_inn=customer_inn,
            customer_region=customer_region,
        )

    @app.post("/api/cart/add", response_model=CartResponsePayload)
    async def add_to_cart(payload: CartAddRequest, request: Request) -> CartResponsePayload:
        return require_service(request).add_to_cart(payload)

    @app.get("/api/cart", response_model=CartResponsePayload)
    async def get_cart(request: Request, user_id: str = Query(min_length=1)) -> CartResponsePayload:
        return require_service(request).get_cart(user_id)

    @app.post("/api/cart/create-procurement", response_model=ProcurementPayload)
    async def create_procurement(payload: ProcurementCreateRequest, request: Request) -> ProcurementPayload:
        return require_service(request).create_procurement(payload)

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    if FRONTEND_DIST_ROOT.exists():
        assets_dir = FRONTEND_DIST_ROOT / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        async def frontend_index():
            return FileResponse(FRONTEND_DIST_ROOT / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def frontend_spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")

            candidate = (FRONTEND_DIST_ROOT / full_path).resolve()
            try:
                candidate.relative_to(FRONTEND_DIST_ROOT)
            except ValueError:
                raise HTTPException(status_code=404, detail="Not found")

            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST_ROOT / "index.html")

    return app


app = create_app()
