from __future__ import annotations

import math
import os
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.cache import CacheService
from tenderhack.cart_boost import CartBoostModifier, InMemoryCartStorage
from tenderhack.descriptions import CatalogDescriptionService
from tenderhack.online_state import OnlineStateService
from tenderhack.offers import OfferLookupService
from tenderhack.penalization import InMemorySkipStorage, InteractionTracker, RankingModifier
from tenderhack.personalization import PersonalizationService
from tenderhack.personalization_runtime import PersonalizationRuntimeService
from tenderhack.search import SearchService
from tenderhack.search_rerank_model import SearchRerankPredictor
from tenderhack.text import normalize_text, stem_token, stem_tokens, tokenize, unique_preserve_order


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


def _optional_env_path(name: str) -> Optional[Path]:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class AppSettings:
    search_db_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_search.sqlite"
    preprocessed_db_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_preprocessed.sqlite"
    synonyms_path: Path = PROJECT_ROOT / "data" / "reference" / "search_synonyms.json"
    fasttext_model_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_fasttext.bin"
    personalization_model_path: Path = PROJECT_ROOT / "artifacts" / "personalization_model.cbm"
    search_rerank_enabled: bool = True
    search_rerank_model_path: Optional[Path] = None
    search_rerank_metadata_path: Optional[Path] = None
    raw_ste_catalog_path: Path = PROJECT_ROOT / "СТЕ_20260403.csv"
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
            search_db_path=_env_path("TENDERHACK_SEARCH_DB", cls.search_db_path),
            preprocessed_db_path=_env_path("TENDERHACK_PREPROCESSED_DB", cls.preprocessed_db_path),
            synonyms_path=_env_path("TENDERHACK_SYNONYMS_PATH", cls.synonyms_path),
            fasttext_model_path=_env_path("TENDERHACK_FASTTEXT_MODEL_PATH", cls.fasttext_model_path),
            personalization_model_path=_env_path("TENDERHACK_PERSONALIZATION_MODEL_PATH", cls.personalization_model_path),
            search_rerank_enabled=_env_bool("TENDERHACK_SEARCH_RERANK_ENABLED", cls.search_rerank_enabled),
            search_rerank_model_path=_optional_env_path("TENDERHACK_SEARCH_RERANK_MODEL_PATH"),
            search_rerank_metadata_path=_optional_env_path("TENDERHACK_SEARCH_RERANK_METADATA_PATH"),
            raw_ste_catalog_path=Path(os.getenv("TENDERHACK_RAW_STE_CATALOG_PATH", cls.raw_ste_catalog_path)),
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


class LoginRequest(BaseModel):
    inn: str = Field(min_length=1)


class UserPayload(BaseModel):
    id: str
    inn: str
    region: str
    viewedCategories: List[str] = Field(default_factory=list)
    topCategories: List[dict] = Field(default_factory=list)
    frequentProducts: List[dict] = Field(default_factory=list)


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
    limit: Optional[int] = Field(default=None, ge=1, le=50)
    offset: int = Field(default=0, ge=0)
    min_score: float = Field(default=0.55, ge=0.0, le=1.0)


class ProductPayload(BaseModel):
    id: str
    name: str
    category: str
    price: float
    offerCount: int = 0
    supplierInn: str
    descriptionPreview: Optional[str] = None
    reasonToShow: Optional[str] = None


class SearchResponsePayload(BaseModel):
    items: List[ProductPayload]
    totalCount: int
    total_found: int
    has_more: bool
    correctedQuery: Optional[str] = None


class SuggestionPayload(BaseModel):
    text: str
    type: Literal["product", "category", "correction", "query"]
    reason: Optional[str] = None
    score: float


class EventRequest(BaseModel):
    userId: Optional[str] = None
    inn: Optional[str] = None
    region: Optional[str] = None
    eventType: Literal[
        "search_result_click",
        "item_open",
        "item_close",
        "bounce",
        "cart_add",
        "cart_remove",
        "purchase",
        "item_click",
    ]
    steId: Optional[str] = None
    category: Optional[str] = None
    durationMs: Optional[int] = Field(default=None, ge=0)


class EventResponsePayload(BaseModel):
    status: str
    userId: str
    sessionVersion: int
    recentCategories: List[str] = Field(default_factory=list)
    clickedSteIds: List[str] = Field(default_factory=list)
    cartSteIds: List[str] = Field(default_factory=list)
    bouncedCategories: List[str] = Field(default_factory=list)


class TenderHackApiService:
    LOGIN_CACHE_VERSION = 6
    SEARCH_CACHE_VERSION = 9
    SUGGESTIONS_CACHE_VERSION = 16
    _SUGGESTION_TYPE_PRIORITY = {
        "correction": 4,
        "product": 3,
        "query": 2,
        "category": 1,
    }

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._validate_required_paths()
        self.cache_service = CacheService(url=settings.redis_url, prefix="tenderhack")
        self.description_service = CatalogDescriptionService(raw_catalog_path=settings.raw_ste_catalog_path)
        self.online_state_service = OnlineStateService(
            cache_service=self.cache_service,
            session_ttl_seconds=settings.session_state_ttl_seconds,
        )
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
        self.search_rerank_predictor = (
            SearchRerankPredictor(
                model_path=settings.search_rerank_model_path,
                metadata_path=settings.search_rerank_metadata_path,
            )
            if settings.search_rerank_enabled
            else None
        )
        self.search_rerank_cache_token = self._build_search_rerank_cache_token()
        self.offer_lookup_service = OfferLookupService(
            db_path=settings.preprocessed_db_path,
            cache_service=self.cache_service,
            lookup_ttl_seconds=settings.offer_lookup_cache_ttl_seconds,
        )
        self.skip_storage = InMemorySkipStorage()
        self.interaction_tracker = InteractionTracker(self.skip_storage)
        self.ranking_modifier = RankingModifier(self.skip_storage)
        self.cart_storage = InMemoryCartStorage()
        self.cart_boost_modifier = CartBoostModifier(self.cart_storage)

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
            raise FileNotFoundError(
                "Missing required search assets:\n" + "\n".join(f"- {path}" for path in missing)
            )

    def login(self, inn: str) -> UserPayload:
        cache_key = self.cache_service.build_key(
            "login",
            data={"inn": inn, "version": self.LOGIN_CACHE_VERSION},
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, dict):
            return UserPayload(**cached_payload)

        profile = self.personalization_service.build_customer_profile(customer_inn=inn)
        recommended_categories = list(profile.get("recommended_categories") or profile.get("top_categories") or [])
        recommended_ste = list(profile.get("recommended_ste") or profile.get("top_ste") or [])
        viewed_categories = [str(item.get("category") or "") for item in recommended_categories[:5] if item.get("category")]
        region = str(profile.get("customer_region") or "")
        frequent_products = self._load_frequent_products(recommended_ste[:6])
        payload = UserPayload(
            id=f"user-{inn}",
            inn=inn,
            region=region,
            viewedCategories=viewed_categories,
            topCategories=[
                {
                    "category": str(item.get("category") or ""),
                    "purchaseCount": int(item.get("purchase_count") or 0),
                    "totalAmount": round(float(item.get("total_amount") or 0.0), 2),
                    "reason": str(item.get("reason") or "Часто закупалось учреждением"),
                    "recommendationScore": round(float(item.get("recommendation_score") or item.get("weight") or 0.0), 4),
                }
                for item in recommended_categories[:6]
            ],
            frequentProducts=frequent_products,
        )
        self.cache_service.set_json(cache_key, self._model_dump(payload), ttl_seconds=self.settings.login_cache_ttl_seconds)
        return payload

    def _load_frequent_products(self, top_ste: List[dict]) -> List[dict]:
        if not top_ste:
            return []

        ste_ids = [str(item.get("ste_id") or "") for item in top_ste if item.get("ste_id")]
        if not ste_ids:
            return []

        placeholders = ", ".join("?" for _ in ste_ids)
        try:
            rows = self.search_service.conn.execute(
                f"""
                SELECT ste_id, clean_name
                FROM ste_catalog
                WHERE ste_id IN ({placeholders})
                """,
                ste_ids,
            ).fetchall()
        except Exception:
            rows = []

        name_by_ste_id = {str(row["ste_id"]): str(row["clean_name"]) for row in rows}
        result: List[dict] = []
        for item in top_ste:
            ste_id = str(item.get("ste_id") or "")
            if not ste_id:
                continue
            result.append(
                {
                    "steId": ste_id,
                    "name": name_by_ste_id.get(ste_id, ste_id),
                    "category": str(item.get("category") or ""),
                    "purchaseCount": int(item.get("purchase_count") or 0),
                    "totalAmount": round(float(item.get("total_amount") or 0.0), 2),
                    "reason": str(item.get("reason") or "Часто закупалось учреждением"),
                    "recommendationScore": round(float(item.get("recommendation_score") or item.get("weight") or 0.0), 4),
                }
            )
        return result

    def search(self, payload: SearchRequest) -> SearchResponsePayload:
        page_limit = int(payload.limit or payload.topK)
        user_context = payload.userContext or SearchUserContext()
        resolved_user_id = user_context.id or (f"user-{user_context.inn}" if user_context.inn else "anonymous")
        normalized_query = normalize_text(payload.query)
        short_personalized_prefix = bool(
            user_context.inn
            and len(tokenize(normalized_query)) == 1
            and 2 <= len(normalized_query) <= 4
        )
        server_session = self.online_state_service.get_session_state(
            user_id=user_context.id or (f"user-{user_context.inn}" if user_context.inn else None),
            customer_inn=user_context.inn,
            customer_region=user_context.region,
        )
        session_categories = unique_preserve_order(
            list(server_session.get("recent_categories", [])) + payload.viewedCategories + user_context.viewedCategories
        )
        bounced_categories = {
            normalize_text(value)
            for value in [*server_session.get("bounced_categories", []), *payload.bouncedCategories]
            if value
        }
        merged_session_state = {
            "recent_categories": session_categories,
            "clicked_ste_ids": list(server_session.get("clicked_ste_ids", [])),
            "cart_ste_ids": list(server_session.get("cart_ste_ids", [])),
            "bounced_categories": list(bounced_categories),
            "version": int(server_session.get("version", 0) or 0),
        }

        cache_key = self.cache_service.build_key(
            "search",
            data=self._search_cache_data(
                payload,
                server_session=merged_session_state,
                search_rerank_token=self.search_rerank_cache_token,
            ),
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, dict):
            return SearchResponsePayload(**cached_payload)

        same_type_prefix_products = (
            self._resolve_same_type_prefix_products(user_inn=user_context.inn, query=payload.query)
            if short_personalized_prefix
            else []
        )
        raw_limit = max((payload.offset + page_limit) * 5, 60)
        if short_personalized_prefix:
            raw_limit = max(raw_limit, 160)
        raw_payload = self.search_service.search(
            query=payload.query,
            limit=raw_limit,
            offset=0,
            min_score=payload.min_score,
            candidate_limit=max(raw_limit * 4, 250),
        )
        results = list(raw_payload["results"])
        if self.search_rerank_predictor and self.search_rerank_predictor.enabled:
            results = self.search_rerank_predictor.rerank_candidates(
                query=payload.query,
                query_meta=dict(raw_payload["query"]),
                candidates=results,
            )

        if user_context.inn or user_context.region or session_categories:
            personalization_query = (
                raw_payload["query"].get("corrected_query")
                or raw_payload["query"].get("normalized_query")
                or payload.query
            )
            results = self.personalization_runtime_service.rerank_candidates(
                query=str(personalization_query),
                candidates=results,
                user_id=resolved_user_id,
                customer_inn=user_context.inn,
                customer_region=user_context.region,
                session_categories=session_categories,
                session_state=merged_session_state,
            )
        else:
            for item in results:
                item["session_priority"] = 0.0
                item["final_score"] = item.get("search_score", 0.0)
                item["history_priority"] = 0.0
                item["top_reason_codes"] = []
                item["reasons"] = ["оставлено выше за счёт базовой текстовой релевантности"]

        # Шаг 1: мягкий буст за категории из корзины
        results = self.cart_boost_modifier.apply_boost(
            recommendations=results,
            user_id=user_context.id or (f"user-{user_context.inn}" if user_context.inn else "anonymous"),
        )

        # Шаг 2: эвристическая пессимизация (штраф за быстрый отказ)
        results = self.ranking_modifier.apply_penalties(
            recommendations=results, 
            user_id=user_context.id or (f"user-{user_context.inn}" if user_context.inn else "anonymous")
        )

        if same_type_prefix_products:
            self._apply_same_type_prefix_boost(results, same_type_prefix_products)

        for item in results:
            category_norm = normalize_text(str(item.get("category", "")))
            if category_norm and category_norm in bounced_categories:
                item["final_score"] = round(float(item.get("final_score", item.get("search_score", 0.0))) - 100.0, 4)
                item["reason_to_hide"] = "Категория пессимизирована после быстрого отказа"

        results.sort(
            key=lambda item: (
                float(item.get("session_priority", 0.0)),
                float(item.get("final_score", item.get("search_score", 0.0))),
                float(item.get("search_score", 0.0)),
                float(item.get("history_priority", 0.0)),
                float(item.get("personalization_score", 0.0)),
            ),
            reverse=True,
        )

        total_found = int(raw_payload.get("total_found", len(results)))
        paginated_results = results[payload.offset : payload.offset + page_limit]

        ste_ids = [str(item["ste_id"]) for item in paginated_results]
        offer_lookup = self.offer_lookup_service.get_offer_lookup(ste_ids)
        description_lookup = self.description_service.get_previews(
            ste_ids,
            fallback_by_ste_id={
                str(item["ste_id"]): {
                    "attribute_keys": str(item.get("attribute_keys") or ""),
                }
                for item in paginated_results
            },
        )

        products: List[ProductPayload] = []
        for item in paginated_results:
            ste_id = str(item["ste_id"])
            offer = offer_lookup.get(ste_id, {})
            reason_to_show = self._map_reason_to_show(
                reason_codes=item.get("top_reason_codes", []),
                category=str(item.get("category") or ""),
                session_categories=session_categories,
                is_bounced=bool(item.get("reason_to_hide")),
            )
            products.append(
                ProductPayload(
                    id=ste_id,
                    name=str(item.get("clean_name") or item.get("normalized_name") or ste_id),
                    category=str(item.get("category") or ""),
                    price=round(float(offer.get("price", 0.0)), 2),
                    offerCount=int(offer.get("offer_count") or 0),
                    supplierInn=str(offer.get("supplier_inn") or "не указан"),
                    descriptionPreview=description_lookup.get(ste_id),
                    reasonToShow=reason_to_show,
                )
            )

        corrected_query = raw_payload["query"].get("corrected_query") or None
        normalized_query = raw_payload["query"].get("normalized_query") or None
        if corrected_query == normalized_query:
            corrected_query = None

        response_payload = SearchResponsePayload(
            items=products,
            totalCount=total_found,
            total_found=total_found,
            has_more=payload.offset + page_limit < total_found,
            correctedQuery=corrected_query,
        )
        self.cache_service.set_json(
            cache_key,
            self._model_dump(response_payload),
            ttl_seconds=self.settings.search_cache_ttl_seconds,
        )
        return response_payload

    def record_event(self, payload: EventRequest) -> EventResponsePayload:
        resolved_user_id = payload.userId or (f"user-{payload.inn}" if payload.inn else "anonymous")
        if payload.eventType == "item_close" and payload.category:
            self.interaction_tracker.register_view(
                user_id=resolved_user_id,
                category_id=str(payload.category),
                dwell_time_ms=payload.durationMs or 0,
            )

        # Трекинг корзины: обновляем счётчик добавлений/удалений по категории
        if payload.eventType == "cart_add" and payload.category:
            self.cart_storage.increment_cart(resolved_user_id, str(payload.category))
        elif payload.eventType == "cart_remove" and payload.category:
            self.cart_storage.decrement_cart(resolved_user_id, str(payload.category))
        session_state = self.online_state_service.record_event(
            user_id=resolved_user_id,
            customer_inn=payload.inn,
            customer_region=payload.region,
            event_type=payload.eventType,
            ste_id=payload.steId,
            category=payload.category,
            duration_ms=payload.durationMs,
        )
        return EventResponsePayload(
            status="ok",
            userId=resolved_user_id,
            sessionVersion=int(session_state.get("version", 0) or 0),
            recentCategories=[str(value) for value in session_state.get("recent_categories", [])],
            clickedSteIds=[str(value) for value in session_state.get("clicked_ste_ids", [])],
            cartSteIds=[str(value) for value in session_state.get("cart_ste_ids", [])],
            bouncedCategories=[str(value) for value in session_state.get("bounced_categories", [])],
        )

    def suggestions(
        self,
        query: str,
        top_k: int = 5,
        user_inn: Optional[str] = None,
        viewed_categories: Optional[List[str]] = None,
        top_categories: Optional[List[str]] = None,
    ) -> List[SuggestionPayload]:
        cache_key = self.cache_service.build_key(
            "suggestions",
            data={
                "version": self.SUGGESTIONS_CACHE_VERSION,
                "query": query,
                "top_k": top_k,
                "user_inn": user_inn,
                "viewed_categories": unique_preserve_order([str(value) for value in (viewed_categories or []) if value]),
                "top_categories": unique_preserve_order([str(value) for value in (top_categories or []) if value]),
            },
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, list):
            cached_items = []
            for item in cached_payload:
                if isinstance(item, dict):
                    cached_items.append(SuggestionPayload(**item))
            if cached_items:
                return cached_items

        payload = self.search_service.search(query=query, top_k=max(top_k * 3, 12))
        query_payload = payload["query"]
        corrected_query = query_payload.get("corrected_query")
        normalized_query = query_payload.get("normalized_query")
        short_personalized_prefix = bool(
            user_inn
            and len(tokenize(normalized_query or query)) == 1
            and 2 <= len(normalize_text(normalized_query or query)) <= 4
        )
        same_type_prefix_suggestions = self._build_personalized_product_suggestions(
            query=query,
            products=self._resolve_same_type_prefix_products(user_inn=user_inn, query=query),
        )
        product_suggestions = self._build_personalized_product_suggestions(
            query=query,
            products=self._resolve_suggestion_products(user_inn=user_inn),
        )
        if short_personalized_prefix and same_type_prefix_suggestions:
            product_suggestions = [
                item
                for item in product_suggestions
                if str(item.reason or "") == "Часто закупалось"
            ]
        category_suggestions = self._build_personalized_category_suggestions(
            query=query,
            categories=self._resolve_suggestion_categories(
                user_inn=user_inn,
                viewed_categories=viewed_categories or [],
                top_categories=top_categories or [],
            ),
        )
        suggestion_groups = [product_suggestions, same_type_prefix_suggestions]
        if not (short_personalized_prefix and same_type_prefix_suggestions):
            suggestion_groups.append(category_suggestions)
        suggestions = self._merge_suggestion_groups(*suggestion_groups)
        if corrected_query and corrected_query != normalized_query:
            suggestions.append(
                self._build_suggestion(
                    text=corrected_query,
                    suggestion_type="correction",
                    reason="Исправление запроса",
                    score=180.0,
                )
            )
        abstract_suggestions = self._build_abstract_suggestions(
            query=query,
            query_payload=query_payload,
            results=payload["results"],
        )
        if short_personalized_prefix and same_type_prefix_suggestions:
            abstract_suggestions = []
        if suggestions:
            remaining_slots = max(0, top_k - len(self._dedupe_suggestions(suggestions)))
            suggestions.extend(abstract_suggestions[: min(2, remaining_slots)])
        else:
            suggestions.extend(abstract_suggestions)
        result = self._dedupe_suggestions(suggestions)[:top_k]
        self.cache_service.set_json(
            cache_key,
            [self._model_dump(item) for item in result],
            ttl_seconds=self.settings.suggestions_cache_ttl_seconds,
        )
        return result

    @staticmethod
    def _model_dump(model: BaseModel) -> dict:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    @staticmethod
    def _build_suggestion(
        *,
        text: str,
        suggestion_type: Literal["product", "category", "correction", "query"],
        reason: Optional[str],
        score: float,
    ) -> SuggestionPayload:
        return SuggestionPayload(
            text=text,
            type=suggestion_type,
            reason=reason,
            score=round(float(score), 4),
        )

    @staticmethod
    def _trim_trailing_connector_tokens(tokens: List[str]) -> List[str]:
        connector_tokens = {
            "и",
            "или",
            "а",
            "но",
            "на",
            "в",
            "во",
            "с",
            "со",
            "к",
            "ко",
            "о",
            "об",
            "обо",
            "у",
            "от",
            "до",
            "за",
            "из",
            "по",
            "под",
            "над",
            "при",
            "для",
            "без",
            "через",
            "между",
        }
        trimmed = list(tokens)
        while trimmed and (len(trimmed[-1]) <= 1 or trimmed[-1] in connector_tokens):
            trimmed.pop()
        return trimmed

    @staticmethod
    def _abstract_name_phrase(name: str, query: str) -> str:
        query_tokens = tokenize(query)
        name_tokens = tokenize(name)
        if not name_tokens:
            return ""

        phrase_tokens: List[str] = []
        for token in name_tokens:
            if token.isdigit() or any(char.isdigit() for char in token):
                break
            phrase_tokens.append(token)
        phrase_tokens = TenderHackApiService._trim_trailing_connector_tokens(phrase_tokens)
        min_tokens = 1 if len(query_tokens) <= 1 else 2
        if len(phrase_tokens) < min_tokens:
            return ""
        return " ".join(phrase_tokens)

    @staticmethod
    def _compact_category_phrase(category: str) -> str:
        category_tokens = [token for token in tokenize(category) if not token.isdigit()]
        category_tokens = TenderHackApiService._trim_trailing_connector_tokens(category_tokens)
        if not category_tokens:
            return ""
        return " ".join(category_tokens)

    @staticmethod
    def _significant_tokens(value: str) -> List[str]:
        tokens = tokenize(value)
        result: List[str] = []
        for token in tokens:
            if token.isdigit():
                continue
            if len(token) <= 2:
                continue
            if token in {"мг", "мл", "шт", "таб", "кап", "фл", "амп", "гр", "г", "дл", "д", "№"}:
                continue
            result.append(token)
        return result

    @classmethod
    def _token_prefix_match_score(cls, query: str, candidate: str, *, allow_secondary_tokens: bool = True) -> float:
        query_norm = normalize_text(query)
        if not query_norm:
            return 0.0

        query_tokens = cls._significant_tokens(query)
        candidate_tokens = cls._significant_tokens(candidate)
        if not candidate_tokens:
            return 0.0

        first_token = candidate_tokens[0]
        score = 0.0
        if first_token.startswith(query_norm):
            score += 120.0

        if not query_tokens:
            if allow_secondary_tokens and any(token.startswith(query_norm) for token in candidate_tokens[1:5]):
                score += 55.0
            return score

        query_stems = [stem_token(token) for token in query_tokens]
        candidate_stems = [stem_token(token) for token in candidate_tokens]

        for query_token, query_stem in zip(query_tokens, query_stems):
            if first_token.startswith(query_token):
                score += 60.0
                continue
            if query_stem and stem_token(first_token).startswith(query_stem):
                score += 38.0
                continue
            if allow_secondary_tokens and len(query_token) >= 3:
                if any(token.startswith(query_token) for token in candidate_tokens[1:4]):
                    score += 26.0
                    continue
                if query_stem and any(stem.startswith(query_stem) for stem in candidate_stems[1:4]):
                    score += 18.0

        return score

    @classmethod
    def _product_suggestion_phrase(cls, name: str) -> str:
        tokens = tokenize(name)
        if not tokens:
            return ""

        stop_tokens = {
            "раствор",
            "растворы",
            "таблетки",
            "таблетка",
            "табл",
            "таб",
            "капсулы",
            "капсула",
            "капс",
            "мазь",
            "крем",
            "суспензия",
            "сусп",
            "сироп",
            "порошок",
            "спрей",
            "аэрозоль",
            "ампула",
            "амп",
            "флакон",
            "фл",
            "капли",
            "концентрат",
            "конц",
            "инф",
            "наруж",
            "приема",
            "прием",
            "внутрь",
            "введение",
        }

        phrase_tokens: List[str] = []
        for token in tokens:
            if any(char.isdigit() for char in token):
                break
            if len(token) <= 1:
                continue
            if token in {"ооо", "ао", "оао", "зао", "пао", "россия", "германия", "австрия", "италия", "швейцария"}:
                break
            if phrase_tokens and token in stop_tokens:
                break
            if len(token) == 2 and not phrase_tokens:
                continue
            phrase_tokens.append(token)

        phrase_tokens = cls._trim_trailing_connector_tokens(phrase_tokens)
        phrase = " ".join(phrase_tokens)
        if not phrase:
            return ""
        return phrase[:1].upper() + phrase[1:]

    @classmethod
    def _build_abstract_suggestions(
        cls,
        *,
        query: str,
        query_payload: dict,
        results: List[dict],
    ) -> List[SuggestionPayload]:
        query_norm = normalize_text(query)
        if len(query_norm) <= 2:
            return []

        expanded_tokens = [str(token) for token in query_payload.get("expanded_tokens", []) if token]
        corrected_query = str(query_payload.get("corrected_query") or "")
        query_tokens = unique_preserve_order(tokenize(query) + tokenize(corrected_query) + expanded_tokens)
        ranked_suggestions: List[SuggestionPayload] = []

        for synonym_rule in query_payload.get("applied_synonyms", []):
            for target in synonym_rule.get("targets", []):
                candidate = normalize_text(str(target))
                if not candidate or candidate == query_norm:
                    continue
                ranked_suggestions.append(
                    cls._build_suggestion(
                        text=candidate,
                        suggestion_type="query",
                        reason="Синоним запроса",
                        score=160.0,
                    )
                )

        for item in results:
            name_phrase = cls._abstract_name_phrase(str(item.get("clean_name") or ""), query)
            category_phrase = cls._compact_category_phrase(str(item.get("category") or ""))

            for candidate in [name_phrase, category_phrase]:
                candidate_norm = normalize_text(candidate)
                if not candidate_norm or candidate_norm == query_norm:
                    continue
                if query_tokens and not any(
                    token.startswith(query_norm) or query_norm.startswith(token[: max(2, min(len(token), len(query_norm)))])
                    for token in cls._significant_tokens(candidate)
                ):
                    continue
                score = cls._token_prefix_match_score(query, candidate)
                if score > 0:
                    suggestion_type: Literal["product", "category", "correction", "query"] = (
                        "category" if candidate == category_phrase else "query"
                    )
                    reason = "Популярная категория" if suggestion_type == "category" else "Продолжение запроса"
                    ranked_suggestions.append(
                        cls._build_suggestion(
                            text=candidate,
                            suggestion_type=suggestion_type,
                            reason=reason,
                            score=score,
                        )
                    )

        ranked_suggestions.sort(
            key=lambda item: (item.score, -len(item.text), item.text),
            reverse=True,
        )
        return cls._dedupe_suggestions(ranked_suggestions)

    def _resolve_suggestion_categories(
        self,
        *,
        user_inn: Optional[str],
        viewed_categories: List[str],
        top_categories: List[str],
    ) -> List[str]:
        categories = unique_preserve_order([str(value) for value in [*viewed_categories, *top_categories] if value])
        if categories or not user_inn:
            return categories

        login_cache_key = self.cache_service.build_key(
            "login",
            data={"inn": user_inn, "version": self.LOGIN_CACHE_VERSION},
        )
        cached_payload = self.cache_service.get_json(login_cache_key)
        if isinstance(cached_payload, dict):
            cached_top_categories = [
                str(item.get("category") or "")
                for item in cached_payload.get("topCategories", [])
                if isinstance(item, dict) and item.get("category")
            ]
            return unique_preserve_order(cached_top_categories)

        try:
            payload = self.login(user_inn)
        except Exception:
            return []
        return unique_preserve_order([str(item.get("category") or "") for item in payload.topCategories if item.get("category")])

    def _resolve_suggestion_products(self, *, user_inn: Optional[str]) -> List[dict]:
        if not user_inn:
            return []

        cache_key = self.cache_service.build_key("suggestion_products", data={"inn": user_inn})
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, list):
            return [item for item in cached_payload if isinstance(item, dict)]

        try:
            profile = self.personalization_service.build_customer_profile(customer_inn=user_inn, top_ste=150)
            recommended_ste = list(profile.get("recommended_ste") or profile.get("top_ste") or [])
            frequent_products = self._load_frequent_products(recommended_ste[:150])
        except Exception:
            frequent_products = []

        self.cache_service.set_json(
            cache_key,
            frequent_products,
            ttl_seconds=self.settings.user_profile_cache_ttl_seconds,
        )
        return frequent_products

    def _resolve_same_type_prefix_products(self, *, user_inn: Optional[str], query: str) -> List[dict]:
        query_norm = normalize_text(query)
        if not user_inn or len(tokenize(query_norm)) != 1 or len(query_norm) < 2 or len(query_norm) > 4:
            return []

        cache_key = self.cache_service.build_key(
            "same_type_prefix_products",
            data={
                "version": 3,
                "inn": user_inn,
                "query": query_norm,
            },
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, list):
            return [item for item in cached_payload if isinstance(item, dict)]

        try:
            profile = self.personalization_service.build_customer_profile(customer_inn=user_inn, top_ste=200)
        except Exception:
            return []

        same_type_peer_inns = [str(value) for value in profile.get("same_type_peer_inns", []) if value]
        archetype = str(profile.get("institution_archetype") or "general")
        if not same_type_peer_inns and archetype == "general":
            return []

        candidate_rows = self.search_service.conn.execute(
            """
            SELECT ste_id, clean_name, normalized_name, category, normalized_category
            FROM ste_catalog
            WHERE normalized_name LIKE ?
            ORDER BY length(normalized_name) ASC, normalized_name ASC
            LIMIT 120
            """,
            (f"{query_norm}%",),
        ).fetchall()
        if not candidate_rows:
            return []

        candidate_ids = [str(row["ste_id"]) for row in candidate_rows if row["ste_id"]]
        same_type_stats = self._load_same_type_prefix_stats(
            ste_ids=candidate_ids,
            peer_inns=same_type_peer_inns,
            archetype_categories=list(profile.get("archetype_categories") or []),
            customer_inn=user_inn,
        )
        if not same_type_stats:
            return []

        archetype_categories: set[str] = set()
        category_recommendation_scores: dict[str, float] = {}
        profile_category_codes = set()
        for item in profile.get("recommended_categories", []):
            category_norm = normalize_text(str(item.get("category") or item.get("normalized_category") or ""))
            recommendation_score = float(item.get("recommendation_score") or item.get("weight") or 0.0)
            if not category_norm or recommendation_score < 2.0:
                continue
            category_recommendation_scores[category_norm] = max(
                category_recommendation_scores.get(category_norm, 0.0),
                recommendation_score,
            )
            profile_category_codes.update(self._extract_category_codes(category_norm))
        for item in profile.get("archetype_categories", []):
            category_norm = normalize_text(str(item.get("category") or item.get("normalized_category") or ""))
            archetype_weight = float(item.get("weight") or item.get("recommendation_score") or 0.0)
            if not category_norm or archetype_weight < 0.35:
                continue
            archetype_categories.add(category_norm)
            category_recommendation_scores[category_norm] = max(
                category_recommendation_scores.get(category_norm, 0.0),
                archetype_weight,
            )
            profile_category_codes.update(self._extract_category_codes(category_norm))
        recommended_by_ste = {
            str(item.get("ste_id") or ""): float(item.get("recommendation_score") or 0.0)
            for item in profile.get("recommended_ste", [])
            if item.get("ste_id")
        }

        ranked_products: List[dict] = []
        for row in candidate_rows:
            ste_id = str(row["ste_id"] or "")
            if not ste_id:
                continue
            stats = same_type_stats.get(ste_id)
            if not stats:
                continue
            prefix_score = max(
                self._token_prefix_match_score(query, str(row["clean_name"] or ""), allow_secondary_tokens=False),
                self._token_prefix_match_score(query, str(row["normalized_name"] or ""), allow_secondary_tokens=False),
            )
            if prefix_score <= 0:
                continue
            category_norm = normalize_text(str(row["category"] or row["normalized_category"] or ""))
            exact_category_score = float(category_recommendation_scores.get(category_norm, 0.0))
            candidate_codes = self._extract_category_codes(category_norm)
            code_match_score = self._category_code_match_score(candidate_codes, profile_category_codes)
            code_boost = 14.0 * code_match_score
            category_boost = 0.0
            if category_norm and category_norm in archetype_categories:
                category_boost = 12.0
            elif exact_category_score > 0:
                category_boost = min(10.0, exact_category_score * 1.8)
            recommendation_score = recommended_by_ste.get(ste_id, 0.0)
            same_type_count = float(stats.get("purchase_count", 0.0) or 0.0)
            global_count = max(same_type_count, float(stats.get("global_purchase_count", 0.0) or 0.0))
            specificity = same_type_count / global_count if global_count > 0 else 0.0
            type_relevance_score = max(category_boost, code_boost)
            if specificity < 0.20 and type_relevance_score <= 0:
                continue
            # For very short prefixes we only keep broadly-supported items when they are
            # professional for this institution type; generic one-off matches create noisy suggestions.
            if len(query_norm) <= 4 and type_relevance_score <= 0 and same_type_count < 3:
                continue
            if type_relevance_score <= 0 and specificity < 0.45:
                continue

            popularity_score = min(14.0, 2.5 * math.log1p(same_type_count))
            specificity_score = 18.0 * specificity
            peer_recommendation_boost = min(6.0, recommendation_score)
            generic_penalty = 10.0 if len(query_norm) <= 4 and type_relevance_score <= 0 else 0.0
            ranked_products.append(
                {
                    "steId": ste_id,
                    "name": str(row["clean_name"] or ste_id),
                    "category": str(row["category"] or ""),
                    "purchaseCount": int(same_type_count),
                    "totalAmount": round(float(stats.get("total_amount", 0.0) or 0.0), 2),
                    "reason": "Популярно у учреждений того же типа",
                    "recommendationScore": round(
                        prefix_score * 0.06
                        + popularity_score
                        + specificity_score
                        + category_boost
                        + code_boost
                        + peer_recommendation_boost
                        - generic_penalty,
                        4,
                    ),
                }
            )

        ranked_products.sort(
            key=lambda item: (
                float(item.get("recommendationScore") or 0.0),
                float(item.get("purchaseCount") or 0.0),
                str(item.get("name") or ""),
            ),
            reverse=True,
        )
        result = ranked_products[:30]
        self.cache_service.set_json(
            cache_key,
            result,
            ttl_seconds=self.settings.suggestions_cache_ttl_seconds,
        )
        return result

    @staticmethod
    def _extract_category_codes(value: str) -> set[str]:
        normalized = normalize_text(value)
        if not normalized:
            return set()
        return {
            match.group(0)
            for match in re.finditer(r"\b[a-zа-я]\d{2}[a-zа-я]{0,2}\b", normalized)
        }

    @staticmethod
    def _category_code_match_score(candidate_codes: set[str], reference_codes: set[str]) -> float:
        best_score = 0.0
        for candidate_code in candidate_codes:
            for reference_code in reference_codes:
                if candidate_code == reference_code:
                    best_score = max(best_score, 1.0)
                    continue
                if candidate_code.startswith(reference_code) or reference_code.startswith(candidate_code):
                    best_score = max(best_score, 0.8)
        return best_score

    @staticmethod
    def _apply_same_type_prefix_boost(results: List[dict], same_type_prefix_products: List[dict]) -> None:
        if not results or not same_type_prefix_products:
            return

        score_by_ste = {
            str(item.get("steId") or ""): float(item.get("recommendationScore") or 0.0)
            for item in same_type_prefix_products
            if item.get("steId")
        }
        for item in results:
            ste_id = str(item.get("ste_id") or "")
            prefix_score = score_by_ste.get(ste_id)
            if prefix_score is None:
                continue
            boost = min(24.0, 6.0 + prefix_score * 0.4)
            item["institution_type_prefix_boost"] = round(boost, 4)
            item["final_score"] = round(float(item.get("final_score", item.get("search_score", 0.0))) + boost, 4)
            item["top_reason_codes"] = unique_preserve_order(
                ["INSTITUTION_TYPE_PREFIX_MATCH"] + [str(code) for code in item.get("top_reason_codes", []) if code]
            )
            item["reasons"] = unique_preserve_order(
                ["поднято выше по короткому префиксу с учетом типа учреждения"]
                + [str(reason) for reason in item.get("reasons", []) if reason]
            )

    def _load_same_type_prefix_stats(
        self,
        *,
        ste_ids: List[str],
        peer_inns: List[str],
        archetype_categories: List[dict],
        customer_inn: str,
    ) -> dict[str, dict]:
        if not ste_ids:
            return {}

        ste_placeholders = ", ".join("?" for _ in ste_ids)
        rows = []
        if peer_inns:
            peer_placeholders = ", ".join("?" for _ in peer_inns)
            rows = self.personalization_service.conn.execute(
                f"""
                SELECT
                    cs.ste_id,
                    SUM(cs.purchase_count) AS purchase_count,
                    SUM(cs.total_amount) AS total_amount
                FROM customer_ste_stats cs
                WHERE cs.customer_inn IN ({peer_placeholders})
                  AND cs.ste_id IN ({ste_placeholders})
                GROUP BY cs.ste_id
                ORDER BY purchase_count DESC, total_amount DESC
                """,
                [*peer_inns, *ste_ids],
            ).fetchall()

        if not rows:
            category_ids = [int(item.get("category_id") or 0) for item in archetype_categories if int(item.get("category_id") or 0) > 0]
            if not category_ids:
                return {}
            category_placeholders = ", ".join("?" for _ in category_ids)
            rows = self.personalization_service.conn.execute(
                f"""
                SELECT
                    cs.ste_id,
                    SUM(cs.purchase_count) AS purchase_count,
                    SUM(cs.total_amount) AS total_amount
                FROM customer_ste_stats cs
                WHERE cs.customer_inn <> ?
                  AND cs.category_id IN ({category_placeholders})
                  AND cs.ste_id IN ({ste_placeholders})
                GROUP BY cs.ste_id
                ORDER BY purchase_count DESC, total_amount DESC
                """,
                [customer_inn, *category_ids, *ste_ids],
            ).fetchall()

        result = {
            str(row["ste_id"]): {
                "purchase_count": float(row["purchase_count"] or 0.0),
                "total_amount": float(row["total_amount"] or 0.0),
                "global_purchase_count": 0.0,
            }
            for row in rows
            if row["ste_id"]
        }
        if not result:
            return {}

        ste_placeholders = ", ".join("?" for _ in result)
        global_rows = self.personalization_service.conn.execute(
            f"""
            SELECT
                ste_id,
                SUM(purchase_count) AS purchase_count
            FROM customer_ste_stats
            WHERE ste_id IN ({ste_placeholders})
            GROUP BY ste_id
            """,
            list(result.keys()),
        ).fetchall()
        for row in global_rows:
            ste_id = str(row["ste_id"] or "")
            if ste_id in result:
                result[ste_id]["global_purchase_count"] = float(row["purchase_count"] or 0.0)
        return result

    @classmethod
    def _build_personalized_category_suggestions(
        cls,
        *,
        query: str,
        categories: List[str],
    ) -> List[SuggestionPayload]:
        query_norm = normalize_text(query)
        if not query_norm:
            return []

        ranked_candidates: List[SuggestionPayload] = []
        for category in categories:
            candidate = cls._compact_category_phrase(str(category))
            candidate_norm = normalize_text(candidate)
            if not candidate_norm or candidate_norm == query_norm:
                continue
            score = cls._token_prefix_match_score(query, candidate)
            if score > 0:
                ranked_candidates.append(
                    cls._build_suggestion(
                        text=candidate,
                        suggestion_type="category",
                        reason="Категория из истории",
                        score=score + 20.0,
                    )
                )

        ranked_candidates.sort(
            key=lambda item: (item.score, -len(item.text), item.text),
            reverse=True,
        )
        return cls._dedupe_suggestions(ranked_candidates)

    @classmethod
    def _build_personalized_product_suggestions(
        cls,
        *,
        query: str,
        products: List[dict],
    ) -> List[SuggestionPayload]:
        query_norm = normalize_text(query)
        if not query_norm or not products:
            return []

        ranked_candidates: List[SuggestionPayload] = []

        for item in products:
            full_name = str(item.get("name") or "").strip()
            if not full_name:
                continue
            suggestion_phrase = cls._product_suggestion_phrase(full_name) or full_name
            candidate_norm = normalize_text(suggestion_phrase)
            full_name_norm = normalize_text(full_name)
            if not full_name_norm or full_name_norm == query_norm:
                continue

            score = max(
                cls._token_prefix_match_score(query, suggestion_phrase),
                cls._token_prefix_match_score(query, full_name),
            )
            if score <= 0:
                continue

            score += min(float(item.get("purchaseCount") or 0), 15.0)
            score += min(float(item.get("recommendationScore") or 0.0), 12.0)
            if candidate_norm.startswith(query_norm):
                score += 18.0
            if full_name_norm.startswith(query_norm):
                score += 10.0

            item_reason = str(item.get("reason") or "")
            normalized_reason = item_reason.lower()
            if "часто закупалось учреждением" in normalized_reason:
                reason = "Часто закупалось"
            elif "того же типа" in normalized_reason or "рекомендуется для" in normalized_reason:
                reason = "По типу учреждения"
            elif "похожих учреждений" in normalized_reason:
                reason = "Популярно у похожих учреждений"
            elif "учреждени" in normalized_reason:
                reason = "Часто закупалось"
            elif "регион" in normalized_reason:
                reason = "Популярно в регионе"
            else:
                reason = "Часто закупалось"

            ranked_candidates.append(
                cls._build_suggestion(
                    text=suggestion_phrase,
                    suggestion_type="product",
                    reason=reason,
                    score=score + 35.0,
                )
            )

        ranked_candidates.sort(
            key=lambda item: (item.score, -len(item.text), item.text),
            reverse=True,
        )
        return cls._dedupe_suggestions(ranked_candidates)

    @staticmethod
    def _merge_suggestion_groups(*groups: List[SuggestionPayload]) -> List[SuggestionPayload]:
        merged: List[SuggestionPayload] = []
        if not groups:
            return merged

        max_len = max((len(group) for group in groups), default=0)
        for index in range(max_len):
            for group in groups:
                if index < len(group):
                    merged.append(group[index])
        return TenderHackApiService._dedupe_suggestions(merged)

    @classmethod
    def _suggestion_dedupe_key(cls, text: str) -> str:
        normalized_text = normalize_text(text)
        if not normalized_text:
            return ""

        stemmed_tokens = stem_tokens(tokenize(normalized_text))
        if stemmed_tokens:
            return " ".join(stemmed_tokens)
        return normalized_text

    @classmethod
    def _is_better_suggestion(cls, candidate: SuggestionPayload, current: SuggestionPayload) -> bool:
        if candidate.score != current.score:
            return candidate.score > current.score

        candidate_type_rank = cls._SUGGESTION_TYPE_PRIORITY.get(candidate.type, 0)
        current_type_rank = cls._SUGGESTION_TYPE_PRIORITY.get(current.type, 0)
        if candidate_type_rank != current_type_rank:
            return candidate_type_rank > current_type_rank

        candidate_text = normalize_text(candidate.text)
        current_text = normalize_text(current.text)
        if len(candidate_text) != len(current_text):
            return len(candidate_text) < len(current_text)
        return candidate_text < current_text

    @classmethod
    def _dedupe_suggestions(cls, suggestions: List[SuggestionPayload]) -> List[SuggestionPayload]:
        deduped_by_key: dict[str, SuggestionPayload] = {}
        order: List[str] = []
        for item in suggestions:
            dedupe_key = cls._suggestion_dedupe_key(item.text)
            if not dedupe_key:
                continue
            if dedupe_key not in deduped_by_key:
                deduped_by_key[dedupe_key] = item
                order.append(dedupe_key)
                continue
            if cls._is_better_suggestion(item, deduped_by_key[dedupe_key]):
                deduped_by_key[dedupe_key] = item
        return [deduped_by_key[key] for key in order]

    @staticmethod
    def _search_cache_data(
        payload: SearchRequest,
        server_session: Optional[dict] = None,
        search_rerank_token: str = "disabled",
    ) -> dict:
        user_context = payload.userContext or SearchUserContext()
        page_limit = int(payload.limit or payload.topK)
        return {
            "version": TenderHackApiService.SEARCH_CACHE_VERSION,
            "query": payload.query,
            "search_rerank_token": search_rerank_token,
            "user_id": user_context.id,
            "user_inn": user_context.inn,
            "user_region": user_context.region,
            "user_viewed_categories": unique_preserve_order([str(value) for value in user_context.viewedCategories if value]),
            "viewed_categories": unique_preserve_order([str(value) for value in payload.viewedCategories if value]),
            "bounced_categories": unique_preserve_order(
                [normalize_text(str(value)) for value in payload.bouncedCategories if value]
            ),
            "limit": page_limit,
            "offset": int(payload.offset),
            "min_score": round(float(payload.min_score), 4),
            "server_session_version": int((server_session or {}).get("version", 0) or 0),
            "server_recent_categories": unique_preserve_order(
                [str(value) for value in (server_session or {}).get("recent_categories", []) if value]
            ),
            "server_clicked_ste_ids": unique_preserve_order(
                [str(value) for value in (server_session or {}).get("clicked_ste_ids", []) if value]
            ),
            "server_cart_ste_ids": unique_preserve_order(
                [str(value) for value in (server_session or {}).get("cart_ste_ids", []) if value]
            ),
            "server_bounced_categories": unique_preserve_order(
                [normalize_text(str(value)) for value in (server_session or {}).get("bounced_categories", []) if value]
            ),
        }

    def _build_search_rerank_cache_token(self) -> str:
        predictor = self.search_rerank_predictor
        if predictor is None or not predictor.enabled:
            return "disabled"
        try:
            modified_at = int(predictor.model_path.stat().st_mtime)
        except OSError:
            modified_at = 0
        return f"{predictor.model_path.name}:{modified_at}"

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

        if "SESSION_CART_BOOST" in codes or "SESSION_CLICK_BOOST" in codes:
            return "Продолжить подбор в этой категории"
        if "INSTITUTION_TYPE_PREFIX_MATCH" in codes:
            return "По типу учреждения"
        if codes & {"USER_CATEGORY_AFFINITY", "USER_REPEAT_BUY", "RECENT_SIMILAR_PURCHASE", "SUPPLIER_AFFINITY"}:
            return "На основе ваших закупок"
        if "REGIONAL_POPULARITY" in codes:
            return "Популярно в вашем регионе"
        if "SIMILAR_CUSTOMER_POPULARITY" in codes:
            return "Популярно у похожих заказчиков"
        if category_norm and category_norm in session_category_set:
            return "Продолжить подбор в этой категории"
        return None


def create_app(settings: Optional[AppSettings] = None) -> FastAPI:
    active_settings = settings or AppSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = TenderHackApiService(active_settings)
        app.state.service = service
        yield
        service.close()

    app = FastAPI(
        title="TenderHack Search API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/auth/login", response_model=UserPayload)
    async def login(payload: LoginRequest, request: Request) -> UserPayload:
        return request.app.state.service.login(payload.inn)

    @app.post("/api/search", response_model=SearchResponsePayload)
    async def search(payload: SearchRequest, request: Request) -> SearchResponsePayload:
        return request.app.state.service.search(payload)

    @app.post("/api/event", response_model=EventResponsePayload)
    async def event(payload: EventRequest, request: Request) -> EventResponsePayload:
        return request.app.state.service.record_event(payload)

    @app.get("/api/search/suggestions", response_model=List[SuggestionPayload])
    async def suggestions(
        request: Request,
        q: str = Query(min_length=1),
        top_k: int = Query(default=5, ge=1, le=10),
        inn: Optional[str] = Query(default=None),
        viewed_categories: Optional[str] = Query(default=None),
        top_categories: Optional[str] = Query(default=None),
    ) -> List[SuggestionPayload]:
        def _split_values(raw: Optional[str]) -> List[str]:
            if not raw:
                return []
            return [value.strip() for value in raw.split("|") if value.strip()]

        return request.app.state.service.suggestions(
            query=q,
            top_k=top_k,
            user_inn=inn,
            viewed_categories=_split_values(viewed_categories),
            top_categories=_split_values(top_categories),
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


app = create_app()
