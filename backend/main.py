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
from tenderhack.personalization import INSTITUTION_ARCHETYPE_PROFILE_LABELS, PersonalizationService
from tenderhack.personalization_runtime import PersonalizationRuntimeService
from tenderhack.search import SearchService
from tenderhack.search_rerank_model import SearchRerankPredictor
from tenderhack.text import normalize_text, stem_token, stem_tokens, tokenize, unique_preserve_order
from tenderhack.user_profile_scorer import SQLiteUserHistoryRepository, UserProfileScorer, apply_personalization


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


def _default_synonyms_path() -> Path:
    local_path = PROJECT_ROOT / "search_synonyms.json"
    if local_path.exists():
        return local_path
    return PROJECT_ROOT / "data" / "reference" / "search_synonyms.json"


def _default_contracts_path() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "contracts_clean.csv",
        PROJECT_ROOT / "data" / "processed" / "contracts_flat.csv",
        PROJECT_ROOT / "data" / "processed" / "contracts.csv",
    ]
    candidates.extend(sorted(PROJECT_ROOT.glob("Контракты_*.csv")))
    for path in candidates:
        if path.exists():
            return path
    return PROJECT_ROOT / "Контракты_20260403.csv"


@dataclass
class AppSettings:
    search_db_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_search.sqlite"
    preprocessed_db_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_preprocessed.sqlite"
    synonyms_path: Path = _default_synonyms_path()
    contracts_path: Path = _default_contracts_path()
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
            contracts_path=_env_path("TENDERHACK_CONTRACTS_PATH", cls.contracts_path),
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
    entityType: Optional[str] = None
    customerName: Optional[str] = None
    organizationTypeCode: Optional[str] = None
    organizationTypeLabel: Optional[str] = None
    organizationTypeSource: Optional[str] = None
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
    closeReason: Optional[Literal["dismiss", "after_cart_add"]] = None


class EventResponsePayload(BaseModel):
    status: str
    userId: str
    sessionVersion: int
    recentCategories: List[str] = Field(default_factory=list)
    clickedSteIds: List[str] = Field(default_factory=list)
    cartSteIds: List[str] = Field(default_factory=list)
    bouncedCategories: List[str] = Field(default_factory=list)
    itemCloseOutcome: Literal["none", "forgiven", "applied", "suppressed"] = "none"


class TenderHackApiService:
    LOGIN_CACHE_VERSION = 13
    SEARCH_CACHE_VERSION = 15
    SUGGESTIONS_CACHE_VERSION = 32
    PROFILE_TOP_CATEGORIES_LIMIT = 6
    PROFILE_FREQUENT_PRODUCTS_LIMIT = 18
    MAX_HISTORY_CATEGORY_SUGGESTIONS = 1
    MAX_HISTORY_REASON_SUGGESTIONS = 3
    MAX_INSTITUTION_TYPE_SUGGESTIONS = 3
    MIN_QUERY_COMPLETION_SUGGESTIONS = 3
    MAX_CART_CONTEXT_BOOSTED_RESULTS = 3
    MAX_CART_CONTEXT_SIGNATURE_RESULT_SHARE = 0.45
    MAX_CART_CATEGORY_BADGED_RESULTS = 2
    MIN_CART_CATEGORY_QUERY_SCORE = 35.0
    MAX_CART_CATEGORY_BOOST = 0.045
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
        self._profile_db_attached = False
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
        self.personalization_service = PersonalizationService(
            db_path=settings.preprocessed_db_path,
            contracts_path=settings.contracts_path,
        )
        self.user_profile_scorer = UserProfileScorer(SQLiteUserHistoryRepository(self.personalization_service.conn))
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

    def _ensure_profile_db_attached(self) -> None:
        if self._profile_db_attached:
            return
        attached_rows = self.search_service.conn.execute("PRAGMA database_list").fetchall()
        if any(str(row[1]) == "profile_db" for row in attached_rows if len(row) >= 2):
            self._profile_db_attached = True
            return
        self.search_service.conn.execute(
            "ATTACH DATABASE ? AS profile_db",
            (str(self.settings.preprocessed_db_path),),
        )
        self._profile_db_attached = True

    @staticmethod
    def _resolve_runtime_user_id(*, user_id: Optional[str], user_inn: Optional[str]) -> str:
        if user_id:
            return str(user_id)
        if user_inn:
            return f"user-{user_inn}"
        return "anonymous"

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

    @staticmethod
    def _resolve_organization_type_payload(*, profile: dict, customer_name_context: dict) -> dict:
        if str(profile.get("entity_type") or "") == "supplier":
            return {
                "organizationTypeCode": "supplier",
                "organizationTypeLabel": "Поставщик",
                "organizationTypeSource": "По профилю поставщика",
            }
        name_code = str(customer_name_context.get("institution_name_archetype") or "").strip()
        name_label = str(customer_name_context.get("institution_name_archetype_label") or "").strip()
        history_code = str(profile.get("institution_archetype") or "").strip()
        history_label = INSTITUTION_ARCHETYPE_PROFILE_LABELS.get(
            history_code,
            INSTITUTION_ARCHETYPE_PROFILE_LABELS["general"],
        )

        if name_code and name_code != "general":
            return {
                "organizationTypeCode": name_code,
                "organizationTypeLabel": name_label or INSTITUTION_ARCHETYPE_PROFILE_LABELS.get(
                    name_code,
                    INSTITUTION_ARCHETYPE_PROFILE_LABELS["general"],
                ),
                "organizationTypeSource": "По наименованию заказчика",
            }

        if history_code and history_code != "general":
            return {
                "organizationTypeCode": history_code,
                "organizationTypeLabel": history_label,
                "organizationTypeSource": "По истории закупок",
            }

        fallback_code = name_code or history_code or "general"
        fallback_label = (
            name_label
            or INSTITUTION_ARCHETYPE_PROFILE_LABELS.get(
                fallback_code,
                INSTITUTION_ARCHETYPE_PROFILE_LABELS["general"],
            )
        )
        fallback_source = "По наименованию заказчика" if customer_name_context.get("customer_name") else "По истории закупок"
        return {
            "organizationTypeCode": fallback_code,
            "organizationTypeLabel": fallback_label,
            "organizationTypeSource": fallback_source,
        }

    def login(self, inn: str) -> UserPayload:
        cache_key = self.cache_service.build_key(
            "login",
            data={"inn": inn, "version": self.LOGIN_CACHE_VERSION},
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, dict):
            return UserPayload(**cached_payload)

        profile = self.personalization_service.build_profile_by_inn(
            inn,
            top_ste=self.PROFILE_FREQUENT_PRODUCTS_LIMIT,
        )
        customer_name_context = self.personalization_service.get_entity_name_context(inn)
        organization_type_payload = self._resolve_organization_type_payload(
            profile=profile,
            customer_name_context=customer_name_context,
        )
        recommended_categories = list(profile.get("recommended_categories") or profile.get("top_categories") or [])
        recommended_ste = list(profile.get("recommended_ste") or profile.get("top_ste") or [])
        viewed_categories = [
            str(item.get("category") or "")
            for item in recommended_categories[:5]
            if item.get("category")
        ]
        region = str(profile.get("customer_region") or "")
        entity_type = str(profile.get("entity_type") or "customer")
        frequent_products = self._load_frequent_products(
            recommended_ste[: self.PROFILE_FREQUENT_PRODUCTS_LIMIT],
            entity_type=entity_type,
        )
        profile_history_reason = (
            "Часто поставлялось поставщиком" if entity_type == "supplier" else "Часто закупалось учреждением"
        )
        payload = UserPayload(
            id=f"user-{inn}",
            inn=inn,
            region=region,
            entityType=entity_type,
            customerName=str(customer_name_context.get("customer_name") or "") or None,
            organizationTypeCode=str(organization_type_payload.get("organizationTypeCode") or "") or None,
            organizationTypeLabel=str(organization_type_payload.get("organizationTypeLabel") or "") or None,
            organizationTypeSource=str(organization_type_payload.get("organizationTypeSource") or "") or None,
            viewedCategories=viewed_categories,
            topCategories=[
                {
                    "category": str(item.get("category") or ""),
                    "purchaseCount": int(item.get("purchase_count") or 0),
                    "totalAmount": round(float(item.get("total_amount") or 0.0), 2),
                    "reason": str(item.get("reason") or profile_history_reason),
                    "recommendationScore": round(float(item.get("recommendation_score") or item.get("weight") or 0.0), 4),
                }
                for item in recommended_categories[: self.PROFILE_TOP_CATEGORIES_LIMIT]
            ],
            frequentProducts=frequent_products,
        )
        self.cache_service.set_json(cache_key, self._model_dump(payload), ttl_seconds=self.settings.login_cache_ttl_seconds)
        return payload

    def _load_frequent_products(self, top_ste: List[dict], *, entity_type: str = "customer") -> List[dict]:
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
        profile_history_reason = (
            "Часто поставлялось поставщиком" if str(entity_type or "") == "supplier" else "Часто закупалось учреждением"
        )
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
                    "reason": str(item.get("reason") or profile_history_reason),
                    "recommendationScore": round(float(item.get("recommendation_score") or item.get("weight") or 0.0), 4),
                }
            )
        return result

    def search(self, payload: SearchRequest) -> SearchResponsePayload:
        page_limit = int(payload.limit or payload.topK)
        user_context = payload.userContext or SearchUserContext()
        resolved_user_id = self._resolve_runtime_user_id(user_id=user_context.id, user_inn=user_context.inn)
        normalized_query = normalize_text(payload.query)
        short_personalized_prefix = bool(
            user_context.inn
            and len(tokenize(normalized_query)) == 1
            and 2 <= len(normalized_query) <= 4
        )
        server_session = self.online_state_service.get_session_state(
            user_id=resolved_user_id,
            customer_inn=user_context.inn,
            customer_region=user_context.region,
        )
        if user_context.id or user_context.inn:
            session_categories = unique_preserve_order(
                [str(value) for value in server_session.get("recent_categories", []) if value]
            )
        else:
            session_categories = unique_preserve_order(
                list(server_session.get("recent_categories", [])) + payload.viewedCategories + user_context.viewedCategories
            )
        server_bounced_categories = {
            normalize_text(value)
            for value in server_session.get("bounced_categories", [])
            if value
        }
        if user_context.id or user_context.inn:
            bounced_categories = server_bounced_categories
        else:
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
        has_session_signal = any(
            merged_session_state.get(key)
            for key in ("recent_categories", "clicked_ste_ids", "cart_ste_ids", "bounced_categories")
        )

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

        if user_context.inn or user_context.region or has_session_signal:
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

        cart_items = self._load_cart_context_items(list(merged_session_state.get("cart_ste_ids", [])))

        # Шаг 1: точечный буст за конкретные товары в корзине и их близкие аналоги.
        results = self._apply_cart_context_boost(
            results,
            query=payload.query,
            cart_items=cart_items,
        )

        # Шаг 2: мягкий lift для товаров той же категории на category-specific запросах.
        results = self._apply_cart_category_boost(
            results,
            query=payload.query,
            cart_items=cart_items,
        )

        # Шаг 3: эвристическая пессимизация (штраф за быстрый отказ)
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
                0 if item.get("reason_to_hide") else 1,
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

        resolved_entity_type = self.personalization_service.detect_entity_type(user_context.inn) if user_context.inn else "unknown"
        products: List[ProductPayload] = []
        for item in paginated_results:
            ste_id = str(item["ste_id"])
            offer = offer_lookup.get(ste_id, {})
            reason_to_show = self._map_reason_to_show(
                reason_codes=item.get("top_reason_codes", []),
                category=str(item.get("category") or ""),
                session_categories=session_categories,
                is_bounced=bool(item.get("reason_to_hide")),
                entity_type=resolved_entity_type,
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
        resolved_user_id = self._resolve_runtime_user_id(user_id=payload.userId, user_inn=payload.inn)
        session_state = self.online_state_service.record_event(
            user_id=resolved_user_id,
            customer_inn=payload.inn,
            customer_region=payload.region,
            event_type=payload.eventType,
            ste_id=payload.steId,
            category=payload.category,
            duration_ms=payload.durationMs,
            close_reason=payload.closeReason,
        )
        item_close_outcome: Literal["none", "forgiven", "applied", "suppressed"] = "none"
        if payload.eventType == "item_close":
            if session_state.get("last_item_close_suppressed"):
                item_close_outcome = "suppressed"
            elif payload.category:
                item_close_outcome = self.interaction_tracker.register_view(
                    user_id=resolved_user_id,
                    category_id=str(payload.category),
                    dwell_time_ms=payload.durationMs or 0,
                )
        return EventResponsePayload(
            status="ok",
            userId=resolved_user_id,
            sessionVersion=int(session_state.get("version", 0) or 0),
            recentCategories=[str(value) for value in session_state.get("recent_categories", [])],
            clickedSteIds=[str(value) for value in session_state.get("clicked_ste_ids", [])],
            cartSteIds=[str(value) for value in session_state.get("cart_ste_ids", [])],
            bouncedCategories=[str(value) for value in session_state.get("bounced_categories", [])],
            itemCloseOutcome=item_close_outcome,
        )

    def suggestions(
        self,
        query: str,
        top_k: int = 8,
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
        user_weights = self._resolve_suggestion_user_weights(user_inn=user_inn)
        user_ste_weights = self._resolve_suggestion_user_ste_weights(user_inn=user_inn)
        same_type_prefix_suggestions = self._build_personalized_product_suggestions(
            query=query,
            products=self._resolve_same_type_prefix_products(user_inn=user_inn, query=query),
            user_weights=user_weights,
            user_item_weights=user_ste_weights,
            require_positive_boost=False,
        )
        product_suggestions = self._build_personalized_product_suggestions(
            query=query,
            products=self._resolve_suggestion_products(user_inn=user_inn),
            user_weights=user_weights,
            user_item_weights=user_ste_weights,
            require_positive_boost=True,
        )
        if short_personalized_prefix and same_type_prefix_suggestions:
            product_suggestions = [
                item
                for item in product_suggestions
                if str(item.reason or "") in {"Часто закупалось", "Часто поставлялось"}
            ]
        category_suggestions = self._build_personalized_category_suggestions(
            query=query,
            categories=self._resolve_suggestion_categories(
                user_inn=user_inn,
                viewed_categories=viewed_categories or [],
                top_categories=top_categories or [],
            ),
            user_weights=user_weights,
            require_positive_boost=True,
        )
        suggestion_groups = [product_suggestions, same_type_prefix_suggestions]
        if not (short_personalized_prefix and same_type_prefix_suggestions):
            suggestion_groups.append(category_suggestions)
        suggestions = self._merge_suggestion_groups(*suggestion_groups, query=query)
        if corrected_query and corrected_query != normalized_query:
            suggestions.append(
                self._build_suggestion(
                    text=corrected_query,
                    suggestion_type="correction",
                    reason="Исправление запроса",
                    score=180.0,
                )
            )
        completion_suggestions = self._build_completion_suggestions(
            query=query,
            query_payload=query_payload,
        )
        abstract_suggestions = self._build_abstract_suggestions(
            query=query,
            query_payload=query_payload,
            results=payload["results"],
        )
        abstract_suggestions = self._diversify_suggestions_by_family(
            self._merge_suggestion_groups(completion_suggestions, abstract_suggestions, query=query)
        )
        reserved_abstract_slots = 0
        if short_personalized_prefix and same_type_prefix_suggestions:
            query_first_abstract_suggestions = [
                item for item in abstract_suggestions if item.type in {"query", "correction"}
            ]
            other_abstract_suggestions = [
                item for item in abstract_suggestions if item.type not in {"query", "correction"}
            ]
            abstract_suggestions = [*query_first_abstract_suggestions, *other_abstract_suggestions]
            reserved_abstract_slots = min(
                top_k,
                self.MIN_QUERY_COMPLETION_SUGGESTIONS,
                len(query_first_abstract_suggestions),
            )
        primary_suggestions = self._dedupe_suggestions(suggestions, query=query)
        diversified_primary_suggestions, _limited_reason_overflow = self._partition_suggestions_by_history_limit(
            primary_suggestions,
            top_k=top_k,
        )
        if diversified_primary_suggestions:
            max_primary_slots = max(0, top_k - reserved_abstract_slots)
            suggestions = list(diversified_primary_suggestions[:max_primary_slots])
            remaining_slots = max(0, top_k - len(suggestions))
            suggestions.extend(abstract_suggestions[:remaining_slots])
        else:
            suggestions = list(abstract_suggestions[:top_k])
        result = self._dedupe_suggestions(suggestions, query=query)[:top_k]
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
    def _suggestion_name_source(name: str) -> str:
        raw_name = str(name).strip()
        if not raw_name:
            return ""
        compact_source = re.split(r"[,;()]", raw_name, maxsplit=1)[0].strip()
        return compact_source or raw_name

    @classmethod
    def _trim_verbose_product_phrase_tokens(cls, tokens: List[str]) -> List[str]:
        if not tokens:
            return []

        clause_stop_tokens = {"при", "за", "вследствие"}
        detail_stop_tokens = {
            "рядовой",
            "карьерный",
            "фракция",
            "фракции",
            "типоразмер",
            "типоразмера",
            "марка",
            "сорт",
            "класс",
            "зернистость",
            "гранулометрия",
            "диаметр",
            "толщина",
            "длина",
            "ширина",
            "высота",
            "масса",
            "вес",
            "объем",
            "объём",
            "серия",
            "модель",
            "артикул",
            "комплектность",
        }
        secondary_clause_tokens = {"с", "со", "без"}

        compact_tokens: List[str] = []
        for token in tokens:
            if compact_tokens:
                if token in clause_stop_tokens and len(compact_tokens) >= 2:
                    break
                if token in secondary_clause_tokens and len(compact_tokens) >= 2:
                    break
                if token in detail_stop_tokens and len(compact_tokens) >= 2:
                    break
            compact_tokens.append(token)

        return cls._trim_trailing_connector_tokens(compact_tokens)

    @classmethod
    def _abstract_name_phrase(cls, name: str, query: str) -> str:
        query_tokens = tokenize(query)
        name_tokens = tokenize(cls._suggestion_name_source(name))
        if not name_tokens:
            return ""

        phrase_tokens: List[str] = []
        for token in name_tokens:
            if token.isdigit() or any(char.isdigit() for char in token):
                break
            phrase_tokens.append(token)
        phrase_tokens = cls._trim_verbose_product_phrase_tokens(phrase_tokens)
        min_tokens = 1 if len(query_tokens) <= 1 else 2
        if len(phrase_tokens) < min_tokens:
            return ""
        return " ".join(phrase_tokens)

    @classmethod
    def _query_anchored_name_phrase(cls, name: str, query: str) -> str:
        query_tokens = cls._significant_tokens(query)
        if len(query_tokens) != 1:
            return ""

        base_phrase = cls._product_suggestion_phrase(name) or cls._abstract_name_phrase(name, query)
        phrase_tokens = tokenize(base_phrase)
        if len(phrase_tokens) < 2:
            return ""

        query_token = query_tokens[0]
        query_stem = stem_token(query_token)
        anchor_index = -1
        for index, token in enumerate(phrase_tokens[1:], start=1):
            token_stem = stem_token(token)
            if token.startswith(query_token) or query_token.startswith(token):
                anchor_index = index
                break
            if query_stem and token_stem and (token_stem.startswith(query_stem) or query_stem.startswith(token_stem)):
                anchor_index = index
                break

        if anchor_index < 1:
            return ""

        anchored_tokens = cls._trim_verbose_product_phrase_tokens(
            [
                phrase_tokens[anchor_index],
                *phrase_tokens[:anchor_index],
                *phrase_tokens[anchor_index + 1 :],
            ]
        )
        if len(anchored_tokens) < 2:
            return ""

        anchored_phrase = " ".join(anchored_tokens)
        if not anchored_phrase or normalize_text(anchored_phrase) == normalize_text(base_phrase):
            return ""
        return anchored_phrase[:1].upper() + anchored_phrase[1:]

    @classmethod
    def _matched_query_token_phrase(cls, *, source_text: str, query: str) -> str:
        query_stems = unique_preserve_order(
            stem_token(token)
            for token in cls._significant_tokens(query)
            if stem_token(token)
        )
        if len(query_stems) < 2:
            return ""

        matched_tokens: List[str] = []
        matched_stems: set[str] = set()
        for token in cls._significant_tokens(source_text):
            token_stem = stem_token(token)
            if not token_stem or token_stem not in query_stems or token_stem in matched_stems:
                continue
            matched_tokens.append(token)
            matched_stems.add(token_stem)

        if len(matched_stems) < len(query_stems):
            return ""

        phrase = " ".join(matched_tokens)
        if not phrase:
            return ""
        return phrase[:1].upper() + phrase[1:]

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

    @staticmethod
    def _secondary_token_like_pattern(query_norm: str) -> str:
        return f"% {query_norm}%"

    @classmethod
    def _query_head_token_bonus(cls, query: str, candidate: str) -> float:
        query_tokens = cls._significant_tokens(query)
        candidate_tokens = cls._significant_tokens(candidate)
        if len(query_tokens) != 1 or not candidate_tokens:
            return 0.0

        query_token = query_tokens[0]
        candidate_token = candidate_tokens[0]
        if not query_token or not candidate_token:
            return 0.0
        if candidate_token == query_token:
            return 18.0

        query_stem = stem_token(query_token)
        candidate_stem = stem_token(candidate_token)
        if query_stem and candidate_stem and candidate_stem == query_stem:
            return 12.0
        return 0.0

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
        tokens = tokenize(cls._suggestion_name_source(name))
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

        phrase_tokens = cls._trim_verbose_product_phrase_tokens(phrase_tokens)
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
            clean_name = str(item.get("clean_name") or "")
            matched_query_phrase = cls._matched_query_token_phrase(source_text=clean_name, query=query)
            anchored_name_phrase = cls._query_anchored_name_phrase(clean_name, query)
            name_phrase = cls._abstract_name_phrase(clean_name, query)
            category_phrase = cls._compact_category_phrase(str(item.get("category") or ""))

            for candidate, suggestion_type, force_include in [
                (matched_query_phrase, "query", True),
                (anchored_name_phrase, "query", False),
                (name_phrase, "query", False),
                (category_phrase, "category", False),
            ]:
                candidate_norm = normalize_text(candidate)
                if not candidate_norm or candidate_norm == query_norm:
                    continue
                if not force_include and query_tokens and not any(
                    token.startswith(query_norm) or query_norm.startswith(token[: max(2, min(len(token), len(query_norm)))])
                    for token in cls._significant_tokens(candidate)
                ):
                    continue
                score = cls._token_prefix_match_score(query, candidate)
                if score > 0:
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
        return cls._dedupe_suggestions(ranked_suggestions, query=query)

    def _build_completion_suggestions(
        self,
        *,
        query: str,
        query_payload: dict,
    ) -> List[SuggestionPayload]:
        query_norm = normalize_text(query)
        if len(query_norm) <= 2:
            return []

        completion_tokens = unique_preserve_order(
            [
                normalize_text(str(token))
                for token in query_payload.get("completion_expansions", [])
                if token
            ]
        )
        if not completion_tokens:
            return []

        ranked_suggestions: List[SuggestionPayload] = []
        seen_texts: set[str] = set()

        for candidate_norm in completion_tokens[:8]:
            if not candidate_norm or candidate_norm == query_norm or not candidate_norm.startswith(query_norm):
                continue

            secondary_token_pattern = self._secondary_token_like_pattern(candidate_norm)

            rows = self.search_service.conn.execute(
                """
                SELECT clean_name, normalized_name, category, normalized_category
                FROM ste_catalog
                WHERE normalized_name LIKE ?
                   OR normalized_name LIKE ?
                   OR normalized_category LIKE ?
                   OR normalized_category LIKE ?
                ORDER BY
                    CASE
                        WHEN normalized_name = ? THEN 0
                        WHEN normalized_name LIKE ? THEN 1
                        WHEN normalized_category = ? THEN 2
                        WHEN normalized_category LIKE ? THEN 3
                        ELSE 4
                    END,
                    length(normalized_name) ASC,
                    length(normalized_category) ASC,
                    normalized_name ASC,
                    normalized_category ASC
                LIMIT 12
                """,
                (
                    f"{candidate_norm}%",
                    secondary_token_pattern,
                    f"{candidate_norm}%",
                    secondary_token_pattern,
                    candidate_norm,
                    f"{candidate_norm}%",
                    candidate_norm,
                    f"{candidate_norm}%",
                ),
            ).fetchall()
            if not rows:
                continue

            distinct_categories = {
                normalize_text(str(row["normalized_category"] or row["category"] or ""))
                for row in rows
                if str(row["normalized_category"] or row["category"] or "").strip()
            }
            support_score = min(14.0, 3.0 * math.log1p(len(rows)) + 1.5 * len(distinct_categories))

            query_text = ""
            for row in rows:
                clean_name = str(row["clean_name"] or "").strip()
                if not clean_name:
                    continue
                normalized_name = normalize_text(str(row["normalized_name"] or clean_name))
                anchored_phrase = self._query_anchored_name_phrase(clean_name, query)
                phrase = anchored_phrase or self._product_suggestion_phrase(clean_name) or self._abstract_name_phrase(clean_name, query)
                normalized_phrase = normalize_text(phrase)
                if normalized_phrase == candidate_norm or normalized_name == candidate_norm or normalized_phrase.startswith(candidate_norm):
                    query_text = phrase or clean_name
                    break
            normalized_query_text = normalize_text(query_text)
            if normalized_query_text and normalized_query_text not in seen_texts and normalized_query_text != query_norm:
                seen_texts.add(normalized_query_text)
                ranked_suggestions.append(
                    self._build_suggestion(
                        text=query_text,
                        suggestion_type="query",
                        reason="Популярное продолжение",
                        score=round(174.0 + support_score, 4),
                    )
                )

            category_phrases: List[str] = []
            seen_categories: set[str] = set()
            for row in rows:
                normalized_category = normalize_text(str(row["normalized_category"] or row["category"] or ""))
                if not normalized_category or not normalized_category.startswith(candidate_norm):
                    continue
                category_phrase = self._compact_category_phrase(str(row["category"] or row["normalized_category"] or ""))
                normalized_phrase = normalize_text(category_phrase)
                if not normalized_phrase or normalized_phrase in seen_categories:
                    continue
                seen_categories.add(normalized_phrase)
                category_phrases.append(category_phrase)
                if len(category_phrases) >= 2:
                    break

            for index, category_phrase in enumerate(category_phrases):
                normalized_phrase = normalize_text(category_phrase)
                if normalized_phrase in seen_texts or normalized_phrase == query_norm:
                    continue
                seen_texts.add(normalized_phrase)
                ranked_suggestions.append(
                    self._build_suggestion(
                        text=category_phrase,
                        suggestion_type="category",
                        reason="Популярная категория",
                        score=round(168.0 + support_score - index * 4.0, 4),
                    )
                )

        ranked_suggestions.sort(
            key=lambda item: (
                item.score,
                self._SUGGESTION_TYPE_PRIORITY.get(item.type, 0),
                -len(item.text),
                item.text,
            ),
            reverse=True,
        )
        return self._dedupe_suggestions(ranked_suggestions, query=query)

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
            profile = self.personalization_service.build_profile_by_inn(user_inn, top_ste=150)
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

    def _resolve_suggestion_user_weights(self, *, user_inn: Optional[str]) -> dict[str, float]:
        if not user_inn:
            return {}

        cache_key = self.cache_service.build_key(
            "suggestion_user_category_weights",
            data={"version": 2, "inn": user_inn},
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, dict):
            return {
                str(category_id): float(weight or 0.0)
                for category_id, weight in cached_payload.items()
                if category_id
            }

        try:
            weights = self.user_profile_scorer.compute_category_weights(user_inn)
        except Exception:
            weights = {}

        self.cache_service.set_json(
            cache_key,
            weights,
            ttl_seconds=self.settings.user_profile_cache_ttl_seconds,
        )
        return weights

    def _resolve_suggestion_user_ste_weights(self, *, user_inn: Optional[str]) -> dict[str, float]:
        if not user_inn:
            return {}

        cache_key = self.cache_service.build_key(
            "suggestion_user_ste_weights",
            data={"version": 1, "inn": user_inn},
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, dict):
            return {
                str(ste_id): float(weight or 0.0)
                for ste_id, weight in cached_payload.items()
                if ste_id
            }

        try:
            weights = self.user_profile_scorer.compute_ste_weights(user_inn)
        except Exception:
            weights = {}

        self.cache_service.set_json(
            cache_key,
            weights,
            ttl_seconds=self.settings.user_profile_cache_ttl_seconds,
        )
        return weights

    def _resolve_same_type_prefix_products(self, *, user_inn: Optional[str], query: str) -> List[dict]:
        query_norm = normalize_text(query)
        if not user_inn or len(tokenize(query_norm)) != 1 or len(query_norm) < 2 or len(query_norm) > 4:
            return []

        cache_key = self.cache_service.build_key(
            "same_type_prefix_products",
            data={
                "version": 7,
                "inn": user_inn,
                "query": query_norm,
            },
        )
        cached_payload = self.cache_service.get_json(cache_key)
        if isinstance(cached_payload, list):
            return [item for item in cached_payload if isinstance(item, dict)]

        try:
            name_context = self.personalization_service.get_entity_name_context(user_inn, limit=180)
        except Exception:
            return []

        same_type_peer_inns = [str(value) for value in name_context.get("same_type_peer_inns", []) if value]
        archetype = str(name_context.get("institution_name_archetype") or "general")
        if not same_type_peer_inns or archetype == "general":
            return []

        self._ensure_profile_db_attached()
        peer_placeholders = ", ".join("?" for _ in same_type_peer_inns)
        secondary_pattern = self._secondary_token_like_pattern(query_norm)
        candidate_rows = self.search_service.conn.execute(
            f"""
            WITH peer_stats AS (
                SELECT
                    cs.ste_id,
                    SUM(cs.purchase_count) AS purchase_count,
                    SUM(cs.total_amount) AS total_amount
                FROM profile_db.customer_ste_stats cs
                WHERE cs.customer_inn IN ({peer_placeholders})
                GROUP BY cs.ste_id
            )
            SELECT
                sc.ste_id,
                sc.clean_name,
                sc.normalized_name,
                sc.category,
                sc.normalized_category,
                sc.key_tokens,
                ps.purchase_count,
                ps.total_amount
            FROM peer_stats ps
            JOIN ste_catalog sc ON sc.ste_id = ps.ste_id
            WHERE sc.normalized_name LIKE ?
               OR sc.normalized_name LIKE ?
               OR sc.normalized_category LIKE ?
               OR sc.normalized_category LIKE ?
               OR sc.key_tokens LIKE ?
               OR sc.key_tokens LIKE ?
            ORDER BY
                ps.purchase_count DESC,
                ps.total_amount DESC,
                length(sc.normalized_name) ASC,
                sc.normalized_name ASC
            LIMIT 320
            """,
            [
                *same_type_peer_inns,
                f"{query_norm}%",
                secondary_pattern,
                f"{query_norm}%",
                secondary_pattern,
                f"{query_norm}%",
                secondary_pattern,
            ],
        ).fetchall()
        if not candidate_rows:
            return []

        candidate_ids = [str(row["ste_id"]) for row in candidate_rows if row["ste_id"]]
        candidate_placeholders = ", ".join("?" for _ in candidate_ids)
        global_rows = self.personalization_service.conn.execute(
            f"""
            SELECT
                ste_id,
                SUM(purchase_count) AS purchase_count
            FROM customer_ste_stats
            WHERE ste_id IN ({candidate_placeholders})
            GROUP BY ste_id
            """,
            candidate_ids,
        ).fetchall()
        global_purchase_count_by_ste = {
            str(row["ste_id"]): float(row["purchase_count"] or 0.0)
            for row in global_rows
            if row["ste_id"]
        }

        ranked_products: List[dict] = []
        for row in candidate_rows:
            ste_id = str(row["ste_id"] or "")
            if not ste_id:
                continue
            anchored_phrase = self._query_anchored_name_phrase(str(row["clean_name"] or ""), query)
            prefix_score = max(
                self._token_prefix_match_score(query, anchored_phrase, allow_secondary_tokens=False),
                self._token_prefix_match_score(query, str(row["clean_name"] or "")),
                self._token_prefix_match_score(query, str(row["normalized_name"] or "")),
                self._token_prefix_match_score(query, str(row["category"] or row["normalized_category"] or "")),
                self._token_prefix_match_score(query, str(row["key_tokens"] or "")),
            )
            if prefix_score <= 0:
                continue
            category_norm = normalize_text(str(row["category"] or row["normalized_category"] or ""))
            same_type_count = float(row["purchase_count"] or 0.0)
            if same_type_count <= 0:
                continue
            global_count = max(same_type_count, float(global_purchase_count_by_ste.get(ste_id, 0.0) or 0.0))
            specificity = same_type_count / global_count if global_count > 0 else 0.0
            name_type_relevance_score = self.personalization_service.customer_name_archetype_match_score(
                archetype=archetype,
                texts=[str(row["clean_name"] or "")],
            )
            type_relevance_score = self.personalization_service.customer_name_archetype_match_score(
                archetype=archetype,
                texts=[
                    str(row["clean_name"] or ""),
                    str(row["category"] or row["normalized_category"] or ""),
                ],
            )
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
            type_relevance_boost = 10.0 * type_relevance_score
            explicit_name_type_bonus = 8.0 * name_type_relevance_score
            head_token_bonus = self._query_head_token_bonus(query, anchored_phrase or str(row["clean_name"] or ""))
            generic_penalty = 10.0 if len(query_norm) <= 4 and type_relevance_score <= 0 else 0.0
            ranked_products.append(
                {
                    "steId": ste_id,
                    "name": anchored_phrase or str(row["clean_name"] or ste_id),
                    "category": str(row["category"] or ""),
                    "purchaseCount": int(same_type_count),
                    "totalAmount": round(float(row["total_amount"] or 0.0), 2),
                    "reason": "Популярно у организаций того же типа",
                    "recommendationScore": round(
                        prefix_score * 0.06
                        + popularity_score
                        + specificity_score
                        + type_relevance_boost
                        + explicit_name_type_bonus
                        + head_token_bonus
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

    def _load_cart_context_items(self, cart_ste_ids: List[str]) -> List[dict]:
        if not cart_ste_ids:
            return []

        unique_ids = unique_preserve_order([str(value) for value in cart_ste_ids if value])
        if not unique_ids:
            return []

        placeholders = ", ".join("?" for _ in unique_ids)
        rows = self.search_service.conn.execute(
            f"""
            SELECT
                ste_id,
                clean_name,
                normalized_name,
                category,
                normalized_category,
                key_tokens
            FROM ste_catalog
            WHERE ste_id IN ({placeholders})
            """,
            unique_ids,
        ).fetchall()
        return [dict(row) for row in rows]

    @classmethod
    def _cart_context_signature_stems(cls, item: dict, query: str) -> set[str]:
        query_stems = {stem_token(token) for token in cls._significant_tokens(query) if stem_token(token)}
        item_tokens = cls._significant_tokens(
            " ".join(
                part
                for part in [
                    str(item.get("normalized_name") or item.get("clean_name") or ""),
                    str(item.get("key_tokens") or ""),
                ]
                if part
            )
        )
        item_stems = {stem_token(token) for token in item_tokens if stem_token(token)}
        return item_stems - query_stems

    @classmethod
    def _cart_context_candidate_stems(cls, item: dict) -> set[str]:
        candidate_tokens = cls._significant_tokens(
            " ".join(
                part
                for part in [
                    str(item.get("normalized_name") or item.get("clean_name") or ""),
                    str(item.get("key_tokens") or ""),
                ]
                if part
            )
        )
        return {stem_token(token) for token in candidate_tokens if stem_token(token)}

    @classmethod
    def _cart_context_discriminative_stems(
        cls,
        *,
        cart_item: dict,
        query: str,
        results: List[dict],
    ) -> set[str]:
        signature_stems = cls._cart_context_signature_stems(cart_item, query)
        if not signature_stems or not results:
            return set()

        result_stem_sets = [cls._cart_context_candidate_stems(item) for item in results]
        result_count = max(1, len(result_stem_sets))
        return {
            stem
            for stem in signature_stems
            if sum(1 for stem_set in result_stem_sets if stem in stem_set) / result_count
            <= cls.MAX_CART_CONTEXT_SIGNATURE_RESULT_SHARE
        }

    @classmethod
    def _cart_context_similarity(cls, *, candidate: dict, cart_item: dict, signature_stems: set[str]) -> float:
        candidate_id = str(candidate.get("ste_id") or candidate.get("candidate_id") or "")
        cart_id = str(cart_item.get("ste_id") or "")
        if candidate_id and cart_id and candidate_id == cart_id:
            return 1.0

        if not signature_stems:
            return 0.0

        candidate_stems = cls._cart_context_candidate_stems(candidate)
        if not candidate_stems:
            return 0.0

        overlap_count = len(signature_stems & candidate_stems)
        if overlap_count == 0:
            return 0.0

        category_norm = normalize_text(str(candidate.get("category") or candidate.get("normalized_category") or ""))
        cart_category_norm = normalize_text(str(cart_item.get("category") or cart_item.get("normalized_category") or ""))
        same_category = bool(category_norm and cart_category_norm and category_norm == cart_category_norm)
        coverage = overlap_count / max(1, len(signature_stems))
        if not same_category and coverage < 0.75:
            return 0.0
        if coverage < 0.5:
            return 0.0
        return min(1.0, coverage + (0.2 if same_category else 0.0))

    @classmethod
    def _apply_cart_context_boost(
        cls,
        results: List[dict],
        *,
        query: str,
        cart_items: List[dict],
    ) -> List[dict]:
        if not results or not cart_items:
            return results

        cart_ids = {str(item.get("ste_id") or "") for item in cart_items if item.get("ste_id")}
        discriminative_signature_by_cart_id = {
            str(cart_item.get("ste_id") or ""): cls._cart_context_discriminative_stems(
                cart_item=cart_item,
                query=query,
                results=results,
            )
            for cart_item in cart_items
            if cart_item.get("ste_id")
        }
        candidate_boosts: dict[int, tuple[float, float]] = {}
        for index, item in enumerate(results):
            ste_id = str(item.get("ste_id") or item.get("candidate_id") or "")
            if ste_id and ste_id in cart_ids:
                continue

            best_similarity = 0.0
            best_base_score = float(item.get("final_score", item.get("search_score", 0.0)) or 0.0)
            for cart_item in cart_items:
                signature_stems = discriminative_signature_by_cart_id.get(str(cart_item.get("ste_id") or ""), set())
                best_similarity = max(
                    best_similarity,
                    cls._cart_context_similarity(
                        candidate=item,
                        cart_item=cart_item,
                        signature_stems=signature_stems,
                    ),
                )
            if best_similarity < 0.5:
                continue
            candidate_boosts[index] = (best_similarity, best_base_score)

        boosted_indexes = {
            index
            for index, _ in sorted(
                candidate_boosts.items(),
                key=lambda entry: (entry[1][0], entry[1][1]),
                reverse=True,
            )[: cls.MAX_CART_CONTEXT_BOOSTED_RESULTS]
        }
        updated_results: List[dict] = []
        for index, item in enumerate(results):
            ste_id = str(item.get("ste_id") or item.get("candidate_id") or "")
            if ste_id and ste_id in cart_ids:
                updated_results.append(item)
                continue

            boost_payload = candidate_boosts.get(index)
            if not boost_payload or index not in boosted_indexes:
                updated_results.append(item)
                continue

            boosted = dict(item)
            base_score = float(boosted.get("final_score", boosted.get("search_score", 0.0)) or 0.0)
            best_similarity = boost_payload[0]
            context_multiplier = 1.0 + 0.06 * best_similarity
            context_boost = base_score * (context_multiplier - 1.0)
            boosted["final_score"] = round(base_score * context_multiplier, 6)
            boosted["cart_context_similarity"] = round(best_similarity, 6)
            boosted["cart_context_boost"] = round(context_boost, 6)
            boosted["cart_context_multiplier"] = round(context_multiplier, 6)
            existing_codes = [str(code) for code in boosted.get("top_reason_codes", []) if code]
            if "SESSION_CART_CONTEXT_BOOST" not in existing_codes:
                existing_codes = ["SESSION_CART_CONTEXT_BOOST", *existing_codes]
            boosted["top_reason_codes"] = existing_codes
            boosted["reasons"] = unique_preserve_order(
                ["Поднято после добавления похожего товара в корзину"]
                + [str(reason) for reason in boosted.get("reasons", []) if reason]
            )
            updated_results.append(boosted)

        updated_results.sort(
            key=lambda item: (
                float(item.get("session_priority", 0.0)),
                float(item.get("final_score", item.get("search_score", 0.0))),
                float(item.get("search_score", 0.0)),
                float(item.get("history_priority", 0.0)),
                float(item.get("personalization_score", 0.0)),
            ),
            reverse=True,
        )
        return updated_results

    @classmethod
    def _cart_category_query_score(cls, query: str, category: str) -> float:
        return cls._token_prefix_match_score(query, category, allow_secondary_tokens=False)

    @classmethod
    def _apply_cart_category_boost(
        cls,
        results: List[dict],
        *,
        query: str,
        cart_items: List[dict],
    ) -> List[dict]:
        if not results or not cart_items:
            return results

        cart_ids = {str(item.get("ste_id") or "") for item in cart_items if item.get("ste_id")}
        category_query_scores: dict[str, float] = {}
        for cart_item in cart_items:
            category_norm = normalize_text(str(cart_item.get("category") or cart_item.get("normalized_category") or ""))
            if not category_norm:
                continue
            category_query_scores[category_norm] = max(
                category_query_scores.get(category_norm, 0.0),
                cls._cart_category_query_score(query, str(cart_item.get("category") or cart_item.get("normalized_category") or "")),
            )

        if not category_query_scores:
            return results

        candidate_boosts: dict[int, tuple[float, float, float]] = {}
        for index, item in enumerate(results):
            ste_id = str(item.get("ste_id") or item.get("candidate_id") or "")
            category_norm = normalize_text(str(item.get("category") or item.get("normalized_category") or ""))
            if not category_norm or (ste_id and ste_id in cart_ids):
                continue

            existing_codes = [str(code) for code in item.get("top_reason_codes", []) if code]
            if "SESSION_CART_BOOST" in existing_codes or "SESSION_CART_CONTEXT_BOOST" in existing_codes:
                continue

            category_query_score = category_query_scores.get(category_norm, 0.0)
            if category_query_score < cls.MIN_CART_CATEGORY_QUERY_SCORE:
                continue

            base_score = float(item.get("final_score", item.get("search_score", 0.0)) or 0.0)
            category_strength = min(1.0, category_query_score / 120.0)
            category_boost_multiplier = 1.0 + min(
                cls.MAX_CART_CATEGORY_BOOST,
                0.025 + 0.02 * category_strength,
            )
            candidate_boosts[index] = (category_query_score, base_score, category_boost_multiplier)

        boosted_indexes = {
            index
            for index, _ in sorted(
                candidate_boosts.items(),
                key=lambda entry: (entry[1][0], entry[1][1]),
                reverse=True,
            )[: cls.MAX_CART_CATEGORY_BADGED_RESULTS]
        }

        updated_results: List[dict] = []
        for index, item in enumerate(results):
            boost_payload = candidate_boosts.get(index)
            if not boost_payload or index not in boosted_indexes:
                updated_results.append(item)
                continue

            boosted = dict(item)
            base_score = float(boosted.get("final_score", boosted.get("search_score", 0.0)) or 0.0)
            category_query_score = boost_payload[0]
            category_boost_multiplier = boost_payload[2]
            category_boost = base_score * (category_boost_multiplier - 1.0)
            boosted["final_score"] = round(base_score * category_boost_multiplier, 6)
            boosted["cart_category_query_score"] = round(category_query_score, 6)
            boosted["cart_category_boost"] = round(category_boost, 6)
            boosted["cart_category_multiplier"] = round(category_boost_multiplier, 6)
            boosted["top_reason_codes"] = ["SESSION_CART_CATEGORY_BOOST", *existing_codes]
            boosted["reasons"] = unique_preserve_order(
                ["Поднято как товар из категории, уже добавленной в корзину"]
                + [str(reason) for reason in boosted.get("reasons", []) if reason]
            )
            updated_results.append(boosted)

        updated_results.sort(
            key=lambda item: (
                float(item.get("session_priority", 0.0)),
                float(item.get("final_score", item.get("search_score", 0.0))),
                float(item.get("search_score", 0.0)),
                float(item.get("history_priority", 0.0)),
                float(item.get("personalization_score", 0.0)),
            ),
            reverse=True,
        )
        return updated_results

    def _load_same_type_prefix_stats(
        self,
        *,
        ste_ids: List[str],
        peer_inns: List[str],
        customer_inn: str,
    ) -> dict[str, dict]:
        if not ste_ids:
            return {}

        ste_placeholders = ", ".join("?" for _ in ste_ids)
        if not peer_inns:
            return {}
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
        user_weights: Optional[dict[str, float]] = None,
        require_positive_boost: bool = False,
    ) -> List[SuggestionPayload]:
        query_norm = normalize_text(query)
        if not query_norm:
            return []

        raw_candidates: List[dict] = []
        for category in categories:
            candidate = cls._compact_category_phrase(str(category))
            candidate_norm = normalize_text(candidate)
            if not candidate_norm or candidate_norm == query_norm:
                continue
            score = cls._token_prefix_match_score(query, candidate)
            if score > 0:
                raw_candidates.append(
                    {
                        "text": candidate,
                        "suggestion_type": "category",
                        "reason": "Категория из истории",
                        "category": candidate_norm,
                        "base_score": score + 20.0,
                    }
                )

        personalized_candidates = apply_personalization(raw_candidates, user_weights or {})
        if require_positive_boost:
            personalized_candidates = [
                item
                for item in personalized_candidates
                if float(item.get("boost_weight", 0.0) or 0.0) > 0.0
            ]
        ranked_candidates = [
            cls._build_suggestion(
                text=str(item.get("text") or ""),
                suggestion_type="category",
                reason=str(item.get("reason") or "Категория из истории"),
                score=float(item.get("final_score", item.get("base_score", 0.0)) or 0.0),
            )
            for item in personalized_candidates
            if item.get("text")
        ]
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
        user_weights: Optional[dict[str, float]] = None,
        user_item_weights: Optional[dict[str, float]] = None,
        require_positive_boost: bool = False,
    ) -> List[SuggestionPayload]:
        query_norm = normalize_text(query)
        if not query_norm or not products:
            return []

        raw_candidates: List[dict] = []

        for source_index, item in enumerate(products):
            full_name = str(item.get("name") or "").strip()
            if not full_name:
                continue
            suggestion_phrase = cls._product_suggestion_phrase(full_name) or full_name
            anchored_phrase = cls._query_anchored_name_phrase(full_name, query)
            display_phrase = anchored_phrase or suggestion_phrase
            candidate_norm = normalize_text(display_phrase)
            full_name_norm = normalize_text(full_name)
            if not full_name_norm or full_name_norm == query_norm:
                continue

            score = max(
                cls._token_prefix_match_score(query, display_phrase),
                cls._token_prefix_match_score(query, suggestion_phrase),
                cls._token_prefix_match_score(query, full_name),
            )
            if score <= 0:
                continue

            score += min(float(item.get("purchaseCount") or 0), 15.0)
            score += min(float(item.get("recommendationScore") or 0.0), 12.0)
            score += cls._query_head_token_bonus(query, display_phrase)
            if candidate_norm.startswith(query_norm):
                score += 18.0
            if full_name_norm.startswith(query_norm):
                score += 10.0

            item_reason = str(item.get("reason") or "")
            normalized_reason = item_reason.lower()
            if "часто закупалось учреждением" in normalized_reason:
                reason = "Часто закупалось"
            elif "часто поставлялось поставщиком" in normalized_reason:
                reason = "Часто поставлялось"
            elif "того же типа" in normalized_reason or "рекомендуется для" in normalized_reason:
                reason = "По типу учреждения"
            elif "похожих учреждений" in normalized_reason:
                reason = "Популярно у похожих учреждений"
            elif "поставлялось" in normalized_reason or "поставщик" in normalized_reason:
                reason = "Часто поставлялось"
            elif "учреждени" in normalized_reason:
                reason = "Часто закупалось"
            elif "регион" in normalized_reason:
                reason = "Популярно в регионе"
            else:
                reason = "Часто закупалось"

            raw_candidates.append(
                {
                    "ste_id": str(item.get("steId") or item.get("ste_id") or ""),
                    "text": display_phrase,
                    "suggestion_type": "product",
                    "reason": reason,
                    "category": str(item.get("category") or ""),
                    "base_score": score + 35.0,
                    "source_rank": -source_index,
                }
            )

        personalized_candidates = apply_personalization(
            raw_candidates,
            user_weights or {},
            user_item_weights=user_item_weights or {},
        )
        if require_positive_boost:
            personalized_candidates = [
                item
                for item in personalized_candidates
                if (
                    float(item.get("item_boost_weight", 0.0) or 0.0) > 0.0
                    if str(item.get("reason") or "") in {"Часто закупалось", "Часто поставлялось"}
                    else float(item.get("boost_weight", 0.0) or 0.0) > 0.0
                )
            ]
        ranked_candidates = [
            (
                float(item.get("source_rank", 0.0) or 0.0),
                cls._build_suggestion(
                    text=str(item.get("text") or ""),
                    suggestion_type="product",
                    reason=str(item.get("reason") or "Часто закупалось"),
                    score=float(item.get("final_score", item.get("base_score", 0.0)) or 0.0),
                ),
            )
            for item in personalized_candidates
            if item.get("text")
        ]
        ranked_candidates.sort(
            key=lambda item: (item[1].score, item[0], -len(item[1].text), item[1].text),
            reverse=True,
        )
        return cls._dedupe_suggestions([item[1] for item in ranked_candidates])

    @staticmethod
    def _merge_suggestion_groups(*groups: List[SuggestionPayload], query: str = "") -> List[SuggestionPayload]:
        merged: List[SuggestionPayload] = []
        if not groups:
            return merged

        max_len = max((len(group) for group in groups), default=0)
        for index in range(max_len):
            for group in groups:
                if index < len(group):
                    merged.append(group[index])
        return TenderHackApiService._dedupe_suggestions(merged, query=query)

    @classmethod
    def _suggestion_family_key_from_text(cls, text: str) -> str:
        significant_stems = [stem_token(token) for token in cls._significant_tokens(text) if stem_token(token)]
        if significant_stems:
            return significant_stems[0]
        return normalize_text(text)

    @classmethod
    def _diversify_suggestions_by_family(cls, suggestions: List[SuggestionPayload]) -> List[SuggestionPayload]:
        if len(suggestions) <= 1:
            return suggestions

        groups: dict[str, List[SuggestionPayload]] = {}
        family_order: List[str] = []
        for item in suggestions:
            family_key = cls._suggestion_family_key_from_text(item.text)
            if not family_key:
                family_key = normalize_text(item.text)
            if family_key not in groups:
                groups[family_key] = []
                family_order.append(family_key)
            groups[family_key].append(item)

        for group in groups.values():
            group.sort(
                key=lambda item: (
                    item.score,
                    cls._SUGGESTION_TYPE_PRIORITY.get(item.type, 0),
                    -len(item.text),
                    item.text,
                ),
                reverse=True,
            )

        diversified: List[SuggestionPayload] = []
        max_group_len = max((len(group) for group in groups.values()), default=0)
        for index in range(max_group_len):
            round_items = [groups[key][index] for key in family_order if index < len(groups[key])]
            round_items.sort(
                key=lambda item: (
                    item.score,
                    cls._SUGGESTION_TYPE_PRIORITY.get(item.type, 0),
                    -len(item.text),
                    item.text,
                ),
                reverse=True,
            )
            diversified.extend(round_items)
        return diversified

    @classmethod
    def _suggestion_history_bucket(cls, suggestion: SuggestionPayload) -> str:
        reason = str(suggestion.reason or "")
        if reason == "Категория из истории":
            return "history_category"
        if reason in {"Часто закупалось", "Часто поставлялось"}:
            return "history_product"
        return ""

    @classmethod
    def _partition_suggestions_by_history_limit(
        cls,
        suggestions: List[SuggestionPayload],
        *,
        top_k: int,
    ) -> tuple[List[SuggestionPayload], List[SuggestionPayload]]:
        if top_k <= 0 or not suggestions:
            return ([], list(suggestions))

        max_history_categories = min(cls.MAX_HISTORY_CATEGORY_SUGGESTIONS, top_k)
        max_history_total = min(
            top_k,
            max(1, min(cls.MAX_HISTORY_REASON_SUGGESTIONS, math.ceil(top_k * 0.4))),
        )

        kept: List[SuggestionPayload] = []
        overflow: List[SuggestionPayload] = []
        history_category_count = 0
        history_total_count = 0
        institution_type_count = 0

        for item in suggestions:
            history_bucket = cls._suggestion_history_bucket(item)
            suggestion_reason = str(item.reason or "")
            if not history_bucket:
                if suggestion_reason == "По типу учреждения":
                    if institution_type_count >= min(cls.MAX_INSTITUTION_TYPE_SUGGESTIONS, top_k):
                        overflow.append(item)
                        continue
                    institution_type_count += 1
                kept.append(item)
                continue

            is_history_category = history_bucket == "history_category"
            exceeds_category_limit = is_history_category and history_category_count >= max_history_categories
            exceeds_history_limit = history_total_count >= max_history_total
            if exceeds_category_limit or exceeds_history_limit:
                overflow.append(item)
                continue

            kept.append(item)
            history_total_count += 1
            if is_history_category:
                history_category_count += 1

        return (kept, overflow)

    @classmethod
    def _suggestion_dedupe_key(
        cls,
        suggestion: SuggestionPayload,
        *,
        query: str = "",
    ) -> str:
        normalized_text = normalize_text(suggestion.text)
        if not normalized_text:
            return ""

        if suggestion.type in {"query", "correction"}:
            significant_stems = sorted({stem_token(token) for token in cls._significant_tokens(normalized_text) if stem_token(token)})
            if len(significant_stems) >= 3:
                query_stems = {stem_token(token) for token in cls._significant_tokens(query) if stem_token(token)}
                if not query_stems or len(query_stems & set(significant_stems)) >= min(2, len(query_stems)):
                    return f"phrase:{' '.join(significant_stems)}"

        stemmed_tokens = stem_tokens(tokenize(normalized_text))
        if stemmed_tokens:
            return " ".join(stemmed_tokens)
        return normalized_text

    @classmethod
    def _query_alignment_score(cls, suggestion_text: str, query: str) -> tuple[float, int, int]:
        normalized_query = normalize_text(query)
        normalized_suggestion = normalize_text(suggestion_text)
        if not normalized_query or not normalized_suggestion:
            return (0.0, 0, 0)

        if normalized_suggestion == normalized_query:
            return (1000.0, 0, 0)
        if normalized_suggestion.startswith(normalized_query):
            return (500.0, 0, 0)

        query_tokens = [stem_token(token) for token in cls._significant_tokens(normalized_query) if stem_token(token)]
        suggestion_tokens = [stem_token(token) for token in cls._significant_tokens(normalized_suggestion) if stem_token(token)]
        if not query_tokens or not suggestion_tokens:
            return (0.0, 0, 0)

        prefix_matches = 0
        for left, right in zip(query_tokens, suggestion_tokens):
            if left != right:
                break
            prefix_matches += 1

        ordered_matches = 0
        suggestion_index = 0
        for query_token in query_tokens:
            while suggestion_index < len(suggestion_tokens) and suggestion_tokens[suggestion_index] != query_token:
                suggestion_index += 1
            if suggestion_index >= len(suggestion_tokens):
                break
            ordered_matches += 1
            suggestion_index += 1

        score = 25.0 * prefix_matches + 5.0 * ordered_matches
        return (score, prefix_matches, ordered_matches)

    @classmethod
    def _is_better_suggestion(
        cls,
        candidate: SuggestionPayload,
        current: SuggestionPayload,
        *,
        query: str = "",
    ) -> bool:
        if candidate.score != current.score:
            return candidate.score > current.score

        candidate_type_rank = cls._SUGGESTION_TYPE_PRIORITY.get(candidate.type, 0)
        current_type_rank = cls._SUGGESTION_TYPE_PRIORITY.get(current.type, 0)
        if candidate_type_rank != current_type_rank:
            return candidate_type_rank > current_type_rank

        candidate_alignment = cls._query_alignment_score(candidate.text, query)
        current_alignment = cls._query_alignment_score(current.text, query)
        if candidate_alignment != current_alignment:
            return candidate_alignment > current_alignment

        candidate_text = normalize_text(candidate.text)
        current_text = normalize_text(current.text)
        if len(candidate_text) != len(current_text):
            return len(candidate_text) < len(current_text)
        return candidate_text < current_text

    @classmethod
    def _dedupe_suggestions(
        cls,
        suggestions: List[SuggestionPayload],
        *,
        query: str = "",
    ) -> List[SuggestionPayload]:
        deduped_by_key: dict[str, SuggestionPayload] = {}
        order: List[str] = []
        for item in suggestions:
            dedupe_key = cls._suggestion_dedupe_key(item, query=query)
            if not dedupe_key:
                continue
            if dedupe_key not in deduped_by_key:
                deduped_by_key[dedupe_key] = item
                order.append(dedupe_key)
                continue
            if cls._is_better_suggestion(item, deduped_by_key[dedupe_key], query=query):
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
        entity_type: str = "customer",
    ) -> Optional[str]:
        if is_bounced:
            return None
        codes = {str(code) for code in reason_codes}
        session_category_set = {normalize_text(value) for value in session_categories if value}
        category_norm = normalize_text(category)

        if "SESSION_CART_BOOST" in codes or "SESSION_CART_CONTEXT_BOOST" in codes or "SESSION_CART_CATEGORY_BOOST" in codes:
            return "Рекомендовано вам исходя из корзины"
        if "SESSION_CLICK_BOOST" in codes:
            return "Недавно смотрели"
        if "SUPPLIER_OWN_PRODUCT" in codes:
            return "Вы поставляете этот товар"
        if "SUPPLIER_CATEGORY_MATCH" in codes:
            return "Похожий товар из вашего ассортимента"
        if "INSTITUTION_TYPE_PREFIX_MATCH" in codes:
            return "По типу учреждения"
        if codes & {"USER_CATEGORY_AFFINITY", "USER_REPEAT_BUY", "RECENT_SIMILAR_PURCHASE", "SUPPLIER_AFFINITY"}:
            return "На основе ваших поставок" if str(entity_type or "") == "supplier" else "На основе ваших закупок"
        if "SIMILAR_CUSTOMER_POPULARITY" in codes:
            return "Популярно у похожих заказчиков"
        if category_norm and category_norm in session_category_set:
            return "Недавно смотрели"
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
        top_k: int = Query(default=8, ge=1, le=10),
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
