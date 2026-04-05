from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional

from .config import AppSettings
from .constants import LOGIN_CACHE_VERSION, SEARCH_CACHE_VERSION, SUGGESTIONS_CACHE_VERSION
from .paths import ensure_src_root
from .schemas import (
    EventRequest,
    EventResponsePayload,
    ProductPayload,
    SearchRequest,
    SearchResponsePayload,
    SearchUserContext,
    UserPayload,
)
from .suggestions import SearchSuggestionService
from .utils import model_dump

ensure_src_root()

from tenderhack.cache import CacheService
from tenderhack.cart_boost import CartBoostModifier, InMemoryCartStorage
from tenderhack.descriptions import CatalogDescriptionService
from tenderhack.offers import OfferLookupService
from tenderhack.online_state import OnlineStateService
from tenderhack.penalization import InMemorySkipStorage, InteractionTracker, RankingModifier
from tenderhack.personalization import PersonalizationService
from tenderhack.personalization_runtime import PersonalizationRuntimeService
from tenderhack.search import SearchService
from tenderhack.search_rerank_model import SearchRerankPredictor
from tenderhack.text import normalize_text, tokenize, unique_preserve_order


class TenderHackApiService:
    LOGIN_CACHE_VERSION = LOGIN_CACHE_VERSION
    SEARCH_CACHE_VERSION = SEARCH_CACHE_VERSION
    SUGGESTIONS_CACHE_VERSION = SUGGESTIONS_CACHE_VERSION

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
        self.suggestion_service = SearchSuggestionService(
            cache_service=self.cache_service,
            search_service=self.search_service,
            personalization_service=self.personalization_service,
            login_loader=self.login,
            frequent_product_loader=self._load_frequent_products,
            same_type_prefix_loader=lambda user_inn, query: self._resolve_same_type_prefix_products(
                user_inn=user_inn,
                query=query,
            ),
            user_profile_cache_ttl_seconds=settings.user_profile_cache_ttl_seconds,
            suggestions_cache_ttl_seconds=settings.suggestions_cache_ttl_seconds,
        )

    def close(self) -> None:
        self.search_service.close()
        self.personalization_service.close()
        self.personalization_runtime_service.close()
        self.offer_lookup_service.close()
        self.description_service.close()
        self.cache_service.close()

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
        self.cache_service.set_json(cache_key, model_dump(payload), ttl_seconds=self.settings.login_cache_ttl_seconds)
        return payload

    def search(self, payload: SearchRequest) -> SearchResponsePayload:
        page_limit = int(payload.limit or payload.topK)
        user_context = payload.userContext or SearchUserContext()
        resolved_user_id = self._resolve_user_id(user_id=user_context.id, user_inn=user_context.inn)
        normalized_query = normalize_text(payload.query)
        short_personalized_prefix = bool(
            user_context.inn
            and len(tokenize(normalized_query)) == 1
            and 2 <= len(normalized_query) <= 4
        )
        session_categories, bounced_categories, merged_session_state = self._build_merged_session_state(
            payload=payload,
            user_context=user_context,
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
        results = self._maybe_rerank_search_results(query=payload.query, raw_payload=raw_payload, results=results)
        results = self._maybe_apply_personalization(
            payload=payload,
            raw_payload=raw_payload,
            results=results,
            user_context=user_context,
            resolved_user_id=resolved_user_id,
            session_categories=session_categories,
            merged_session_state=merged_session_state,
        )

        results = self.cart_boost_modifier.apply_boost(recommendations=results, user_id=resolved_user_id)
        results = self.ranking_modifier.apply_penalties(recommendations=results, user_id=resolved_user_id)

        if same_type_prefix_products:
            self._apply_same_type_prefix_boost(results, same_type_prefix_products)

        for item in results:
            category_norm = normalize_text(str(item.get("category", "")))
            if category_norm and category_norm in bounced_categories:
                item["final_score"] = round(float(item.get("final_score", item.get("search_score", 0.0))) - 100.0, 4)
                item["reason_to_hide"] = "Категория пессимизирована после быстрого отказа"

        self._sort_results(results)
        total_found = int(raw_payload.get("total_found", len(results)))
        paginated_results = results[payload.offset : payload.offset + page_limit]

        response_payload = SearchResponsePayload(
            items=self._build_product_payloads(
                paginated_results=paginated_results,
                session_categories=session_categories,
            ),
            totalCount=total_found,
            total_found=total_found,
            has_more=payload.offset + page_limit < total_found,
            correctedQuery=self._resolve_corrected_query(raw_payload["query"]),
        )
        self.cache_service.set_json(
            cache_key,
            model_dump(response_payload),
            ttl_seconds=self.settings.search_cache_ttl_seconds,
        )
        return response_payload

    def record_event(self, payload: EventRequest) -> EventResponsePayload:
        resolved_user_id = self._resolve_user_id(user_id=payload.userId, user_inn=payload.inn)
        if payload.eventType == "item_close" and payload.category:
            self.interaction_tracker.register_view(
                user_id=resolved_user_id,
                category_id=str(payload.category),
                dwell_time_ms=payload.durationMs or 0,
            )

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
    ):
        return self.suggestion_service.suggestions(
            query=query,
            top_k=top_k,
            user_inn=user_inn,
            viewed_categories=viewed_categories,
            top_categories=top_categories,
        )

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

    def _build_merged_session_state(
        self,
        *,
        payload: SearchRequest,
        user_context: SearchUserContext,
    ) -> tuple[List[str], set[str], dict]:
        server_session = self.online_state_service.get_session_state(
            user_id=self._resolve_session_user_id(user_id=user_context.id, user_inn=user_context.inn),
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
        return session_categories, bounced_categories, merged_session_state

    def _maybe_rerank_search_results(self, *, query: str, raw_payload: dict, results: List[dict]) -> List[dict]:
        if self.search_rerank_predictor and self.search_rerank_predictor.enabled:
            return self.search_rerank_predictor.rerank_candidates(
                query=query,
                query_meta=dict(raw_payload["query"]),
                candidates=results,
            )
        return results

    def _maybe_apply_personalization(
        self,
        *,
        payload: SearchRequest,
        raw_payload: dict,
        results: List[dict],
        user_context: SearchUserContext,
        resolved_user_id: str,
        session_categories: List[str],
        merged_session_state: dict,
    ) -> List[dict]:
        if user_context.inn or user_context.region or session_categories:
            personalization_query = (
                raw_payload["query"].get("corrected_query")
                or raw_payload["query"].get("normalized_query")
                or payload.query
            )
            return self.personalization_runtime_service.rerank_candidates(
                query=str(personalization_query),
                candidates=results,
                user_id=resolved_user_id,
                customer_inn=user_context.inn,
                customer_region=user_context.region,
                session_categories=session_categories,
                session_state=merged_session_state,
            )

        for item in results:
            item["session_priority"] = 0.0
            item["final_score"] = item.get("search_score", 0.0)
            item["history_priority"] = 0.0
            item["top_reason_codes"] = []
            item["reasons"] = ["оставлено выше за счёт базовой текстовой релевантности"]
        return results

    def _build_product_payloads(
        self,
        *,
        paginated_results: List[dict],
        session_categories: List[str],
    ) -> List[ProductPayload]:
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
        return products

    @staticmethod
    def _sort_results(results: List[dict]) -> None:
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

    @staticmethod
    def _resolve_corrected_query(query_payload: dict) -> Optional[str]:
        corrected_query = query_payload.get("corrected_query") or None
        normalized_query = query_payload.get("normalized_query") or None
        if corrected_query == normalized_query:
            return None
        return corrected_query

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
                SearchSuggestionService._token_prefix_match_score(
                    query,
                    str(row["clean_name"] or ""),
                    allow_secondary_tokens=False,
                ),
                SearchSuggestionService._token_prefix_match_score(
                    query,
                    str(row["normalized_name"] or ""),
                    allow_secondary_tokens=False,
                ),
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

    @staticmethod
    def _resolve_session_user_id(*, user_id: Optional[str], user_inn: Optional[str]) -> Optional[str]:
        return user_id or (f"user-{user_inn}" if user_inn else None)

    @staticmethod
    def _resolve_user_id(*, user_id: Optional[str], user_inn: Optional[str]) -> str:
        return user_id or (f"user-{user_inn}" if user_inn else "anonymous")

    @staticmethod
    def _search_cache_data(
        payload: SearchRequest,
        server_session: Optional[dict] = None,
        search_rerank_token: str = "disabled",
    ) -> dict:
        user_context = payload.userContext or SearchUserContext()
        page_limit = int(payload.limit or payload.topK)
        return {
            "version": SEARCH_CACHE_VERSION,
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


__all__ = ["TenderHackApiService"]
