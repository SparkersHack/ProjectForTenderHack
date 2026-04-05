from __future__ import annotations

from typing import Callable, List, Literal, Optional

from .constants import LOGIN_CACHE_VERSION, SUGGESTIONS_CACHE_VERSION
from .paths import ensure_src_root
from .schemas import SuggestionPayload, UserPayload
from .utils import model_dump

ensure_src_root()

from tenderhack.text import normalize_text, stem_token, stem_tokens, tokenize, unique_preserve_order


LoginLoader = Callable[[str], UserPayload]
FrequentProductLoader = Callable[[List[dict]], List[dict]]
SameTypePrefixLoader = Callable[[Optional[str], str], List[dict]]


class SearchSuggestionService:
    _SUGGESTION_TYPE_PRIORITY = {
        "correction": 4,
        "product": 3,
        "query": 2,
        "category": 1,
    }

    def __init__(
        self,
        *,
        cache_service,
        search_service,
        personalization_service,
        login_loader: LoginLoader,
        frequent_product_loader: FrequentProductLoader,
        same_type_prefix_loader: SameTypePrefixLoader,
        user_profile_cache_ttl_seconds: int,
        suggestions_cache_ttl_seconds: int,
    ) -> None:
        self.cache_service = cache_service
        self.search_service = search_service
        self.personalization_service = personalization_service
        self.login_loader = login_loader
        self.frequent_product_loader = frequent_product_loader
        self.same_type_prefix_loader = same_type_prefix_loader
        self.user_profile_cache_ttl_seconds = user_profile_cache_ttl_seconds
        self.suggestions_cache_ttl_seconds = suggestions_cache_ttl_seconds

    def suggestions(
        self,
        *,
        query: str,
        top_k: int = 5,
        user_inn: Optional[str] = None,
        viewed_categories: Optional[List[str]] = None,
        top_categories: Optional[List[str]] = None,
    ) -> List[SuggestionPayload]:
        cache_key = self.cache_service.build_key(
            "suggestions",
            data={
                "version": SUGGESTIONS_CACHE_VERSION,
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
        normalized_input = normalize_text(normalized_query or query)
        short_personalized_prefix = bool(
            user_inn
            and len(tokenize(normalized_query or query)) == 1
            and 2 <= len(normalized_input) <= 4
        )

        same_type_prefix_suggestions = self._build_personalized_product_suggestions(
            query=query,
            products=self.same_type_prefix_loader(user_inn, query),
        )
        product_suggestions = self._build_personalized_product_suggestions(
            query=query,
            products=self._resolve_suggestion_products(user_inn=user_inn),
        )
        if short_personalized_prefix and same_type_prefix_suggestions:
            product_suggestions = [item for item in product_suggestions if str(item.reason or "") == "Часто закупалось"]

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
            [model_dump(item) for item in result],
            ttl_seconds=self.suggestions_cache_ttl_seconds,
        )
        return result

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
            data={"inn": user_inn, "version": LOGIN_CACHE_VERSION},
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
            payload = self.login_loader(user_inn)
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
            frequent_products = self.frequent_product_loader(recommended_ste[:150])
        except Exception:
            frequent_products = []

        self.cache_service.set_json(
            cache_key,
            frequent_products,
            ttl_seconds=self.user_profile_cache_ttl_seconds,
        )
        return frequent_products

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

    @classmethod
    def _abstract_name_phrase(cls, name: str, query: str) -> str:
        query_tokens = tokenize(query)
        name_tokens = tokenize(name)
        if not name_tokens:
            return ""

        phrase_tokens: List[str] = []
        for token in name_tokens:
            if token.isdigit() or any(char.isdigit() for char in token):
                break
            phrase_tokens.append(token)
        phrase_tokens = cls._trim_trailing_connector_tokens(phrase_tokens)
        min_tokens = 1 if len(query_tokens) <= 1 else 2
        if len(phrase_tokens) < min_tokens:
            return ""
        return " ".join(phrase_tokens)

    @classmethod
    def _compact_category_phrase(cls, category: str) -> str:
        category_tokens = [token for token in tokenize(category) if not token.isdigit()]
        category_tokens = cls._trim_trailing_connector_tokens(category_tokens)
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
            first_token_stem = stem_token(first_token)
            if query_stem and first_token_stem.startswith(query_stem):
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
                if score <= 0:
                    continue
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
            if score <= 0:
                continue
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

    @classmethod
    def _merge_suggestion_groups(cls, *groups: List[SuggestionPayload]) -> List[SuggestionPayload]:
        merged: List[SuggestionPayload] = []
        if not groups:
            return merged

        max_len = max((len(group) for group in groups), default=0)
        for index in range(max_len):
            for group in groups:
                if index < len(group):
                    merged.append(group[index])
        return cls._dedupe_suggestions(merged)

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


__all__ = ["SearchSuggestionService"]
