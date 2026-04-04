from __future__ import annotations

import copy
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tenderhack.cache import CacheService
from features.personalization_features import derive_item_kind
from tenderhack.dataset_layout import resolve_preprocessed_db_path
from tenderhack.personalization_model import PersonalizationPredictor
from tenderhack.text import normalize_text, unique_preserve_order


DEFAULT_PREPROCESSED_DB = resolve_preprocessed_db_path()
DEFAULT_PERSONALIZATION_MODEL_PATH = Path("artifacts/personalization_model.cbm")


def _chunked(values: List[str], size: int = 400) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _parse_iso_date(value: object) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _quantile(values: List[float], q: float) -> float:
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return float(lower + (upper - lower) * fraction)


class PersonalizationRuntimeService:
    def __init__(
        self,
        db_path: Path | str = DEFAULT_PREPROCESSED_DB,
        model_path: Path | str = DEFAULT_PERSONALIZATION_MODEL_PATH,
        personalization_weight: float = 0.35,
        cache_service: CacheService | None = None,
        base_profile_ttl_seconds: int = 1800,
    ) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.predictor = PersonalizationPredictor(model_path=model_path)
        self.personalization_weight = float(personalization_weight)
        self.cache_service = cache_service
        self.base_profile_ttl_seconds = int(base_profile_ttl_seconds)

    def close(self) -> None:
        self.conn.close()

    def rerank_candidates(
        self,
        *,
        query: str,
        candidates: List[Dict[str, object]],
        user_id: str,
        customer_inn: Optional[str] = None,
        customer_region: Optional[str] = None,
        session_categories: Optional[List[str]] = None,
        session_state: Optional[Dict[str, object]] = None,
        reference_date: Optional[date] = None,
    ) -> List[Dict[str, object]]:
        if not candidates:
            return []

        active_date = reference_date or date.today()
        user_profile = self.build_user_profile(
            user_id=user_id,
            customer_inn=customer_inn,
            customer_region=customer_region,
            session_categories=session_categories or [],
            reference_date=active_date,
        )
        enriched_candidates = self.enrich_candidates(
            candidates=candidates,
            user_profile=user_profile,
            customer_region=str(user_profile.get("customer_region") or customer_region or ""),
            reference_date=active_date,
        )
        query_features = {
            "query": query,
            "normalized_query": normalize_text(query),
            "reference_date": active_date.isoformat(),
            "user_id": user_id,
        }
        predictions = self.predictor.predict_personalization(
            candidates=enriched_candidates,
            user_profile=user_profile,
            query_features=query_features,
        )
        predictions_by_id = {str(item["candidate_id"]): item for item in predictions}
        active_session_state = session_state or {}

        reranked: List[Dict[str, object]] = []
        for index, candidate in enumerate(enriched_candidates, start=1):
            candidate_id = str(candidate.get("candidate_id") or candidate.get("ste_id") or "")
            prediction = predictions_by_id.get(candidate_id, {})
            personalization_score = float(prediction.get("personalization_score", 0.0) or 0.0)
            dynamic_score, dynamic_reason_codes, dynamic_reasons = self._dynamic_session_adjustment(
                candidate=candidate,
                session_state=active_session_state,
            )
            reason_codes = unique_preserve_order(
                [str(code) for code in prediction.get("top_reason_codes", [])] + dynamic_reason_codes
            )
            reasons = unique_preserve_order(
                [str(text) for text in prediction.get("reasons", [])] + dynamic_reasons
            )
            final_score = (
                float(candidate.get("search_score", 0.0))
                + self.personalization_weight * personalization_score
                + dynamic_score
            )
            enriched = dict(candidate)
            enriched["base_search_rank"] = index
            enriched["personalization_score"] = round(personalization_score, 6)
            enriched["dynamic_session_score"] = round(dynamic_score, 6)
            enriched["top_reason_codes"] = reason_codes
            enriched["reasons"] = reasons
            enriched["final_score"] = round(final_score, 6)
            reranked.append(enriched)

        reranked.sort(
            key=lambda item: (
                float(item.get("final_score", item.get("search_score", 0.0))),
                float(item.get("personalization_score", 0.0)),
                float(item.get("search_score", 0.0)),
            ),
            reverse=True,
        )
        return reranked

    def _dynamic_session_adjustment(
        self,
        *,
        candidate: Dict[str, object],
        session_state: Dict[str, object],
    ) -> tuple[float, List[str], List[str]]:
        category_norm = normalize_text(str(candidate.get("category") or candidate.get("normalized_category") or ""))
        ste_id = str(candidate.get("ste_id") or candidate.get("candidate_id") or "")
        clicked_ids = {str(value) for value in session_state.get("clicked_ste_ids", [])}
        cart_ids = {str(value) for value in session_state.get("cart_ste_ids", [])}
        recent_categories = {normalize_text(str(value)) for value in session_state.get("recent_categories", []) if value}
        bounced_categories = {normalize_text(str(value)) for value in session_state.get("bounced_categories", []) if value}

        score = 0.0
        reason_codes: List[str] = []
        reasons: List[str] = []

        if ste_id and ste_id in clicked_ids:
            score += 12.0
            reason_codes.append("SESSION_CLICK_BOOST")
            reasons.append("Поднято после недавнего клика пользователя")
        if ste_id and ste_id in cart_ids:
            # Cart actions are the strongest online signal and should dominate
            # weaker historical priors for the exact candidate.
            score += 35.0
            reason_codes.append("SESSION_CART_BOOST")
            reasons.append("Поднято после добавления похожей позиции в корзину")
        if category_norm and category_norm in recent_categories:
            score += 1.5
            reason_codes.append("SESSION_CATEGORY_BOOST")
            reasons.append("Категория была недавно просмотрена в текущей сессии")
        if category_norm and category_norm in bounced_categories:
            score -= 10.0
            reason_codes.append("SESSION_BOUNCE_PENALTY")
            reasons.append("Категория понижена после быстрого отказа")

        return score, reason_codes, reasons

    def build_user_profile(
        self,
        *,
        user_id: str,
        customer_inn: Optional[str],
        customer_region: Optional[str],
        session_categories: List[str],
        reference_date: date,
    ) -> Dict[str, object]:
        profile = self._load_base_profile(customer_inn=customer_inn, customer_region=customer_region)
        profile["user_id"] = user_id
        if customer_region:
            profile["customer_region"] = customer_region
        self._apply_session_categories(profile=profile, session_categories=session_categories, reference_date=reference_date)
        return profile

    def _new_profile_template(self, region: Optional[str]) -> Dict[str, object]:
        return {
            "user_id": "UNKNOWN",
            "customer_region": region or "UNKNOWN",
            "total_purchases": 0,
            "total_amount": 0.0,
            "recent_amounts": [],
            "category_counts": {},
            "ste_counts": {},
            "supplier_counts": {},
            "item_kind_counts": {},
            "last_purchase_dt": None,
            "last_category_purchase_dt": {},
            "last_ste_purchase_dt": {},
            "last_supplier_purchase_dt": {},
            "last_item_kind_purchase_dt": {},
            "recent_purchase_dates": [],
            "recent_category_dates": {},
        }

    def _load_base_profile(
        self,
        *,
        customer_inn: Optional[str],
        customer_region: Optional[str],
    ) -> Dict[str, object]:
        if not customer_inn:
            return self._new_profile_template(region=customer_region)

        cache_key = None
        if self.cache_service and self.cache_service.enabled:
            cache_key = self.cache_service.build_key("user-profile", suffix=str(customer_inn))
            cached_profile = self.cache_service.get_json(cache_key)
            if isinstance(cached_profile, dict):
                profile = copy.deepcopy(cached_profile)
                if customer_region:
                    profile["customer_region"] = customer_region
                return profile

        region = customer_region or self._infer_customer_region(customer_inn)
        profile = self._new_profile_template(region=region)
        self._fill_profile_from_history(profile=profile, customer_inn=customer_inn)

        if cache_key and self.cache_service:
            self.cache_service.set_json(cache_key, profile, ttl_seconds=self.base_profile_ttl_seconds)
        return copy.deepcopy(profile)

    def enrich_candidates(
        self,
        *,
        candidates: List[Dict[str, object]],
        user_profile: Dict[str, object],
        customer_region: str,
        reference_date: date,
    ) -> List[Dict[str, object]]:
        ste_ids = unique_preserve_order([str(item.get("ste_id") or item.get("candidate_id") or "") for item in candidates if item.get("ste_id") or item.get("candidate_id")])
        normalized_categories = unique_preserve_order(
            [
                str(item.get("normalized_category") or normalize_text(str(item.get("category") or "")))
                for item in candidates
                if item.get("normalized_category") or item.get("category")
            ]
        )

        offer_lookup = self._load_offer_lookup(ste_ids)
        global_ste_stats = self._load_global_ste_stats(ste_ids)
        regional_ste_stats = self._load_regional_ste_stats(ste_ids, customer_region)
        global_category_stats = self._load_global_category_stats(normalized_categories)
        regional_category_stats = self._load_regional_category_stats(normalized_categories, customer_region)
        recent_ste_stats = self._load_recent_ste_stats(ste_ids, cutoff_date=reference_date - timedelta(days=30))
        recent_category_stats = self._load_recent_category_stats(
            normalized_categories,
            cutoff_date=reference_date - timedelta(days=90),
        )
        seasonal_category_stats = self._load_seasonal_category_stats(normalized_categories, month=reference_date.month)
        category_price_bands = self._load_category_price_bands(normalized_categories)
        dominant_category = self._dominant_category(user_profile)
        similar_customer_stats = self._load_similar_customer_ste_stats(
            ste_ids=ste_ids,
            customer_region=customer_region,
            normalized_category=dominant_category,
        )

        enriched: List[Dict[str, object]] = []
        for candidate in candidates:
            payload = dict(candidate)
            ste_id = str(payload.get("ste_id") or payload.get("candidate_id") or "")
            normalized_category = str(payload.get("normalized_category") or normalize_text(str(payload.get("category") or "")))
            offer = offer_lookup.get(ste_id, {})
            ste_stats = global_ste_stats.get(ste_id, {})
            price_bands = category_price_bands.get(normalized_category, {})
            payload["candidate_id"] = ste_id
            payload["customer_region"] = customer_region or ""
            payload["candidate_primary_supplier_inn"] = str(offer.get("supplier_inn") or "")
            payload["candidate_primary_supplier_region"] = str(offer.get("supplier_region") or "")
            payload["candidate_primary_supplier_share"] = 0.0
            payload["candidate_price_proxy"] = round(
                float(
                    offer.get("min_price")
                    or offer.get("avg_price")
                    or ste_stats.get("avg_amount_per_purchase")
                    or 0.0
                ),
                4,
            )
            payload["category_price_p25"] = round(float(price_bands.get("p25", 0.0) or 0.0), 4)
            payload["category_price_p75"] = round(float(price_bands.get("p75", 0.0) or 0.0), 4)
            payload["global_ste_popularity"] = float(ste_stats.get("purchase_count", 0.0) or 0.0)
            payload["global_category_popularity"] = float(
                global_category_stats.get(normalized_category, {}).get("purchase_count", 0.0) or 0.0
            )
            payload["regional_ste_popularity"] = float(
                regional_ste_stats.get(ste_id, {}).get("purchase_count", 0.0) or 0.0
            )
            payload["regional_category_popularity"] = float(
                regional_category_stats.get(normalized_category, {}).get("purchase_count", 0.0) or 0.0
            )
            payload["similar_customer_ste_popularity"] = float(
                similar_customer_stats.get(ste_id, {}).get("purchase_count", 0.0) or 0.0
            )
            payload["seasonal_category_popularity"] = float(
                seasonal_category_stats.get(normalized_category, {}).get("purchase_count", 0.0) or 0.0
            )
            payload["candidate_ste_recent_30d_popularity"] = float(
                recent_ste_stats.get(ste_id, {}).get("purchase_count", 0.0) or 0.0
            )
            payload["candidate_category_recent_90d_popularity"] = float(
                recent_category_stats.get(normalized_category, {}).get("purchase_count", 0.0) or 0.0
            )
            enriched.append(payload)
        return enriched

    def _fill_profile_from_history(self, *, profile: Dict[str, object], customer_inn: str) -> None:
        ste_rows = self.conn.execute(
            """
            SELECT
                cs.ste_id,
                cl.category,
                cl.normalized_category,
                cs.purchase_count,
                cs.total_amount,
                cs.first_purchase_dt,
                cs.last_purchase_dt,
                sol.supplier_inn,
                sol.supplier_region,
                COALESCE(sol.min_price, sol.avg_price, 0.0) AS price_proxy
            FROM customer_ste_stats cs
            JOIN category_lookup cl ON cl.category_id = cs.category_id
            LEFT JOIN ste_offer_lookup sol ON sol.ste_id = cs.ste_id
            WHERE cs.customer_inn = ?
            ORDER BY cs.last_purchase_dt DESC
            """,
            (customer_inn,),
        ).fetchall()
        category_rows = self.conn.execute(
            """
            SELECT
                cl.category,
                cl.normalized_category,
                cc.purchase_count,
                cc.total_amount,
                cc.first_purchase_dt,
                cc.last_purchase_dt
            FROM customer_category_stats cc
            JOIN category_lookup cl ON cl.category_id = cc.category_id
            WHERE cc.customer_inn = ?
            ORDER BY cc.last_purchase_dt DESC
            """,
            (customer_inn,),
        ).fetchall()

        total_purchases = 0
        total_amount = 0.0
        recent_amounts: List[float] = []
        recent_purchase_dates: List[str] = []
        category_counts: Dict[str, int] = {}
        ste_counts: Dict[str, int] = {}
        supplier_counts: Dict[str, int] = {}
        item_kind_counts: Dict[str, int] = {}
        last_category_purchase_dt: Dict[str, str] = {}
        last_ste_purchase_dt: Dict[str, str] = {}
        last_supplier_purchase_dt: Dict[str, str] = {}
        last_item_kind_purchase_dt: Dict[str, str] = {}
        recent_category_dates: Dict[str, List[str]] = {}
        last_purchase_dt: Optional[str] = None

        for row in ste_rows:
            ste_id = str(row["ste_id"])
            category = str(row["category"])
            supplier_inn = str(row["supplier_inn"] or "")
            item_kind = derive_item_kind("", category)
            purchase_count = int(row["purchase_count"] or 0)
            total_purchases += purchase_count
            total_amount += float(row["total_amount"] or 0.0)
            ste_counts[ste_id] = purchase_count
            if supplier_inn:
                supplier_counts[supplier_inn] = supplier_counts.get(supplier_inn, 0) + purchase_count
            item_kind_counts[item_kind] = item_kind_counts.get(item_kind, 0) + purchase_count

            last_dt = str(row["last_purchase_dt"] or "")
            if last_dt:
                last_ste_purchase_dt[ste_id] = last_dt
                last_purchase_dt = max(filter(None, [last_purchase_dt, last_dt])) if last_purchase_dt else last_dt
                if supplier_inn:
                    existing_supplier_dt = last_supplier_purchase_dt.get(supplier_inn)
                    if not existing_supplier_dt or last_dt > existing_supplier_dt:
                        last_supplier_purchase_dt[supplier_inn] = last_dt
                existing_item_kind_dt = last_item_kind_purchase_dt.get(item_kind)
                if not existing_item_kind_dt or last_dt > existing_item_kind_dt:
                    last_item_kind_purchase_dt[item_kind] = last_dt
                for _ in range(min(purchase_count, 5)):
                    if len(recent_purchase_dates) >= 180:
                        break
                    recent_purchase_dates.append(last_dt)

            average_amount = float(row["total_amount"] or 0.0) / purchase_count if purchase_count else 0.0
            for _ in range(min(purchase_count, 5)):
                if len(recent_amounts) >= 180:
                    break
                recent_amounts.append(round(average_amount, 4))

        for row in category_rows:
            category = str(row["category"])
            purchase_count = int(row["purchase_count"] or 0)
            category_counts[category] = purchase_count
            last_dt = str(row["last_purchase_dt"] or "")
            if last_dt:
                last_category_purchase_dt[category] = last_dt
                recent_category_dates[category] = [last_dt for _ in range(min(purchase_count, 5))]

        profile["total_purchases"] = total_purchases
        profile["total_amount"] = round(total_amount, 4)
        profile["recent_amounts"] = recent_amounts[:180]
        profile["category_counts"] = category_counts
        profile["ste_counts"] = ste_counts
        profile["supplier_counts"] = supplier_counts
        profile["item_kind_counts"] = item_kind_counts
        profile["last_purchase_dt"] = last_purchase_dt
        profile["last_category_purchase_dt"] = last_category_purchase_dt
        profile["last_ste_purchase_dt"] = last_ste_purchase_dt
        profile["last_supplier_purchase_dt"] = last_supplier_purchase_dt
        profile["last_item_kind_purchase_dt"] = last_item_kind_purchase_dt
        profile["recent_purchase_dates"] = recent_purchase_dates[:180]
        profile["recent_category_dates"] = recent_category_dates

    def _apply_session_categories(
        self,
        *,
        profile: Dict[str, object],
        session_categories: List[str],
        reference_date: date,
    ) -> None:
        categories = unique_preserve_order([str(item).strip() for item in session_categories if str(item).strip()])
        if not categories:
            return

        total_purchases = int(profile.get("total_purchases", 0) or 0)
        category_counts = dict(profile.get("category_counts", {}))
        item_kind_counts = dict(profile.get("item_kind_counts", {}))
        last_category_purchase_dt = dict(profile.get("last_category_purchase_dt", {}))
        last_item_kind_purchase_dt = dict(profile.get("last_item_kind_purchase_dt", {}))
        recent_category_dates = {
            str(key): list(values) for key, values in dict(profile.get("recent_category_dates", {})).items()
        }
        session_event_date = reference_date.isoformat()

        if total_purchases == 0:
            total_purchases = len(categories)

        for category in categories:
            category_counts[category] = int(category_counts.get(category, 0)) + 1
            last_category_purchase_dt[category] = session_event_date
            recent_category_dates.setdefault(category, [])
            recent_category_dates[category].insert(0, session_event_date)
            recent_category_dates[category] = recent_category_dates[category][:10]

            item_kind = derive_item_kind("", category)
            item_kind_counts[item_kind] = int(item_kind_counts.get(item_kind, 0)) + 1
            last_item_kind_purchase_dt[item_kind] = session_event_date

        profile["total_purchases"] = total_purchases
        profile["category_counts"] = category_counts
        profile["item_kind_counts"] = item_kind_counts
        profile["last_category_purchase_dt"] = last_category_purchase_dt
        profile["last_item_kind_purchase_dt"] = last_item_kind_purchase_dt
        profile["recent_category_dates"] = recent_category_dates

    def _infer_customer_region(self, customer_inn: Optional[str]) -> Optional[str]:
        if not customer_inn:
            return None
        try:
            row = self.conn.execute(
                """
                SELECT customer_region
                FROM customer_region_lookup
                WHERE customer_inn = ?
                LIMIT 1
                """,
                (customer_inn,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return str(row["customer_region"]) if row and row["customer_region"] else None

    def _load_offer_lookup(self, ste_ids: List[str]) -> Dict[str, Dict[str, object]]:
        if not ste_ids:
            return {}
        result: Dict[str, Dict[str, object]] = {}
        for chunk in _chunked(ste_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    ste_id,
                    supplier_inn,
                    supplier_region,
                    offer_count,
                    avg_price,
                    min_price,
                    last_contract_dt
                FROM ste_offer_lookup
                WHERE ste_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                result[str(row["ste_id"])] = {
                    "supplier_inn": row["supplier_inn"] or "",
                    "supplier_region": row["supplier_region"] or "",
                    "offer_count": int(row["offer_count"] or 0),
                    "avg_price": float(row["avg_price"] or 0.0),
                    "min_price": float(row["min_price"] or 0.0),
                    "last_contract_dt": row["last_contract_dt"],
                }
        return result

    def _load_global_ste_stats(self, ste_ids: List[str]) -> Dict[str, Dict[str, float]]:
        if not ste_ids:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for chunk in _chunked(ste_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    ste_id,
                    SUM(purchase_count) AS purchase_count,
                    SUM(total_amount) AS total_amount
                FROM customer_ste_stats
                WHERE ste_id IN ({placeholders})
                GROUP BY ste_id
                """,
                chunk,
            ).fetchall()
            for row in rows:
                purchase_count = float(row["purchase_count"] or 0.0)
                total_amount = float(row["total_amount"] or 0.0)
                result[str(row["ste_id"])] = {
                    "purchase_count": purchase_count,
                    "total_amount": total_amount,
                    "avg_amount_per_purchase": total_amount / purchase_count if purchase_count else 0.0,
                }
        return result

    def _load_regional_ste_stats(self, ste_ids: List[str], customer_region: str) -> Dict[str, Dict[str, float]]:
        if not ste_ids or not customer_region:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for chunk in _chunked(ste_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    cs.ste_id,
                    SUM(cs.purchase_count) AS purchase_count
                FROM customer_ste_stats cs
                JOIN customer_region_lookup cr ON cr.customer_inn = cs.customer_inn
                WHERE cs.ste_id IN ({placeholders})
                  AND cr.customer_region = ?
                GROUP BY cs.ste_id
                """,
                [*chunk, customer_region],
            ).fetchall()
            for row in rows:
                result[str(row["ste_id"])] = {"purchase_count": float(row["purchase_count"] or 0.0)}
        return result

    def _load_global_category_stats(self, normalized_categories: List[str]) -> Dict[str, Dict[str, float]]:
        if not normalized_categories:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for chunk in _chunked(normalized_categories):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    cl.normalized_category,
                    SUM(cc.purchase_count) AS purchase_count
                FROM customer_category_stats cc
                JOIN category_lookup cl ON cl.category_id = cc.category_id
                WHERE cl.normalized_category IN ({placeholders})
                GROUP BY cl.normalized_category
                """,
                chunk,
            ).fetchall()
            for row in rows:
                result[str(row["normalized_category"])] = {"purchase_count": float(row["purchase_count"] or 0.0)}
        return result

    def _load_regional_category_stats(
        self,
        normalized_categories: List[str],
        customer_region: str,
    ) -> Dict[str, Dict[str, float]]:
        if not normalized_categories or not customer_region:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for chunk in _chunked(normalized_categories):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    cl.normalized_category,
                    SUM(rc.purchase_count) AS purchase_count
                FROM region_category_stats rc
                JOIN category_lookup cl ON cl.category_id = rc.category_id
                WHERE cl.normalized_category IN ({placeholders})
                  AND rc.customer_region = ?
                GROUP BY cl.normalized_category
                """,
                [*chunk, customer_region],
            ).fetchall()
            for row in rows:
                result[str(row["normalized_category"])] = {"purchase_count": float(row["purchase_count"] or 0.0)}
        return result

    def _load_recent_ste_stats(self, ste_ids: List[str], cutoff_date: date) -> Dict[str, Dict[str, float]]:
        if not ste_ids:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for chunk in _chunked(ste_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    ste_id,
                    SUM(purchase_count) AS purchase_count
                FROM customer_ste_stats
                WHERE ste_id IN ({placeholders})
                  AND last_purchase_dt >= ?
                GROUP BY ste_id
                """,
                [*chunk, cutoff_date.isoformat()],
            ).fetchall()
            for row in rows:
                result[str(row["ste_id"])] = {"purchase_count": float(row["purchase_count"] or 0.0)}
        return result

    def _load_recent_category_stats(
        self,
        normalized_categories: List[str],
        cutoff_date: date,
    ) -> Dict[str, Dict[str, float]]:
        if not normalized_categories:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for chunk in _chunked(normalized_categories):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    cl.normalized_category,
                    SUM(cc.purchase_count) AS purchase_count
                FROM customer_category_stats cc
                JOIN category_lookup cl ON cl.category_id = cc.category_id
                WHERE cl.normalized_category IN ({placeholders})
                  AND cc.last_purchase_dt >= ?
                GROUP BY cl.normalized_category
                """,
                [*chunk, cutoff_date.isoformat()],
            ).fetchall()
            for row in rows:
                result[str(row["normalized_category"])] = {"purchase_count": float(row["purchase_count"] or 0.0)}
        return result

    def _load_seasonal_category_stats(
        self,
        normalized_categories: List[str],
        month: int,
    ) -> Dict[str, Dict[str, float]]:
        if not normalized_categories:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        month_token = f"{month:02d}"
        for chunk in _chunked(normalized_categories):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    cl.normalized_category,
                    SUM(cc.purchase_count) AS purchase_count
                FROM customer_category_stats cc
                JOIN category_lookup cl ON cl.category_id = cc.category_id
                WHERE cl.normalized_category IN ({placeholders})
                  AND substr(COALESCE(cc.last_purchase_dt, ''), 6, 2) = ?
                GROUP BY cl.normalized_category
                """,
                [*chunk, month_token],
            ).fetchall()
            for row in rows:
                result[str(row["normalized_category"])] = {"purchase_count": float(row["purchase_count"] or 0.0)}
        return result

    def _load_category_price_bands(self, normalized_categories: List[str]) -> Dict[str, Dict[str, float]]:
        if not normalized_categories:
            return {}
        grouped: Dict[str, List[float]] = {}
        for chunk in _chunked(normalized_categories):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    cl.normalized_category,
                    COALESCE(sol.min_price, sol.avg_price, AVG(cs.total_amount * 1.0 / NULLIF(cs.purchase_count, 0))) AS price_proxy
                FROM customer_ste_stats cs
                JOIN category_lookup cl ON cl.category_id = cs.category_id
                LEFT JOIN ste_offer_lookup sol ON sol.ste_id = cs.ste_id
                WHERE cl.normalized_category IN ({placeholders})
                GROUP BY cl.normalized_category, cs.ste_id
                HAVING price_proxy > 0
                """,
                chunk,
            ).fetchall()
            for row in rows:
                grouped.setdefault(str(row["normalized_category"]), []).append(float(row["price_proxy"] or 0.0))

        return {
            category: {
                "p25": round(_quantile(values, 0.25), 4),
                "p75": round(_quantile(values, 0.75), 4),
            }
            for category, values in grouped.items()
        }

    def _load_similar_customer_ste_stats(
        self,
        *,
        ste_ids: List[str],
        customer_region: str,
        normalized_category: str,
    ) -> Dict[str, Dict[str, float]]:
        if not ste_ids or not customer_region or not normalized_category:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for chunk in _chunked(ste_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    cs.ste_id,
                    SUM(cs.purchase_count) AS purchase_count
                FROM customer_ste_stats cs
                JOIN customer_region_lookup cr ON cr.customer_inn = cs.customer_inn
                JOIN category_lookup cl ON cl.category_id = cs.category_id
                WHERE cs.ste_id IN ({placeholders})
                  AND cr.customer_region = ?
                  AND cl.normalized_category = ?
                GROUP BY cs.ste_id
                """,
                [*chunk, customer_region, normalized_category],
            ).fetchall()
            for row in rows:
                result[str(row["ste_id"])] = {"purchase_count": float(row["purchase_count"] or 0.0)}
        return result

    @staticmethod
    def _dominant_category(user_profile: Dict[str, object]) -> str:
        category_counts = dict(user_profile.get("category_counts", {}))
        if not category_counts:
            return ""
        dominant_category = max(category_counts.items(), key=lambda item: int(item[1]))[0]
        return normalize_text(str(dominant_category))
