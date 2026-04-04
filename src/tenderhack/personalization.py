from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .dataset_layout import resolve_preprocessed_db_path
from .text import normalize_text, stem_tokens, tokenize


DEFAULT_PREPROCESSED_DB = resolve_preprocessed_db_path()


@dataclass
class SessionState:
    clicked_ste_ids: List[str]
    cart_ste_ids: List[str]
    recent_categories: List[str]

    @classmethod
    def from_mapping(cls, payload: Optional[Dict[str, object]]) -> "SessionState":
        payload = payload or {}
        clicked_ste_ids = [str(value) for value in payload.get("clicked_ste_ids", [])]
        cart_ste_ids = [str(value) for value in payload.get("cart_ste_ids", [])]
        recent_categories = [normalize_text(str(value)) for value in payload.get("recent_categories", [])]
        return cls(clicked_ste_ids=clicked_ste_ids, cart_ste_ids=cart_ste_ids, recent_categories=recent_categories)


class PersonalizationService:
    def __init__(self, db_path: Path | str = DEFAULT_PREPROCESSED_DB) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def build_customer_profile(
        self,
        customer_inn: str,
        customer_region: Optional[str] = None,
        top_categories: int = 12,
        top_ste: int = 20,
        top_region_categories: int = 12,
    ) -> Dict[str, object]:
        customer_inn = str(customer_inn)
        customer_region = customer_region or self._infer_customer_region(customer_inn)

        top_customer_categories = self.conn.execute(
            """
            SELECT
                cl.category,
                cc.purchase_count,
                cc.total_amount,
                cc.first_purchase_dt,
                cc.last_purchase_dt
            FROM customer_category_stats cc
            JOIN category_lookup cl ON cl.category_id = cc.category_id
            WHERE cc.customer_inn = ?
            ORDER BY cc.purchase_count DESC, cc.total_amount DESC
            LIMIT ?
            """,
            (customer_inn, top_categories),
        ).fetchall()

        top_customer_ste = self.conn.execute(
            """
            SELECT
                cs.ste_id,
                cl.category,
                cs.purchase_count,
                cs.total_amount,
                cs.first_purchase_dt,
                cs.last_purchase_dt
            FROM customer_ste_stats cs
            JOIN category_lookup cl ON cl.category_id = cs.category_id
            WHERE cs.customer_inn = ?
            ORDER BY cs.purchase_count DESC, cs.total_amount DESC
            LIMIT ?
            """,
            (customer_inn, top_ste),
        ).fetchall()

        top_regional_categories: List[sqlite3.Row] = []
        if customer_region:
            top_regional_categories = self.conn.execute(
                """
                SELECT
                    cl.category,
                    rc.purchase_count,
                    rc.total_amount,
                    rc.first_purchase_dt,
                    rc.last_purchase_dt
                FROM region_category_stats rc
                JOIN category_lookup cl ON cl.category_id = rc.category_id
                WHERE rc.customer_region = ?
                ORDER BY rc.purchase_count DESC, rc.total_amount DESC
                LIMIT ?
                """,
                (customer_region, top_region_categories),
            ).fetchall()

        category_preferences = self._weight_category_rows(top_customer_categories)
        ste_preferences = self._weight_ste_rows(top_customer_ste)
        region_preferences = self._weight_category_rows(top_regional_categories)

        return {
            "customer_inn": customer_inn,
            "customer_region": customer_region,
            "top_categories": category_preferences,
            "top_ste": ste_preferences,
            "regional_categories": region_preferences,
            "category_affinity": {item["normalized_category"]: item["weight"] for item in category_preferences},
            "ste_affinity": {item["ste_id"]: item["weight"] for item in ste_preferences},
            "regional_affinity": {item["normalized_category"]: item["weight"] for item in region_preferences},
        }

    def _infer_customer_region(self, customer_inn: str) -> Optional[str]:
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
        return row["customer_region"] if row else None

    def _weight_category_rows(self, rows: Iterable[sqlite3.Row]) -> List[Dict[str, object]]:
        rows = list(rows)
        if not rows:
            return []
        max_count = max(row["purchase_count"] for row in rows) or 1
        max_amount_log = max(math.log1p(float(row["total_amount"])) for row in rows) or 1.0
        result = []
        for rank, row in enumerate(rows, start=1):
            count_component = float(row["purchase_count"]) / max_count
            amount_component = math.log1p(float(row["total_amount"])) / max_amount_log if max_amount_log else 0.0
            rank_component = 1.0 / rank
            weight = 0.55 * count_component + 0.25 * amount_component + 0.20 * rank_component
            result.append(
                {
                    "category": row["category"],
                    "normalized_category": normalize_text(row["category"]),
                    "purchase_count": int(row["purchase_count"]),
                    "total_amount": round(float(row["total_amount"]), 2),
                    "first_purchase_dt": row["first_purchase_dt"],
                    "last_purchase_dt": row["last_purchase_dt"],
                    "weight": round(weight, 4),
                }
            )
        return result

    def _weight_ste_rows(self, rows: Iterable[sqlite3.Row]) -> List[Dict[str, object]]:
        rows = list(rows)
        if not rows:
            return []
        max_count = max(row["purchase_count"] for row in rows) or 1
        max_amount_log = max(math.log1p(float(row["total_amount"])) for row in rows) or 1.0
        result = []
        for rank, row in enumerate(rows, start=1):
            count_component = float(row["purchase_count"]) / max_count
            amount_component = math.log1p(float(row["total_amount"])) / max_amount_log if max_amount_log else 0.0
            rank_component = 1.0 / rank
            weight = 0.60 * count_component + 0.25 * amount_component + 0.15 * rank_component
            result.append(
                {
                    "ste_id": row["ste_id"],
                    "category": row["category"],
                    "normalized_category": normalize_text(row["category"]),
                    "purchase_count": int(row["purchase_count"]),
                    "total_amount": round(float(row["total_amount"]), 2),
                    "first_purchase_dt": row["first_purchase_dt"],
                    "last_purchase_dt": row["last_purchase_dt"],
                    "weight": round(weight, 4),
                }
            )
        return result

    def rerank_ste(
        self,
        results: List[Dict[str, object]],
        customer_profile: Dict[str, object],
        session_state: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        session = SessionState.from_mapping(session_state)
        category_affinity = customer_profile.get("category_affinity", {})
        ste_affinity = customer_profile.get("ste_affinity", {})
        regional_affinity = customer_profile.get("regional_affinity", {})

        reranked: List[Dict[str, object]] = []
        for index, result in enumerate(results, start=1):
            category_norm = normalize_text(str(result.get("category", "")))
            ste_id = str(result.get("ste_id"))
            base_score = float(result.get("search_score", 0.0))

            history_affinity = float(ste_affinity.get(ste_id, 0.0))
            category_score = self._best_category_affinity(category_norm, customer_profile.get("top_categories", []))
            region_score = self._best_category_affinity(category_norm, customer_profile.get("regional_categories", []))
            session_boost = self._session_boost(result, session)

            final_score = (
                base_score
                + 5.0 * history_affinity
                + 3.0 * category_score
                + 2.0 * region_score
                + 4.0 * session_boost
            )

            explanation = self._build_explanation(
                result=result,
                history_affinity=history_affinity,
                category_affinity=category_score,
                region_affinity=region_score,
                session_boost=session_boost,
            )

            enriched = dict(result)
            enriched["base_search_rank"] = index
            enriched["personalization_features"] = {
                "history_affinity": round(history_affinity, 4),
                "category_affinity": round(category_score, 4),
                "region_affinity": round(region_score, 4),
                "session_action_boost": round(session_boost, 4),
            }
            enriched["final_score"] = round(final_score, 4)
            enriched["explanation"] = explanation
            reranked.append(enriched)

        reranked.sort(
            key=lambda item: (
                item["final_score"],
                item["personalization_features"]["history_affinity"],
                item["personalization_features"]["category_affinity"],
                item["search_score"],
            ),
            reverse=True,
        )
        return reranked

    def rerank_offers(
        self,
        offers: List[Dict[str, object]],
        customer_profile: Dict[str, object],
        session_state: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        session = SessionState.from_mapping(session_state)
        max_price = max((float(offer.get("unit_price", 0.0) or 0.0) for offer in offers), default=0.0)
        reranked: List[Dict[str, object]] = []
        for index, offer in enumerate(offers, start=1):
            ste_id = str(offer.get("ste_id", ""))
            category_norm = normalize_text(str(offer.get("category", "")))
            supplier_region = normalize_text(str(offer.get("supplier_region", "")))
            base_score = float(offer.get("offer_score", offer.get("search_score", 0.0)))

            history_affinity = float(customer_profile.get("ste_affinity", {}).get(ste_id, 0.0))
            category_affinity = self._best_category_affinity(category_norm, customer_profile.get("top_categories", []))
            region_affinity = self._best_category_affinity(category_norm, customer_profile.get("regional_categories", []))
            session_boost = self._session_boost({"ste_id": ste_id, "category": category_norm}, session)

            region_match_boost = 0.0
            customer_region = normalize_text(str(customer_profile.get("customer_region", "")))
            if customer_region and supplier_region and customer_region == supplier_region:
                region_match_boost = 0.35

            unit_price = float(offer.get("unit_price", 0.0) or 0.0)
            price_bonus = 0.0
            if max_price > 0 and unit_price > 0:
                price_bonus = max(0.0, 1.0 - (unit_price / max_price))

            final_score = (
                base_score
                + 4.0 * history_affinity
                + 3.0 * category_affinity
                + 2.0 * region_affinity
                + 3.0 * session_boost
                + 1.5 * region_match_boost
                + 1.5 * price_bonus
            )

            explanation = []
            if history_affinity >= 0.20:
                explanation.append("СТЕ уже часто закупалось этой организацией")
            if category_affinity >= 0.20:
                explanation.append("оффер относится к предпочитаемой категории")
            if region_match_boost > 0:
                explanation.append("регион поставки совпадает с регионом заказчика")
            if price_bonus >= 0.20:
                explanation.append("цена выгоднее части альтернатив")
            if session_boost >= 0.35:
                explanation.append("поднято после действий пользователя в текущей сессии")
            if not explanation:
                explanation.append("оставлено выше за счёт базовой релевантности оферты")

            enriched = dict(offer)
            enriched["base_offer_rank"] = index
            enriched["offer_personalization_features"] = {
                "history_affinity": round(history_affinity, 4),
                "category_affinity": round(category_affinity, 4),
                "region_affinity": round(region_affinity, 4),
                "session_action_boost": round(session_boost, 4),
                "region_match_boost": round(region_match_boost, 4),
                "price_bonus": round(price_bonus, 4),
            }
            enriched["final_offer_score"] = round(final_score, 4)
            enriched["offer_explanation"] = explanation
            reranked.append(enriched)

        reranked.sort(
            key=lambda item: (
                item["final_offer_score"],
                item["offer_personalization_features"]["history_affinity"],
                item["offer_personalization_features"]["category_affinity"],
                item["offer_personalization_features"]["price_bonus"],
            ),
            reverse=True,
        )
        return reranked

    def _session_boost(self, result: Dict[str, object], session: SessionState) -> float:
        category_norm = normalize_text(str(result.get("category", "")))
        ste_id = str(result.get("ste_id"))
        boost = 0.0
        if ste_id in session.clicked_ste_ids:
            boost += 0.45
        if ste_id in session.cart_ste_ids:
            boost += 0.75
        if category_norm and category_norm in session.recent_categories:
            boost += 0.35
        return min(boost, 1.5)

    def _build_explanation(
        self,
        result: Dict[str, object],
        history_affinity: float,
        category_affinity: float,
        region_affinity: float,
        session_boost: float,
    ) -> List[str]:
        explanation: List[str] = []
        if history_affinity >= 0.25:
            explanation.append("часто закупалось этой организацией")
        if category_affinity >= 0.20:
            explanation.append("похоже на ранее выбранные категории")
        if region_affinity >= 0.20:
            explanation.append("популярно у заказчиков того же региона")
        if session_boost >= 0.35:
            explanation.append("поднято после клика или добавления в корзину")
        if not explanation:
            explanation.append("оставлено выше за счёт базовой текстовой релевантности")
        return explanation

    def _best_category_affinity(self, result_category: str, profile_categories: List[Dict[str, object]]) -> float:
        result_stems = set(stem_tokens(tokenize(result_category)))
        if not result_stems:
            return 0.0
        best = 0.0
        for item in profile_categories:
            profile_stems = set(stem_tokens(tokenize(str(item.get("normalized_category", "")))))
            if not profile_stems:
                continue
            overlap = len(result_stems & profile_stems) / max(1, min(len(result_stems), len(profile_stems)))
            best = max(best, float(item.get("weight", 0.0)) * overlap)
        return round(best, 4)


def build_customer_profile(
    customer_inn: str,
    customer_region: Optional[str] = None,
    db_path: Path | str = DEFAULT_PREPROCESSED_DB,
) -> Dict[str, object]:
    service = PersonalizationService(db_path=db_path)
    try:
        return service.build_customer_profile(customer_inn=customer_inn, customer_region=customer_region)
    finally:
        service.close()


def rerank_ste(
    results: List[Dict[str, object]],
    customer_profile: Dict[str, object],
    session_state: Optional[Dict[str, object]] = None,
    db_path: Path | str = DEFAULT_PREPROCESSED_DB,
) -> List[Dict[str, object]]:
    service = PersonalizationService(db_path=db_path)
    try:
        return service.rerank_ste(results=results, customer_profile=customer_profile, session_state=session_state)
    finally:
        service.close()


def rerank_offers(
    offers: List[Dict[str, object]],
    customer_profile: Dict[str, object],
    session_state: Optional[Dict[str, object]] = None,
    db_path: Path | str = DEFAULT_PREPROCESSED_DB,
) -> List[Dict[str, object]]:
    service = PersonalizationService(db_path=db_path)
    try:
        return service.rerank_offers(offers=offers, customer_profile=customer_profile, session_state=session_state)
    finally:
        service.close()
