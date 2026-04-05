from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Mapping, Protocol, Sequence

from .text import normalize_text


def _normalize_category_key(value: object) -> str:
    if value is None:
        return ""
    raw_value = str(value).strip()
    if not raw_value:
        return ""
    normalized = normalize_text(raw_value)
    return normalized or raw_value


def _normalize_entity_key(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


class UserHistoryRepository(Protocol):
    def get_category_purchase_counts(self, user_id: str) -> Dict[str, int]:
        ...

    def get_ste_purchase_counts(self, user_id: str) -> Dict[str, int]:
        ...


@dataclass(frozen=True)
class InMemoryUserHistoryRepository(UserHistoryRepository):
    user_history: Mapping[str, Mapping[str, int]]
    ste_history: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def get_category_purchase_counts(self, user_id: str) -> Dict[str, int]:
        category_counts = self.user_history.get(str(user_id), {})
        result: Dict[str, int] = {}
        for category_id, purchase_count in dict(category_counts).items():
            raw_category = str(category_id).strip()
            if not raw_category:
                continue
            result[raw_category] = int(purchase_count or 0)
        return result

    def get_ste_purchase_counts(self, user_id: str) -> Dict[str, int]:
        ste_counts = self.ste_history.get(str(user_id), {})
        result: Dict[str, int] = {}
        for ste_id, purchase_count in dict(ste_counts).items():
            raw_ste_id = str(ste_id).strip()
            if not raw_ste_id:
                continue
            result[raw_ste_id] = int(purchase_count or 0)
        return result


class SQLiteUserHistoryRepository(UserHistoryRepository):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get_category_purchase_counts(self, user_id: str) -> Dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT
                cl.normalized_category,
                SUM(cc.purchase_count) AS purchase_count
            FROM customer_category_stats cc
            JOIN category_lookup cl ON cl.category_id = cc.category_id
            WHERE cc.customer_inn = ?
            GROUP BY cl.normalized_category
            ORDER BY purchase_count DESC, cl.normalized_category ASC
            """,
            (str(user_id),),
        ).fetchall()
        result: Dict[str, int] = {}
        for row in rows:
            category_key = str(row["normalized_category"] or "").strip()
            if not category_key:
                continue
            result[category_key] = int(row["purchase_count"] or 0)
        return result

    def get_ste_purchase_counts(self, user_id: str) -> Dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT
                cs.ste_id,
                SUM(cs.purchase_count) AS purchase_count
            FROM customer_ste_stats cs
            WHERE cs.customer_inn = ?
            GROUP BY cs.ste_id
            ORDER BY purchase_count DESC, cs.ste_id ASC
            """,
            (str(user_id),),
        ).fetchall()
        result: Dict[str, int] = {}
        for row in rows:
            ste_id = str(row["ste_id"] or "").strip()
            if not ste_id:
                continue
            result[ste_id] = int(row["purchase_count"] or 0)
        return result


class UserProfileScorer:
    def __init__(self, repository: UserHistoryRepository) -> None:
        self.repository = repository

    @staticmethod
    def _compute_repeat_weights(raw_counts: Mapping[str, int]) -> Dict[str, float]:
        positive_counts = {
            category_id: int(purchase_count)
            for category_id, purchase_count in dict(raw_counts).items()
            if int(purchase_count or 0) > 0
        }
        total_unique_categories = len(positive_counts)
        if total_unique_categories == 0:
            return {}

        denominator = math.sqrt(float(total_unique_categories))
        if denominator <= 0:
            return {category_id: 0.0 for category_id in positive_counts}

        total_repeat_purchases = sum(max(purchase_count - 1, 0) for purchase_count in positive_counts.values())
        if total_repeat_purchases <= 0:
            repeat_support = {
                category_id: 1.0 if total_unique_categories == 1 else 0.0
                for category_id in positive_counts
            }
        else:
            repeat_support = {
                category_id: max(purchase_count - 1, 0) / float(total_repeat_purchases)
                for category_id, purchase_count in positive_counts.items()
            }

        return {
            category_id: (float(purchase_count) / denominator) * float(repeat_support.get(category_id, 0.0))
            for category_id, purchase_count in positive_counts.items()
        }

    def compute_category_weights(self, user_id: str) -> Dict[str, float]:
        category_counts = self.repository.get_category_purchase_counts(user_id)
        return self._compute_repeat_weights(category_counts)

    def compute_ste_weights(self, user_id: str) -> Dict[str, float]:
        ste_counts = self.repository.get_ste_purchase_counts(user_id)
        return self._compute_repeat_weights(ste_counts)


def apply_personalization(
    search_results: Sequence[dict],
    user_weights: Mapping[str, float],
    user_item_weights: Mapping[str, float] | None = None,
) -> list[dict]:
    personalized_results: list[dict] = []
    normalized_weights = {
        _normalize_category_key(category_id): float(weight or 0.0)
        for category_id, weight in dict(user_weights).items()
        if _normalize_category_key(category_id)
    }
    normalized_item_weights = {
        _normalize_entity_key(item_id): float(weight or 0.0)
        for item_id, weight in dict(user_item_weights or {}).items()
        if _normalize_entity_key(item_id)
    }
    max_weight = max(normalized_weights.values(), default=0.0) + max(normalized_item_weights.values(), default=0.0)

    for item in search_results:
        base_score = float(
            item.get("base_score", item.get("score", item.get("search_score", item.get("final_score", 0.0)))) or 0.0
        )
        category_key = ""
        for field_name in ("category_id", "categoryId", "normalized_category", "category"):
            category_key = _normalize_category_key(item.get(field_name))
            if category_key:
                break
        item_key = ""
        for field_name in ("ste_id", "steId", "candidate_id", "id"):
            item_key = _normalize_entity_key(item.get(field_name))
            if item_key:
                break
        category_boost_weight = float(normalized_weights.get(category_key, 0.0) or 0.0)
        item_boost_weight = float(normalized_item_weights.get(item_key, 0.0) or 0.0)
        boost_weight = category_boost_weight + item_boost_weight
        final_score = base_score * (1.0 + boost_weight)
        relative_weight = min(1.0, boost_weight / max_weight) if max_weight > 0 else 0.0
        absolute_weight = boost_weight / (1.0 + boost_weight) if boost_weight > 0 else 0.0
        personalization_confidence = relative_weight * absolute_weight

        enriched = dict(item)
        enriched["base_score"] = round(base_score, 4)
        enriched["category_boost_weight"] = round(category_boost_weight, 6)
        enriched["item_boost_weight"] = round(item_boost_weight, 6)
        enriched["boost_weight"] = round(boost_weight, 6)
        enriched["personalization_confidence"] = round(personalization_confidence, 6)
        enriched["final_score"] = round(final_score, 4)
        personalized_results.append(enriched)

    personalized_results.sort(
        key=lambda item: (
            float(item.get("final_score", 0.0)),
            float(item.get("base_score", 0.0)),
            str(item.get("text") or item.get("name") or ""),
        ),
        reverse=True,
    )
    return personalized_results
