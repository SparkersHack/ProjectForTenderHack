from __future__ import annotations

import math
import time
from typing import Dict, List, Protocol, Tuple


QUICK_EXIT_THRESHOLD_MS = 2000


class SkipStorage(Protocol):
    def increment_skip(self, user_id: str, category_id: str) -> int:
        ...

    def get_skips(self, user_id: str, category_id: str) -> int:
        ...


class InMemorySkipStorage(SkipStorage):
    def __init__(self, ttl_seconds: int = 4 * 3600) -> None:
        self._store: Dict[str, Tuple[int, float]] = {}
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _make_key(user_id: str, category_id: str) -> str:
        return f"user:{user_id}:cat:{category_id}:skips"

    def increment_skip(self, user_id: str, category_id: str) -> int:
        current_time = time.time()
        key = self._make_key(user_id, category_id)
        count, expires_at = self._store.get(key, (0, 0.0))
        if 0 < expires_at < current_time:
            count = 0

        next_count = count + 1
        self._store[key] = (next_count, current_time + self._ttl_seconds)
        return next_count

    def get_skips(self, user_id: str, category_id: str) -> int:
        current_time = time.time()
        key = self._make_key(user_id, category_id)
        count, expires_at = self._store.get(key, (0, 0.0))
        if 0 < expires_at < current_time:
            return 0
        return count


class InteractionTracker:
    SKIP_THRESHOLD_MS = QUICK_EXIT_THRESHOLD_MS
    FORGIVEN_QUICK_EXITS = 1

    def __init__(self, storage: SkipStorage, grace_ttl_seconds: int = 4 * 3600) -> None:
        self.storage = storage
        self._grace_ttl_seconds = int(grace_ttl_seconds)
        self._quick_exit_store: Dict[str, Tuple[int, float]] = {}

    @staticmethod
    def _make_grace_key(user_id: str, category_id: str) -> str:
        return f"user:{user_id}:cat:{category_id}:quick-exits"

    def _get_quick_exit_count(self, user_id: str, category_id: str) -> int:
        current_time = time.time()
        key = self._make_grace_key(user_id, category_id)
        count, expires_at = self._quick_exit_store.get(key, (0, 0.0))
        if 0 < expires_at < current_time:
            self._quick_exit_store.pop(key, None)
            return 0
        return count

    def _store_quick_exit_count(self, user_id: str, category_id: str, count: int) -> None:
        self._quick_exit_store[self._make_grace_key(user_id, category_id)] = (
            max(0, int(count)),
            time.time() + self._grace_ttl_seconds,
        )

    def register_view(self, user_id: str, category_id: str, dwell_time_ms: int) -> str:
        if dwell_time_ms >= self.SKIP_THRESHOLD_MS:
            return "none"

        quick_exit_count = self._get_quick_exit_count(user_id, category_id) + 1
        self._store_quick_exit_count(user_id, category_id, quick_exit_count)
        if quick_exit_count <= self.FORGIVEN_QUICK_EXITS:
            return "forgiven"

        self.storage.increment_skip(user_id, category_id)
        return "applied"


class RankingModifier:
    def __init__(self, storage: SkipStorage) -> None:
        self.storage = storage

    @staticmethod
    def calculate_multiplier(skip_count: int) -> float:
        if skip_count == 0:
            return 1.0

        penalty = 0.2 * math.log2(1 + skip_count)
        return max(0.4, 1.0 - penalty)

    def apply_penalties(self, recommendations: List[Dict], user_id: str) -> List[Dict]:
        updated_recommendations: List[Dict] = []
        for recommendation in recommendations:
            category_id = recommendation.get("category_id") or recommendation.get("category")
            if not category_id:
                updated_recommendations.append(recommendation)
                continue

            base_score = float(recommendation.get("base_score", recommendation.get("search_score", 0.0)))
            current_score = float(recommendation.get("final_score", base_score))
            skip_count = self.storage.get_skips(user_id, str(category_id))
            multiplier = self.calculate_multiplier(skip_count)

            updated = dict(recommendation)
            updated["final_score"] = round(current_score * multiplier, 4)
            updated["penalty_multiplier"] = multiplier
            if multiplier < 1.0 and "reason_to_hide" not in updated:
                updated["reason_to_hide"] = (
                    f"Пессимизация категории. Множитель: {multiplier:.2f} (Скипов: {skip_count})"
                )
            updated_recommendations.append(updated)

        updated_recommendations.sort(key=lambda item: float(item.get("final_score", 0.0)), reverse=True)
        return updated_recommendations
