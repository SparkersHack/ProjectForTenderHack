from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .cache import CacheService
from .penalization import QUICK_EXIT_THRESHOLD_MS
from .text import normalize_text, unique_preserve_order


SESSION_KEEP_LIMIT = 20


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dedupe_trim(values: List[str], limit: int = SESSION_KEEP_LIMIT) -> List[str]:
    return unique_preserve_order([value for value in values if value])[:limit]


def _normalize_int_mapping(values: Dict[str, object]) -> Dict[str, int]:
    cleaned: Dict[str, int] = {}
    for key, raw_value in values.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        try:
            normalized_value = int(raw_value)
        except (TypeError, ValueError):
            continue
        cleaned[normalized_key] = normalized_value
    return cleaned


@dataclass
class OnlineStateService:
    cache_service: CacheService
    session_ttl_seconds: int = 86400

    def get_session_state(
        self,
        *,
        user_id: Optional[str],
        customer_inn: Optional[str] = None,
        customer_region: Optional[str] = None,
    ) -> Dict[str, object]:
        state = self._load_session_state(
            user_id=user_id,
            customer_inn=customer_inn,
            customer_region=customer_region,
        )
        return state

    def record_event(
        self,
        *,
        user_id: Optional[str],
        customer_inn: Optional[str],
        customer_region: Optional[str],
        event_type: str,
        ste_id: Optional[str] = None,
        category: Optional[str] = None,
        duration_ms: Optional[int] = None,
        close_reason: Optional[str] = None,
    ) -> Dict[str, object]:
        state = self._load_session_state(
            user_id=user_id,
            customer_inn=customer_inn,
            customer_region=customer_region,
        )
        event_type = str(event_type or "").strip().lower()
        normalized_category = normalize_text(category or "")
        ste_id = str(ste_id or "").strip()
        duration_ms = int(duration_ms or 0)
        close_reason = str(close_reason or "").strip().lower()

        event_counts = {str(key): int(value) for key, value in dict(state.get("event_counts", {})).items()}
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        state["event_counts"] = event_counts
        state["last_event_type"] = event_type
        state["last_event_at"] = _utc_now_iso()
        state["version"] = int(state.get("version", 0) or 0) + 1

        recent_categories = [normalize_text(value) for value in state.get("recent_categories", [])]
        clicked_ste_ids = [str(value) for value in state.get("clicked_ste_ids", [])]
        cart_ste_ids = [str(value) for value in state.get("cart_ste_ids", [])]
        bounced_categories = [normalize_text(value) for value in state.get("bounced_categories", [])]
        bounce_counts = {normalize_text(str(key)): int(value) for key, value in dict(state.get("bounce_counts", {})).items()}
        open_ste_sessions = _normalize_int_mapping(dict(state.get("open_ste_sessions", {})))
        pending_post_cart_close_sessions = _normalize_int_mapping(
            dict(state.get("pending_post_cart_close_sessions", {}))
        )
        open_sequence = int(state.get("open_sequence", 0) or 0)
        state["last_item_close_penalizable"] = False
        state["last_item_close_suppressed"] = False

        if event_type in {"search_result_click", "item_click"}:
            if ste_id:
                clicked_ste_ids = _dedupe_trim([ste_id, *clicked_ste_ids])
            if normalized_category:
                recent_categories = _dedupe_trim([normalized_category, *recent_categories])

        if event_type == "item_open":
            if ste_id:
                pending_post_cart_close_sessions.pop(ste_id, None)
                open_sequence += 1
                open_ste_sessions[ste_id] = open_sequence

        if event_type == "cart_add":
            if ste_id:
                cart_ste_ids = _dedupe_trim([ste_id, *cart_ste_ids])
                active_open_session = open_ste_sessions.get(ste_id)
                if active_open_session is not None:
                    pending_post_cart_close_sessions[ste_id] = active_open_session
            if normalized_category:
                bounced_categories = [value for value in bounced_categories if value != normalized_category]
                bounce_counts.pop(normalized_category, None)

        if event_type == "cart_remove" and ste_id:
            cart_ste_ids = [value for value in cart_ste_ids if value != ste_id]
            pending_post_cart_close_sessions.pop(ste_id, None)

        is_quick_item_close = event_type == "item_close" and 0 < duration_ms < QUICK_EXIT_THRESHOLD_MS
        cart_positive_close = False
        if is_quick_item_close:
            cart_positive_close = close_reason == "after_cart_add"
            if not cart_positive_close and ste_id:
                active_open_session = open_ste_sessions.get(ste_id)
                pending_close_session = pending_post_cart_close_sessions.get(ste_id)
                cart_positive_close = (
                    active_open_session is not None
                    and pending_close_session is not None
                    and active_open_session == pending_close_session
                )
            if cart_positive_close and normalized_category:
                bounced_categories = [value for value in bounced_categories if value != normalized_category]
                bounce_counts.pop(normalized_category, None)
            state["last_item_close_penalizable"] = not cart_positive_close
            state["last_item_close_suppressed"] = cart_positive_close

        is_bounce = event_type == "bounce" or (is_quick_item_close and not cart_positive_close)
        if is_bounce and normalized_category:
            next_bounce_count = bounce_counts.get(normalized_category, 0) + 1
            bounce_counts[normalized_category] = next_bounce_count
            if next_bounce_count > 1:
                bounced_categories = _dedupe_trim([normalized_category, *bounced_categories])

        if event_type == "item_close" and not is_quick_item_close:
            if ste_id:
                clicked_ste_ids = _dedupe_trim([ste_id, *clicked_ste_ids])
            if normalized_category:
                recent_categories = _dedupe_trim([normalized_category, *recent_categories])

        if event_type == "item_close" and is_quick_item_close:
            if ste_id:
                clicked_ste_ids = [value for value in clicked_ste_ids if value != ste_id]
            if normalized_category:
                recent_categories = [value for value in recent_categories if value != normalized_category]

        if event_type == "item_close" and ste_id:
            active_open_session = open_ste_sessions.get(ste_id)
            pending_close_session = pending_post_cart_close_sessions.get(ste_id)
            if active_open_session is not None and pending_close_session == active_open_session:
                pending_post_cart_close_sessions.pop(ste_id, None)
            open_ste_sessions.pop(ste_id, None)

        if event_type == "purchase" and normalized_category:
            recent_categories = _dedupe_trim([normalized_category, *recent_categories])
            bounced_categories = [value for value in bounced_categories if value != normalized_category]
            bounce_counts.pop(normalized_category, None)

        state["recent_categories"] = recent_categories
        state["clicked_ste_ids"] = clicked_ste_ids
        state["cart_ste_ids"] = cart_ste_ids
        state["open_sequence"] = open_sequence
        state["open_ste_sessions"] = open_ste_sessions
        state["pending_post_cart_close_sessions"] = pending_post_cart_close_sessions
        state["bounced_categories"] = bounced_categories
        state["bounce_counts"] = bounce_counts

        self._store_session_state(user_id=user_id, state=state)
        return state

    def _load_session_state(
        self,
        *,
        user_id: Optional[str],
        customer_inn: Optional[str],
        customer_region: Optional[str],
    ) -> Dict[str, object]:
        state = self._empty_state(
            user_id=user_id,
            customer_inn=customer_inn,
            customer_region=customer_region,
        )
        if not user_id or not self.cache_service.enabled:
            return state

        cache_key = self.cache_service.build_key("session", suffix=str(user_id))
        cached_payload = self.cache_service.get_json(cache_key)
        if not isinstance(cached_payload, dict):
            return state

        state.update(cached_payload)
        if customer_inn:
            state["customer_inn"] = customer_inn
        if customer_region:
            state["customer_region"] = customer_region
        state["recent_categories"] = _dedupe_trim([normalize_text(value) for value in state.get("recent_categories", [])])
        state["clicked_ste_ids"] = _dedupe_trim([str(value) for value in state.get("clicked_ste_ids", [])])
        state["cart_ste_ids"] = _dedupe_trim([str(value) for value in state.get("cart_ste_ids", [])])
        state["open_sequence"] = int(state.get("open_sequence", 0) or 0)
        state["open_ste_sessions"] = _normalize_int_mapping(dict(state.get("open_ste_sessions", {})))
        state["pending_post_cart_close_sessions"] = _normalize_int_mapping(
            dict(state.get("pending_post_cart_close_sessions", {}))
        )
        state["bounced_categories"] = _dedupe_trim([normalize_text(value) for value in state.get("bounced_categories", [])])
        state["bounce_counts"] = {
            normalize_text(str(key)): int(value) for key, value in dict(state.get("bounce_counts", {})).items()
        }
        state["event_counts"] = {str(key): int(value) for key, value in dict(state.get("event_counts", {})).items()}
        state["version"] = int(state.get("version", 0) or 0)
        state["last_item_close_penalizable"] = bool(state.get("last_item_close_penalizable", False))
        state["last_item_close_suppressed"] = bool(state.get("last_item_close_suppressed", False))
        return state

    def _store_session_state(self, *, user_id: Optional[str], state: Dict[str, object]) -> None:
        if not user_id or not self.cache_service.enabled:
            return
        cache_key = self.cache_service.build_key("session", suffix=str(user_id))
        self.cache_service.set_json(cache_key, state, ttl_seconds=self.session_ttl_seconds)

    @staticmethod
    def _empty_state(
        *,
        user_id: Optional[str],
        customer_inn: Optional[str],
        customer_region: Optional[str],
    ) -> Dict[str, object]:
        return {
            "user_id": user_id or "anonymous",
            "customer_inn": customer_inn or "",
            "customer_region": customer_region or "",
            "recent_categories": [],
            "clicked_ste_ids": [],
            "cart_ste_ids": [],
            "open_sequence": 0,
            "open_ste_sessions": {},
            "pending_post_cart_close_sessions": {},
            "bounced_categories": [],
            "bounce_counts": {},
            "event_counts": {},
            "last_event_type": None,
            "last_event_at": None,
            "last_item_close_penalizable": False,
            "last_item_close_suppressed": False,
            "version": 0,
        }
