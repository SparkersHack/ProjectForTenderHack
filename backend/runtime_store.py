from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from tenderhack.text import normalize_text, unique_preserve_order


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CartItem:
    ste_id: str
    name: str
    category: str
    quantity: int
    price: float
    supplier_inn: str
    added_at: str = field(default_factory=_utc_now_iso)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "steId": self.ste_id,
            "name": self.name,
            "category": self.category,
            "quantity": self.quantity,
            "price": round(float(self.price), 2),
            "supplierInn": self.supplier_inn,
            "addedAt": self.added_at,
        }


@dataclass
class SessionSnapshot:
    user_id: str
    clicked_ste_ids: List[str] = field(default_factory=list)
    cart_ste_ids: List[str] = field(default_factory=list)
    recent_categories: List[str] = field(default_factory=list)
    viewed_categories: List[str] = field(default_factory=list)
    bounced_categories: List[str] = field(default_factory=list)
    events: List[Dict[str, object]] = field(default_factory=list)

    def to_mapping(self) -> Dict[str, object]:
        return {
            "userId": self.user_id,
            "clickedSteIds": list(self.clicked_ste_ids),
            "cartSteIds": list(self.cart_ste_ids),
            "recentCategories": list(self.recent_categories),
            "viewedCategories": list(self.viewed_categories),
            "bouncedCategories": list(self.bounced_categories),
            "eventCount": len(self.events),
        }


class RuntimeStateStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionSnapshot] = {}
        self._cart_items: Dict[str, Dict[str, CartItem]] = {}

    def get_session_snapshot(self, user_id: str) -> SessionSnapshot:
        if not user_id:
            user_id = "anonymous"
        snapshot = self._sessions.get(user_id)
        if snapshot is None:
            snapshot = SessionSnapshot(user_id=user_id)
            self._sessions[user_id] = snapshot
        return snapshot

    def track_event(
        self,
        *,
        user_id: str,
        event_type: str,
        ste_id: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> SessionSnapshot:
        snapshot = self.get_session_snapshot(user_id)
        normalized_category = normalize_text(category or "")
        event_type = str(event_type or "").strip().lower()
        payload = {
            "type": event_type,
            "steId": ste_id,
            "category": category or "",
            "query": query or "",
            "metadata": dict(metadata or {}),
            "createdAt": _utc_now_iso(),
        }
        snapshot.events.append(payload)
        snapshot.events = snapshot.events[-100:]

        if ste_id and event_type in {"click", "open", "view"}:
            snapshot.clicked_ste_ids = unique_preserve_order([ste_id] + snapshot.clicked_ste_ids)[:20]

        if normalized_category and event_type in {"click", "open", "view", "query_select"}:
            snapshot.viewed_categories = unique_preserve_order([category or normalized_category] + snapshot.viewed_categories)[:20]
            snapshot.recent_categories = unique_preserve_order([normalized_category] + snapshot.recent_categories)[:20]

        if normalized_category and event_type in {"bounce", "fast_bounce"}:
            snapshot.bounced_categories = unique_preserve_order([normalized_category] + snapshot.bounced_categories)[:20]

        if ste_id and event_type in {"cart", "cart_add"}:
            snapshot.cart_ste_ids = unique_preserve_order([ste_id] + snapshot.cart_ste_ids)[:20]
            if normalized_category:
                snapshot.recent_categories = unique_preserve_order([normalized_category] + snapshot.recent_categories)[:20]

        return snapshot

    def add_to_cart(
        self,
        *,
        user_id: str,
        ste_id: str,
        name: str,
        category: str,
        quantity: int,
        price: float,
        supplier_inn: str,
    ) -> CartItem:
        snapshot = self.get_session_snapshot(user_id)
        cart = self._cart_items.setdefault(snapshot.user_id, {})
        existing = cart.get(ste_id)
        if existing is not None:
            existing.quantity += max(1, int(quantity))
            return existing

        item = CartItem(
            ste_id=ste_id,
            name=name,
            category=category,
            quantity=max(1, int(quantity)),
            price=price,
            supplier_inn=supplier_inn,
        )
        cart[ste_id] = item
        snapshot.cart_ste_ids = unique_preserve_order([ste_id] + snapshot.cart_ste_ids)[:20]
        if category:
            normalized_category = normalize_text(category)
            snapshot.recent_categories = unique_preserve_order([normalized_category] + snapshot.recent_categories)[:20]
            snapshot.viewed_categories = unique_preserve_order([category] + snapshot.viewed_categories)[:20]
        return item

    def get_cart(self, user_id: str) -> List[CartItem]:
        snapshot = self.get_session_snapshot(user_id)
        return list(self._cart_items.get(snapshot.user_id, {}).values())

    def clear_cart(self, user_id: str) -> None:
        snapshot = self.get_session_snapshot(user_id)
        self._cart_items.pop(snapshot.user_id, None)
