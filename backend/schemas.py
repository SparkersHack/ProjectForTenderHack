from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


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


__all__ = [
    "EventRequest",
    "EventResponsePayload",
    "LoginRequest",
    "ProductPayload",
    "SearchRequest",
    "SearchResponsePayload",
    "SearchUserContext",
    "SuggestionPayload",
    "UserPayload",
]
