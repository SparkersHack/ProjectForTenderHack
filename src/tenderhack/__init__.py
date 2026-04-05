from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "CacheService",
    "CatalogDescriptionService",
    "OnlineStateService",
    "OfferLookupService",
    "PersonalizationService",
    "PersonalizationRuntimeService",
    "SemanticExpander",
    "SearchService",
    "SearchRerankPredictor",
    "build_customer_profile",
    "rerank_offers",
    "rerank_ste",
    "search_ste",
]

_EXPORTS = {
    "CacheService": (".cache", "CacheService"),
    "CatalogDescriptionService": (".descriptions", "CatalogDescriptionService"),
    "OnlineStateService": (".online_state", "OnlineStateService"),
    "OfferLookupService": (".offers", "OfferLookupService"),
    "PersonalizationService": (".personalization", "PersonalizationService"),
    "PersonalizationRuntimeService": (".personalization_runtime", "PersonalizationRuntimeService"),
    "SemanticExpander": (".semantic", "SemanticExpander"),
    "SearchService": (".search", "SearchService"),
    "SearchRerankPredictor": (".search_rerank_model", "SearchRerankPredictor"),
    "build_customer_profile": (".personalization", "build_customer_profile"),
    "rerank_offers": (".personalization", "rerank_offers"),
    "rerank_ste": (".personalization", "rerank_ste"),
    "search_ste": (".search", "search_ste"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, export_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, export_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .cache import CacheService
    from .descriptions import CatalogDescriptionService
    from .online_state import OnlineStateService
    from .offers import OfferLookupService
    from .personalization import PersonalizationService, build_customer_profile, rerank_offers, rerank_ste
    from .personalization_runtime import PersonalizationRuntimeService
    from .search import SearchService, search_ste
    from .search_rerank_model import SearchRerankPredictor
    from .semantic import SemanticExpander
