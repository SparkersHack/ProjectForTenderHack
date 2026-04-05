from __future__ import annotations

from importlib import import_module


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
    "SQLiteUserHistoryRepository": (".user_profile_scorer", "SQLiteUserHistoryRepository"),
    "UserProfileScorer": (".user_profile_scorer", "UserProfileScorer"),
    "build_customer_profile": (".personalization", "build_customer_profile"),
    "apply_personalization": (".user_profile_scorer", "apply_personalization"),
    "rerank_offers": (".personalization", "rerank_offers"),
    "rerank_ste": (".personalization", "rerank_ste"),
    "search_ste": (".search", "search_ste"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
