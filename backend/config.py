from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import PROJECT_ROOT


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


def _optional_env_path(name: str) -> Optional[Path]:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class AppSettings:
    search_db_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_search.sqlite"
    preprocessed_db_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_preprocessed.sqlite"
    synonyms_path: Path = PROJECT_ROOT / "data" / "reference" / "search_synonyms.json"
    fasttext_model_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_fasttext.bin"
    personalization_model_path: Path = PROJECT_ROOT / "artifacts" / "personalization_model.cbm"
    search_rerank_enabled: bool = True
    search_rerank_model_path: Optional[Path] = None
    search_rerank_metadata_path: Optional[Path] = None
    raw_ste_catalog_path: Path = PROJECT_ROOT / "СТЕ_20260403.csv"
    redis_url: Optional[str] = "memory://"
    semantic_backend: str = "auto"
    login_cache_ttl_seconds: int = 1800
    search_cache_ttl_seconds: int = 120
    suggestions_cache_ttl_seconds: int = 300
    user_profile_cache_ttl_seconds: int = 1800
    offer_lookup_cache_ttl_seconds: int = 1800
    session_state_ttl_seconds: int = 86400

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            search_db_path=_env_path("TENDERHACK_SEARCH_DB", cls.search_db_path),
            preprocessed_db_path=_env_path("TENDERHACK_PREPROCESSED_DB", cls.preprocessed_db_path),
            synonyms_path=_env_path("TENDERHACK_SYNONYMS_PATH", cls.synonyms_path),
            fasttext_model_path=_env_path("TENDERHACK_FASTTEXT_MODEL_PATH", cls.fasttext_model_path),
            personalization_model_path=_env_path("TENDERHACK_PERSONALIZATION_MODEL_PATH", cls.personalization_model_path),
            search_rerank_enabled=_env_bool("TENDERHACK_SEARCH_RERANK_ENABLED", cls.search_rerank_enabled),
            search_rerank_model_path=_optional_env_path("TENDERHACK_SEARCH_RERANK_MODEL_PATH"),
            search_rerank_metadata_path=_optional_env_path("TENDERHACK_SEARCH_RERANK_METADATA_PATH"),
            raw_ste_catalog_path=_env_path("TENDERHACK_RAW_STE_CATALOG_PATH", cls.raw_ste_catalog_path),
            redis_url=os.getenv("TENDERHACK_REDIS_URL") or cls.redis_url,
            semantic_backend=os.getenv("TENDERHACK_SEMANTIC_BACKEND", cls.semantic_backend),
            login_cache_ttl_seconds=int(os.getenv("TENDERHACK_LOGIN_CACHE_TTL_SECONDS", cls.login_cache_ttl_seconds)),
            search_cache_ttl_seconds=int(
                os.getenv("TENDERHACK_SEARCH_CACHE_TTL_SECONDS", cls.search_cache_ttl_seconds)
            ),
            suggestions_cache_ttl_seconds=int(
                os.getenv("TENDERHACK_SUGGESTIONS_CACHE_TTL_SECONDS", cls.suggestions_cache_ttl_seconds)
            ),
            user_profile_cache_ttl_seconds=int(
                os.getenv("TENDERHACK_USER_PROFILE_CACHE_TTL_SECONDS", cls.user_profile_cache_ttl_seconds)
            ),
            offer_lookup_cache_ttl_seconds=int(
                os.getenv("TENDERHACK_OFFER_LOOKUP_CACHE_TTL_SECONDS", cls.offer_lookup_cache_ttl_seconds)
            ),
            session_state_ttl_seconds=int(
                os.getenv("TENDERHACK_SESSION_STATE_TTL_SECONDS", cls.session_state_ttl_seconds)
            ),
        )


__all__ = ["AppSettings"]
