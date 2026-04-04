from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.offers import OfferLookupService
from tenderhack.personalization import PersonalizationService
from tenderhack.personalization_runtime import PersonalizationRuntimeService
from tenderhack.search import SearchService
from tenderhack.search_rerank_model import SearchRerankPredictor
from tenderhack.text import normalize_text, unique_preserve_order


def _discover_search_rerank_model_path(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "processed" / "tenderhack_yeti_ranker.cbm",
        project_root / "data" / "processed" / "tenderhack_yeti_ranker_small.cbm",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _discover_search_rerank_metadata_path(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "processed" / "tenderhack_yeti_ranker.json",
        project_root / "data" / "processed" / "tenderhack_yeti_ranker_small.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class AppSettings:
    search_db_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_search.sqlite"
    preprocessed_db_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_preprocessed.sqlite"
    synonyms_path: Path = PROJECT_ROOT / "data" / "reference" / "search_synonyms.json"
    fasttext_model_path: Path = PROJECT_ROOT / "data" / "processed" / "tenderhack_fasttext.bin"
    personalization_model_path: Path = PROJECT_ROOT / "artifacts" / "personalization_model.cbm"
    search_rerank_model_path: Path = _discover_search_rerank_model_path(PROJECT_ROOT)
    search_rerank_metadata_path: Path = _discover_search_rerank_metadata_path(PROJECT_ROOT)
    semantic_backend: str = "auto"

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            search_db_path=Path(os.getenv("TENDERHACK_SEARCH_DB", cls.search_db_path)),
            preprocessed_db_path=Path(os.getenv("TENDERHACK_PREPROCESSED_DB", cls.preprocessed_db_path)),
            synonyms_path=Path(os.getenv("TENDERHACK_SYNONYMS_PATH", cls.synonyms_path)),
            fasttext_model_path=Path(os.getenv("TENDERHACK_FASTTEXT_MODEL_PATH", cls.fasttext_model_path)),
            personalization_model_path=Path(
                os.getenv("TENDERHACK_PERSONALIZATION_MODEL_PATH", cls.personalization_model_path)
            ),
            search_rerank_model_path=Path(
                os.getenv("TENDERHACK_SEARCH_RERANK_MODEL_PATH", cls.search_rerank_model_path)
            ),
            search_rerank_metadata_path=Path(
                os.getenv("TENDERHACK_SEARCH_RERANK_METADATA_PATH", cls.search_rerank_metadata_path)
            ),
            semantic_backend=os.getenv("TENDERHACK_SEMANTIC_BACKEND", cls.semantic_backend),
        )


class LoginRequest(BaseModel):
    inn: str = Field(min_length=1)


class UserPayload(BaseModel):
    id: str
    inn: str
    region: str
    viewedCategories: List[str] = Field(default_factory=list)


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


class ProductPayload(BaseModel):
    id: str
    name: str
    category: str
    price: float
    supplierInn: str
    reasonToShow: Optional[str] = None


class SearchResponsePayload(BaseModel):
    items: List[ProductPayload]
    totalCount: int
    correctedQuery: Optional[str] = None


class TenderHackApiService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._validate_required_paths()
        self.search_service = SearchService(
            search_db_path=settings.search_db_path,
            synonyms_path=settings.synonyms_path,
            semantic_backend=settings.semantic_backend,
            fasttext_model_path=settings.fasttext_model_path,
        )
        self.personalization_service = PersonalizationService(db_path=settings.preprocessed_db_path)
        self.personalization_runtime_service = PersonalizationRuntimeService(
            db_path=settings.preprocessed_db_path,
            model_path=settings.personalization_model_path,
        )
        self.offer_lookup_service = OfferLookupService(db_path=settings.preprocessed_db_path)
        self.search_rerank_predictor = SearchRerankPredictor(
            model_path=settings.search_rerank_model_path,
            metadata_path=settings.search_rerank_metadata_path,
        )

    def close(self) -> None:
        self.search_service.close()
        self.personalization_service.close()
        self.personalization_runtime_service.close()
        self.offer_lookup_service.close()

    def _validate_required_paths(self) -> None:
        missing = [
            str(path)
            for path in [self.settings.search_db_path, self.settings.preprocessed_db_path, self.settings.synonyms_path]
            if not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required search assets:\n" + "\n".join(f"- {path}" for path in missing)
            )

    def login(self, inn: str) -> UserPayload:
        profile = self.personalization_service.build_customer_profile(customer_inn=inn)
        viewed_categories = [item["category"] for item in profile.get("top_categories", [])[:5]]
        region = str(profile.get("customer_region") or "")
        return UserPayload(
            id=f"user-{inn}",
            inn=inn,
            region=region,
            viewedCategories=viewed_categories,
        )

    def search(self, payload: SearchRequest) -> SearchResponsePayload:
        raw_payload = self.search_service.search(query=payload.query, top_k=max(payload.topK * 5, 60))
        results = list(raw_payload["results"])
        if self.search_rerank_predictor.enabled:
            results = self.search_rerank_predictor.rerank_candidates(
                query=payload.query,
                query_meta=dict(raw_payload["query"]),
                candidates=results,
            )

        user_context = payload.userContext or SearchUserContext()
        session_categories = unique_preserve_order(payload.viewedCategories + user_context.viewedCategories)
        bounced_categories = {normalize_text(value) for value in payload.bouncedCategories if value}

        if user_context.inn or user_context.region or session_categories:
            personalization_query = (
                raw_payload["query"].get("corrected_query")
                or raw_payload["query"].get("normalized_query")
                or payload.query
            )
            results = self.personalization_runtime_service.rerank_candidates(
                query=str(personalization_query),
                candidates=results,
                user_id=user_context.id or (f"user-{user_context.inn}" if user_context.inn else "anonymous"),
                customer_inn=user_context.inn,
                customer_region=user_context.region,
                session_categories=session_categories,
            )
        else:
            for item in results:
                item["final_score"] = item.get("search_score", 0.0)
                item["top_reason_codes"] = []
                item["reasons"] = ["оставлено выше за счёт базовой текстовой релевантности"]

        for item in results:
            category_norm = normalize_text(str(item.get("category", "")))
            if category_norm and category_norm in bounced_categories:
                item["final_score"] = round(float(item.get("final_score", item.get("search_score", 0.0))) - 100.0, 4)
                item["reason_to_hide"] = "Категория пессимизирована после быстрого отказа"

        results.sort(
            key=lambda item: (
                float(item.get("final_score", item.get("search_score", 0.0))),
                float(item.get("search_score", 0.0)),
            ),
            reverse=True,
        )

        ste_ids = [str(item["ste_id"]) for item in results[: payload.topK * 2]]
        offer_lookup = self.offer_lookup_service.get_offer_lookup(ste_ids)

        products: List[ProductPayload] = []
        for item in results:
            if len(products) >= payload.topK:
                break
            ste_id = str(item["ste_id"])
            offer = offer_lookup.get(ste_id, {})
            reason_to_show = self._map_reason_to_show(
                reason_codes=item.get("top_reason_codes", []),
                category=str(item.get("category") or ""),
                session_categories=session_categories,
                is_bounced=bool(item.get("reason_to_hide")),
            )
            products.append(
                ProductPayload(
                    id=ste_id,
                    name=str(item.get("clean_name") or item.get("normalized_name") or ste_id),
                    category=str(item.get("category") or ""),
                    price=round(float(offer.get("price", 0.0)), 2),
                    supplierInn=str(offer.get("supplier_inn") or "не указан"),
                    reasonToShow=reason_to_show,
                )
            )

        corrected_query = raw_payload["query"].get("corrected_query") or None
        normalized_query = raw_payload["query"].get("normalized_query") or None
        if corrected_query == normalized_query:
            corrected_query = None

        return SearchResponsePayload(
            items=products,
            totalCount=len(results),
            correctedQuery=corrected_query,
        )

    def suggestions(self, query: str, top_k: int = 5) -> List[str]:
        payload = self.search_service.search(query=query, top_k=max(top_k, 5))
        suggestions: List[str] = []
        corrected_query = payload["query"].get("corrected_query")
        normalized_query = payload["query"].get("normalized_query")
        if corrected_query and corrected_query != normalized_query:
            suggestions.append(corrected_query)
        suggestions.extend(item["clean_name"] for item in payload["results"])
        return unique_preserve_order(suggestions)[:top_k]

    @staticmethod
    def _map_reason_to_show(
        reason_codes: List[str],
        category: str,
        session_categories: List[str],
        is_bounced: bool,
    ) -> Optional[str]:
        if is_bounced:
            return None
        codes = {str(code) for code in reason_codes}
        session_category_set = {normalize_text(value) for value in session_categories if value}
        category_norm = normalize_text(category)

        if codes & {"USER_CATEGORY_AFFINITY", "USER_REPEAT_BUY", "RECENT_SIMILAR_PURCHASE", "SUPPLIER_AFFINITY"}:
            return "На основе ваших закупок"
        if "REGIONAL_POPULARITY" in codes:
            return "Популярно в вашем регионе"
        if "SIMILAR_CUSTOMER_POPULARITY" in codes:
            return "Популярно у похожих заказчиков"
        if category_norm and category_norm in session_category_set:
            return "Продолжить подбор в этой категории"
        return None


def create_app(settings: Optional[AppSettings] = None) -> FastAPI:
    active_settings = settings or AppSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = TenderHackApiService(active_settings)
        app.state.service = service
        yield
        service.close()

    app = FastAPI(
        title="TenderHack Search API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/auth/login", response_model=UserPayload)
    async def login(payload: LoginRequest, request: Request) -> UserPayload:
        return request.app.state.service.login(payload.inn)

    @app.post("/api/search", response_model=SearchResponsePayload)
    async def search(payload: SearchRequest, request: Request) -> SearchResponsePayload:
        return request.app.state.service.search(payload)

    @app.get("/api/search/suggestions", response_model=List[str])
    async def suggestions(
        request: Request,
        q: str = Query(min_length=1),
        top_k: int = Query(default=5, ge=1, le=10),
    ) -> List[str]:
        return request.app.state.service.suggestions(query=q, top_k=top_k)

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


app = create_app()
