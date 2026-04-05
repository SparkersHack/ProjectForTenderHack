from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import AppSettings
from .schemas import (
    EventRequest,
    EventResponsePayload,
    LoginRequest,
    SearchRequest,
    SearchResponsePayload,
    SuggestionPayload,
    UserPayload,
)
from .service import TenderHackApiService
from .utils import split_pipe_separated_values


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

    @app.post("/api/event", response_model=EventResponsePayload)
    async def event(payload: EventRequest, request: Request) -> EventResponsePayload:
        return request.app.state.service.record_event(payload)

    @app.get("/api/search/suggestions", response_model=List[SuggestionPayload])
    async def suggestions(
        request: Request,
        q: str = Query(min_length=1),
        top_k: int = Query(default=5, ge=1, le=10),
        inn: Optional[str] = Query(default=None),
        viewed_categories: Optional[str] = Query(default=None),
        top_categories: Optional[str] = Query(default=None),
    ) -> List[SuggestionPayload]:
        return request.app.state.service.suggestions(
            query=q,
            top_k=top_k,
            user_inn=inn,
            viewed_categories=split_pipe_separated_values(viewed_categories),
            top_categories=split_pipe_separated_values(top_categories),
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


app = create_app()


__all__ = ["app", "create_app"]
