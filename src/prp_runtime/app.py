"""ASGI application factory.

Importing this module must not create an event loop, open a database
connection, or start a background task. All wiring happens inside
``create_app`` and its lifespan.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from prp_runtime import __version__
from prp_runtime.api.errors import install_error_handlers
from prp_runtime.api.native import create_router
from prp_runtime.control.controller import RunController
from prp_runtime.domain.errors import ErrorCode, ErrorDetail
from prp_runtime.providers.base import ProviderAdapter
from prp_runtime.providers.openai_compatible import OpenAICompatibleProvider
from prp_runtime.settings import Settings
from prp_runtime.storage.recovery import recover_after_restart
from prp_runtime.storage.sqlite import SqliteStore

__all__ = ["HealthResponse", "build_adapters", "create_app"]


class HealthResponse(BaseModel):
    """Liveness payload. It never exposes configuration or environment."""

    status: str
    version: str


def build_adapters(settings: Settings) -> dict[str, ProviderAdapter]:
    """Build one outbound adapter per configured model profile."""
    return {
        profile.alias: OpenAICompatibleProvider(profile) for profile in settings.profiles
    }


def create_app(
    settings: Settings,
    *,
    adapters: Mapping[str, ProviderAdapter] | None = None,
    store: SqliteStore | None = None,
) -> FastAPI:
    """Build the ASGI application for the given settings.

    ``adapters`` and ``store`` exist so a test can inject a fake provider and a
    temporary database. When they are omitted the application owns both.
    """
    injected_store = store
    resolved_adapters = dict(adapters) if adapters is not None else None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owned = injected_store is None
        active = injected_store or SqliteStore(Path(settings.database_path))
        await active.open()
        try:
            application.state.store = active
            application.state.recovery = await recover_after_restart(active)
            application.state.adapters = (
                resolved_adapters if resolved_adapters is not None else build_adapters(settings)
            )
            application.state.controller = RunController(
                active, settings, application.state.adapters
            )
            yield
        finally:
            application.state.controller = None
            if owned:
                await active.close()

    app = FastAPI(title="PRP Runtime", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = None
    app.state.controller = None

    install_error_handlers(app)

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Refuse an oversized body before it is read."""
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit():
            if int(declared) > settings.max_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": ErrorDetail.for_code(
                            ErrorCode.INPUT_TOO_LARGE,
                            f"the request body exceeds {settings.max_request_bytes} bytes",
                            field="body",
                        ).model_dump(mode="json")
                    },
                    headers={"Cache-Control": "no-store"},
                )
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Report process liveness only. No provider, database or network check."""
        return HealthResponse(status="ok", version=__version__)

    app.include_router(create_router())

    return app
