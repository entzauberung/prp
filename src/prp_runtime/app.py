"""ASGI application factory.

Importing this module must not create an event loop, open a database
connection, or start a background task. All wiring happens inside
``create_app`` and its lifespan.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from prp_runtime import __version__
from prp_runtime.api.anthropic_messages import create_router as create_anthropic_router
from prp_runtime.api.errors import install_error_handlers
from prp_runtime.api.native import create_router as create_native_router
from prp_runtime.api.native_agent import create_router as create_native_agent_router
from prp_runtime.api.openai_chat import create_router as create_openai_chat_router
from prp_runtime.api.openai_responses import (
    create_router as create_openai_responses_router,
)
from prp_runtime.domain.enums import ExecutionLocation, IsolationMode
from prp_runtime.domain.errors import ErrorCode, ErrorDetail, PrpError
from prp_runtime.providers.base import ProviderAdapter
from prp_runtime.runtime.composition import RuntimeComposition, build_adapters
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.workspace.sandbox import SandboxCapabilities, probe_bwrap

__all__ = [
    "HealthResponse",
    "ReadinessResponse",
    "build_adapters",
    "create_app",
]


class HealthResponse(BaseModel):
    """Liveness payload. It never exposes configuration or environment."""

    status: str
    version: str


class ReadinessResponse(BaseModel):
    """Local readiness facts, without provider calls or configuration values."""

    status: str
    store_open: bool
    controller_present: bool
    profiles_configured: bool
    adapters_ready: bool
    path_boundary_ready: bool
    sandbox_ready: bool
    sandbox_required: bool
    execution_location: ExecutionLocation
    isolation_mode: IsolationMode


def create_app(
    settings: Settings,
    *,
    adapters: Mapping[str, ProviderAdapter] | None = None,
    store: SqliteStore | None = None,
    execution_location: ExecutionLocation = ExecutionLocation.CLOUD,
    isolation_mode: IsolationMode = IsolationMode.SANDBOXED,
    sandbox_capabilities: SandboxCapabilities | None = None,
) -> FastAPI:
    """Build the ASGI application for the given settings.

    ``adapters`` and ``store`` exist so a test can inject a fake provider and a
    temporary database. When they are omitted the application owns both.
    """
    injected_store = store
    resolved_adapters = dict(adapters) if adapters is not None else None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        composition = RuntimeComposition(
            settings,
            adapters=resolved_adapters,
            store=injected_store,
            execution_location=execution_location,
            isolation_mode=isolation_mode,
        )
        try:
            await composition.open()
            application.state.event_bus = composition.event_bus
            application.state.store = composition.store
            application.state.recovery = composition.recovery
            application.state.adapters = composition.adapters
            application.state.sandbox_capabilities = (
                sandbox_capabilities
                if sandbox_capabilities is not None
                else probe_bwrap()
            )
            application.state.execution_location = composition.execution_location
            application.state.isolation_mode = composition.isolation_mode
            application.state.tool_runtime_provider = composition.tool_runtime_provider
            application.state.controller = composition.controller
            application.state.supervisor = composition.supervisor
            yield
        finally:
            application.state.event_bus = None
            application.state.supervisor = None
            application.state.tool_runtime_provider = None
            application.state.controller = None
            await composition.close()

    app = FastAPI(title="PRP Runtime", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.execution_location = execution_location
    app.state.isolation_mode = isolation_mode
    app.state.sandbox_capabilities = sandbox_capabilities
    app.state.store = None
    app.state.controller = None
    app.state.event_bus = None
    app.state.supervisor = None
    app.state.tool_runtime_provider = None

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

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready() -> ReadinessResponse:
        """Report local readiness without contacting a Provider."""
        active_store = getattr(app.state, "store", None)
        store_open = bool(active_store is not None and active_store.is_open)
        controller_present = app.state.controller is not None
        profiles_configured = (
            settings.leader_profile is not None and settings.worker_profile is not None
        )
        configured_aliases = {profile.alias for profile in settings.profiles}
        active_adapters: Mapping[str, Any] = getattr(app.state, "adapters", {})
        adapters_ready = configured_aliases <= set(active_adapters)
        capabilities = app.state.sandbox_capabilities
        sandbox_ready = isinstance(capabilities, SandboxCapabilities) and capabilities.ready
        selected_location = getattr(app.state, "execution_location", execution_location)
        selected_isolation = getattr(app.state, "isolation_mode", isolation_mode)
        path_boundary_ready = getattr(app.state, "tool_runtime_provider", None) is not None
        sandbox_required = selected_isolation is IsolationMode.SANDBOXED
        components_ready = (
            store_open
            and controller_present
            and profiles_configured
            and adapters_ready
            and path_boundary_ready
        )
        sandbox_ok = sandbox_ready if sandbox_required else True
        response = ReadinessResponse(
            status="ready" if components_ready and sandbox_ok else "not_ready",
            store_open=store_open,
            controller_present=controller_present,
            profiles_configured=profiles_configured,
            adapters_ready=adapters_ready,
            path_boundary_ready=path_boundary_ready,
            sandbox_ready=sandbox_ready,
            sandbox_required=sandbox_required,
            execution_location=selected_location,
            isolation_mode=selected_isolation,
        )
        return JSONResponse(
            status_code=200 if response.status == "ready" else 503,
            content=response.model_dump(mode="json"),
            headers={"Cache-Control": "no-store"},
        )  # type: ignore[return-value]

    app.include_router(create_native_router())
    app.include_router(create_native_agent_router())
    app.include_router(create_openai_responses_router())
    app.include_router(create_openai_chat_router())
    app.include_router(create_anthropic_router())

    return app
