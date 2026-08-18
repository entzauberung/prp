"""ASGI application factory.

Importing this module must not create an event loop, open a database
connection, or start a background task. All wiring happens inside
``create_app`` and its lifespan.
"""

from collections.abc import AsyncIterator, Collection, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
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
from prp_runtime.control.controller import RunController
from prp_runtime.control.routing import facts_from_request
from prp_runtime.domain.enums import RunStatus
from prp_runtime.domain.errors import ErrorCode, ErrorDetail, ProviderError, PrpError
from prp_runtime.domain.models import ErrorCategory, ErrorInfo, Run
from prp_runtime.providers.base import ProviderAdapter
from prp_runtime.providers.openai_compatible import OpenAICompatibleProvider
from prp_runtime.runtime.event_bus import EventBus
from prp_runtime.runtime.supervisor import RunSupervisor
from prp_runtime.runtime.tooling import ScopeToolRuntimeProvider
from prp_runtime.settings import Settings
from prp_runtime.storage.recovery import recover_after_restart
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.workspace.sandbox import SandboxCapabilities, probe_bwrap

__all__ = ["HealthResponse", "ReadinessResponse", "build_adapters", "create_app"]


class _SqlitePendingRunScanner:
    """Read pending run ids from the Store without making them queue state."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    async def list_pending_runs(self) -> Collection[Run]:
        return await self._store.list_recoverable_runs()


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
    sandbox_ready: bool


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
    owns_adapters = adapters is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owns_store = injected_store is None
        event_bus = EventBus()
        active = (
            SqliteStore(Path(settings.database_path), event_bus=event_bus)
            if injected_store is None
            else injected_store
        )
        active.set_event_bus(event_bus)
        application.state.event_bus = event_bus
        runtime_adapters: Mapping[str, ProviderAdapter] = {}
        supervisor: RunSupervisor | None = None
        tool_runtime_provider: ScopeToolRuntimeProvider | None = None
        try:
            await active.open()
            application.state.store = active
            recovery = await recover_after_restart(active)
            application.state.recovery = recovery
            runtime_adapters = (
                resolved_adapters if resolved_adapters is not None else build_adapters(settings)
            )
            application.state.adapters = runtime_adapters
            application.state.sandbox_capabilities = probe_bwrap()
            tool_runtime_provider = ScopeToolRuntimeProvider(active, settings)
            application.state.tool_runtime_provider = tool_runtime_provider
            controller = RunController(
                active,
                settings,
                application.state.adapters,
                tool_executor_provider=tool_runtime_provider.executor_for,
            )
            application.state.controller = controller

            async def execute_persisted(run_id: str) -> Run:
                run = await active.get_run(run_id)
                try:
                    return await controller.execute(
                        run_id,
                        routing_facts=facts_from_request(run.request),
                        principal_id=settings.service_principal,
                    )
                except PrpError as error:
                    current = await active.get_run(run_id)
                    if current.status.is_terminal:
                        return current
                    if current.status is RunStatus.CANCELLING:
                        return await controller._finish_run(
                            current, RunStatus.CANCELLED
                        )
                    category = (
                        ErrorCategory.PROVIDER_ERROR
                        if isinstance(error, ProviderError)
                        else ErrorCategory.UNKNOWN
                    )
                    return await controller._finish_run(
                        current,
                        RunStatus.FAILED,
                        error=ErrorInfo(category=category, message=str(error)),
                    )

            supervisor = RunSupervisor(
                _SqlitePendingRunScanner(active), execute_persisted
            )
            application.state.supervisor = supervisor
            await supervisor.start(
                recoverable_run_ids=recovery.recoverable_run_ids,
                blocked_run_ids=recovery.blocked_run_ids,
            )
            yield
        finally:
            close_error: BaseException | None = None
            if supervisor is not None:
                try:
                    await supervisor.stop(drain=True)
                except BaseException as error:
                    close_error = error
            if tool_runtime_provider is not None:
                try:
                    tool_runtime_provider.close()
                except BaseException as error:
                    if close_error is None:
                        close_error = error
            try:
                await event_bus.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
            active.set_event_bus(None)
            application.state.event_bus = None
            application.state.supervisor = None
            application.state.tool_runtime_provider = None
            application.state.controller = None
            if owns_adapters:
                for adapter in runtime_adapters.values():
                    try:
                        await adapter.aclose()
                    except BaseException as error:
                        if close_error is None:
                            close_error = error
            if owns_store:
                try:
                    await active.close()
                except BaseException as error:
                    if close_error is None:
                        close_error = error
            if close_error is not None:
                raise close_error

    app = FastAPI(title="PRP Runtime", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.sandbox_capabilities = None
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
        response = ReadinessResponse(
            status=(
                "ready"
                if store_open
                and controller_present
                and profiles_configured
                and adapters_ready
                and sandbox_ready
                else "not_ready"
            ),
            store_open=store_open,
            controller_present=controller_present,
            profiles_configured=profiles_configured,
            adapters_ready=adapters_ready,
            sandbox_ready=sandbox_ready,
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
