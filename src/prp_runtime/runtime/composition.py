"""Reusable in-process runtime composition without an HTTP listener."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from prp_runtime.domain.enums import ExecutionLocation, IsolationMode, RunStatus
from prp_runtime.domain.errors import ProviderError, PrpError
from prp_runtime.domain.models import ErrorCategory, ErrorInfo, Run
from prp_runtime.providers.base import ProviderAdapter
from prp_runtime.providers.factory import build_provider_adapter
from prp_runtime.runtime.event_bus import EventBus
from prp_runtime.settings import Settings
from prp_runtime.storage.recovery import RecoveryReport, recover_after_restart
from prp_runtime.storage.sqlite import SqliteStore

__all__ = [
    "RuntimeComposition",
    "build_adapters",
    "open_runtime_composition",
]


class _SqlitePendingRunScanner:
    """Read pending run ids from the Store without making them queue state."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    async def list_pending_runs(self) -> Collection[Run]:
        return await self._store.list_recoverable_runs()


def build_adapters(settings: Settings) -> dict[str, ProviderAdapter]:
    """Build one outbound adapter per configured model profile."""
    return {profile.alias: build_provider_adapter(profile) for profile in settings.profiles}


class RuntimeComposition:
    """Owned store/adapters/controller/supervisor wiring for one process."""

    __slots__ = (
        "settings",
        "owns_store",
        "owns_adapters",
        "event_bus",
        "store",
        "recovery",
        "adapters",
        "tool_runtime_provider",
        "controller",
        "supervisor",
        "_opened",
        "_closed",
        "_injected_adapters",
        "_injected_store",
        "execution_location",
        "isolation_mode",
    )

    def __init__(
        self,
        settings: Settings,
        *,
        adapters: Mapping[str, ProviderAdapter] | None = None,
        store: SqliteStore | None = None,
        execution_location: ExecutionLocation = ExecutionLocation.CLOUD,
        isolation_mode: IsolationMode = IsolationMode.SANDBOXED,
    ) -> None:
        self.settings = settings
        self.owns_store = store is None
        self.owns_adapters = adapters is None
        self.execution_location = execution_location
        self.isolation_mode = isolation_mode
        self.event_bus: EventBus | None = None
        self.store = store
        self.recovery: RecoveryReport | None = None
        self.adapters: dict[str, ProviderAdapter] = dict(adapters) if adapters is not None else {}
        self.tool_runtime_provider = None
        self.controller = None
        self.supervisor = None
        self._opened = False
        self._closed = False
        self._injected_adapters = adapters
        self._injected_store = store

    def public_facts(self) -> dict[str, Any]:
        """Return lifecycle facts without host paths or credentials."""
        return {
            "opened": self._opened,
            "closed": self._closed,
            "owns_store": self.owns_store,
            "owns_adapters": self.owns_adapters,
            "store_open": self.store is not None and self.store.is_open,
            "controller_present": self.controller is not None,
            "supervisor_running": bool(self.supervisor and self.supervisor.running),
            "execution_location": self.execution_location.value,
            "isolation_mode": self.isolation_mode.value,
            "path_boundary_ready": self.tool_runtime_provider is not None,
        }

    async def open(self) -> RuntimeComposition:
        """Open store, recover, adapters, tools, controller and supervisor."""
        if self._opened:
            return self
        if self._closed:
            raise RuntimeError("runtime composition is closed")
        try:
            event_bus = EventBus()
            self.event_bus = event_bus
            store = self._injected_store
            if store is None:
                store = SqliteStore(Path(self.settings.database_path), event_bus=event_bus)
            store.set_event_bus(event_bus)
            await store.open()
            self.store = store
            self.recovery = await recover_after_restart(store)
            from prp_runtime.control.controller import RunController
            from prp_runtime.runtime.supervisor import RunSupervisor
            from prp_runtime.runtime.tooling import ScopeToolRuntimeProvider

            if self._injected_adapters is None:
                self.adapters = build_adapters(self.settings)
            else:
                self.adapters = dict(self._injected_adapters)
            self.tool_runtime_provider = ScopeToolRuntimeProvider(
                store,
                self.settings,
                enable_server_resolver=self.execution_location is not ExecutionLocation.BRIDGE,
            )
            self.controller = RunController(
                store,
                self.settings,
                self.adapters,
                tool_executor_provider=self.tool_runtime_provider.executor_for,
            )
            self.supervisor = RunSupervisor(
                _SqlitePendingRunScanner(store),
                self._execute_persisted,
            )
            await self.supervisor.start(
                recoverable_run_ids=self.recovery.recoverable_run_ids,
                blocked_run_ids=self.recovery.blocked_run_ids,
            )
            self._opened = True
            return self
        except BaseException:
            try:
                await self.close()
            except BaseException:
                pass
            raise

    async def close(self) -> None:
        """Stop owned runtime resources; injected store/adapters stay open."""
        if self._closed:
            return
        close_error: BaseException | None = None
        if self.supervisor is not None:
            try:
                await self.supervisor.stop(drain=True)
            except BaseException as error:
                close_error = error
        if self.tool_runtime_provider is not None:
            try:
                self.tool_runtime_provider.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if self.event_bus is not None:
            try:
                await self.event_bus.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        self.event_bus = None
        if self.store is not None:
            self.store.set_event_bus(None)
        self.supervisor = None
        self.tool_runtime_provider = None
        self.controller = None
        if self.owns_adapters:
            for adapter in self.adapters.values():
                try:
                    await adapter.aclose()
                except BaseException as error:
                    if close_error is None:
                        close_error = error
        if self.owns_store and self.store is not None:
            try:
                await self.store.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        self._closed = True
        if close_error is not None:
            raise close_error

    async def _execute_persisted(self, run_id: str) -> Run:
        from prp_runtime.control.routing import facts_from_request

        store = self.store
        controller = self.controller
        if store is None or controller is None:
            raise RuntimeError("runtime composition is not open")
        run = await store.get_run(run_id)
        try:
            return await controller.execute(
                run_id,
                routing_facts=facts_from_request(run.request),
                principal_id=self.settings.service_principal,
            )
        except PrpError as error:
            current = await store.get_run(run_id)
            if current.status.is_terminal:
                return current
            if current.status is RunStatus.CANCELLING:
                return await controller._finish_run(current, RunStatus.CANCELLED)
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

    async def __aenter__(self) -> RuntimeComposition:
        return await self.open()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        facts = self.public_facts()
        return (
            "RuntimeComposition("
            f"opened={facts['opened']!r}, "
            f"closed={facts['closed']!r}, "
            f"owns_store={facts['owns_store']!r}, "
            f"owns_adapters={facts['owns_adapters']!r})"
        )


async def open_runtime_composition(
    settings: Settings,
    *,
    adapters: Mapping[str, ProviderAdapter] | None = None,
    store: SqliteStore | None = None,
) -> RuntimeComposition:
    """Open one reusable runtime composition without starting HTTP."""
    composition = RuntimeComposition(settings, adapters=adapters, store=store)
    try:
        return await composition.open()
    except BaseException:
        await composition.close()
        raise
