"""The single run controller.

One controller drives every strategy. It is the only writer of run and work unit
state, and every state change it makes is paired with an event in the same
transaction. A worker may produce facts; only the controller decides what those
facts mean for the run.

This version implements the DIRECT strategy: one work unit, one attempt, no
planner and no verifier.
"""

from collections.abc import Mapping

from pydantic import JsonValue

from prp_runtime.domain.enums import (
    ExecutionStrategy,
    ModelRole,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import (
    DomainValidationError,
    ErrorCode,
    ProviderError,
    StateError,
)
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import (
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
    Run,
    WorkUnit,
)
from prp_runtime.domain.transitions import (
    AttemptNotAllowedError,
    resolve_run_outcome,
    transition_run,
    transition_work_unit,
)
from prp_runtime.domain.values import new_run_id, new_work_unit_id, utc_now
from prp_runtime.providers.base import ProviderAdapter
from prp_runtime.runtime.context import build_worker_context
from prp_runtime.runtime.worker import Worker
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

__all__ = ["DIRECT_WORK_UNIT_NAME", "SUPPORTED_STRATEGIES", "RunController"]

DIRECT_WORK_UNIT_NAME = "direct"

#: Strategies this version can execute. A request for any other strategy is
#: refused explicitly instead of being silently downgraded.
SUPPORTED_STRATEGIES: frozenset[ExecutionStrategy] = frozenset({ExecutionStrategy.DIRECT})

_TERMINAL_RUN_EVENTS: dict[RunStatus, EventType] = {
    RunStatus.SUCCEEDED: EventType.RUN_SUCCEEDED,
    RunStatus.FAILED: EventType.RUN_FAILED,
    RunStatus.CANCELLED: EventType.RUN_CANCELLED,
}

_WORK_UNIT_EVENTS: dict[WorkUnitStatus, EventType] = {
    WorkUnitStatus.READY: EventType.WORK_UNIT_READY,
    WorkUnitStatus.RUNNING: EventType.WORK_UNIT_STARTED,
    WorkUnitStatus.SUCCEEDED: EventType.WORK_UNIT_SUCCEEDED,
    WorkUnitStatus.FAILED: EventType.WORK_UNIT_FAILED,
    WorkUnitStatus.CANCELLED: EventType.WORK_UNIT_CANCELLED,
    WorkUnitStatus.BLOCKED: EventType.WORK_UNIT_BLOCKED,
    WorkUnitStatus.INVALIDATED: EventType.WORK_UNIT_INVALIDATED,
}


class RunController:
    """Creates, executes and cancels runs."""

    def __init__(
        self,
        store: SqliteStore,
        settings: Settings,
        adapters: Mapping[str, ProviderAdapter],
    ) -> None:
        self._store = store
        self._settings = settings
        self._adapters = dict(adapters)

    async def create_run(self, request: NativeRunRequest) -> Run:
        """Persist a new run in PENDING and record its creation."""
        run = Run(run_id=new_run_id(), request=request, created_at=utc_now())
        async with self._store.transaction():
            await self._store.create_run(run)
            await self._store.append_event(
                run.run_id,
                EventType.RUN_CREATED,
                {"request": request.model_dump(mode="json")},
            )
        return run

    async def cancel(self, run_id: str) -> Run:
        """Request cancellation.

        A run that has not started is cancelled at once. A running run enters
        CANCELLING, which immediately blocks any further attempt. Cancelling an
        already terminal run changes nothing.
        """
        run = await self._store.get_run(run_id)
        if run.status.is_terminal:
            return run
        if run.status is RunStatus.CANCELLING:
            return run
        if run.status is RunStatus.PENDING:
            return await self._finish_run(run, RunStatus.CANCELLED)
        cancelling = Run.model_validate(
            run.model_dump()
            | {"status": transition_run(run.status, RunStatus.CANCELLING)}
        )
        async with self._store.transaction():
            await self._store.update_run(cancelling)
            await self._store.append_event(run_id, EventType.RUN_CANCELLING, {})
        return cancelling

    async def execute(self, run_id: str) -> Run:
        """Execute a pending run to a terminal status."""
        run = await self._store.get_run(run_id)
        if run.status.is_terminal or run.status is RunStatus.CANCELLING:
            return run
        if run.status is not RunStatus.PENDING:
            raise StateError(
                f"run {run_id} is already {run.status.value}",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        strategy = self._select_strategy(run.request)
        run = await self._start_run(run, strategy)
        return await self._execute_direct(run)

    # --- strategy routing -------------------------------------------------------

    def _select_strategy(self, request: NativeRunRequest) -> ExecutionStrategy:
        """Choose the strategy. AUTO picks the weakest sufficient control level."""
        if request.routing_policy is RoutingPolicy.MANUAL:
            # The request model already guarantees a pinned strategy here.
            if request.strategy is None:
                raise DomainValidationError(
                    "manual routing requires a strategy",
                    code=ErrorCode.INVALID_REQUEST,
                    field="strategy",
                )
            chosen = request.strategy
        else:
            chosen = ExecutionStrategy.DIRECT
        if chosen not in SUPPORTED_STRATEGIES:
            raise DomainValidationError(
                f"strategy {chosen.value} is not available in this version",
                code=ErrorCode.INVALID_REQUEST,
                field="strategy",
            )
        return chosen

    # --- run lifecycle ----------------------------------------------------------

    async def _start_run(self, run: Run, strategy: ExecutionStrategy) -> Run:
        started = Run.model_validate(
            run.model_dump()
            | {
                "status": transition_run(run.status, RunStatus.RUNNING),
                "strategy": strategy,
                "started_at": utc_now(),
            }
        )
        async with self._store.transaction():
            await self._store.append_event(
                run.run_id,
                EventType.STRATEGY_SELECTED,
                {
                    "strategy": strategy.value,
                    "routing_policy": run.request.routing_policy.value,
                    "rationale": self._strategy_rationale(run.request, strategy),
                },
            )
            await self._store.update_run(started)
            await self._store.append_event(run.run_id, EventType.RUN_STARTED, {})
        return started

    @staticmethod
    def _strategy_rationale(request: NativeRunRequest, strategy: ExecutionStrategy) -> str:
        if request.routing_policy is RoutingPolicy.MANUAL:
            return f"the caller pinned {strategy.value}"
        return (
            f"{strategy.value} is the weakest sufficient control level for a single "
            "text request"
        )

    async def _finish_run(
        self, run: Run, status: RunStatus, error: ErrorInfo | None = None
    ) -> Run:
        now = utc_now()
        changes: dict[str, object] = {
            "status": transition_run(run.status, status),
            "completed_at": now,
        }
        if run.started_at is None:
            # A run cancelled before dispatch still needs a start marker: the
            # domain model requires one for every status past PENDING.
            changes["started_at"] = now
        if error is not None:
            changes["error"] = error
        finished = Run.model_validate(run.model_dump() | changes)
        payload: dict[str, JsonValue] = {}
        if error is not None:
            payload["error"] = error.model_dump(mode="json")
        async with self._store.transaction():
            await self._store.update_run(finished)
            await self._store.append_event(
                run.run_id, _TERMINAL_RUN_EVENTS[status], payload
            )
        return finished

    # --- DIRECT -----------------------------------------------------------------

    async def _execute_direct(self, run: Run) -> Run:
        """One work unit, one attempt, no planner and no verifier."""
        work_unit = await self._create_direct_work_unit(run)
        work_unit = await self._advance_work_unit(work_unit, WorkUnitStatus.READY)
        work_unit = await self._advance_work_unit(work_unit, WorkUnitStatus.RUNNING)

        worker = self._worker_for(ModelRole.WORKER)
        context = build_worker_context(work_unit, instructions=run.request.instructions)
        try:
            result = await worker.execute(run=run, work_unit=work_unit, context=context)
        except AttemptNotAllowedError:
            # Cancellation landed between starting the unit and dispatching it.
            await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
            return await self._settle_run(run, [WorkUnitStatus.CANCELLED])

        outcome = (
            WorkUnitStatus.SUCCEEDED if result.succeeded else WorkUnitStatus.FAILED
        )
        await self._advance_work_unit(work_unit, outcome, error=result.error)
        return await self._settle_run(run, [outcome], error=result.error)

    async def _create_direct_work_unit(self, run: Run) -> WorkUnit:
        work_unit = WorkUnit(
            work_unit_id=new_work_unit_id(),
            run_id=run.run_id,
            graph_version=run.graph_version,
            name=DIRECT_WORK_UNIT_NAME,
            instruction=run.request.input,
            output=run.request.output,
            created_at=utc_now(),
        )
        async with self._store.transaction():
            await self._store.create_work_unit(work_unit)
            await self._store.append_event(
                run.run_id,
                EventType.WORK_UNIT_CREATED,
                {"work_unit_id": work_unit.work_unit_id, "name": work_unit.name},
            )
        return work_unit

    async def _advance_work_unit(
        self, work_unit: WorkUnit, status: WorkUnitStatus, error: ErrorInfo | None = None
    ) -> WorkUnit:
        updated = WorkUnit.model_validate(
            work_unit.model_dump()
            | {"status": transition_work_unit(work_unit.status, status)}
        )
        payload: dict[str, JsonValue] = {"work_unit_id": work_unit.work_unit_id}
        if status is WorkUnitStatus.FAILED:
            payload["error"] = (
                error.model_dump(mode="json")
                if error is not None
                else ErrorInfo(
                    category=ErrorCategory.UNKNOWN, message="the work unit failed"
                ).model_dump(mode="json")
            )
        async with self._store.transaction():
            await self._store.update_work_unit(updated)
            await self._store.append_event(
                work_unit.run_id, _WORK_UNIT_EVENTS[status], payload
            )
        return updated

    async def _settle_run(
        self,
        run: Run,
        work_unit_statuses: list[WorkUnitStatus],
        error: ErrorInfo | None = None,
    ) -> Run:
        """Decide the run outcome from its work units and any cancel request."""
        current = await self._store.get_run(run.run_id)
        if current.status.is_terminal:
            return current
        cancel_requested = current.status is RunStatus.CANCELLING
        outcome = resolve_run_outcome(work_unit_statuses, cancel_requested=cancel_requested)
        return await self._finish_run(
            current, outcome, error if outcome is RunStatus.FAILED else None
        )

    # --- workers ----------------------------------------------------------------

    def _worker_for(self, role: ModelRole) -> Worker:
        profile = self._settings.require_profile(role)
        adapter = self._adapters.get(profile.alias)
        if adapter is None:
            raise ProviderError(
                f"no adapter is registered for model alias {profile.alias!r}",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        return Worker(self._store, adapter, profile)
