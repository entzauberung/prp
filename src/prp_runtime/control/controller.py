"""The single run controller.

One controller drives every strategy. It is the only writer of run and work unit
state, and every state change it makes is paired with an event in the same
transaction. A worker may produce facts; only the controller decides what those
facts mean for the run.

DIRECT uses one work unit and one attempt. CASCADE reuses that same lifecycle
with a bounded, ordered worker profile chain; the verifier evaluates every
produced artifact and the controller decides acceptance.
"""

from collections.abc import Awaitable, Callable, Mapping

from pydantic import JsonValue

from prp_runtime.control.budget import (
    check_attempt_budget,
    check_deadline,
    check_token_budget_postflight,
    check_token_budget_preflight,
)
from prp_runtime.control.cascade import (
    CascadeChain,
    CascadeDisposition,
    build_cascade_chain,
    decide_cascade,
    provider_failure_is_retryable,
)
from prp_runtime.control.progressive import (
    RevisionDecision,
    RevisionDisposition,
    RevisionStopReason,
    decide_revision,
)
from prp_runtime.control.routing import RoutingFacts, StrategyDecision, route
from prp_runtime.domain.enums import (
    AttemptStatus,
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
from prp_runtime.domain.events import EventType, payload_from_model
from prp_runtime.domain.models import (
    Attempt,
    ControllerAction,
    ControllerDecision,
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
    Run,
    VerificationResult,
    WorkUnit,
)
from prp_runtime.domain.transitions import (
    AttemptNotAllowedError,
    resolve_run_outcome,
    transition_attempt,
    transition_run,
    transition_work_unit,
)
from prp_runtime.domain.values import new_attempt_id, new_run_id, new_work_unit_id, utc_now
from prp_runtime.planning.compiler import CompiledPlan, compile_plan
from prp_runtime.planning.frontier import compute_frontier
from prp_runtime.planning.models import (
    PlanProposal,
    PlanRejection,
    PlanRevision,
    PlanRevisionReason,
)
from prp_runtime.planning.planner import (
    Planner,
    PlannerCallResult,
    new_planning_work_unit,
)
from prp_runtime.providers.base import ModelProfile, ProviderAdapter
from prp_runtime.runtime.context import DependencyArtifact, build_worker_context
from prp_runtime.runtime.scheduler import (
    PlannedExecutor,
    Scheduler,
    WaveOutcome,
    WaveResult,
    WaveStatus,
)
from prp_runtime.runtime.worker import Worker
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.verification.rules import plan_for_output
from prp_runtime.verification.verifier import RuleVerifier

__all__ = ["DIRECT_WORK_UNIT_NAME", "SUPPORTED_STRATEGIES", "RunController"]

DIRECT_WORK_UNIT_NAME = "direct"

#: Strategies this version can execute. A request for any other strategy is
#: refused explicitly instead of being silently downgraded.
SUPPORTED_STRATEGIES: frozenset[ExecutionStrategy] = frozenset(
    {
        ExecutionStrategy.DIRECT,
        ExecutionStrategy.CASCADE,
        ExecutionStrategy.PLANNED,
        ExecutionStrategy.PROGRESSIVE,
    }
)

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

    async def commit_plan(
        self,
        run_id: str,
        result: CompiledPlan | PlanRejection,
        *,
        target_graph_version: int,
        revision: PlanRevision | None = None,
    ) -> tuple[WorkUnit, ...] | PlanRejection:
        """Commit one compiled graph or record its structured rejection."""
        run = await self._store.get_run(run_id)
        rejection: PlanRejection | None = None
        if isinstance(result, PlanRejection):
            rejection = result
        elif result.run_id != run_id:
            rejection = PlanRejection(
                summary="The compiled plan was rejected",
                reasons=("compiled plan run_id does not match the target run",),
            )
        elif result.graph_version != target_graph_version:
            rejection = PlanRejection(
                summary="The compiled plan was rejected",
                reasons=("compiled plan graph_version does not match the target version",),
            )
        elif target_graph_version <= run.graph_version:
            rejection = PlanRejection(
                summary="The compiled plan was rejected",
                reasons=("target graph_version must be greater than the current version",),
            )
        elif run.status.is_terminal:
            rejection = PlanRejection(
                summary="The compiled plan was rejected",
                reasons=("a terminal run cannot accept a new graph",),
            )

        if rejection is not None:
            await self._record_plan_rejection(
                run, target_graph_version, rejection
            )
            return rejection

        assert isinstance(result, CompiledPlan)
        created_at = utc_now()
        work_units = tuple(
            WorkUnit(
                work_unit_id=draft.work_unit_id,
                run_id=draft.run_id,
                graph_version=draft.graph_version,
                name=draft.name,
                instruction=draft.instruction,
                acceptance_criteria=draft.acceptance_criteria,
                output=draft.output,
                depends_on=draft.depends_on,
                resource_claims=draft.resource_claims,
                created_at=created_at,
            )
            for draft in result.nodes
        )
        updated_run = Run.model_validate(
            run.model_dump() | {"graph_version": target_graph_version}
        )
        decision = ControllerDecision(
            run_id=run.run_id,
            action=(
                ControllerAction.REVISE_PLAN
                if revision is not None
                else ControllerAction.COMMIT_PLAN
            ),
            rationale=(
                f"compiled graph version {target_graph_version} passed validation"
            ),
        )
        try:
            async with self._store.transaction():
                if revision is not None:
                    await self._store.append_event(
                        run.run_id,
                        EventType.PLAN_REVISED,
                        {
                            "graph_version": target_graph_version,
                            "base_graph_version": revision.base_graph_version,
                            "reason": revision.reason.value,
                            "summary": revision.summary,
                        },
                    )
                await self._store.append_event(
                    run.run_id,
                    EventType.PLAN_PROPOSED,
                    {
                        "graph_version": target_graph_version,
                        "node_count": len(work_units),
                    },
                )
                await self._store.create_graph(work_units)
                for work_unit in work_units:
                    await self._store.append_event(
                        run.run_id,
                        EventType.WORK_UNIT_CREATED,
                        {
                            "work_unit_id": work_unit.work_unit_id,
                            "name": work_unit.name,
                            "graph_version": target_graph_version,
                        },
                    )
                await self._store.update_run(updated_run)
                await self._store.append_event(
                    run.run_id,
                    EventType.CONTROLLER_DECISION,
                    payload_from_model("decision", decision),
                )
                await self._store.append_event(
                    run.run_id,
                    EventType.PLAN_COMMITTED,
                    {
                        "graph_version": target_graph_version,
                        "work_unit_ids": [unit.work_unit_id for unit in work_units],
                    },
                )
        except StateError:
            rejection = PlanRejection(
                summary="The compiled plan could not be committed",
                reasons=("the graph write violated persisted state",),
            )
            await self._record_plan_rejection(run, target_graph_version, rejection)
            return rejection
        return work_units

    async def commit_revision(
        self,
        run_id: str,
        revision: PlanRevision,
        *,
        target_graph_version: int | None = None,
    ) -> tuple[WorkUnit, ...] | PlanRejection:
        """Compile and atomically commit one strictly next graph version."""
        run = await self._store.get_run(run_id)
        target = (
            revision.base_graph_version + 1
            if target_graph_version is None
            else target_graph_version
        )
        rejection: PlanRejection | None = None
        if run.status.is_terminal:
            rejection = PlanRejection(
                summary="The plan revision was rejected",
                reasons=("a terminal run cannot accept a new graph",),
            )
        elif revision.base_graph_version != run.graph_version:
            rejection = PlanRejection(
                summary="The plan revision was rejected",
                reasons=("revision base_graph_version is stale",),
            )
        elif target != revision.base_graph_version + 1:
            rejection = PlanRejection(
                summary="The plan revision was rejected",
                reasons=("a revision must advance exactly one graph version",),
            )
        if rejection is not None:
            await self._record_plan_rejection(run, target, rejection)
            return rejection

        compiled = compile_plan(
            revision.proposal,
            run_id=run_id,
            graph_version=target,
        )
        return await self.commit_plan(
            run_id,
            compiled,
            target_graph_version=target,
            revision=revision,
        )

    async def dispatch_planned_wave(
        self,
        run_id: str,
        *,
        scheduler: Scheduler,
        execute: PlannedExecutor,
        graph_version: int | None = None,
        max_concurrency: int | None = None,
        settle_run: bool = True,
    ) -> WaveResult:
        """Dispatch one committed frontier wave through Controller callbacks."""
        run = await self._store.get_run(run_id)
        version = run.graph_version if graph_version is None else graph_version
        work_units = await self._store.list_work_units(
            run_id, graph_version=version
        )

        if run.status is RunStatus.CANCELLED:
            await self._cancel_planned_units(work_units)
            return WaveResult(graph_version=version, status=WaveStatus.CANCELLED)
        if run.status is RunStatus.SUCCEEDED:
            return WaveResult(graph_version=version, status=WaveStatus.COMPLETE)
        if run.status is RunStatus.FAILED:
            return WaveResult(graph_version=version, status=WaveStatus.BLOCKED)
        if run.status is RunStatus.PENDING:
            run = await self._start_run(run, ExecutionStrategy.PLANNED)
        if run.status is RunStatus.CANCELLING:
            await self._cancel_planned_units(work_units)
            current = await self._store.get_run(run_id)
            if current.status is RunStatus.CANCELLING:
                await self._finish_run(current, RunStatus.CANCELLED)
            return WaveResult(graph_version=version, status=WaveStatus.CANCELLED)

        async def on_started(unit: WorkUnit) -> WorkUnit | None:
            current = await self._store.get_run(run_id)
            if current.status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
                if not unit.status.is_terminal:
                    await self._cancel_planned_unit(unit)
                return None
            ready = unit
            if unit.status is WorkUnitStatus.PENDING:
                ready = await self._advance_work_unit(unit, WorkUnitStatus.READY)
            current = await self._store.get_run(run_id)
            if current.status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
                if not ready.status.is_terminal:
                    await self._cancel_planned_unit(ready)
                return None
            return await self._advance_work_unit(ready, WorkUnitStatus.RUNNING)

        async def on_succeeded(unit: WorkUnit) -> None:
            await self._advance_work_unit(unit, WorkUnitStatus.SUCCEEDED)

        async def on_failed(unit: WorkUnit, error: ErrorInfo) -> None:
            current = await self._store.get_run(run_id)
            if current.status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
                await self._cancel_planned_unit(unit)
                return
            await self._advance_work_unit(unit, WorkUnitStatus.FAILED, error=error)

        concurrency = max_concurrency
        if concurrency is None:
            concurrency = run.request.budget.max_concurrency or 1
        result = await scheduler.run_wave(
            work_units,
            graph_version=version,
            max_concurrency=concurrency,
            execute=execute,
            on_started=on_started,
            on_succeeded=on_succeeded,
            on_failed=on_failed,
        )
        current = await self._store.get_run(run_id)
        if current.status in (RunStatus.CANCELLING, RunStatus.CANCELLED) or (
            result.status is WaveStatus.CANCELLED
        ):
            await self._cancel_planned_units(
                await self._store.list_work_units(run_id, graph_version=version)
            )
            current = await self._store.get_run(run_id)
            if current.status is RunStatus.CANCELLING:
                await self._finish_run(current, RunStatus.CANCELLED)
            return result.model_copy(update={"status": WaveStatus.CANCELLED})

        await self._propagate_planned_blocked(run_id, version)
        if settle_run:
            await self._settle_planned_run(run_id, version)
        return result

    async def _cancel_planned_unit(self, unit: WorkUnit) -> None:
        """Cancel one non-terminal unit while preserving the event ledger."""
        if unit.status.is_terminal:
            return
        await self._advance_work_unit(unit, WorkUnitStatus.CANCELLED)

    async def _cancel_planned_units(self, units: tuple[WorkUnit, ...]) -> None:
        """Cancel every still-runnable unit in a committed graph version."""
        for unit in units:
            await self._cancel_planned_unit(unit)

    async def _propagate_planned_blocked(
        self, run_id: str, graph_version: int
    ) -> tuple[WorkUnit, ...]:
        """Materialize dependency failures as stable BLOCKED work-unit facts."""
        while True:
            units = await self._store.list_work_units(
                run_id, graph_version=graph_version
            )
            frontier = compute_frontier(units, graph_version=graph_version)
            by_id = {unit.work_unit_id: unit for unit in units}
            changed = False
            for detail in frontier.blocked_details:
                unit = by_id[detail.work_unit_id]
                if unit.status not in (WorkUnitStatus.PENDING, WorkUnitStatus.READY):
                    continue
                dependency_suffix = ", ".join(detail.dependency_ids)
                reason = detail.reason.value
                if dependency_suffix:
                    reason = f"{reason}: {dependency_suffix}"
                await self._advance_work_unit(
                    unit, WorkUnitStatus.BLOCKED, reason=reason
                )
                changed = True
            if not changed:
                return units

    async def _settle_planned_run(self, run_id: str, graph_version: int) -> Run:
        """Settle a planned run only after its current graph has no work left."""
        run = await self._store.get_run(run_id)
        if run.status.is_terminal:
            return run
        units = await self._store.list_work_units(run_id, graph_version=graph_version)
        frontier = compute_frontier(units, graph_version=graph_version)
        if frontier.ready or frontier.waiting:
            return run
        if not units:
            return run
        statuses = tuple(unit.status for unit in units)
        if run.status is RunStatus.CANCELLING:
            await self._cancel_planned_units(units)
            return await self._finish_run(
                await self._store.get_run(run_id), RunStatus.CANCELLED
            )
        if any(
            status in (WorkUnitStatus.FAILED, WorkUnitStatus.BLOCKED, WorkUnitStatus.INVALIDATED)
            for status in statuses
        ):
            failed_ids = tuple(
                unit.work_unit_id
                for unit in units
                if unit.status
                in (WorkUnitStatus.FAILED, WorkUnitStatus.BLOCKED, WorkUnitStatus.INVALIDATED)
            )
            error: ErrorInfo | None = None
            failed_id_set = set(failed_ids)
            for event in reversed(await self._store.list_events(run_id)):
                if event.event_type is not EventType.WORK_UNIT_FAILED:
                    continue
                if event.payload.get("work_unit_id") not in failed_id_set:
                    continue
                error = ErrorInfo.model_validate(event.payload["error"])
                break
            if error is None:
                error = ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="planned graph failed: " + ", ".join(failed_ids),
                )
            return await self._finish_run(run, RunStatus.FAILED, error=error)
        if all(status is WorkUnitStatus.SUCCEEDED for status in statuses):
            return await self._finish_run(run, RunStatus.SUCCEEDED)
        if any(status is WorkUnitStatus.CANCELLED for status in statuses):
            return await self._finish_run(run, RunStatus.CANCELLED)
        return run

    async def _record_plan_rejection(
        self,
        run: Run,
        target_graph_version: int,
        rejection: PlanRejection,
    ) -> None:
        rationale = "; ".join(rejection.reasons)
        decision = ControllerDecision(
            run_id=run.run_id,
            action=ControllerAction.REJECT_PLAN,
            rationale=rationale,
        )
        async with self._store.transaction():
            await self._store.append_event(
                run.run_id,
                EventType.PLAN_PROPOSED,
                {"graph_version": target_graph_version, "node_count": 0},
            )
            await self._store.append_event(
                run.run_id,
                EventType.CONTROLLER_DECISION,
                payload_from_model("decision", decision),
            )
            await self._store.append_event(
                run.run_id,
                EventType.PLAN_REJECTED,
                {
                    "graph_version": target_graph_version,
                    "reason": rejection.summary,
                    "reasons": list(rejection.reasons),
                },
            )

    async def execute(
        self,
        run_id: str,
        *,
        routing_facts: RoutingFacts | None = None,
    ) -> Run:
        """Execute a pending run to a terminal status."""
        run = await self._store.get_run(run_id)
        if run.status.is_terminal or run.status is RunStatus.CANCELLING:
            return run
        if run.status is not RunStatus.PENDING:
            raise StateError(
                f"run {run_id} is already {run.status.value}",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        selection = self._select_strategy(run.request, facts=routing_facts)
        assert selection.strategy is not None
        strategy = selection.strategy
        cascade_chain: CascadeChain | None = None
        cascade_workers: tuple[Worker, ...] | None = None
        planner: Planner | None = None
        planned_worker: Worker | None = None
        if strategy is ExecutionStrategy.CASCADE:
            cascade_chain = build_cascade_chain(
                self._settings.require_profile(ModelRole.WORKER),
                self._settings.cascade_profiles,
            )
            # Validate the complete finite chain before starting the run. A
            # missing adapter is configuration failure, never a partial run.
            cascade_workers = tuple(
                self._worker_for_profile(profile) for profile in cascade_chain
            )
        elif strategy in (ExecutionStrategy.PLANNED, ExecutionStrategy.PROGRESSIVE):
            planner = self._planner()
            planned_worker = self._worker_for(ModelRole.WORKER)
        run = await self._start_run(run, strategy, selection=selection)
        if strategy is ExecutionStrategy.CASCADE:
            assert cascade_chain is not None
            assert cascade_workers is not None
            return await self._execute_cascade(run, cascade_chain, cascade_workers)
        if strategy is ExecutionStrategy.PLANNED:
            assert planner is not None
            assert planned_worker is not None
            return await self._execute_planned(run, planner, planned_worker)
        if strategy is ExecutionStrategy.PROGRESSIVE:
            assert planner is not None
            assert planned_worker is not None
            return await self._execute_progressive(run, planner, planned_worker)
        return await self._execute_direct(run)

    # --- strategy routing -------------------------------------------------------

    def _select_strategy(
        self,
        request: NativeRunRequest,
        *,
        facts: RoutingFacts | None = None,
    ) -> StrategyDecision:
        """Choose a strategy through the deterministic routing contract."""
        decision = route(
            request,
            facts=facts,
            available_strategies=SUPPORTED_STRATEGIES,
        )
        if decision.rejected:
            assert decision.rejection is not None
            raise DomainValidationError(
                f"{decision.reason} ({decision.rejection.code.value})",
                code=ErrorCode.INVALID_REQUEST,
                field=(
                    "strategy"
                    if request.routing_policy is RoutingPolicy.MANUAL
                    else "routing_policy"
                ),
            )
        assert decision.strategy is not None
        return decision

    # --- run lifecycle ----------------------------------------------------------

    async def _start_run(
        self,
        run: Run,
        strategy: ExecutionStrategy,
        *,
        selection: StrategyDecision | None = None,
    ) -> Run:
        rationale = (
            selection.reason
            if selection is not None
            else self._strategy_rationale(run.request, strategy)
        )
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
                    "rationale": rationale,
                },
            )
            await self._store.append_event(
                run.run_id,
                EventType.CONTROLLER_DECISION,
                payload_from_model(
                    "decision",
                    ControllerDecision(
                        run_id=run.run_id,
                        action=ControllerAction.SELECT_STRATEGY,
                        rationale=rationale,
                        to_strategy=strategy,
                    ),
                ),
            )
            await self._store.update_run(started)
            await self._store.append_event(run.run_id, EventType.RUN_STARTED, {})
        return started

    @staticmethod
    def _strategy_rationale(request: NativeRunRequest, strategy: ExecutionStrategy) -> str:
        if request.routing_policy is RoutingPolicy.MANUAL:
            return f"the caller pinned {strategy.value}"
        if strategy is ExecutionStrategy.PLANNED:
            return "a committed work graph requires PLANNED execution"
        if strategy is ExecutionStrategy.PROGRESSIVE:
            return "evidence-gated revision requires PROGRESSIVE execution"
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
        """One work unit, one attempt. Budget is checked before and after dispatch."""
        work_unit = await self._create_direct_work_unit(run)
        work_unit = await self._advance_work_unit(work_unit, WorkUnitStatus.READY)

        # preflight: deadline and attempt count before any provider call
        preflight = await self._preflight_budget(run)
        if preflight is not None:
            # READY cannot transition to FAILED; cancel the unit instead
            await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
            return await self._settle_run(run, [WorkUnitStatus.CANCELLED], error=preflight)

        work_unit = await self._advance_work_unit(work_unit, WorkUnitStatus.RUNNING)

        worker = self._worker_for(ModelRole.WORKER)
        context = build_worker_context(work_unit, instructions=run.request.instructions)
        try:
            result = await worker.execute(run=run, work_unit=work_unit, context=context)
        except AttemptNotAllowedError:
            # Cancellation landed between starting the unit and dispatching it.
            await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
            return await self._settle_run(run, [WorkUnitStatus.CANCELLED])

        if not result.succeeded or result.artifact is None:
            await self._advance_work_unit(work_unit, WorkUnitStatus.FAILED, error=result.error)
            return await self._settle_run(run, [WorkUnitStatus.FAILED], error=result.error)

        # postflight: token budget after usage is recorded
        postflight = await self._postflight_budget(run)
        if postflight is not None:
            await self._advance_work_unit(work_unit, WorkUnitStatus.FAILED, error=postflight)
            return await self._settle_run(run, [WorkUnitStatus.FAILED], error=postflight)

        report = await RuleVerifier().verify_and_record(
            self._store,
            result.artifact,
            plan_for_output(work_unit.output),
        )

        evidence_ids = await self._record_evidence_events(
            run, work_unit, result.artifact.artifact_id
        )

        if report.result is VerificationResult.PASS:
            action = ControllerAction.ACCEPT_ARTIFACT
            rationale = report.summary()
            verify_error: ErrorInfo | None = None
        else:
            action = ControllerAction.REJECT_ARTIFACT
            rationale = report.summary()
            verify_error = ErrorInfo(
                category=ErrorCategory.VERIFICATION_FAILED, message=rationale
            )

        decision = ControllerDecision(
            run_id=run.run_id,
            action=action,
            rationale=rationale,
            work_unit_id=work_unit.work_unit_id,
            evidence_ids=evidence_ids,
        )
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload_from_model("decision", decision),
        )

        outcome = (
            WorkUnitStatus.SUCCEEDED
            if report.result is VerificationResult.PASS
            else WorkUnitStatus.FAILED
        )
        await self._advance_work_unit(work_unit, outcome, error=verify_error)
        return await self._settle_run(run, [outcome], error=verify_error)

    async def _execute_cascade(
        self,
        run: Run,
        chain: CascadeChain,
        workers: tuple[Worker, ...],
    ) -> Run:
        """Run the finite CASCADE chain until verification passes or it is exhausted."""
        work_unit = await self._create_work_unit(run, name="cascade")
        work_unit = await self._advance_work_unit(work_unit, WorkUnitStatus.READY)
        last_error: ErrorInfo | None = None
        pending_escalation: (
            tuple[ModelProfile, ModelProfile, str, tuple[str, ...]] | None
        ) = None

        for attempt_index, (profile, worker) in enumerate(
            zip(chain, workers, strict=True), start=1
        ):
            preflight = await self._preflight_budget(run)
            if preflight is not None:
                terminal = (
                    WorkUnitStatus.CANCELLED
                    if work_unit.status is WorkUnitStatus.READY
                    else WorkUnitStatus.FAILED
                )
                await self._advance_work_unit(work_unit, terminal, error=preflight)
                return await self._settle_run(run, [terminal], error=preflight)

            current_run = await self._store.get_run(run.run_id)
            if current_run.status is RunStatus.CANCELLING:
                await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
                return await self._settle_run(current_run, [WorkUnitStatus.CANCELLED])

            if pending_escalation is not None:
                from_profile, to_profile, rationale, evidence_ids = pending_escalation
                await self._record_cascade_escalation(
                    run,
                    work_unit,
                    from_profile,
                    to_profile,
                    rationale,
                    evidence_ids=evidence_ids,
                )
                pending_escalation = None

            if work_unit.status is WorkUnitStatus.READY:
                work_unit = await self._advance_work_unit(
                    work_unit, WorkUnitStatus.RUNNING
                )

            context = build_worker_context(
                work_unit, instructions=run.request.instructions
            )
            try:
                result = await worker.execute(
                    run=current_run,
                    work_unit=work_unit,
                    context=context,
                    attempt_index=attempt_index,
                )
            except AttemptNotAllowedError:
                await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
                return await self._settle_run(
                    current_run, [WorkUnitStatus.CANCELLED]
                )

            if result.succeeded and result.artifact is not None:
                postflight = await self._postflight_budget(run)
                if postflight is not None:
                    await self._advance_work_unit(
                        work_unit, WorkUnitStatus.FAILED, error=postflight
                    )
                    return await self._settle_run(
                        run, [WorkUnitStatus.FAILED], error=postflight
                    )

                report = await RuleVerifier().verify_and_record(
                    self._store,
                    result.artifact,
                    plan_for_output(work_unit.output),
                )
                evidence_ids = await self._record_evidence_events(
                    run, work_unit, result.artifact.artifact_id
                )
                policy = decide_cascade(
                    verification_result=report.result,
                    has_next_profile=attempt_index < len(chain),
                )

                if policy.disposition is CascadeDisposition.ACCEPT:
                    action = ControllerAction.ACCEPT_ARTIFACT
                    verify_error = None
                else:
                    action = ControllerAction.REJECT_ARTIFACT
                    verify_error = ErrorInfo(
                        category=ErrorCategory.VERIFICATION_FAILED,
                        message=report.summary(),
                    )
                decision = ControllerDecision(
                    run_id=run.run_id,
                    action=action,
                    rationale=report.summary(),
                    work_unit_id=work_unit.work_unit_id,
                    evidence_ids=evidence_ids,
                )
                await self._store.append_event(
                    run.run_id,
                    EventType.CONTROLLER_DECISION,
                    payload_from_model("decision", decision),
                )

                if policy.disposition is CascadeDisposition.ACCEPT:
                    work_unit = await self._advance_work_unit(
                        work_unit, WorkUnitStatus.SUCCEEDED
                    )
                    return await self._settle_run(run, [WorkUnitStatus.SUCCEEDED])
                last_error = verify_error
                current_run = await self._store.get_run(run.run_id)
                if current_run.status is RunStatus.CANCELLING:
                    await self._advance_work_unit(
                        work_unit, WorkUnitStatus.CANCELLED
                    )
                    return await self._settle_run(
                        current_run, [WorkUnitStatus.CANCELLED]
                    )
                if policy.disposition is CascadeDisposition.ESCALATE:
                    pending_escalation = (
                        profile,
                        chain[attempt_index],
                        policy.rationale,
                        evidence_ids,
                    )
                    continue
            else:
                last_error = result.error
                current_run = await self._store.get_run(run.run_id)
                if current_run.status is RunStatus.CANCELLING:
                    await self._advance_work_unit(
                        work_unit, WorkUnitStatus.CANCELLED
                    )
                    return await self._settle_run(
                        current_run, [WorkUnitStatus.CANCELLED]
                    )
                policy = decide_cascade(
                    provider_retryable=provider_failure_is_retryable(result.error),
                    has_next_profile=attempt_index < len(chain),
                )
                if policy.disposition is CascadeDisposition.ESCALATE:
                    pending_escalation = (
                        profile,
                        chain[attempt_index],
                        policy.rationale,
                        (),
                    )
                    continue
            break

        await self._advance_work_unit(
            work_unit, WorkUnitStatus.FAILED, error=last_error
        )
        return await self._settle_run(run, [WorkUnitStatus.FAILED], error=last_error)

    async def _execute_planned(
        self,
        run: Run,
        planner: Planner,
        worker: Worker,
    ) -> Run:
        """Plan, compile, commit, and execute one bounded graph to a terminal run."""
        preflight = await self._preflight_budget(run)
        if preflight is not None:
            return await self._finish_run(run, RunStatus.FAILED, error=preflight)

        planner_result, postflight = await self._execute_planner_propose(run, planner)
        if postflight is not None:
            return await self._finish_run(
                await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=postflight,
            )
        current = await self._store.get_run(run.run_id)
        if current.status is RunStatus.CANCELLING:
            return await self._finish_run(current, RunStatus.CANCELLED)

        target_graph_version = run.graph_version + 1
        proposal: PlanProposal | PlanRejection
        if planner_result.rejection is not None:
            proposal = planner_result.rejection
        else:
            assert isinstance(planner_result.proposal, PlanProposal)
            proposal = planner_result.proposal
        compiled = (
            proposal
            if isinstance(proposal, PlanRejection)
            else compile_plan(
                proposal,
                run_id=run.run_id,
                graph_version=target_graph_version,
            )
        )
        committed = await self.commit_plan(
            run.run_id,
            compiled,
            target_graph_version=target_graph_version,
        )
        if isinstance(committed, PlanRejection):
            error = ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message=(
                    f"{committed.summary}: " + "; ".join(committed.reasons)
                ),
            )
            return await self._finish_run(
                await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=error,
            )

        scheduler = Scheduler()

        async def execute_unit(unit: WorkUnit) -> WaveOutcome:
            return await self._execute_planned_work_unit(
                run.run_id,
                unit,
                worker,
            )

        # Every dispatched wave terminalizes at least one current-graph unit.
        # The extra iteration is a defensive bound for an immediately empty or
        # blocked frontier; no unbounded scheduling loop is permitted.
        for _ in range(len(committed) + 1):
            await self.dispatch_planned_wave(
                run.run_id,
                scheduler=scheduler,
                execute=execute_unit,
                graph_version=target_graph_version,
            )
            current = await self._store.get_run(run.run_id)
            if current.status.is_terminal:
                return current

        error = ErrorInfo(
            category=ErrorCategory.UNKNOWN,
            message="planned graph did not converge within its bounded wave count",
        )
        return await self._finish_run(
            await self._store.get_run(run.run_id),
            RunStatus.FAILED,
            error=error,
        )

    async def _execute_planned_work_unit(
        self,
        run_id: str,
        work_unit: WorkUnit,
        worker: Worker,
    ) -> WaveOutcome:
        """Execute and deterministically verify one Controller-started graph unit."""
        run = await self._store.get_run(run_id)
        if run.status is RunStatus.CANCELLING:
            return WaveOutcome.failure(
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="run cancellation prevented planned dispatch",
                )
            )
        preflight = await self._preflight_budget(run)
        if preflight is not None:
            return WaveOutcome.failure(preflight)

        dependencies: list[DependencyArtifact] = []
        for dependency_id in work_unit.depends_on:
            dependency = await self._store.get_work_unit(dependency_id)
            artifacts = await self._store.list_artifacts(dependency_id)
            if not artifacts:
                return WaveOutcome.failure(
                    ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message=(
                            "succeeded dependency has no persisted artifact: "
                            f"{dependency_id}"
                        ),
                    )
                )
            dependencies.extend(
                DependencyArtifact(
                    work_unit_name=dependency.name,
                    artifact_name=artifact.name,
                    content=artifact.content,
                )
                for artifact in artifacts
            )

        context = build_worker_context(
            work_unit,
            instructions=run.request.instructions,
            dependencies=tuple(dependencies),
        )
        try:
            result = await worker.execute(
                run=run,
                work_unit=work_unit,
                context=context,
            )
        except AttemptNotAllowedError:
            return WaveOutcome.failure(
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="run cancellation prevented planned attempt",
                )
            )
        if not result.succeeded or result.artifact is None:
            return WaveOutcome.failure(
                result.error
                or ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="planned worker failed without a classified error",
                )
            )

        postflight = await self._postflight_budget(run)
        if postflight is not None:
            return WaveOutcome.failure(postflight)

        report = await RuleVerifier().verify_and_record(
            self._store,
            result.artifact,
            plan_for_output(work_unit.output),
        )
        evidence_ids = await self._record_evidence_events(
            run,
            work_unit,
            result.artifact.artifact_id,
        )
        accepted = report.result is VerificationResult.PASS
        error = None
        if not accepted:
            error = ErrorInfo(
                category=ErrorCategory.VERIFICATION_FAILED,
                message=report.summary(),
            )
        decision = ControllerDecision(
            run_id=run.run_id,
            action=(
                ControllerAction.ACCEPT_ARTIFACT
                if accepted
                else ControllerAction.REJECT_ARTIFACT
            ),
            rationale=report.summary(),
            work_unit_id=work_unit.work_unit_id,
            evidence_ids=evidence_ids,
        )
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload_from_model("decision", decision),
        )
        if error is not None:
            return WaveOutcome.failure(error)
        return WaveOutcome.success()

    async def _execute_progressive(
        self,
        run: Run,
        planner: Planner,
        worker: Worker,
    ) -> Run:
        """Execute bounded graph revisions until evidence or budget settles the run."""
        preflight = await self._preflight_budget(run)
        if preflight is not None:
            return await self._finish_run(run, RunStatus.FAILED, error=preflight)

        planner_result, postflight = await self._execute_planner_propose(run, planner)
        if postflight is not None:
            return await self._finish_run(
                await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=postflight,
            )
        current = await self._store.get_run(run.run_id)
        if current.status is RunStatus.CANCELLING:
            return await self._finish_run(current, RunStatus.CANCELLED)
        target_graph_version = run.graph_version + 1
        proposal: PlanProposal | PlanRejection
        if planner_result.rejection is not None:
            proposal = planner_result.rejection
        else:
            assert isinstance(planner_result.proposal, PlanProposal)
            proposal = planner_result.proposal
        compiled = (
            proposal
            if isinstance(proposal, PlanRejection)
            else compile_plan(
                proposal,
                run_id=run.run_id,
                graph_version=target_graph_version,
            )
        )
        committed = await self.commit_plan(
            run.run_id,
            compiled,
            target_graph_version=target_graph_version,
        )
        if isinstance(committed, PlanRejection):
            error = ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message=f"{committed.summary}: " + "; ".join(committed.reasons),
            )
            return await self._finish_run(
                await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=error,
            )

        scheduler = Scheduler()
        revision_count = 0
        max_rounds = (
            (run.request.budget.max_plan_revisions + 1)
            if run.request.budget.max_plan_revisions is not None
            else 1
        )
        for _ in range(max_rounds):
            graph = await self._store.list_work_units(
                run.run_id,
                graph_version=(await self._store.get_run(run.run_id)).graph_version,
            )
            converged = False
            for _ in range(len(graph) + 1):
                current = await self._store.get_run(run.run_id)
                if current.status.is_terminal:
                    return current
                await self.dispatch_planned_wave(
                    run.run_id,
                    scheduler=scheduler,
                    execute=lambda unit: self._execute_planned_work_unit(
                        run.run_id, unit, worker
                    ),
                    graph_version=current.graph_version,
                    settle_run=False,
                )
                current = await self._store.get_run(run.run_id)
                if current.status.is_terminal:
                    return current
                graph = await self._store.list_work_units(
                    run.run_id, graph_version=current.graph_version
                )
                frontier = compute_frontier(
                    graph, graph_version=current.graph_version
                )
                if not frontier.ready and not frontier.waiting:
                    converged = True
                    break
            if not converged:
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message="Progressive graph did not converge within its wave bound",
                    ),
                )

            graph = await self._store.list_work_units(
                run.run_id, graph_version=current.graph_version
            )
            if all(unit.status is WorkUnitStatus.SUCCEEDED for unit in graph):
                return await self._settle_planned_run(
                    run.run_id, current.graph_version
                )
            verification_result, failure_error = (
                await self._progressive_failure_signal(run.run_id, graph)
            )
            decision = decide_revision(
                budget=run.request.budget,
                revision_count=revision_count,
                graph_version=current.graph_version,
                verification_result=verification_result,
                error=None if verification_result is not None else failure_error,
                run_status=current.status,
                usage=current.usage,
            )
            if decision.disposition is RevisionDisposition.STOP:
                return await self._finish_progressive_stop(
                    current,
                    decision,
                    failure_error=failure_error,
                )

            assert decision.reason is not None
            latest = await self._store.get_run(run.run_id)
            if latest.status is RunStatus.CANCELLING:
                return await self._finish_run(latest, RunStatus.CANCELLED)
            preflight = await self._preflight_budget(latest)
            if preflight is not None:
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=preflight,
                )
            revision_result, postflight = await self._execute_planner_revise(
                latest,
                planner,
                base_graph_version=current.graph_version,
                reason=decision.reason,
                feedback=decision.rationale,
            )
            if postflight is not None:
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=postflight,
                )
            if revision_result.rejection is not None:
                rejection_error = revision_result.error or ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="Planner revision failed without a classified error",
                )
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=rejection_error,
                )
            assert isinstance(revision_result.proposal, PlanRevision)
            committed_revision = await self.commit_revision(
                run.run_id,
                revision_result.proposal,
            )
            if isinstance(committed_revision, PlanRejection):
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message=(
                            f"{committed_revision.summary}: "
                            + "; ".join(committed_revision.reasons)
                        ),
                    ),
                )
            revision_count += 1

        return await self._finish_run(
            await self._store.get_run(run.run_id),
            RunStatus.FAILED,
            error=ErrorInfo(
                category=ErrorCategory.BUDGET_EXCEEDED,
                message="the Progressive revision loop exhausted its hard bound",
            ),
        )

    async def _progressive_failure_signal(
        self,
        run_id: str,
        graph: tuple[WorkUnit, ...],
    ) -> tuple[VerificationResult | None, ErrorInfo | None]:
        """Recover one current-graph verdict/error without reading model text."""
        failed_ids = {
            unit.work_unit_id
            for unit in graph
            if unit.status is WorkUnitStatus.FAILED
        }
        evidence_results: list[VerificationResult] = []
        for work_unit_id in failed_ids:
            evidence_results.extend(
                row.result
                for row in await self._store.list_evidence(work_unit_id)
            )
        if evidence_results:
            if VerificationResult.FAIL in evidence_results:
                result = VerificationResult.FAIL
            elif VerificationResult.INCONCLUSIVE in evidence_results:
                result = VerificationResult.INCONCLUSIVE
            else:
                result = VerificationResult.PASS
            error = await self._failure_event_error(run_id, failed_ids)
            return result, error
        error = await self._failure_event_error(run_id, failed_ids)
        if error is None and any(unit.status is WorkUnitStatus.BLOCKED for unit in graph):
            error = ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message="the current Progressive graph contains blocked work",
            )
        return None, error

    async def _failure_event_error(
        self, run_id: str, work_unit_ids: set[str]
    ) -> ErrorInfo | None:
        for event in reversed(await self._store.list_events(run_id)):
            if event.event_type is not EventType.WORK_UNIT_FAILED:
                continue
            if event.payload.get("work_unit_id") not in work_unit_ids:
                continue
            return ErrorInfo.model_validate(event.payload["error"])
        return None

    async def _finish_progressive_stop(
        self,
        run: Run,
        decision: RevisionDecision,
        *,
        failure_error: ErrorInfo | None,
    ) -> Run:
        if decision.stop_reason is RevisionStopReason.CANCELLED:
            if run.status is RunStatus.CANCELLING:
                return await self._finish_run(run, RunStatus.CANCELLED)
            return run
        if decision.stop_reason is RevisionStopReason.BUDGET:
            error = ErrorInfo(
                category=ErrorCategory.BUDGET_EXCEEDED,
                message=decision.rationale,
            )
        else:
            error = failure_error or ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message=decision.rationale,
            )
        return await self._finish_run(run, RunStatus.FAILED, error=error)

    async def _record_evidence_events(
        self,
        run: Run,
        work_unit: WorkUnit,
        artifact_id: str,
    ) -> tuple[str, ...]:
        """Record ledger entries that reference the Evidence rows just persisted."""
        evidence = tuple(
            row
            for row in await self._store.list_evidence(work_unit.work_unit_id)
            if row.artifact_id == artifact_id
        )
        for row in evidence:
            await self._store.append_event(
                run.run_id,
                EventType.EVIDENCE_RECORDED,
                {
                    "work_unit_id": work_unit.work_unit_id,
                    "evidence_id": row.evidence_id,
                    "result": row.result.value,
                },
            )
        return tuple(row.evidence_id for row in evidence)

    async def _record_cascade_escalation(
        self,
        run: Run,
        work_unit: WorkUnit,
        from_profile: ModelProfile,
        to_profile: ModelProfile,
        rationale: str,
        *,
        evidence_ids: tuple[str, ...],
    ) -> None:
        decision = ControllerDecision(
            run_id=run.run_id,
            action=ControllerAction.ESCALATE_MODEL,
            rationale=rationale,
            work_unit_id=work_unit.work_unit_id,
            evidence_ids=evidence_ids,
        )
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload_from_model("decision", decision),
        )
        await self._store.append_event(
            run.run_id,
            EventType.STRATEGY_ESCALATED,
            {
                "from_strategy": ExecutionStrategy.CASCADE.value,
                "to_strategy": ExecutionStrategy.CASCADE.value,
                "from_profile": from_profile.alias,
                "to_profile": to_profile.alias,
                "reason": rationale,
            },
        )

    async def _create_direct_work_unit(self, run: Run) -> WorkUnit:
        return await self._create_work_unit(run, name=DIRECT_WORK_UNIT_NAME)

    async def _create_work_unit(self, run: Run, *, name: str) -> WorkUnit:
        work_unit = WorkUnit(
            work_unit_id=new_work_unit_id(),
            run_id=run.run_id,
            graph_version=run.graph_version,
            name=name,
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
        self,
        work_unit: WorkUnit,
        status: WorkUnitStatus,
        error: ErrorInfo | None = None,
        *,
        reason: str | None = None,
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
        if status in (WorkUnitStatus.BLOCKED, WorkUnitStatus.INVALIDATED):
            payload["reason"] = reason or f"work unit became {status.value}"
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

    async def _execute_planner_propose(
        self,
        run: Run,
        planner: Planner,
    ) -> tuple[PlannerCallResult, ErrorInfo | None]:
        """Record one initial Planner call before any user graph is compiled."""
        return await self._execute_planner_call(
            run,
            planner,
            attempt_index=await self._next_planner_attempt_index(run.run_id),
            call=lambda: planner.propose_call(run.request),
        )

    async def _execute_planner_revise(
        self,
        run: Run,
        planner: Planner,
        *,
        base_graph_version: int,
        reason: PlanRevisionReason,
        feedback: str,
    ) -> tuple[PlannerCallResult, ErrorInfo | None]:
        """Record one bounded revision call before committing its replacement graph."""
        return await self._execute_planner_call(
            run,
            planner,
            attempt_index=await self._next_planner_attempt_index(run.run_id),
            call=lambda: planner.revise_call(
                run.request,
                base_graph_version=base_graph_version,
                reason=reason,
                feedback=feedback,
            ),
        )

    async def _next_planner_attempt_index(self, run_id: str) -> int:
        attempts = await self._store.list_run_attempts(run_id)
        return sum(attempt.role is ModelRole.PLANNER for attempt in attempts) + 1

    async def _execute_planner_call(
        self,
        run: Run,
        planner: Planner,
        *,
        attempt_index: int,
        call: Callable[[], Awaitable[PlannerCallResult]],
    ) -> tuple[PlannerCallResult, ErrorInfo | None]:
        """Persist one Planner call and its measured provider facts."""
        work_unit = new_planning_work_unit(run.run_id)
        started_at = utc_now()
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=work_unit.work_unit_id,
            attempt_index=attempt_index,
            role=ModelRole.PLANNER,
            model=planner.profile.model_ref,
            status=transition_attempt(AttemptStatus.PENDING, AttemptStatus.RUNNING),
            created_at=started_at,
            started_at=started_at,
        )
        async with self._store.transaction():
            await self._store.create_work_unit(work_unit)
            await self._store.append_event(
                run.run_id,
                EventType.WORK_UNIT_CREATED,
                {
                    "work_unit_id": work_unit.work_unit_id,
                    "name": work_unit.name,
                    "graph_version": work_unit.graph_version,
                },
            )
            await self._store.append_event(
                run.run_id,
                EventType.WORK_UNIT_STARTED,
                {"work_unit_id": work_unit.work_unit_id},
            )
            await self._store.create_attempt(attempt)
            await self._store.append_event(
                run.run_id,
                EventType.ATTEMPT_STARTED,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "model": attempt.model.identifier,
                    "role": attempt.role.value,
                    "attempt_index": attempt.attempt_index,
                },
            )

        try:
            result = await call()
        except BaseException:
            unknown = Attempt.model_validate(
                attempt.model_dump()
                | {
                    "status": transition_attempt(
                        attempt.status, AttemptStatus.UNKNOWN
                    ),
                    "completed_at": utc_now(),
                }
            )
            cancelled = WorkUnit.model_validate(
                work_unit.model_dump()
                | {
                    "status": transition_work_unit(
                        work_unit.status, WorkUnitStatus.CANCELLED
                    )
                }
            )
            async with self._store.transaction():
                await self._store.update_attempt(unknown)
                await self._store.update_work_unit(cancelled)
                await self._store.append_event(
                    run.run_id,
                    EventType.ATTEMPT_UNKNOWN,
                    {
                        "work_unit_id": attempt.work_unit_id,
                        "attempt_id": attempt.attempt_id,
                    },
                )
                await self._store.append_event(
                    run.run_id,
                    EventType.WORK_UNIT_CANCELLED,
                    {"work_unit_id": work_unit.work_unit_id},
                )
            raise

        terminal_status = (
            AttemptStatus.SUCCEEDED if result.succeeded else AttemptStatus.FAILED
        )
        error = result.error
        if terminal_status is AttemptStatus.FAILED and error is None:
            error = ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message="Planner call failed without a classified error",
            )
        completed_at = utc_now()
        completed_attempt = Attempt.model_validate(
            attempt.model_dump()
            | {
                "status": transition_attempt(attempt.status, terminal_status),
                "provider_request_id": result.provider_request_id,
                "usage": result.usage,
                "error": error,
                "completed_at": completed_at,
            }
        )
        completed_unit_status = (
            WorkUnitStatus.SUCCEEDED
            if terminal_status is AttemptStatus.SUCCEEDED
            else WorkUnitStatus.FAILED
        )
        completed_unit = WorkUnit.model_validate(
            work_unit.model_dump()
            | {"status": transition_work_unit(work_unit.status, completed_unit_status)}
        )
        async with self._store.transaction():
            await self._store.update_attempt(completed_attempt)
            await self._store.update_work_unit(completed_unit)
            await self._store.append_event(
                run.run_id,
                EventType.ATTEMPT_SUCCEEDED
                if terminal_status is AttemptStatus.SUCCEEDED
                else EventType.ATTEMPT_FAILED,
                (
                    {
                        "work_unit_id": attempt.work_unit_id,
                        "attempt_id": attempt.attempt_id,
                    }
                    if error is None
                    else {
                        "work_unit_id": attempt.work_unit_id,
                        "attempt_id": attempt.attempt_id,
                        "error": error.model_dump(mode="json"),
                    }
                ),
            )
            await self._store.append_event(
                run.run_id,
                _WORK_UNIT_EVENTS[completed_unit_status],
                (
                    {"work_unit_id": work_unit.work_unit_id}
                    if error is None
                    else {
                        "work_unit_id": work_unit.work_unit_id,
                        "error": error.model_dump(mode="json"),
                    }
                ),
            )
            if result.usage is not None:
                total = await self._store.add_run_usage(run.run_id, result.usage)
                await self._store.append_event(
                    run.run_id,
                    EventType.USAGE_UPDATED,
                    {"usage": total.model_dump(mode="json")},
                )

        postflight = await self._postflight_budget(run)
        return result, postflight

    def _planner(self) -> Planner:
        profile = self._settings.require_profile(ModelRole.PLANNER)
        adapter = self._adapters.get(profile.alias)
        if adapter is None:
            raise ProviderError(
                f"no adapter is registered for model alias {profile.alias!r}",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        return Planner(adapter, profile)

    def _worker_for(self, role: ModelRole) -> Worker:
        profile = self._settings.require_profile(role)
        return self._worker_for_profile(profile)

    def _worker_for_profile(self, profile: ModelProfile) -> Worker:
        adapter = self._adapters.get(profile.alias)
        if adapter is None:
            raise ProviderError(
                f"no adapter is registered for model alias {profile.alias!r}",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        return Worker(self._store, adapter, profile)

    # --- budget helpers ---------------------------------------------------------

    async def _preflight_budget(self, run: Run) -> ErrorInfo | None:
        """Check deadline, attempts, and reached tokens before provider dispatch."""
        budget = run.request.budget
        now = utc_now()

        deadline_decision = check_deadline(budget, now)
        if not deadline_decision.allowed:
            assert deadline_decision.error is not None
            return await self._record_budget_stop(run, deadline_decision.error)

        attempts = await self._store.list_run_attempts(run.run_id)
        attempt_decision = check_attempt_budget(budget, attempt_count=len(attempts))
        if not attempt_decision.allowed:
            assert attempt_decision.error is not None
            return await self._record_budget_stop(run, attempt_decision.error)

        token_decision = check_token_budget_preflight(
            budget, await self._store.get_run_usage(run.run_id)
        )
        if not token_decision.allowed:
            assert token_decision.error is not None
            return await self._record_budget_stop(run, token_decision.error)

        return None

    async def _postflight_budget(self, run: Run) -> ErrorInfo | None:
        """Check token budget after usage has been recorded for the last attempt."""
        budget = run.request.budget
        usage = await self._store.get_run_usage(run.run_id)
        token_decision = check_token_budget_postflight(budget, usage)
        if not token_decision.allowed:
            assert token_decision.error is not None
            return await self._record_budget_stop(run, token_decision.error)
        return None

    async def _record_budget_stop(self, run: Run, error: Exception) -> ErrorInfo:
        """Persist the budget stop decision and BUDGET_EXHAUSTED event."""
        from prp_runtime.domain.errors import BudgetError

        assert isinstance(error, BudgetError)
        error_info = ErrorInfo(
            category=ErrorCategory.BUDGET_EXCEEDED, message=error.detail.message
        )
        decision = ControllerDecision(
            run_id=run.run_id,
            action=ControllerAction.STOP_ON_BUDGET,
            rationale=error.detail.message,
        )
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload_from_model("decision", decision),
        )
        await self._store.append_event(
            run.run_id,
            EventType.BUDGET_EXHAUSTED,
            {"error": error_info.model_dump(mode="json")},
        )
        return error_info
