"""Single-model worker.

A worker executes exactly one attempt for one work unit: it calls the provider,
records the attempt outcome, the produced artifact and the measured usage. It
never decides whether a run is finished and never writes run or work unit state.
"""

from datetime import datetime
from enum import StrEnum, unique

from pydantic import ValidationError, model_validator

from prp_runtime.domain.enums import AttemptStatus, ModelRole, ToolCallStatus
from prp_runtime.control.budget import check_role_dispatch
from prp_runtime.domain.errors import BudgetError, ErrorCode, ProviderError
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import (
    AgentHistoryItem,
    AgentHistoryRecord,
    AgentToolCall,
    AgentToolResult,
    AgentTurn,
    Artifact,
    Attempt,
    DomainModel,
    ErrorCategory,
    ErrorInfo,
    ExecutionScope,
    Run,
    Usage,
    WorkUnit,
    new_artifact_id,
)
from prp_runtime.domain.transitions import assert_can_start_attempt, transition_attempt
from prp_runtime.domain.values import new_attempt_id, utc_now
from prp_runtime.policy.models import ApprovalDecision, ApprovalOutcome, ApprovalRequest, Lease
from prp_runtime.providers.base import ModelProfile, ProviderAdapter
from prp_runtime.runtime.agent_executor import AgentToolExecutor as ProductionAgentToolExecutor
from prp_runtime.runtime.agent_loop import (
    AgentHistoryWriter,
    AgentLoop,
    AgentLoopStatus,
    AgentToolExecutor,
)
from prp_runtime.runtime.context import ANSWER_ARTIFACT_NAME, WorkerContext
from prp_runtime.storage.sqlite import MissingEntityError, SqliteStore
from prp_runtime.tools.models import ToolCall

__all__ = ["ResumeAction", "ResumeState", "Worker", "WorkerResult"]

_CATEGORY_BY_CODE: dict[ErrorCode, ErrorCategory] = {
    ErrorCode.PROVIDER_TIMEOUT: ErrorCategory.TIMEOUT,
    ErrorCode.PROVIDER_RATE_LIMITED: ErrorCategory.RATE_LIMIT,
    ErrorCode.PROVIDER_AUTH_FAILED: ErrorCategory.AUTH,
    ErrorCode.PROVIDER_UNAVAILABLE: ErrorCategory.NETWORK,
    ErrorCode.PROVIDER_INVALID_RESPONSE: ErrorCategory.PROVIDER_ERROR,
    ErrorCode.PROVIDER_NOT_CONFIGURED: ErrorCategory.PROVIDER_ERROR,
}


def _category_for(code: ErrorCode) -> ErrorCategory:
    """Classify a provider failure for the attempt record."""
    return _CATEGORY_BY_CODE.get(code, ErrorCategory.UNKNOWN)


def _completed_at(started_at: datetime) -> datetime:
    """Now, never before the recorded start, so a clock step cannot invalidate it."""
    now = utc_now()
    return started_at if now < started_at else now


class WorkerResult(DomainModel):
    """What one attempt produced."""

    attempt: Attempt
    artifact: Artifact | None = None
    error: ErrorInfo | None = None
    paused: bool = False
    awaiting_remote_result: bool = False
    pending_call_ids: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.attempt.status is AttemptStatus.SUCCEEDED and self.artifact is not None


@unique
class ResumeAction(StrEnum):
    """The only actions a persisted resume inspection may authorize."""

    WAIT = "WAIT"
    WAIT_REMOTE = "WAIT_REMOTE"
    REJECT = "REJECT"
    EXECUTE = "EXECUTE"
    CONTINUE = "CONTINUE"
    BLOCK = "BLOCK"


class ResumeState(DomainModel):
    """A closed decision assembled only from persisted attempt facts."""

    action: ResumeAction
    attempt: Attempt
    history: tuple[AgentHistoryItem, ...] = ()
    pending_call: ToolCall | None = None
    pending_public_call: AgentToolCall | None = None
    approval_request: ApprovalRequest | None = None
    approval_decision: ApprovalDecision | None = None
    lease: Lease | None = None
    replay_results: tuple[AgentToolResult, ...] = ()
    reason: str | None = None

    @classmethod
    def blocked(
        cls,
        attempt: Attempt,
        history: tuple[AgentHistoryItem, ...],
        reason: str,
    ) -> "ResumeState":
        return cls(
            action=ResumeAction.BLOCK,
            attempt=attempt,
            history=history,
            reason=reason,
        )

    @model_validator(mode="after")
    def _decision_shape_is_closed(self) -> "ResumeState":
        if self.action is ResumeAction.BLOCK:
            if not self.reason:
                raise ValueError("blocked resume state requires a reason")
            return self
        if self.action is ResumeAction.CONTINUE:
            if (
                self.pending_call is not None
                or self.approval_request is not None
                or self.approval_decision is not None
                or self.lease is not None
            ):
                raise ValueError(
                    "continue resume state must not contain pending approval facts"
                )
            return self
        if self.action is ResumeAction.WAIT_REMOTE:
            if self.pending_call is None or self.pending_public_call is None:
                raise ValueError("remote wait resume state requires a pending call")
            if (
                self.approval_request is not None
                or self.approval_decision is not None
                or self.lease is not None
            ):
                raise ValueError("remote wait must not contain approval facts")
            if self.pending_call.status is not ToolCallStatus.RUNNING:
                raise ValueError("remote wait requires a running tool call")
            if (
                self.pending_public_call.tool_name != self.pending_call.tool_name
                or self.pending_public_call.arguments != self.pending_call.arguments
            ):
                raise ValueError("resume public call does not match the pending call")
            return self
        if (
            self.pending_call is None
            or self.pending_public_call is None
            or self.approval_request is None
        ):
            raise ValueError(
                "pending resume action requires a call and approval request"
            )
        if self.pending_call.call_id != self.approval_request.call_id:
            raise ValueError("resume approval does not match the pending call")
        if (
            self.pending_public_call.tool_name != self.pending_call.tool_name
            or self.pending_public_call.arguments != self.pending_call.arguments
        ):
            raise ValueError("resume public call does not match the pending call")
        if self.action is ResumeAction.WAIT:
            if self.approval_decision is not None or self.lease is not None:
                raise ValueError("wait resume state must not contain a decision or lease")
        elif self.action is ResumeAction.REJECT:
            if (
                self.approval_decision is None
                or self.approval_decision.outcome is not ApprovalOutcome.DENY
            ):
                raise ValueError("reject resume state requires a DENY decision")
            if self.lease is not None:
                raise ValueError("rejected resume state must not contain a lease")
        elif self.action is ResumeAction.EXECUTE:
            if (
                self.approval_decision is None
                or self.approval_decision.outcome is not ApprovalOutcome.ALLOW
            ):
                raise ValueError("execute resume state requires an ALLOW decision")
            if self.lease is not None and self.lease.call_id != self.pending_call.call_id:
                raise ValueError("execute resume state requires the pending call lease")
        return self


class Worker:
    """Runs one work unit against one configured model."""

    def __init__(
        self,
        store: SqliteStore,
        adapter: ProviderAdapter,
        profile: ModelProfile,
        *,
        tool_executor: AgentToolExecutor | None = None,
        max_tool_rounds: int = 8,
        execution_scope: ExecutionScope | None = None,
    ) -> None:
        self._store = store
        self._adapter = adapter
        self._profile = profile
        self._execution_scope = execution_scope
        self._agent_loop = AgentLoop(
            adapter,
            profile,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
        )

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def execution_scope(self) -> ExecutionScope | None:
        """The scope freshly loaded for this worker, if the run is session-bound."""
        return self._execution_scope

    async def load_resume_state(self, attempt_id: str) -> ResumeState:
        """Hydrate one attempt without guessing an unconfirmed side effect."""
        attempt = await self._store.get_attempt(attempt_id)
        records = await self._store.list_agent_history(attempt_id)
        history = tuple(record.item for record in records)
        if (
            self._execution_scope is not None
            and self._execution_scope.run_id != attempt.run_id
        ):
            return ResumeState.blocked(attempt, history, "execution_scope_mismatch")
        if any(
            record.run_id != attempt.run_id
            or record.work_unit_id != attempt.work_unit_id
            or record.attempt_id != attempt.attempt_id
            for record in records
        ):
            return ResumeState.blocked(attempt, history, "history_scope_mismatch")
        if tuple(record.sequence for record in records) != tuple(
            range(1, len(records) + 1)
        ):
            return ResumeState.blocked(attempt, history, "history_sequence_gap")
        if attempt.status in (AttemptStatus.INTERRUPTED, AttemptStatus.UNKNOWN):
            return ResumeState.blocked(attempt, history, "attempt_outcome_unknown")

        calls = await self._store.list_tool_calls_for_attempt(attempt_id)
        public_calls: dict[str, AgentToolCall] = {}
        history_results: dict[str, list[AgentToolResult]] = {}
        for item in history:
            if isinstance(item, AgentTurn):
                for call in item.tool_calls:
                    previous = public_calls.get(call.call_id)
                    if previous is not None and previous != call:
                        return ResumeState.blocked(
                            attempt, history, "history_tool_call_conflict"
                        )
                    public_calls[call.call_id] = call
            elif isinstance(item, AgentToolResult):
                if item.call_id not in public_calls:
                    return ResumeState.blocked(
                        attempt, history, "history_result_without_call"
                    )
                history_results.setdefault(item.call_id, []).append(item)

        if not public_calls:
            if calls:
                return ResumeState.blocked(
                    attempt, history, "tool_call_missing_from_history"
                )
            return ResumeState(
                action=ResumeAction.CONTINUE,
                attempt=attempt,
                history=history,
            )

        mapped_calls: dict[str, ToolCall] = {}
        for public_call in public_calls.values():
            candidates = [
                persisted
                for persisted in calls
                if persisted.tool_name == public_call.tool_name
                and persisted.arguments == public_call.arguments
            ]
            exact = [
                persisted
                for persisted in candidates
                if persisted.snapshot_id is not None
                and ProductionAgentToolExecutor._internal_call_id(
                    run_id=attempt.run_id,
                    work_unit_id=attempt.work_unit_id,
                    snapshot_id=persisted.snapshot_id,
                    provider_call_id=public_call.call_id,
                    tool_name=public_call.tool_name,
                )
                == persisted.call_id
            ]
            matches = exact if exact else candidates
            if len(matches) != 1:
                return ResumeState.blocked(
                    attempt, history, "tool_call_mapping_ambiguous"
                )
            mapped_calls[public_call.call_id] = matches[0]

        if {call.call_id for call in mapped_calls.values()} != {
            call.call_id for call in calls
        }:
            return ResumeState.blocked(attempt, history, "tool_call_missing_from_history")

        pending: list[tuple[AgentToolCall, ToolCall]] = []
        replay_results: list[AgentToolResult] = []
        for provider_call_id, persisted in mapped_calls.items():
            observed = history_results.get(provider_call_id, [])
            if persisted.status.is_terminal:
                try:
                    stored_result = await self._store.get_tool_result(persisted.call_id)
                except MissingEntityError:
                    return ResumeState.blocked(attempt, history, "tool_result_missing")
                if stored_result.status is not persisted.status:
                    return ResumeState.blocked(
                        attempt, history, "tool_result_status_mismatch"
                    )
                if stored_result.status in (
                    ToolCallStatus.UNKNOWN,
                    ToolCallStatus.INTERRUPTED,
                ):
                    return ResumeState.blocked(attempt, history, "tool_outcome_unknown")
                expected = ProductionAgentToolExecutor.public_result(
                    public_calls[provider_call_id], stored_result
                )
                if any(result != expected for result in observed):
                    return ResumeState.blocked(attempt, history, "history_result_mismatch")
                if not observed:
                    replay_results.append(expected)
            else:
                if observed:
                    return ResumeState.blocked(
                        attempt, history, "history_result_before_terminal"
                    )
                pending.append((public_calls[provider_call_id], persisted))

        if len(pending) > 1:
            return ResumeState.blocked(attempt, history, "multiple_pending_tool_calls")
        if not pending:
            return ResumeState(
                action=ResumeAction.CONTINUE,
                attempt=attempt,
                history=history,
                replay_results=tuple(replay_results),
            )

        public_call, pending_call = pending[0]
        if pending_call.status is ToolCallStatus.RUNNING:
            return ResumeState(
                action=ResumeAction.WAIT_REMOTE,
                attempt=attempt,
                history=history,
                pending_call=pending_call,
                pending_public_call=public_call,
                replay_results=tuple(replay_results),
                reason="remote_assignment_pending",
            )
        if pending_call.status is not ToolCallStatus.AWAITING_APPROVAL:
            return ResumeState.blocked(
                attempt, history, "pending_tool_call_not_awaiting_approval"
            )
        scope = self._execution_scope
        if scope is None or scope.run_id != attempt.run_id:
            return ResumeState.blocked(attempt, history, "approval_owner_scope_missing")

        approvals = await self._store.list_approvals(
            owner_id=scope.principal_id,
            run_id=attempt.run_id,
            call_id=pending_call.call_id,
        )
        if len(approvals) != 1:
            return ResumeState.blocked(attempt, history, "approval_request_not_unique")
        approval = approvals[0]
        if (
            approval.call_id != pending_call.call_id
            or approval.run_id != pending_call.run_id
            or approval.workspace_id != scope.workspace_id
            or approval.tool_name != pending_call.tool_name
            or approval.effect is not pending_call.effect
        ):
            return ResumeState.blocked(attempt, history, "approval_scope_mismatch")

        leases = await self._store.list_leases(
            owner_id=scope.principal_id,
            run_id=attempt.run_id,
            call_id=pending_call.call_id,
        )
        matching_leases = [
            lease
            for lease in leases
            if lease.approval_request_id == approval.request_id
        ]
        if len(matching_leases) != len(leases):
            return ResumeState.blocked(attempt, history, "lease_scope_mismatch")
        if len(matching_leases) > 1:
            return ResumeState.blocked(attempt, history, "lease_not_unique")
        lease = matching_leases[0] if matching_leases else None
        try:
            decision = await self._store.get_approval_decision(
                approval.request_id,
                owner_id=scope.principal_id,
            )
        except MissingEntityError:
            if lease is not None:
                return ResumeState.blocked(attempt, history, "lease_without_decision")
            return ResumeState(
                action=ResumeAction.WAIT,
                attempt=attempt,
                history=history,
                pending_call=pending_call,
                pending_public_call=pending[0][0],
                approval_request=approval,
                replay_results=tuple(replay_results),
            )

        if decision.outcome is ApprovalOutcome.DENY:
            if lease is not None:
                return ResumeState.blocked(attempt, history, "deny_has_lease")
            return ResumeState(
                action=ResumeAction.REJECT,
                attempt=attempt,
                history=history,
                pending_call=pending_call,
                pending_public_call=pending[0][0],
                approval_request=approval,
                approval_decision=decision,
                replay_results=tuple(replay_results),
            )

        if lease is not None and (
            lease.call_id != pending_call.call_id
            or lease.approval_request_id != approval.request_id
            or not lease.scope.is_subset_of(approval.scope)
            or not lease.is_active_at(utc_now())
        ):
            return ResumeState.blocked(attempt, history, "allow_lease_invalid")
        return ResumeState(
            action=ResumeAction.EXECUTE,
            attempt=attempt,
            history=history,
            pending_call=pending_call,
            pending_public_call=pending[0][0],
            approval_request=approval,
            approval_decision=decision,
            lease=lease,
            replay_results=tuple(replay_results),
        )

    async def resume(
        self,
        *,
        run: Run,
        work_unit: WorkUnit,
        context: WorkerContext,
        state: ResumeState,
        role: ModelRole = ModelRole.WORKER,
    ) -> WorkerResult:
        """Continue one hydrated approval path with a fresh provider attempt."""
        if (
            state.attempt.run_id != run.run_id
            or state.attempt.work_unit_id != work_unit.work_unit_id
        ):
            raise ValueError("resume state does not match the requested execution scope")
        if state.action is ResumeAction.WAIT:
            pending = (
                ()
                if state.pending_public_call is None
                else (state.pending_public_call.call_id,)
            )
            return WorkerResult(attempt=state.attempt, paused=True, pending_call_ids=pending)
        if state.action is ResumeAction.WAIT_REMOTE:
            pending = (
                ()
                if state.pending_public_call is None
                else (state.pending_public_call.call_id,)
            )
            return WorkerResult(
                attempt=state.attempt,
                paused=True,
                awaiting_remote_result=True,
                pending_call_ids=pending,
            )
        if state.action is ResumeAction.BLOCK:
            raise ValueError(f"resume state is blocked: {state.reason or 'unknown'}")
        if state.action is ResumeAction.CONTINUE:
            if state.attempt.status is AttemptStatus.SUCCEEDED:
                artifacts = await self._store.list_artifacts(work_unit.work_unit_id)
                matching = tuple(
                    artifact
                    for artifact in artifacts
                    if artifact.attempt_id == state.attempt.attempt_id
                )
                if matching:
                    return WorkerResult(attempt=state.attempt, artifact=matching[-1])
            if state.attempt.status is AttemptStatus.RUNNING:
                return await self._continue_running_attempt(
                    run=run,
                    work_unit=work_unit,
                    context=context,
                    state=state,
                    role=role,
                )

        from prp_runtime.control.reservations import ReservationRequest

        attempt_index = state.attempt.attempt_index + 1
        dispatch_key = f"{work_unit.work_unit_id}:{attempt_index}:{role.value}"
        async with self._store.transaction():
            reservation = await self._store.reserve_reservation(
                ReservationRequest(
                    run_id=run.run_id,
                    work_unit_id=work_unit.work_unit_id,
                    dispatch_key=dispatch_key,
                    capacity_key=self._profile.alias,
                )
            )
            attempt = await self._start_attempt(
                run, work_unit, attempt_index, role
            )
            for sequence, item in enumerate(state.history, start=1):
                await self._store.append_agent_history(
                    AgentHistoryRecord(
                        run_id=attempt.run_id,
                        work_unit_id=attempt.work_unit_id,
                        attempt_id=attempt.attempt_id,
                        sequence=sequence,
                        idempotency_key=f"{attempt.attempt_id}:{sequence}",
                        item=item,
                    )
                )

        blocked = await self._stop_if_role_budget_exceeded(
            run, attempt, reservation.reservation_id
        )
        if blocked is not None:
            return blocked

        async def write_history(sequence: int, item: AgentHistoryItem) -> None:
            await self._store.append_agent_history(
                AgentHistoryRecord(
                    run_id=attempt.run_id,
                    work_unit_id=attempt.work_unit_id,
                    attempt_id=attempt.attempt_id,
                    sequence=sequence,
                    idempotency_key=f"{attempt.attempt_id}:{sequence}",
                    item=item,
                )
            )

        try:
            loop_result = await self._agent_loop.execute(
                input=context.render_input(),
                instructions=context.render_instructions(),
                json_schema=context.output.json_schema,
                history=state.history,
                attempt_id=attempt.attempt_id,
                run_id=run.run_id,
                work_unit_id=work_unit.work_unit_id,
                mode=run.request.agent_options.agent_mode,
                deadline=run.request.budget.deadline,
                max_attempts=run.request.budget.max_attempts,
                history_writer=write_history,
                resume_pending_call=state.pending_public_call,
                replay_results={
                    result.call_id: result for result in state.replay_results
                },
            )
        except ProviderError as error:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                ErrorInfo(category=_category_for(error.code), message=str(error)),
            )
        except BaseException:
            await self._record_unconfirmed(attempt, reservation.reservation_id)
            raise

        if loop_result.status in (
            AgentLoopStatus.PAUSED,
            AgentLoopStatus.WAITING_REMOTE,
        ):
            return await self._record_paused(
                attempt,
                reservation.reservation_id,
                loop_result.usage,
                loop_result.provider_request_id,
                loop_result.pending_call_ids,
                awaiting_remote_result=loop_result.status is AgentLoopStatus.WAITING_REMOTE,
            )
        if loop_result.error is not None:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                loop_result.error,
            )
        try:
            artifact = Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                name=ANSWER_ARTIFACT_NAME,
                kind=context.output.kind,
                content=loop_result.text or "",
            )
        except ValidationError:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                ErrorInfo(
                    category=ErrorCategory.PROVIDER_ERROR,
                    message=(
                        f"upstream {self._profile.alias} returned no usable "
                        f"{context.output.kind.value} result"
                    ),
                ),
            )
        return await self._record_success(
            attempt,
            reservation.reservation_id,
            artifact,
            loop_result.usage,
            loop_result.provider_request_id,
        )

    async def _continue_running_attempt(
        self,
        *,
        run: Run,
        work_unit: WorkUnit,
        context: WorkerContext,
        state: ResumeState,
        role: ModelRole,
    ) -> WorkerResult:
        """Continue one still-open attempt after an accepted remote result."""
        from prp_runtime.control.reservations import ReservationRequest

        attempt = state.attempt
        dispatch_key = f"{work_unit.work_unit_id}:{attempt.attempt_index}:{role.value}:resume"
        async with self._store.transaction():
            reservation = await self._store.reserve_reservation(
                ReservationRequest(
                    run_id=run.run_id,
                    work_unit_id=work_unit.work_unit_id,
                    dispatch_key=dispatch_key,
                    capacity_key=self._profile.alias,
                )
            )

        async def write_history(sequence: int, item: AgentHistoryItem) -> None:
            await self._store.append_agent_history(
                AgentHistoryRecord(
                    run_id=attempt.run_id,
                    work_unit_id=attempt.work_unit_id,
                    attempt_id=attempt.attempt_id,
                    sequence=sequence,
                    idempotency_key=f"{attempt.attempt_id}:{sequence}",
                    item=item,
                )
            )

        try:
            loop_result = await self._agent_loop.execute(
                input=context.render_input(),
                instructions=context.render_instructions(),
                json_schema=context.output.json_schema,
                history=state.history,
                attempt_id=attempt.attempt_id,
                run_id=run.run_id,
                work_unit_id=work_unit.work_unit_id,
                mode=run.request.agent_options.agent_mode,
                deadline=run.request.budget.deadline,
                max_attempts=run.request.budget.max_attempts,
                history_writer=write_history,
                replay_results={
                    result.call_id: result for result in state.replay_results
                },
            )
        except ProviderError as error:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                ErrorInfo(category=_category_for(error.code), message=str(error)),
            )
        except BaseException:
            await self._record_unconfirmed(attempt, reservation.reservation_id)
            raise

        if loop_result.status in (
            AgentLoopStatus.PAUSED,
            AgentLoopStatus.WAITING_REMOTE,
        ):
            return await self._record_paused(
                attempt,
                reservation.reservation_id,
                loop_result.usage,
                loop_result.provider_request_id,
                loop_result.pending_call_ids,
                awaiting_remote_result=loop_result.status is AgentLoopStatus.WAITING_REMOTE,
            )
        if loop_result.error is not None:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                loop_result.error,
            )
        try:
            artifact = Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                name=ANSWER_ARTIFACT_NAME,
                kind=context.output.kind,
                content=loop_result.text or "",
            )
        except ValidationError:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                ErrorInfo(
                    category=ErrorCategory.PROVIDER_ERROR,
                    message=(
                        f"upstream {self._profile.alias} returned no usable "
                        f"{context.output.kind.value} result"
                    ),
                ),
            )
        return await self._record_success(
            attempt,
            reservation.reservation_id,
            artifact,
            loop_result.usage,
            loop_result.provider_request_id,
        )

    async def execute(
        self,
        *,
        run: Run,
        work_unit: WorkUnit,
        context: WorkerContext,
        attempt_index: int = 1,
        role: ModelRole = ModelRole.WORKER,
    ) -> WorkerResult:
        """Perform one attempt and persist its facts.

        Raises ``AttemptNotAllowedError`` when the run or the work unit forbids a
        new attempt, so a cancelled run can never produce another provider call.
        """
        from prp_runtime.control.reservations import ReservationRequest

        assert_can_start_attempt(run.status, work_unit.status)
        dispatch_key = f"{work_unit.work_unit_id}:{attempt_index}:{role.value}"
        async with self._store.transaction():
            reservation = await self._store.reserve_reservation(
                ReservationRequest(
                    run_id=run.run_id,
                    work_unit_id=work_unit.work_unit_id,
                    dispatch_key=dispatch_key,
                    capacity_key=self._profile.alias,
                )
            )
            attempt = await self._start_attempt(
                run, work_unit, attempt_index, role
            )

        blocked = await self._stop_if_role_budget_exceeded(
            run, attempt, reservation.reservation_id
        )
        if blocked is not None:
            return blocked

        persisted_history = await self._store.list_agent_history(attempt.attempt_id)

        async def write_history(sequence: int, item: AgentHistoryItem) -> None:
            await self._store.append_agent_history(
                AgentHistoryRecord(
                    run_id=attempt.run_id,
                    work_unit_id=attempt.work_unit_id,
                    attempt_id=attempt.attempt_id,
                    sequence=sequence,
                    idempotency_key=f"{attempt.attempt_id}:{sequence}",
                    item=item,
                )
            )

        history_writer: AgentHistoryWriter = write_history
        try:
            loop_result = await self._agent_loop.execute(
                input=context.render_input(),
                instructions=context.render_instructions(),
                json_schema=context.output.json_schema,
                history=tuple(record.item for record in persisted_history),
                attempt_id=attempt.attempt_id,
                run_id=run.run_id,
                work_unit_id=work_unit.work_unit_id,
                mode=run.request.agent_options.agent_mode,
                deadline=run.request.budget.deadline,
                max_attempts=run.request.budget.max_attempts,
                history_writer=history_writer,
            )
        except ProviderError as error:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                ErrorInfo(category=_category_for(error.code), message=str(error)),
            )
        except BaseException:
            # Cancellation or a hard interruption: the upstream outcome cannot be
            # proven, so the attempt becomes UNKNOWN and the error is re-raised.
            await self._record_unconfirmed(attempt, reservation.reservation_id)
            raise

        if loop_result.status in (
            AgentLoopStatus.PAUSED,
            AgentLoopStatus.WAITING_REMOTE,
        ):
            return await self._record_paused(
                attempt,
                reservation.reservation_id,
                loop_result.usage,
                loop_result.provider_request_id,
                loop_result.pending_call_ids,
                awaiting_remote_result=loop_result.status is AgentLoopStatus.WAITING_REMOTE,
            )
        if loop_result.error is not None:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                loop_result.error,
            )

        try:
            artifact = Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                name=ANSWER_ARTIFACT_NAME,
                kind=context.output.kind,
                content=loop_result.text or "",
            )
        except ValidationError:
            return await self._record_failure(
                attempt,
                reservation.reservation_id,
                ErrorInfo(
                    category=ErrorCategory.PROVIDER_ERROR,
                    message=(
                        f"upstream {self._profile.alias} returned no usable "
                        f"{context.output.kind.value} result"
                    ),
                ),
            )
        return await self._record_success(
            attempt,
            reservation.reservation_id,
            artifact,
            loop_result.usage,
            loop_result.provider_request_id,
        )

    async def _record_paused(
        self,
        attempt: Attempt,
        reservation_id: str,
        usage: Usage | None,
        provider_request_id: str | None,
        pending_call_ids: tuple[str, ...],
        *,
        awaiting_remote_result: bool = False,
    ) -> WorkerResult:
        """Settle the provider turn while leaving approval or remote wait pending."""
        if awaiting_remote_result:
            updated = attempt
            changes: dict[str, object] = {}
            if usage is not None:
                changes["usage"] = usage
            if provider_request_id is not None:
                changes["provider_request_id"] = provider_request_id
            if changes:
                updated = attempt.model_copy(update=changes)
            async with self._store.transaction():
                if changes:
                    await self._store.update_attempt(updated)
                if usage is not None:
                    total = await self._store.add_run_usage(attempt.run_id, usage)
                    await self._store.append_event(
                        attempt.run_id,
                        EventType.USAGE_UPDATED,
                        {"usage": total.model_dump(mode="json")},
                    )
                await self._store.settle_reservation(
                    reservation_id,
                    measured_usage=usage,
                )
            return WorkerResult(
                attempt=updated,
                paused=True,
                awaiting_remote_result=True,
                pending_call_ids=pending_call_ids,
            )
        completed = self._close_attempt(
            attempt,
            AttemptStatus.SUCCEEDED,
            usage=usage,
            provider_request_id=provider_request_id,
        )
        async with self._store.transaction():
            await self._store.update_attempt(completed)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_SUCCEEDED,
                {"work_unit_id": attempt.work_unit_id, "attempt_id": attempt.attempt_id},
            )
            if usage is not None:
                total = await self._store.add_run_usage(attempt.run_id, usage)
                await self._store.append_event(
                    attempt.run_id,
                    EventType.USAGE_UPDATED,
                    {"usage": total.model_dump(mode="json")},
                )
            await self._store.settle_reservation(
                reservation_id,
                measured_usage=usage,
            )
        return WorkerResult(
            attempt=completed,
            paused=True,
            awaiting_remote_result=awaiting_remote_result,
            pending_call_ids=pending_call_ids,
        )

    async def _start_attempt(
        self, run: Run, work_unit: WorkUnit, attempt_index: int, role: ModelRole
    ) -> Attempt:
        started_at = utc_now()
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=work_unit.work_unit_id,
            attempt_index=attempt_index,
            role=role,
            model=self._profile.model_ref,
            status=transition_attempt(AttemptStatus.PENDING, AttemptStatus.RUNNING),
            created_at=started_at,
            started_at=started_at,
        )
        async with self._store.transaction():
            await self._store.create_attempt(attempt)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_STARTED,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "model": self._profile.model_ref.identifier,
                    "role": role.value,
                    "attempt_index": attempt_index,
                },
            )
        return attempt

    async def _record_success(
        self,
        attempt: Attempt,
        reservation_id: str,
        artifact: Artifact,
        usage: Usage | None,
        provider_request_id: str | None,
    ) -> WorkerResult:
        completed = self._close_attempt(
            attempt,
            AttemptStatus.SUCCEEDED,
            usage=usage,
            provider_request_id=provider_request_id,
        )
        async with self._store.transaction():
            await self._store.update_attempt(completed)
            await self._store.add_artifact(artifact)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_SUCCEEDED,
                {"work_unit_id": attempt.work_unit_id, "attempt_id": attempt.attempt_id},
            )
            await self._store.append_event(
                attempt.run_id,
                EventType.ARTIFACT_PRODUCED,
                {
                    "work_unit_id": artifact.work_unit_id,
                    "artifact_id": artifact.artifact_id,
                    "name": artifact.name,
                    "kind": artifact.kind.value,
                },
            )
            if usage is not None:
                total = await self._store.add_run_usage(attempt.run_id, usage)
                await self._store.append_event(
                    attempt.run_id,
                    EventType.USAGE_UPDATED,
                    {"usage": total.model_dump(mode="json")},
                )
            await self._store.settle_reservation(
                reservation_id,
                measured_usage=usage,
            )
        return WorkerResult(attempt=completed, artifact=artifact)

    async def _stop_if_role_budget_exceeded(
        self,
        run: Run,
        attempt: Attempt,
        reservation_id: str,
    ) -> WorkerResult | None:
        """Refuse provider dispatch when a measured role ceiling is already reached."""
        decision = check_role_dispatch(
            run.request.budget,
            await self._store.get_run_usage(run.run_id),
            attempt_count=max(0, attempt.attempt_index - 1),
            now=utc_now(),
            context_window_tokens=self._profile.context_window_tokens,
            max_output_tokens=self._profile.max_output_tokens,
            timeout_seconds=self._profile.timeout_seconds,
        )
        if decision.allowed:
            return None
        assert isinstance(decision.error, BudgetError)
        return await self._record_failure(
            attempt,
            reservation_id,
            ErrorInfo(
                category=ErrorCategory.BUDGET_EXCEEDED,
                message=decision.error.detail.message,
            ),
        )

    async def _record_failure(
        self, attempt: Attempt, reservation_id: str, error: ErrorInfo
    ) -> WorkerResult:
        failed = self._close_attempt(attempt, AttemptStatus.FAILED, error=error)
        async with self._store.transaction():
            await self._store.update_attempt(failed)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_FAILED,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "error": error.model_dump(mode="json"),
                },
            )
            await self._store.settle_reservation(
                reservation_id,
                measured_usage=None,
            )
        return WorkerResult(attempt=failed, error=error)

    async def _record_unconfirmed(self, attempt: Attempt, reservation_id: str) -> None:
        unknown = self._close_attempt(attempt, AttemptStatus.UNKNOWN)
        async with self._store.transaction():
            await self._store.update_attempt(unknown)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_UNKNOWN,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "reason": "the upstream outcome could not be confirmed",
                },
            )
            await self._store.settle_reservation(
                reservation_id,
                measured_usage=None,
            )

    @staticmethod
    def _close_attempt(
        attempt: Attempt,
        status: AttemptStatus,
        *,
        usage: Usage | None = None,
        error: ErrorInfo | None = None,
        provider_request_id: str | None = None,
    ) -> Attempt:
        """Re-validate the attempt into its terminal shape via the state machine."""
        changes: dict[str, object] = {
            "status": transition_attempt(attempt.status, status),
            "completed_at": _completed_at(attempt.started_at or attempt.created_at),
        }
        if usage is not None:
            changes["usage"] = usage
        if error is not None:
            changes["error"] = error
        if provider_request_id is not None:
            changes["provider_request_id"] = provider_request_id
        return Attempt.model_validate(attempt.model_dump() | changes)
