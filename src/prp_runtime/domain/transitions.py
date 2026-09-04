"""The single state machine shared by all four execution strategies.

Every status change goes through these pure functions. A terminal status never
returns to a running status: recovery creates a new attempt or performs an
explicit terminal transition, and a graph revision creates a new work unit in a
new graph version instead of rewriting history.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from prp_runtime.domain.enums import (
    AttemptStatus,
    BridgeClaimStatus,
    ExecutionStrategy,
    MergeLedgerStatus,
    ReservationStatus,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    WorkUnitStatus,
)

__all__ = [
    "ATTEMPT_TRANSITIONS",
    "BRIDGE_CLAIM_TRANSITIONS",
    "AttemptNotAllowedError",
    "DomainTransitionError",
    "IllegalStatusTransitionError",
    "TOOL_CALL_TRANSITIONS",
    "RUN_TRANSITIONS",
    "RESERVATION_TRANSITIONS",
    "MERGE_TRANSITIONS",
    "RecoveryAction",
    "RecoveryDecision",
    "RunCompletionNotAllowedError",
    "STRATEGY_CONTROL_STRENGTH",
    "StrategyEscalationNotAllowedError",
    "WORK_UNIT_TRANSITIONS",
    "assert_can_start_attempt",
    "can_escalate_strategy",
    "can_start_attempt",
    "can_transition_attempt",
    "can_transition_bridge_claim",
    "can_transition_tool_call",
    "can_transition_run",
    "can_transition_reservation",
    "can_transition_merge",
    "can_transition_work_unit",
    "control_strength",
    "decide_attempt_recovery",
    "decide_reservation_recovery",
    "decide_run_recovery",
    "decide_tool_call_recovery",
    "decide_work_unit_recovery",
    "escalate_strategy",
    "mark_attempt_unconfirmed",
    "recover_attempt_on_restart",
    "resolve_run_outcome",
    "transition_attempt",
    "transition_bridge_claim",
    "transition_tool_call",
    "transition_run",
    "transition_reservation",
    "transition_merge",
    "transition_work_unit",
]


class RecoveryAction(StrEnum):
    """The safe operation selected for one persisted entity on restart."""

    CONTINUE = "continue"
    INTERRUPT = "interrupt"
    TERMINATE = "terminate"
    PRESERVE = "preserve"
    RELEASE = "release"
    BLOCK = "block"


RecoveryStatus = (
    RunStatus
    | WorkUnitStatus
    | AttemptStatus
    | ReservationStatus
    | ToolCallStatus
)


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """A pure, explainable restart decision.

    ``target_status`` is the only status a recovery worker may write.  A
    ``BLOCK`` decision keeps the recorded status and requires diagnosis rather
    than guessing a retry or an upstream outcome.
    """

    action: RecoveryAction
    target_status: RecoveryStatus
    reason: str
    retry_allowed: bool = False

    @property
    def is_blocked(self) -> bool:
        return self.action is RecoveryAction.BLOCK

    @property
    def can_continue(self) -> bool:
        return self.action is RecoveryAction.CONTINUE and self.retry_allowed


class DomainTransitionError(ValueError):
    """Base class for structured state machine violations."""


class IllegalStatusTransitionError(DomainTransitionError):
    """A status change that the state machine does not define."""

    def __init__(self, entity: str, current: str, target: str) -> None:
        self.entity = entity
        self.current = current
        self.target = target
        super().__init__(f"illegal {entity} transition: {current} -> {target}")


class AttemptNotAllowedError(DomainTransitionError):
    """A new attempt was requested while the run or work unit forbids it."""

    def __init__(
        self, reason: str, run_status: RunStatus, work_unit_status: WorkUnitStatus
    ) -> None:
        self.reason = reason
        self.run_status = run_status
        self.work_unit_status = work_unit_status
        super().__init__(
            f"attempt not allowed ({reason}): run={run_status.value} "
            f"work_unit={work_unit_status.value}"
        )


class RunCompletionNotAllowedError(DomainTransitionError):
    """A run cannot be completed with the given work unit statuses."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"run completion not allowed: {reason}")


class StrategyEscalationNotAllowedError(DomainTransitionError):
    """An escalation that is not a one-directional increase, or is manually pinned."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"strategy escalation not allowed: {reason}")


RUN_TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = MappingProxyType(
    {
        RunStatus.PENDING: frozenset(
            {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}
        ),
        RunStatus.RUNNING: frozenset(
            {
                RunStatus.CANCELLING,
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }
        ),
        RunStatus.CANCELLING: frozenset({RunStatus.CANCELLED}),
        RunStatus.SUCCEEDED: frozenset(),
        RunStatus.FAILED: frozenset(),
        RunStatus.CANCELLED: frozenset(),
    }
)

WORK_UNIT_TRANSITIONS: Mapping[WorkUnitStatus, frozenset[WorkUnitStatus]] = MappingProxyType(
    {
        WorkUnitStatus.PENDING: frozenset(
            {
                WorkUnitStatus.READY,
                WorkUnitStatus.BLOCKED,
                WorkUnitStatus.CANCELLED,
                WorkUnitStatus.INVALIDATED,
            }
        ),
        WorkUnitStatus.READY: frozenset(
            {
                WorkUnitStatus.RUNNING,
                WorkUnitStatus.BLOCKED,
                WorkUnitStatus.CANCELLED,
                WorkUnitStatus.INVALIDATED,
            }
        ),
        WorkUnitStatus.RUNNING: frozenset(
            {
                WorkUnitStatus.SUCCEEDED,
                WorkUnitStatus.FAILED,
                WorkUnitStatus.CANCELLED,
            }
        ),
        WorkUnitStatus.BLOCKED: frozenset(
            {
                WorkUnitStatus.READY,
                WorkUnitStatus.CANCELLED,
                WorkUnitStatus.INVALIDATED,
            }
        ),
        WorkUnitStatus.SUCCEEDED: frozenset(),
        WorkUnitStatus.FAILED: frozenset(),
        WorkUnitStatus.CANCELLED: frozenset(),
        WorkUnitStatus.INVALIDATED: frozenset(),
    }
)

BRIDGE_CLAIM_TRANSITIONS: Mapping[
    BridgeClaimStatus, frozenset[BridgeClaimStatus]
] = MappingProxyType(
    {
        BridgeClaimStatus.ACTIVE: frozenset(
            {
                BridgeClaimStatus.EXPIRED,
                BridgeClaimStatus.SETTLED,
                BridgeClaimStatus.RELEASED,
            }
        ),
        BridgeClaimStatus.EXPIRED: frozenset(),
        BridgeClaimStatus.SETTLED: frozenset(),
        BridgeClaimStatus.RELEASED: frozenset(),
    }
)

RESERVATION_TRANSITIONS: Mapping[
    ReservationStatus, frozenset[ReservationStatus]
] = MappingProxyType(
    {
        ReservationStatus.PENDING: frozenset({ReservationStatus.HELD}),
        ReservationStatus.HELD: frozenset(
            {
                ReservationStatus.SETTLED,
                ReservationStatus.RELEASED,
                ReservationStatus.EXPIRED,
            }
        ),
        ReservationStatus.SETTLED: frozenset(),
        ReservationStatus.RELEASED: frozenset(),
        ReservationStatus.EXPIRED: frozenset(),
    }
)

MERGE_TRANSITIONS: Mapping[
    MergeLedgerStatus, frozenset[MergeLedgerStatus]
] = MappingProxyType(
    {
        MergeLedgerStatus.PLANNED: frozenset(
            {MergeLedgerStatus.RUNNING, MergeLedgerStatus.CONFLICT, MergeLedgerStatus.UNKNOWN}
        ),
        MergeLedgerStatus.RUNNING: frozenset(
            {MergeLedgerStatus.MERGED, MergeLedgerStatus.CONFLICT, MergeLedgerStatus.UNKNOWN}
        ),
        MergeLedgerStatus.MERGED: frozenset(
            {MergeLedgerStatus.PROMOTED, MergeLedgerStatus.UNKNOWN}
        ),
        MergeLedgerStatus.PROMOTED: frozenset(),
        MergeLedgerStatus.CONFLICT: frozenset(),
        MergeLedgerStatus.UNKNOWN: frozenset(),
    }
)

ATTEMPT_TRANSITIONS: Mapping[AttemptStatus, frozenset[AttemptStatus]] = MappingProxyType(
    {
        AttemptStatus.PENDING: frozenset({AttemptStatus.RUNNING, AttemptStatus.CANCELLED}),
        AttemptStatus.RUNNING: frozenset(
            {
                AttemptStatus.SUCCEEDED,
                AttemptStatus.FAILED,
                AttemptStatus.CANCELLED,
                AttemptStatus.INTERRUPTED,
                AttemptStatus.UNKNOWN,
            }
        ),
        AttemptStatus.SUCCEEDED: frozenset(),
        AttemptStatus.FAILED: frozenset(),
        AttemptStatus.CANCELLED: frozenset(),
        AttemptStatus.INTERRUPTED: frozenset(),
        AttemptStatus.UNKNOWN: frozenset(),
    }
)

TOOL_CALL_TRANSITIONS: Mapping[
    ToolCallStatus, frozenset[ToolCallStatus]
] = MappingProxyType(
    {
        ToolCallStatus.REQUESTED: frozenset(
            {
                ToolCallStatus.AWAITING_APPROVAL,
                ToolCallStatus.RUNNING,
                ToolCallStatus.CANCELLED,
                ToolCallStatus.REJECTED,
            }
        ),
        ToolCallStatus.AWAITING_APPROVAL: frozenset(
            {
                ToolCallStatus.RUNNING,
                ToolCallStatus.CANCELLED,
                ToolCallStatus.REJECTED,
            }
        ),
        ToolCallStatus.RUNNING: frozenset(
            {
                ToolCallStatus.SUCCEEDED,
                ToolCallStatus.FAILED,
                ToolCallStatus.CANCELLED,
                ToolCallStatus.INTERRUPTED,
                ToolCallStatus.UNKNOWN,
            }
        ),
        ToolCallStatus.SUCCEEDED: frozenset(),
        ToolCallStatus.FAILED: frozenset(),
        ToolCallStatus.CANCELLED: frozenset(),
        ToolCallStatus.REJECTED: frozenset(),
        ToolCallStatus.INTERRUPTED: frozenset(),
        ToolCallStatus.UNKNOWN: frozenset(),
    }
)

STRATEGY_CONTROL_STRENGTH: Mapping[ExecutionStrategy, int] = MappingProxyType(
    {
        ExecutionStrategy.DIRECT: 0,
        ExecutionStrategy.CASCADE: 1,
        ExecutionStrategy.PLANNED: 2,
        ExecutionStrategy.PROGRESSIVE: 3,
    }
)


def can_transition_run(current: RunStatus, target: RunStatus) -> bool:
    """Whether a run may move from ``current`` to ``target``."""
    return target in RUN_TRANSITIONS[current]


def transition_run(current: RunStatus, target: RunStatus) -> RunStatus:
    """Return ``target`` if the run transition is legal, otherwise raise."""
    if not can_transition_run(current, target):
        raise IllegalStatusTransitionError("run", current.value, target.value)
    return target


def can_transition_work_unit(current: WorkUnitStatus, target: WorkUnitStatus) -> bool:
    """Whether a work unit may move from ``current`` to ``target``."""
    return target in WORK_UNIT_TRANSITIONS[current]


def transition_work_unit(current: WorkUnitStatus, target: WorkUnitStatus) -> WorkUnitStatus:
    """Return ``target`` if the work unit transition is legal, otherwise raise."""
    if not can_transition_work_unit(current, target):
        raise IllegalStatusTransitionError("work_unit", current.value, target.value)
    return target


def can_transition_bridge_claim(
    current: BridgeClaimStatus, target: BridgeClaimStatus
) -> bool:
    """Whether a Bridge claim lease may move to a different status."""
    return target in BRIDGE_CLAIM_TRANSITIONS[current]


def transition_bridge_claim(
    current: BridgeClaimStatus,
    target: BridgeClaimStatus,
    *,
    idempotent_terminal: bool = False,
) -> BridgeClaimStatus:
    """Return a legal claim status; terminal replay is explicit."""
    if idempotent_terminal and current is target and current.is_terminal:
        return current
    if not can_transition_bridge_claim(current, target):
        raise IllegalStatusTransitionError("bridge_claim", current.value, target.value)
    return target


def can_transition_reservation(
    current: ReservationStatus, target: ReservationStatus
) -> bool:
    """Whether a reservation may move to a different lifecycle state."""
    return target in RESERVATION_TRANSITIONS[current]


def transition_reservation(
    current: ReservationStatus,
    target: ReservationStatus,
    *,
    idempotent_terminal: bool = False,
) -> ReservationStatus:
    """Return a legal reservation state, optionally replaying one terminal state.

    Normal transitions reject self-transitions. A caller replaying the same
    terminal operation may opt into idempotence; a different terminal target is
    still rejected and never silently rewritten.
    """
    if idempotent_terminal and current is target and current.is_terminal:
        return current
    if not can_transition_reservation(current, target):
        raise IllegalStatusTransitionError("reservation", current.value, target.value)
    return target


def can_transition_merge(current: MergeLedgerStatus, target: MergeLedgerStatus) -> bool:
    """Whether a merge lifecycle may move to a different status."""
    return target in MERGE_TRANSITIONS[current]


def transition_merge(
    current: MergeLedgerStatus,
    target: MergeLedgerStatus,
    *,
    idempotent_terminal: bool = False,
) -> MergeLedgerStatus:
    """Return a legal merge status without reopening an uncertain terminal fact."""
    if idempotent_terminal and current is target and current.is_terminal:
        return current
    if not can_transition_merge(current, target):
        raise IllegalStatusTransitionError("merge", current.value, target.value)
    return target


def can_transition_attempt(current: AttemptStatus, target: AttemptStatus) -> bool:
    """Whether an attempt may move from ``current`` to ``target``."""
    return target in ATTEMPT_TRANSITIONS[current]


def transition_attempt(current: AttemptStatus, target: AttemptStatus) -> AttemptStatus:
    """Return ``target`` if the attempt transition is legal, otherwise raise."""
    if not can_transition_attempt(current, target):
        raise IllegalStatusTransitionError("attempt", current.value, target.value)
    return target


def can_transition_tool_call(current: ToolCallStatus, target: ToolCallStatus) -> bool:
    """Whether a tool call may move from ``current`` to ``target``."""
    return target in TOOL_CALL_TRANSITIONS[current]


def transition_tool_call(
    current: ToolCallStatus,
    target: ToolCallStatus,
    *,
    approved: bool | None = None,
    idempotent_terminal: bool = False,
) -> ToolCallStatus:
    """Return a legal tool-call state, with explicit approval and replay rules."""
    if idempotent_terminal and current is target and current.is_terminal:
        return current
    if current is ToolCallStatus.AWAITING_APPROVAL and target is ToolCallStatus.RUNNING:
        if approved is not True:
            raise IllegalStatusTransitionError(
                "tool_call", current.value, f"{target.value} without approval"
            )
    elif approved is True and current is ToolCallStatus.REQUESTED:
        raise IllegalStatusTransitionError(
            "tool_call", current.value, "RUNNING with approval bypass"
        )
    if not can_transition_tool_call(current, target):
        raise IllegalStatusTransitionError("tool_call", current.value, target.value)
    return target


def can_start_attempt(run_status: RunStatus, work_unit_status: WorkUnitStatus) -> bool:
    """Whether a new attempt may be created.

    A cancelled or cancelling run never accepts a new attempt. A work unit accepts
    attempts while it is ``READY`` (first attempt) or ``RUNNING`` (retry or model
    escalation).
    """
    if run_status not in (RunStatus.PENDING, RunStatus.RUNNING):
        return False
    return work_unit_status in (WorkUnitStatus.READY, WorkUnitStatus.RUNNING)


def assert_can_start_attempt(run_status: RunStatus, work_unit_status: WorkUnitStatus) -> None:
    """Raise ``AttemptNotAllowedError`` when a new attempt is forbidden."""
    if can_start_attempt(run_status, work_unit_status):
        return
    if run_status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
        reason = "run is cancelled"
    elif run_status.is_terminal:
        reason = "run is terminal"
    else:
        reason = "work unit is not runnable"
    raise AttemptNotAllowedError(reason, run_status, work_unit_status)


def decide_attempt_recovery(current: AttemptStatus) -> RecoveryDecision:
    """Classify one persisted attempt without consulting process memory.

    A pending attempt has no recorded dispatch and may be admitted again. A
    running attempt is interrupted, but never retried automatically because
    its provider-side outcome is unknown.
    """
    if current is AttemptStatus.PENDING:
        return RecoveryDecision(
            action=RecoveryAction.CONTINUE,
            target_status=current,
            reason="attempt was not dispatched before restart",
            retry_allowed=True,
        )
    if current is AttemptStatus.RUNNING:
        return RecoveryDecision(
            action=RecoveryAction.INTERRUPT,
            target_status=AttemptStatus.INTERRUPTED,
            reason="provider outcome is unknown after restart",
        )
    if current in (AttemptStatus.UNKNOWN, AttemptStatus.INTERRUPTED):
        return RecoveryDecision(
            action=RecoveryAction.PRESERVE,
            target_status=current,
            reason="provider outcome remains unconfirmed; retry requires diagnosis",
        )
    return RecoveryDecision(
        action=RecoveryAction.PRESERVE,
        target_status=current,
        reason="terminal attempt history is immutable",
    )


def decide_tool_call_recovery(
    current: ToolCallStatus,
    *,
    has_history_call: bool = True,
    has_pending_history: bool = True,
    has_approval: bool = True,
    has_result: bool | None = None,
    has_history_result: bool = False,
    has_conflict: bool = False,
) -> RecoveryDecision:
    """Classify one ToolCall without assuming an unconfirmed side effect.

    ``RUNNING`` is interrupted to ``UNKNOWN``. A pending request may continue
    only when its public history and approval facts agree. Terminal facts are
    preserved; a missing public result is a replay concern, never a new tool
    execution.
    """
    if has_conflict or not has_history_call:
        return RecoveryDecision(
            action=RecoveryAction.BLOCK,
            target_status=current,
            reason="tool call facts do not match public history",
        )
    if current is ToolCallStatus.RUNNING:
        if has_result is True:
            return RecoveryDecision(
                action=RecoveryAction.BLOCK,
                target_status=current,
                reason="running tool call has a conflicting terminal result",
            )
        return RecoveryDecision(
            action=RecoveryAction.INTERRUPT,
            target_status=ToolCallStatus.UNKNOWN,
            reason="tool side effect is unconfirmed after restart",
        )
    if current in (ToolCallStatus.UNKNOWN, ToolCallStatus.INTERRUPTED):
        return RecoveryDecision(
            action=RecoveryAction.BLOCK,
            target_status=current,
            reason="tool side effect remains unconfirmed; retry is forbidden",
        )
    if current in (ToolCallStatus.REQUESTED, ToolCallStatus.AWAITING_APPROVAL):
        if has_result is True:
            return RecoveryDecision(
                action=RecoveryAction.BLOCK,
                target_status=current,
                reason="non-terminal tool call has a conflicting terminal result",
            )
        if not has_pending_history or (
            current is ToolCallStatus.AWAITING_APPROVAL and not has_approval
        ):
            return RecoveryDecision(
                action=RecoveryAction.BLOCK,
                target_status=current,
                reason="pending tool call is missing approval or history facts",
            )
        return RecoveryDecision(
            action=RecoveryAction.CONTINUE,
            target_status=current,
            reason=(
                "approval wait remains resumable"
                if current is ToolCallStatus.AWAITING_APPROVAL
                else "requested tool call was not dispatched"
            ),
            retry_allowed=True,
        )
    if has_result is False:
        return RecoveryDecision(
            action=RecoveryAction.BLOCK,
            target_status=current,
            reason="terminal tool call has no persisted result",
        )
    if has_result and not has_history_result:
        return RecoveryDecision(
            action=RecoveryAction.PRESERVE,
            target_status=current,
            reason="terminal tool result is preserved for public history replay",
        )
    return RecoveryDecision(
        action=RecoveryAction.PRESERVE,
        target_status=current,
        reason="terminal tool call status is immutable",
    )


def decide_run_recovery(
    current: RunStatus,
    *,
    has_dispatched_attempt: bool = False,
    has_conflict: bool = False,
) -> RecoveryDecision:
    """Classify a persisted run using only durable restart facts."""
    if has_conflict or (current is RunStatus.PENDING and has_dispatched_attempt):
        return RecoveryDecision(
            action=RecoveryAction.BLOCK,
            target_status=current,
            reason="run status conflicts with its dispatched attempt history",
        )
    if current.is_terminal:
        if has_dispatched_attempt:
            return RecoveryDecision(
                action=RecoveryAction.BLOCK,
                target_status=current,
                reason="terminal run has an unaccounted dispatched attempt",
            )
        return RecoveryDecision(
            action=RecoveryAction.PRESERVE,
            target_status=current,
            reason="terminal run status is immutable",
        )
    if current is RunStatus.CANCELLING:
        return RecoveryDecision(
            action=RecoveryAction.PRESERVE,
            target_status=current,
            reason="cancellation is already in progress",
        )
    return RecoveryDecision(
        action=RecoveryAction.CONTINUE,
        target_status=current,
        reason="run may continue from durable pending work",
        retry_allowed=True,
    )


def decide_work_unit_recovery(
    current: WorkUnitStatus,
    *,
    has_running_attempt: bool = False,
    has_unconfirmed_attempt: bool = False,
    has_dispatched_attempt: bool = False,
    has_conflict: bool = False,
) -> RecoveryDecision:
    """Classify a work unit and reject contradictory durable facts."""
    if has_conflict:
        return RecoveryDecision(
            action=RecoveryAction.BLOCK,
            target_status=current,
            reason="work unit facts conflict and cannot be reconciled safely",
        )
    if current in (WorkUnitStatus.PENDING, WorkUnitStatus.READY):
        if has_dispatched_attempt:
            return RecoveryDecision(
                action=RecoveryAction.BLOCK,
                target_status=WorkUnitStatus.BLOCKED,
                reason="dispatch history conflicts with a not-started work unit",
            )
        return RecoveryDecision(
            action=RecoveryAction.CONTINUE,
            target_status=current,
            reason="work unit has no dispatched attempt and may be queued",
            retry_allowed=True,
        )
    if current is WorkUnitStatus.RUNNING:
        if has_unconfirmed_attempt:
            return RecoveryDecision(
                action=RecoveryAction.BLOCK,
                target_status=current,
                reason="work unit has an unconfirmed provider outcome",
            )
        if has_running_attempt:
            return RecoveryDecision(
                action=RecoveryAction.TERMINATE,
                target_status=WorkUnitStatus.FAILED,
                reason="running attempt was interrupted by restart",
            )
        return RecoveryDecision(
            action=RecoveryAction.BLOCK,
            target_status=current,
            reason="running work unit has no durable attempt outcome",
        )
    return RecoveryDecision(
        action=RecoveryAction.PRESERVE,
        target_status=current,
        reason="work unit status is not automatically rewritten on restart",
    )


def decide_reservation_recovery(
    current: ReservationStatus,
    *,
    has_inflight_attempt: bool = False,
    has_unconfirmed_attempt: bool = False,
    has_conflict: bool = False,
) -> RecoveryDecision:
    """Classify a reservation without releasing an uncertain side effect."""
    if has_conflict or (
        current.is_terminal and (has_inflight_attempt or has_unconfirmed_attempt)
    ):
        return RecoveryDecision(
            action=RecoveryAction.BLOCK,
            target_status=current,
            reason="reservation lifecycle conflicts with attempt history",
        )
    if current is ReservationStatus.PENDING:
        return RecoveryDecision(
            action=RecoveryAction.CONTINUE,
            target_status=current,
            reason="reservation was not held before restart",
            retry_allowed=True,
        )
    if current is ReservationStatus.HELD:
        if has_inflight_attempt or has_unconfirmed_attempt:
            return RecoveryDecision(
                action=RecoveryAction.PRESERVE,
                target_status=current,
                reason="reservation belongs to an unconfirmed attempt",
            )
        return RecoveryDecision(
            action=RecoveryAction.RELEASE,
            target_status=ReservationStatus.RELEASED,
            reason="held reservation has no corresponding attempt",
        )
    return RecoveryDecision(
        action=RecoveryAction.PRESERVE,
        target_status=current,
        reason="terminal reservation status is immutable",
    )


def recover_attempt_on_restart(current: AttemptStatus) -> AttemptStatus:
    """Map an attempt status to its post-restart status.

    A running attempt becomes ``INTERRUPTED``: the process cannot prove success or
    failure. Every other status is kept as recorded.
    """
    return decide_attempt_recovery(current).target_status  # type: ignore[return-value]


def mark_attempt_unconfirmed(current: AttemptStatus) -> AttemptStatus:
    """Map a running attempt whose upstream outcome cannot be confirmed to ``UNKNOWN``."""
    if current is AttemptStatus.RUNNING:
        return AttemptStatus.UNKNOWN
    if current.is_terminal:
        return current
    raise IllegalStatusTransitionError("attempt", current.value, AttemptStatus.UNKNOWN.value)


def resolve_run_outcome(
    work_unit_statuses: Iterable[WorkUnitStatus],
    *,
    cancel_requested: bool = False,
) -> RunStatus:
    """Decide the terminal run status from its required work units.

    Raises when any required work unit is still non-terminal, so a run can never
    complete while work is outstanding. ``INVALIDATED`` units are historical and
    do not decide the outcome.
    """
    statuses = tuple(work_unit_statuses)
    if not statuses:
        raise RunCompletionNotAllowedError("no work unit")
    outstanding = [status.value for status in statuses if not status.is_terminal]
    if outstanding:
        raise RunCompletionNotAllowedError("non-terminal work unit: " + ", ".join(outstanding))
    if cancel_requested:
        return RunStatus.CANCELLED
    if any(status is WorkUnitStatus.FAILED for status in statuses):
        return RunStatus.FAILED
    if any(status is WorkUnitStatus.CANCELLED for status in statuses):
        return RunStatus.CANCELLED
    if any(status is WorkUnitStatus.SUCCEEDED for status in statuses):
        return RunStatus.SUCCEEDED
    raise RunCompletionNotAllowedError("no work unit produced a result")


def control_strength(strategy: ExecutionStrategy) -> int:
    """Relative control strength of a strategy."""
    return STRATEGY_CONTROL_STRENGTH[strategy]


def can_escalate_strategy(current: ExecutionStrategy, target: ExecutionStrategy) -> bool:
    """Whether ``target`` is a strictly stronger control strategy than ``current``."""
    return control_strength(target) > control_strength(current)


def escalate_strategy(
    current: ExecutionStrategy,
    target: ExecutionStrategy,
    *,
    routing_policy: RoutingPolicy,
) -> ExecutionStrategy:
    """Return ``target`` if escalation is permitted, otherwise raise.

    Escalation is a product policy of ``AUTO`` routing only, and only increases
    control strength. A manually pinned strategy is never upgraded.
    """
    if routing_policy is not RoutingPolicy.AUTO:
        raise StrategyEscalationNotAllowedError("manual routing pins the strategy")
    if not can_escalate_strategy(current, target):
        raise StrategyEscalationNotAllowedError(
            f"{target.value} is not stronger than {current.value}"
        )
    return target
