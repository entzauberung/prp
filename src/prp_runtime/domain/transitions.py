"""The single state machine shared by all four execution strategies.

Every status change goes through these pure functions. A terminal status never
returns to a running status: recovery creates a new attempt or performs an
explicit terminal transition, and a graph revision creates a new work unit in a
new graph version instead of rewriting history.
"""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)

__all__ = [
    "ATTEMPT_TRANSITIONS",
    "AttemptNotAllowedError",
    "DomainTransitionError",
    "IllegalStatusTransitionError",
    "RUN_TRANSITIONS",
    "RunCompletionNotAllowedError",
    "STRATEGY_CONTROL_STRENGTH",
    "StrategyEscalationNotAllowedError",
    "WORK_UNIT_TRANSITIONS",
    "assert_can_start_attempt",
    "can_escalate_strategy",
    "can_start_attempt",
    "can_transition_attempt",
    "can_transition_run",
    "can_transition_work_unit",
    "control_strength",
    "escalate_strategy",
    "mark_attempt_unconfirmed",
    "recover_attempt_on_restart",
    "resolve_run_outcome",
    "transition_attempt",
    "transition_run",
    "transition_work_unit",
]


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


def can_transition_attempt(current: AttemptStatus, target: AttemptStatus) -> bool:
    """Whether an attempt may move from ``current`` to ``target``."""
    return target in ATTEMPT_TRANSITIONS[current]


def transition_attempt(current: AttemptStatus, target: AttemptStatus) -> AttemptStatus:
    """Return ``target`` if the attempt transition is legal, otherwise raise."""
    if not can_transition_attempt(current, target):
        raise IllegalStatusTransitionError("attempt", current.value, target.value)
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


def recover_attempt_on_restart(current: AttemptStatus) -> AttemptStatus:
    """Map an attempt status to its post-restart status.

    A running attempt becomes ``INTERRUPTED``: the process cannot prove success or
    failure. Every other status is kept as recorded.
    """
    if current is AttemptStatus.RUNNING:
        return AttemptStatus.INTERRUPTED
    return current


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
