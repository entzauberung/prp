"""Closed status machines. Terminal states do not return."""

from collections.abc import Mapping
from types import MappingProxyType

from prp.vocabulary import AttemptStatus, ExecutionStrategy, RunStatus, WorkUnitStatus

__all__ = [
    "ATTEMPT_TRANSITIONS",
    "RUN_TRANSITIONS",
    "STRATEGY_CONTROL_STRENGTH",
    "WORK_UNIT_TRANSITIONS",
    "can_escalate_strategy",
    "can_transition_attempt",
    "can_transition_run",
    "can_transition_work_unit",
]

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

WORK_UNIT_TRANSITIONS: Mapping[WorkUnitStatus, frozenset[WorkUnitStatus]] = (
    MappingProxyType(
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
)

ATTEMPT_TRANSITIONS: Mapping[AttemptStatus, frozenset[AttemptStatus]] = MappingProxyType(
    {
        AttemptStatus.PENDING: frozenset(
            {AttemptStatus.RUNNING, AttemptStatus.CANCELLED, AttemptStatus.FAILED}
        ),
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
    return target in RUN_TRANSITIONS[current]


def can_transition_work_unit(current: WorkUnitStatus, target: WorkUnitStatus) -> bool:
    return target in WORK_UNIT_TRANSITIONS[current]


def can_transition_attempt(current: AttemptStatus, target: AttemptStatus) -> bool:
    return target in ATTEMPT_TRANSITIONS[current]


def can_escalate_strategy(current: ExecutionStrategy, target: ExecutionStrategy) -> bool:
    return STRATEGY_CONTROL_STRENGTH[target] > STRATEGY_CONTROL_STRENGTH[current]
