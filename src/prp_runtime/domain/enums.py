"""Domain enumerations.

These values are the only vocabulary for strategies, roles, lifecycle states and
resource access. No third-party API field ever enters this module.
"""

from enum import StrEnum, unique

__all__ = [
    "AttemptStatus",
    "ExecutionStrategy",
    "ModelRole",
    "ResourceAccess",
    "RoutingPolicy",
    "RunStatus",
    "WorkUnitStatus",
]


@unique
class ExecutionStrategy(StrEnum):
    """The four execution strategies. There is no fifth strategy."""

    DIRECT = "DIRECT"
    CASCADE = "CASCADE"
    PLANNED = "PLANNED"
    PROGRESSIVE = "PROGRESSIVE"


@unique
class RoutingPolicy(StrEnum):
    """How the execution strategy is chosen.

    ``AUTO`` lets the controller pick the weakest sufficient strategy. ``MANUAL``
    pins the caller's strategy and forbids automatic escalation.
    """

    AUTO = "AUTO"
    MANUAL = "MANUAL"


@unique
class ModelRole(StrEnum):
    """The role a model plays in one call.

    A planner may only propose. A worker executes one work unit. A verifier
    judges a produced artifact.
    """

    PLANNER = "PLANNER"
    WORKER = "WORKER"
    VERIFIER = "VERIFIER"


@unique
class ResourceAccess(StrEnum):
    """Declared access mode of a resource claim."""

    READ = "READ"
    WRITE = "WRITE"


@unique
class RunStatus(StrEnum):
    """Lifecycle of a run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_RUN_STATUSES


@unique
class WorkUnitStatus(StrEnum):
    """Lifecycle of a work unit inside the execution graph."""

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INVALIDATED = "INVALIDATED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_WORK_UNIT_STATUSES


@unique
class AttemptStatus(StrEnum):
    """Lifecycle of a single provider call.

    ``INTERRUPTED`` marks an attempt that was running when the process stopped.
    ``UNKNOWN`` marks an attempt whose upstream outcome cannot be confirmed.
    Neither is treated as success or failure.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_ATTEMPT_STATUSES


_TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
)

_TERMINAL_WORK_UNIT_STATUSES: frozenset[WorkUnitStatus] = frozenset(
    {
        WorkUnitStatus.SUCCEEDED,
        WorkUnitStatus.FAILED,
        WorkUnitStatus.CANCELLED,
        WorkUnitStatus.INVALIDATED,
    }
)

_TERMINAL_ATTEMPT_STATUSES: frozenset[AttemptStatus] = frozenset(
    {
        AttemptStatus.SUCCEEDED,
        AttemptStatus.FAILED,
        AttemptStatus.CANCELLED,
        AttemptStatus.INTERRUPTED,
        AttemptStatus.UNKNOWN,
    }
)
