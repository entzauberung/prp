"""Domain enumerations.

These values are the only vocabulary for strategies, roles, lifecycle states and
resource access. No third-party API field ever enters this module.
"""

from enum import StrEnum, unique

__all__ = [
    "AgentMode",
    "AttemptStatus",
    "BridgeClaimStatus",
    "ExecutionLocation",
    "ExecutionStrategy",
    "IsolationMode",
    "MergeLedgerStatus",
    "ModelRole",
    "ResourceAccess",
    "ReservationStatus",
    "RoutingPolicy",
    "RunStatus",
    "ToolCallStatus",
    "ToolEffect",
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
class AgentMode(StrEnum):
    """The agent's authorization posture for tool-capable execution."""

    NORMAL = "NORMAL"
    AUTO = "AUTO"
    PLAN = "PLAN"
    YOLO = "YOLO"


@unique
class IsolationMode(StrEnum):
    """The isolation boundary used for execution."""

    SANDBOXED = "SANDBOXED"
    HOST = "HOST"


@unique
class MergeLedgerStatus(StrEnum):
    """Durable merge lifecycle states, including uncertain recovery."""

    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    MERGED = "MERGED"
    PROMOTED = "PROMOTED"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in {
            MergeLedgerStatus.PROMOTED,
            MergeLedgerStatus.CONFLICT,
            MergeLedgerStatus.UNKNOWN,
        }


@unique
class ExecutionLocation(StrEnum):
    """Where the agent executes.

    ``CLOUD`` is remote server-side execution. ``BRIDGE`` is a claimed native
    client workspace. ``LOCAL`` keeps controller, tools and workspace in one
    process.
    """

    CLOUD = "CLOUD"
    BRIDGE = "BRIDGE"
    LOCAL = "LOCAL"


@unique
class ToolEffect(StrEnum):
    """The declared side-effect class of one tool operation."""

    READ = "READ"
    WRITE = "WRITE"
    COMMAND = "COMMAND"
    NETWORK = "NETWORK"


@unique
class ToolCallStatus(StrEnum):
    """Lifecycle of a protocol-independent tool call."""

    REQUESTED = "REQUESTED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    INTERRUPTED = "INTERRUPTED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ToolCallStatus.SUCCEEDED,
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
            ToolCallStatus.REJECTED,
            ToolCallStatus.INTERRUPTED,
            ToolCallStatus.UNKNOWN,
        }


@unique
class BridgeClaimStatus(StrEnum):
    """Lifecycle of one Native Bridge claim.

    ``ACTIVE`` is the only lease-bearing state. Terminal states are immutable;
    an expired claim may be recorded for audit but can never be revived.
    """

    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    SETTLED = "SETTLED"
    RELEASED = "RELEASED"

    @property
    def is_terminal(self) -> bool:
        return self is not BridgeClaimStatus.ACTIVE


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
class ReservationStatus(StrEnum):
    """Lifecycle of one attempt/capacity reservation."""

    PENDING = "PENDING"
    HELD = "HELD"
    SETTLED = "SETTLED"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ReservationStatus.SETTLED,
            ReservationStatus.RELEASED,
            ReservationStatus.EXPIRED,
        }


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
