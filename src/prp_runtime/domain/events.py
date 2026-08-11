"""The append-only run event ledger.

An event is an envelope: run id, sequence, type, timestamp and a restricted JSON
payload. Payloads are produced from domain models; exception objects, stack
traces and provider payloads are never stored.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum, unique
from types import MappingProxyType

from pydantic import BaseModel, Field, JsonValue, model_validator

from prp_runtime.domain.errors import ErrorCode, StateError
from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.values import RunId, UtcTimestamp, utc_now

__all__ = [
    "EVENT_REQUIRED_KEYS",
    "EventType",
    "RunEvent",
    "assert_sequence_chain",
    "next_sequence",
    "payload_from_model",
]


@unique
class EventType(StrEnum):
    """The complete set of persisted event types."""

    RUN_CREATED = "RUN_CREATED"
    RUN_STARTED = "RUN_STARTED"
    RUN_SUCCEEDED = "RUN_SUCCEEDED"
    RUN_FAILED = "RUN_FAILED"
    RUN_CANCELLING = "RUN_CANCELLING"
    RUN_CANCELLED = "RUN_CANCELLED"
    RUN_RESUMED = "RUN_RESUMED"

    STRATEGY_SELECTED = "STRATEGY_SELECTED"
    STRATEGY_ESCALATED = "STRATEGY_ESCALATED"
    CONTROLLER_DECISION = "CONTROLLER_DECISION"

    PLAN_PROPOSED = "PLAN_PROPOSED"
    PLAN_COMMITTED = "PLAN_COMMITTED"
    PLAN_REJECTED = "PLAN_REJECTED"
    PLAN_REVISED = "PLAN_REVISED"

    WORK_UNIT_CREATED = "WORK_UNIT_CREATED"
    WORK_UNIT_READY = "WORK_UNIT_READY"
    WORK_UNIT_STARTED = "WORK_UNIT_STARTED"
    WORK_UNIT_SUCCEEDED = "WORK_UNIT_SUCCEEDED"
    WORK_UNIT_FAILED = "WORK_UNIT_FAILED"
    WORK_UNIT_BLOCKED = "WORK_UNIT_BLOCKED"
    WORK_UNIT_CANCELLED = "WORK_UNIT_CANCELLED"
    WORK_UNIT_INVALIDATED = "WORK_UNIT_INVALIDATED"

    ATTEMPT_STARTED = "ATTEMPT_STARTED"
    ATTEMPT_SUCCEEDED = "ATTEMPT_SUCCEEDED"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    ATTEMPT_CANCELLED = "ATTEMPT_CANCELLED"
    ATTEMPT_INTERRUPTED = "ATTEMPT_INTERRUPTED"
    ATTEMPT_UNKNOWN = "ATTEMPT_UNKNOWN"

    ARTIFACT_PRODUCED = "ARTIFACT_PRODUCED"
    EVIDENCE_RECORDED = "EVIDENCE_RECORDED"

    USAGE_UPDATED = "USAGE_UPDATED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    ERROR_RAISED = "ERROR_RAISED"


_WORK_UNIT = frozenset({"work_unit_id"})
_ATTEMPT = frozenset({"work_unit_id", "attempt_id"})

EVENT_REQUIRED_KEYS: Mapping[EventType, frozenset[str]] = MappingProxyType(
    {
        EventType.RUN_CREATED: frozenset({"request"}),
        EventType.RUN_STARTED: frozenset(),
        EventType.RUN_SUCCEEDED: frozenset(),
        EventType.RUN_FAILED: frozenset({"error"}),
        EventType.RUN_CANCELLING: frozenset(),
        EventType.RUN_CANCELLED: frozenset(),
        EventType.RUN_RESUMED: frozenset(),
        EventType.STRATEGY_SELECTED: frozenset({"strategy"}),
        EventType.STRATEGY_ESCALATED: frozenset(
            {"from_strategy", "to_strategy", "reason"}
        ),
        EventType.CONTROLLER_DECISION: frozenset({"decision"}),
        EventType.PLAN_PROPOSED: frozenset({"graph_version"}),
        EventType.PLAN_COMMITTED: frozenset({"graph_version", "work_unit_ids"}),
        EventType.PLAN_REJECTED: frozenset({"graph_version", "reason"}),
        EventType.PLAN_REVISED: frozenset({"graph_version"}),
        EventType.WORK_UNIT_CREATED: _WORK_UNIT,
        EventType.WORK_UNIT_READY: _WORK_UNIT,
        EventType.WORK_UNIT_STARTED: _WORK_UNIT,
        EventType.WORK_UNIT_SUCCEEDED: _WORK_UNIT,
        EventType.WORK_UNIT_FAILED: frozenset({"work_unit_id", "error"}),
        EventType.WORK_UNIT_BLOCKED: frozenset({"work_unit_id", "reason"}),
        EventType.WORK_UNIT_CANCELLED: _WORK_UNIT,
        EventType.WORK_UNIT_INVALIDATED: frozenset({"work_unit_id", "reason"}),
        EventType.ATTEMPT_STARTED: _ATTEMPT,
        EventType.ATTEMPT_SUCCEEDED: _ATTEMPT,
        EventType.ATTEMPT_FAILED: frozenset({"work_unit_id", "attempt_id", "error"}),
        EventType.ATTEMPT_CANCELLED: _ATTEMPT,
        EventType.ATTEMPT_INTERRUPTED: _ATTEMPT,
        EventType.ATTEMPT_UNKNOWN: _ATTEMPT,
        EventType.ARTIFACT_PRODUCED: frozenset({"work_unit_id", "artifact_id"}),
        # ``result``, not a boolean: FAIL and INCONCLUSIVE share one boolean, so a
        # ledger entry carrying only the projection would leave a reader unable to
        # tell a check that failed from one that could not decide.
        EventType.EVIDENCE_RECORDED: frozenset({"work_unit_id", "evidence_id", "result"}),
        EventType.USAGE_UPDATED: frozenset({"usage"}),
        EventType.BUDGET_EXHAUSTED: frozenset({"error"}),
        EventType.ERROR_RAISED: frozenset({"error"}),
    }
)


def payload_from_model(key: str, model: BaseModel) -> dict[str, JsonValue]:
    """Wrap a domain model as a JSON-safe payload entry."""
    return {key: model.model_dump(mode="json")}


class RunEvent(DomainModel):
    """One immutable ledger entry of one run."""

    run_id: RunId
    sequence: int = Field(ge=1)
    event_type: EventType
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    timestamp: UtcTimestamp = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _payload_has_required_keys(self) -> "RunEvent":
        missing = EVENT_REQUIRED_KEYS[self.event_type] - set(self.payload)
        if missing:
            raise ValueError(
                f"{self.event_type.value} payload is missing: " + ", ".join(sorted(missing))
            )
        return self


def next_sequence(previous: int | None) -> int:
    """Return the next sequence number for a run."""
    if previous is None:
        return 1
    if previous < 1:
        raise StateError(
            f"sequence must be a positive integer, got {previous}",
            code=ErrorCode.EVENT_SEQUENCE_INVALID,
        )
    return previous + 1


def assert_sequence_chain(events: Iterable[RunEvent]) -> None:
    """Verify one run's ledger starts at 1 and increases by exactly one.

    Raises ``StateError`` with ``EVENT_SEQUENCE_INVALID`` on a gap, a repeat, a
    regression or a mixed run id.
    """
    expected = 1
    run_id: str | None = None
    for event in events:
        if run_id is None:
            run_id = event.run_id
        elif event.run_id != run_id:
            raise StateError(
                "event ledger mixes run ids",
                code=ErrorCode.EVENT_SEQUENCE_INVALID,
            )
        if event.sequence != expected:
            raise StateError(
                f"expected sequence {expected}, got {event.sequence}",
                code=ErrorCode.EVENT_SEQUENCE_INVALID,
            )
        expected += 1
