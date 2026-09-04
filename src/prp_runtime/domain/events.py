"""The append-only run event ledger.

An event is an envelope: run id, sequence, type, timestamp and a restricted JSON
payload. Payloads are produced from domain models; exception objects, stack
traces and provider payloads are never stored.
"""

import json
from collections.abc import Iterable, Mapping
from enum import StrEnum, unique
from types import MappingProxyType

from pydantic import BaseModel, Field, JsonValue, model_validator

from prp_runtime.domain.enums import ToolCallStatus
from prp_runtime.domain.errors import ErrorCode, StateError
from prp_runtime.domain.models import DomainModel, MergeLedger
from prp_runtime.domain.values import RunId, UtcTimestamp, utc_now

__all__ = [
    "EVENT_REQUIRED_KEYS",
    "EventType",
    "MAX_EVENT_PAYLOAD_BYTES",
    "MAX_EVENT_PAYLOAD_KEYS",
    "RunEvent",
    "assert_sequence_chain",
    "next_sequence",
    "payload_from_model",
    "payload_from_agent_history",
    "payload_from_bridge_claim",
    "payload_from_bridge_client_skip",
    "payload_from_bridge_result_wake",
    "payload_from_merge_ledger",
    "payload_from_remote_assignment",
    "payload_from_remote_wait",
    "payload_from_tool_call",
    "payload_from_tool_result",
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

    ROUND_CREATED = "ROUND_CREATED"
    ROUND_VERIFIED = "ROUND_VERIFIED"
    ROUND_FAILED = "ROUND_FAILED"

    MERGE_PLANNED = "MERGE_PLANNED"
    MERGE_STARTED = "MERGE_STARTED"
    MERGE_MERGED = "MERGE_MERGED"
    MERGE_PROMOTED = "MERGE_PROMOTED"
    MERGE_CONFLICT = "MERGE_CONFLICT"
    MERGE_UNKNOWN = "MERGE_UNKNOWN"

    WORK_UNIT_CREATED = "WORK_UNIT_CREATED"
    WORK_UNIT_READY = "WORK_UNIT_READY"
    WORK_UNIT_STARTED = "WORK_UNIT_STARTED"
    WORK_UNIT_SUCCEEDED = "WORK_UNIT_SUCCEEDED"
    WORK_UNIT_FAILED = "WORK_UNIT_FAILED"
    WORK_UNIT_BLOCKED = "WORK_UNIT_BLOCKED"
    WORK_UNIT_CANCELLED = "WORK_UNIT_CANCELLED"
    WORK_UNIT_INVALIDATED = "WORK_UNIT_INVALIDATED"
    WORK_UNIT_REUSED = "WORK_UNIT_REUSED"

    RESERVATION_CREATED = "RESERVATION_CREATED"
    RESERVATION_HELD = "RESERVATION_HELD"
    RESERVATION_SETTLED = "RESERVATION_SETTLED"
    RESERVATION_RELEASED = "RESERVATION_RELEASED"
    RESERVATION_EXPIRED = "RESERVATION_EXPIRED"

    ATTEMPT_STARTED = "ATTEMPT_STARTED"
    ATTEMPT_SUCCEEDED = "ATTEMPT_SUCCEEDED"
    ATTEMPT_FAILED = "ATTEMPT_FAILED"
    ATTEMPT_CANCELLED = "ATTEMPT_CANCELLED"
    ATTEMPT_INTERRUPTED = "ATTEMPT_INTERRUPTED"
    ATTEMPT_UNKNOWN = "ATTEMPT_UNKNOWN"

    TOOL_CALL_REQUESTED = "TOOL_CALL_REQUESTED"
    TOOL_CALL_AWAITING_APPROVAL = "TOOL_CALL_AWAITING_APPROVAL"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_SUCCEEDED = "TOOL_CALL_SUCCEEDED"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    TOOL_CALL_CANCELLED = "TOOL_CALL_CANCELLED"
    TOOL_CALL_REJECTED = "TOOL_CALL_REJECTED"
    TOOL_CALL_INTERRUPTED = "TOOL_CALL_INTERRUPTED"
    TOOL_CALL_UNKNOWN = "TOOL_CALL_UNKNOWN"

    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_DECIDED = "APPROVAL_DECIDED"
    LEASE_CREATED = "LEASE_CREATED"
    LEASE_REVOKED = "LEASE_REVOKED"
    LEASE_EXPIRED = "LEASE_EXPIRED"

    ARTIFACT_PRODUCED = "ARTIFACT_PRODUCED"
    EVIDENCE_RECORDED = "EVIDENCE_RECORDED"

    USAGE_UPDATED = "USAGE_UPDATED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    ERROR_RAISED = "ERROR_RAISED"

    AGENT_HISTORY_RECORDED = "AGENT_HISTORY_RECORDED"

    BRIDGE_CLAIM_CREATED = "BRIDGE_CLAIM_CREATED"
    BRIDGE_CLAIM_EXPIRED = "BRIDGE_CLAIM_EXPIRED"
    BRIDGE_CLAIM_SETTLED = "BRIDGE_CLAIM_SETTLED"
    BRIDGE_CLAIM_RELEASED = "BRIDGE_CLAIM_RELEASED"
    BRIDGE_CLIENT_SKIPPED = "BRIDGE_CLIENT_SKIPPED"
    BRIDGE_RESULT_WAKE = "BRIDGE_RESULT_WAKE"


_WORK_UNIT = frozenset({"work_unit_id"})
_ATTEMPT = frozenset({"work_unit_id", "attempt_id"})
_TOOL_CALL = frozenset({"call_id", "status"})
_MERGE = frozenset({"merge_id", "status", "input_digest"})

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
        EventType.ROUND_CREATED: frozenset(
            {"round_id", "round_index", "graph_version", "status"}
        ),
        EventType.ROUND_VERIFIED: frozenset(
            {"round_id", "round_index", "graph_version", "status", "merged_snapshot_id"}
        ),
        EventType.ROUND_FAILED: frozenset(
            {"round_id", "round_index", "graph_version", "status", "reason"}
        ),
        EventType.MERGE_PLANNED: _MERGE,
        EventType.MERGE_STARTED: _MERGE,
        EventType.MERGE_MERGED: _MERGE | frozenset(
            {"merged_snapshot_id", "merged_content_hash"}
        ),
        EventType.MERGE_PROMOTED: _MERGE
        | frozenset({"merged_snapshot_id", "merged_content_hash", "promoted_content_hash"}),
        EventType.MERGE_CONFLICT: _MERGE,
        EventType.MERGE_UNKNOWN: _MERGE,
        EventType.WORK_UNIT_CREATED: _WORK_UNIT,
        EventType.WORK_UNIT_READY: _WORK_UNIT,
        EventType.WORK_UNIT_STARTED: _WORK_UNIT,
        EventType.WORK_UNIT_SUCCEEDED: _WORK_UNIT,
        EventType.WORK_UNIT_FAILED: frozenset({"work_unit_id", "error"}),
        EventType.WORK_UNIT_BLOCKED: frozenset({"work_unit_id", "reason"}),
        EventType.WORK_UNIT_CANCELLED: _WORK_UNIT,
        EventType.WORK_UNIT_INVALIDATED: frozenset({"work_unit_id", "reason"}),
        EventType.WORK_UNIT_REUSED: frozenset(
            {
                "work_unit_id",
                "source_work_unit_id",
                "source_attempt_id",
                "attempt_id",
                "lineage_key",
                "reason",
                "source_artifact_ids",
                "artifact_ids",
            }
        ),
        EventType.RESERVATION_CREATED: frozenset({"reservation_id", "dispatch_key"}),
        EventType.RESERVATION_HELD: frozenset({"reservation_id"}),
        EventType.RESERVATION_SETTLED: frozenset({"reservation_id"}),
        EventType.RESERVATION_RELEASED: frozenset({"reservation_id"}),
        EventType.RESERVATION_EXPIRED: frozenset({"reservation_id"}),
        EventType.ATTEMPT_STARTED: _ATTEMPT,
        EventType.ATTEMPT_SUCCEEDED: _ATTEMPT,
        EventType.ATTEMPT_FAILED: frozenset({"work_unit_id", "attempt_id", "error"}),
        EventType.ATTEMPT_CANCELLED: _ATTEMPT,
        EventType.ATTEMPT_INTERRUPTED: _ATTEMPT,
        EventType.ATTEMPT_UNKNOWN: _ATTEMPT,
        EventType.TOOL_CALL_REQUESTED: frozenset(
            {"call_id", "tool_name", "effect", "status"}
        ),
        EventType.TOOL_CALL_AWAITING_APPROVAL: frozenset(
            {"call_id", "tool_name", "effect", "status", "reason"}
        ),
        EventType.TOOL_CALL_STARTED: _TOOL_CALL,
        EventType.TOOL_CALL_SUCCEEDED: _TOOL_CALL,
        EventType.TOOL_CALL_FAILED: frozenset({"call_id", "status", "error"}),
        EventType.TOOL_CALL_CANCELLED: _TOOL_CALL,
        EventType.TOOL_CALL_REJECTED: frozenset({"call_id", "status", "error"}),
        EventType.TOOL_CALL_INTERRUPTED: frozenset({"call_id", "status", "error"}),
        EventType.TOOL_CALL_UNKNOWN: frozenset({"call_id", "status", "error"}),
        EventType.APPROVAL_REQUESTED: frozenset(),
        EventType.APPROVAL_DECIDED: frozenset(),
        EventType.LEASE_CREATED: frozenset(),
        EventType.LEASE_REVOKED: frozenset(),
        EventType.LEASE_EXPIRED: frozenset(),
        EventType.ARTIFACT_PRODUCED: frozenset({"work_unit_id", "artifact_id"}),
        # ``result``, not a boolean: FAIL and INCONCLUSIVE share one boolean, so a
        # ledger entry carrying only the projection would leave a reader unable to
        # tell a check that failed from one that could not decide.
        EventType.EVIDENCE_RECORDED: frozenset({"work_unit_id", "evidence_id", "result"}),
        EventType.USAGE_UPDATED: frozenset({"usage"}),
        EventType.BUDGET_EXHAUSTED: frozenset({"error"}),
        EventType.ERROR_RAISED: frozenset({"error"}),
        EventType.AGENT_HISTORY_RECORDED: frozenset(
            {"sequence", "kind", "idempotency_key"}
        ),
        EventType.BRIDGE_CLAIM_CREATED: frozenset(
            {"claim_id", "call_id", "client_id", "status", "expires_at"}
        ),
        EventType.BRIDGE_CLAIM_EXPIRED: frozenset(
            {"claim_id", "call_id", "client_id", "status", "expires_at"}
        ),
        EventType.BRIDGE_CLAIM_SETTLED: frozenset(
            {"claim_id", "call_id", "client_id", "status", "expires_at"}
        ),
        EventType.BRIDGE_CLAIM_RELEASED: frozenset(
            {"claim_id", "call_id", "client_id", "status", "expires_at"}
        ),
        EventType.BRIDGE_CLIENT_SKIPPED: frozenset({"call_id", "client_id", "reason"}),
        EventType.BRIDGE_RESULT_WAKE: frozenset({"call_id", "status"}),
    }
)

_TOOL_EVENT_STATUS: Mapping[EventType, ToolCallStatus] = MappingProxyType(
    {
        EventType.TOOL_CALL_REQUESTED: ToolCallStatus.REQUESTED,
        EventType.TOOL_CALL_AWAITING_APPROVAL: ToolCallStatus.AWAITING_APPROVAL,
        EventType.TOOL_CALL_STARTED: ToolCallStatus.RUNNING,
        EventType.TOOL_CALL_SUCCEEDED: ToolCallStatus.SUCCEEDED,
        EventType.TOOL_CALL_FAILED: ToolCallStatus.FAILED,
        EventType.TOOL_CALL_CANCELLED: ToolCallStatus.CANCELLED,
        EventType.TOOL_CALL_REJECTED: ToolCallStatus.REJECTED,
        EventType.TOOL_CALL_INTERRUPTED: ToolCallStatus.INTERRUPTED,
        EventType.TOOL_CALL_UNKNOWN: ToolCallStatus.UNKNOWN,
    }
)
_TOOL_EVENT_TYPES = frozenset(_TOOL_EVENT_STATUS)
_BRIDGE_CLAIM_EVENT_STATUS: Mapping[EventType, str] = MappingProxyType(
    {
        EventType.BRIDGE_CLAIM_CREATED: "ACTIVE",
        EventType.BRIDGE_CLAIM_EXPIRED: "EXPIRED",
        EventType.BRIDGE_CLAIM_SETTLED: "SETTLED",
        EventType.BRIDGE_CLAIM_RELEASED: "RELEASED",
    }
)
_BRIDGE_CLAIM_EVENT_TYPES = frozenset(_BRIDGE_CLAIM_EVENT_STATUS)
_MERGE_EVENT_STATUS: Mapping[EventType, str] = MappingProxyType(
    {
        EventType.MERGE_PLANNED: "PLANNED",
        EventType.MERGE_STARTED: "RUNNING",
        EventType.MERGE_MERGED: "MERGED",
        EventType.MERGE_PROMOTED: "PROMOTED",
        EventType.MERGE_CONFLICT: "CONFLICT",
        EventType.MERGE_UNKNOWN: "UNKNOWN",
    }
)
_MERGE_EVENT_TYPES = frozenset(_MERGE_EVENT_STATUS)
MAX_EVENT_PAYLOAD_BYTES = 64 * 1024
MAX_EVENT_PAYLOAD_KEYS = 128
_FORBIDDEN_PUBLIC_PAYLOAD_KEYS = frozenset(
    {
        "absolute_path",
        "api_key",
        "apikey",
        "arguments",
        "authorization",
        "chain_of_thought",
        "cot",
        "credential",
        "env",
        "host_path",
        "password",
        "path",
        "provider_body",
        "raw_provider_body",
        "raw_request",
        "raw_response",
        "reasoning",
        "root",
        "secret",
        "thoughts",
        "token",
    }
)
_FORBIDDEN_TOOL_PAYLOAD_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "arguments",
        "callable",
        "credential",
        "env",
        "password",
        "result",
        "secret",
        "shell",
        "stderr",
        "stdout",
        "token",
        "output",
    }
)
_MAX_TOOL_EVENT_OUTPUT_BYTES = 8 * 1024
_MAX_TOOL_EVENT_PAYLOAD_BYTES = 16 * 1024


def _assert_no_private_payload_keys(value: JsonValue) -> None:
    """Reject keys that would persist private or host-specific facts."""
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if nested_key.lower() in _FORBIDDEN_PUBLIC_PAYLOAD_KEYS:
                raise ValueError(f"event payload must not contain {nested_key}")
            _assert_no_private_payload_keys(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _assert_no_private_payload_keys(nested_value)


def _assert_safe_tool_payload(value: JsonValue, *, key: str = "payload") -> None:
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if nested_key.lower() in _FORBIDDEN_TOOL_PAYLOAD_KEYS:
                raise ValueError(f"tool event payload must not contain {nested_key}")
            _assert_safe_tool_payload(nested_value, key=nested_key)
    elif isinstance(value, list):
        for nested_value in value:
            _assert_safe_tool_payload(nested_value, key=key)
    elif key in {"output", "output_preview"} and isinstance(value, str):
        if len(value.encode("utf-8")) > _MAX_TOOL_EVENT_OUTPUT_BYTES:
            raise ValueError("tool event output exceeds the size limit")


def payload_from_model(key: str, model: BaseModel) -> dict[str, JsonValue]:
    """Wrap a domain model as a JSON-safe payload entry."""
    return {key: model.model_dump(mode="json")}


def payload_from_merge_ledger(ledger: MergeLedger) -> dict[str, JsonValue]:
    """Project merge lifecycle facts without paths, errors or payload bodies."""
    payload: dict[str, JsonValue] = {
        "merge_id": ledger.merge_id,
        "workspace_id": ledger.workspace_id,
        "base_snapshot_id": ledger.base_snapshot_id,
        "change_set_ids": list(ledger.change_set_ids),
        "input_digest": ledger.input_digest,
        "status": ledger.status.value,
    }
    if ledger.merged_snapshot_id is not None:
        payload["merged_snapshot_id"] = ledger.merged_snapshot_id
    if ledger.merged_content_hash is not None:
        payload["merged_content_hash"] = ledger.merged_content_hash
    if ledger.promoted_content_hash is not None:
        payload["promoted_content_hash"] = ledger.promoted_content_hash
    return payload


def payload_from_agent_history(record: BaseModel) -> dict[str, JsonValue]:
    """Project a history row to audit metadata without copying its item body."""
    data = record.model_dump(mode="json")
    item = data.get("item")
    if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
        raise ValueError("agent history record has an invalid item")
    return {
        "sequence": data["sequence"],
        "kind": item["kind"],
        "idempotency_key": data["idempotency_key"],
    }


def payload_from_bridge_claim(claim: BaseModel) -> dict[str, JsonValue]:
    """Project a Bridge claim to public lifecycle metadata.

    Fingerprints, roots and secrets stay out of the event payload. The assigned
    client identity is included so assignment facts survive without private data.
    """
    data = claim.model_dump(mode="json")
    return {
        "claim_id": data["claim_id"],
        "call_id": data["call_id"],
        "client_id": data["client_id"],
        "status": data["status"],
        "expires_at": data["expires_at"],
    }


def payload_from_bridge_client_skip(
    *, call_id: str, client_id: str, reason: str
) -> dict[str, JsonValue]:
    """Project one bounded skip reason for a concrete call."""
    return {"call_id": call_id, "client_id": client_id, "reason": reason}


def payload_from_bridge_result_wake(*, call_id: str, status: str) -> dict[str, JsonValue]:
    """Project one result-wake without roots, credentials or private authority."""
    return {"call_id": call_id, "status": status}


def payload_from_remote_wait(facts: BaseModel) -> dict[str, JsonValue]:
    """Project a remote wait without approval, result or private authority."""

    data = facts.model_dump(mode="json")
    payload: dict[str, JsonValue] = {
        "call_id": data["call_id"],
        "reason": data["reason"],
    }
    tool_call_id = data.get("tool_call_id")
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    workspace_id = data.get("workspace_id")
    if workspace_id:
        payload["workspace_id"] = workspace_id
    client_id = data.get("client_id")
    if client_id:
        payload["client_id"] = client_id
    return payload


def payload_from_remote_assignment(assignment: BaseModel) -> dict[str, JsonValue]:
    """Project a remote assignment without roots, credentials or provider data."""

    data = assignment.model_dump(mode="json")
    payload: dict[str, JsonValue] = {
        "call_id": data["call_id"],
        "status": data["status"],
        "tool_name": data["tool_name"],
        "workspace_id": data["workspace_id"],
    }
    client_id = data.get("client_id")
    if client_id:
        payload["client_id"] = client_id
    return payload


def payload_from_tool_call(call: BaseModel) -> dict[str, JsonValue]:
    """Build a lifecycle payload without copying tool arguments."""
    data = call.model_dump(mode="json", exclude={"arguments"})
    return {
        "call_id": data["call_id"],
        "status": data["status"],
        "tool_name": data["tool_name"],
        "effect": data["effect"],
    }


def payload_from_tool_result(result: BaseModel) -> dict[str, JsonValue]:
    """Build a bounded lifecycle payload without exposing unrestricted output."""
    data = result.model_dump(mode="json", exclude={"result", "output"})
    payload: dict[str, JsonValue] = {
        "call_id": data["call_id"],
        "status": data["status"],
        "truncated": data["truncated"],
        "changed_paths": data["changed_paths"],
        "exit_code": data["exit_code"],
    }
    if data.get("error") is not None:
        payload["error"] = data["error"]
    return payload


class RunEvent(DomainModel):
    """One immutable ledger entry of one run."""

    run_id: RunId
    sequence: int = Field(ge=1)
    event_type: EventType
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    timestamp: UtcTimestamp = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _payload_has_required_keys(self) -> "RunEvent":
        if len(self.payload) > MAX_EVENT_PAYLOAD_KEYS:
            raise ValueError("event payload contains too many keys")
        _assert_no_private_payload_keys(self.payload)
        try:
            encoded_payload = json.dumps(
                self.payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("event payload must contain standard JSON") from error
        if len(encoded_payload) > MAX_EVENT_PAYLOAD_BYTES:
            raise ValueError("event payload exceeds the size limit")
        missing = EVENT_REQUIRED_KEYS[self.event_type] - set(self.payload)
        if missing:
            raise ValueError(
                f"{self.event_type.value} payload is missing: " + ", ".join(sorted(missing))
            )
        if self.event_type in _TOOL_EVENT_TYPES:
            _assert_safe_tool_payload(self.payload)
            if (
                len(
                    json.dumps(
                        self.payload,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                )
                > _MAX_TOOL_EVENT_PAYLOAD_BYTES
            ):
                raise ValueError("tool event payload exceeds the size limit")
            expected = _TOOL_EVENT_STATUS[self.event_type]
            raw_status = self.payload.get("status")
            if not isinstance(raw_status, str):
                raise ValueError("tool event payload has an invalid status")
            try:
                actual = ToolCallStatus(raw_status)
            except ValueError as error:
                raise ValueError("tool event payload has an invalid status") from error
            if actual is not expected:
                raise ValueError(
                    f"{self.event_type.value} payload status must be {expected.value}"
                )
        elif self.event_type in _BRIDGE_CLAIM_EVENT_TYPES:
            if set(self.payload) != EVENT_REQUIRED_KEYS[self.event_type]:
                raise ValueError("Bridge claim event payload contains unexpected fields")
            raw_status = self.payload.get("status")
            if raw_status != _BRIDGE_CLAIM_EVENT_STATUS[self.event_type]:
                raise ValueError(
                    f"{self.event_type.value} payload status must be "
                    f"{_BRIDGE_CLAIM_EVENT_STATUS[self.event_type]}"
                )
        elif self.event_type in _MERGE_EVENT_TYPES:
            allowed = {
                "merge_id",
                "workspace_id",
                "base_snapshot_id",
                "change_set_ids",
                "input_digest",
                "status",
                "merged_snapshot_id",
                "merged_content_hash",
                "promoted_content_hash",
            }
            if set(self.payload) - allowed:
                raise ValueError("merge event payload contains unsupported fields")
            raw_status = self.payload.get("status")
            if raw_status != _MERGE_EVENT_STATUS[self.event_type]:
                raise ValueError(
                    f"{self.event_type.value} payload status must be "
                    f"{_MERGE_EVENT_STATUS[self.event_type]}"
                )
        elif self.event_type is EventType.AGENT_HISTORY_RECORDED:
            if set(self.payload) != EVENT_REQUIRED_KEYS[self.event_type]:
                raise ValueError("agent history event payload must contain metadata only")
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
