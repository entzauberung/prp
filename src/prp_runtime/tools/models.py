"""Closed, protocol-independent contracts for tool calls and results."""

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated

from pydantic import Field, JsonValue, StringConstraints, model_validator

from prp_runtime.domain.enums import BridgeClaimStatus, ToolCallStatus, ToolEffect
from prp_runtime.domain.models import DomainModel, ErrorCategory, ErrorInfo
from prp_runtime.domain.values import (
    BridgeClaimId,
    RunId,
    SessionId,
    SnapshotId,
    ToolCallId,
    UtcTimestamp,
    WorkspaceId,
    WorkUnitId,
    new_bridge_claim_id,
)

__all__ = [
    "MAX_TOOL_ARGUMENT_BYTES",
    "MAX_TOOL_OUTPUT_BYTES",
    "MAX_CHANGED_PATHS",
    "BridgeClaim",
    "ToolCall",
    "ToolResult",
    "validate_tool_rejection_reason",
]

MAX_TOOL_ARGUMENT_BYTES = 64 * 1024
MAX_TOOL_OUTPUT_BYTES = 256 * 1024
MAX_CHANGED_PATHS = 10_000
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_TOOL_REJECTION_REASON_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset(
    {"api_key", "apikey", "callable", "credential", "env", "password", "secret", "shell", "token"}
)

ToolName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
RelativePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
]


def _json_size(value: Mapping[str, JsonValue]) -> int:
    return len(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _assert_no_forbidden_keys(value: JsonValue) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"tool JSON must not contain {key}")
            _assert_no_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_keys(nested)


def _assert_relative_path(path: str) -> None:
    if (
        path.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", path)
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("changed path must be relative POSIX syntax")


def validate_tool_rejection_reason(reason: str) -> str:
    """Accept only a bounded public reason code, never free-form detail."""
    if not isinstance(reason, str) or not _TOOL_REJECTION_REASON_RE.fullmatch(reason):
        raise ValueError("tool rejection reason must be a safe reason code")
    return reason


class ToolCall(DomainModel):
    """One immutable request to a named, registered tool."""

    call_id: ToolCallId
    run_id: RunId
    work_unit_id: WorkUnitId
    tool_name: ToolName
    effect: ToolEffect
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.REQUESTED
    snapshot_id: SnapshotId | None = None
    requested_at: UtcTimestamp

    @model_validator(mode="after")
    def _contract_is_bounded_and_closed(self) -> "ToolCall":
        if not _TOOL_NAME_RE.fullmatch(self.tool_name):
            raise ValueError("tool_name is not a registered-name format")
        _assert_no_forbidden_keys(self.arguments)
        if _json_size(self.arguments) > MAX_TOOL_ARGUMENT_BYTES:
            raise ValueError("tool arguments exceed the size limit")
        return self

    def transition(
        self,
        target: ToolCallStatus,
        *,
        approved: bool | None = None,
        idempotent_terminal: bool = False,
    ) -> "ToolCall":
        """Return a new call after applying the shared lifecycle rules."""
        from prp_runtime.domain.transitions import transition_tool_call

        status = transition_tool_call(
            self.status,
            target,
            approved=approved,
            idempotent_terminal=idempotent_terminal,
        )
        return self.model_copy(update={"status": status})


class BridgeClaim(DomainModel):
    """One owner- and session-scoped lease for a Native Bridge call.

    The claim carries only opaque identities and a request fingerprint. Host
    roots, bearer tokens and executable details remain outside this contract.
    """

    claim_id: BridgeClaimId = Field(default_factory=new_bridge_claim_id)
    call_id: ToolCallId
    run_id: RunId
    session_id: SessionId
    workspace_id: WorkspaceId
    owner_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
    ]
    claimant_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
    ]
    idempotency_key: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
    ]
    fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    claimed_at: UtcTimestamp
    expires_at: UtcTimestamp
    status: BridgeClaimStatus = BridgeClaimStatus.ACTIVE
    closed_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _scope_and_lifecycle_are_closed(self) -> "BridgeClaim":
        if self.expires_at <= self.claimed_at:
            raise ValueError("bridge claim expiry must be after claimed_at")
        if self.closed_at is not None and self.closed_at < self.claimed_at:
            raise ValueError("bridge claim closed_at must not precede claimed_at")
        if self.status is BridgeClaimStatus.ACTIVE:
            if self.closed_at is not None:
                raise ValueError("active bridge claim cannot have closed_at")
        elif self.closed_at is None:
            raise ValueError("terminal bridge claim requires closed_at")
        if self.status is BridgeClaimStatus.EXPIRED:
            if self.closed_at is None or self.closed_at < self.expires_at:
                raise ValueError("expired bridge claim must close at or after expires_at")
        if not _FINGERPRINT_RE.fullmatch(self.fingerprint):
            raise ValueError("bridge claim fingerprint must be a lowercase SHA-256")
        return self

    @property
    def is_terminal(self) -> bool:
        """Whether this immutable claim has left its lease-bearing state."""
        return self.status.is_terminal

    def is_active_at(self, at: datetime) -> bool:
        """Return whether the claim can accept a result at an aware time."""
        if at.tzinfo is None:
            raise ValueError("claim check time must be timezone-aware")
        return (
            self.status is BridgeClaimStatus.ACTIVE
            and self.claimed_at <= at < self.expires_at
        )

    def expire(self, *, at: UtcTimestamp) -> "BridgeClaim":
        """Close an active claim after its lease window has elapsed."""
        if self.status is not BridgeClaimStatus.ACTIVE:
            raise ValueError("only an active bridge claim can expire")
        if at < self.expires_at:
            raise ValueError("bridge claim cannot expire before expires_at")
        return self.model_copy(
            update={"status": BridgeClaimStatus.EXPIRED, "closed_at": at}
        )

    def settle(self, *, at: UtcTimestamp) -> "BridgeClaim":
        """Mark an active claim as having its result durably submitted."""
        if self.status is not BridgeClaimStatus.ACTIVE:
            raise ValueError("only an active bridge claim can settle")
        if at < self.claimed_at or at >= self.expires_at:
            raise ValueError("bridge claim must settle inside its lease window")
        return self.model_copy(
            update={"status": BridgeClaimStatus.SETTLED, "closed_at": at}
        )

    def release(self, *, at: UtcTimestamp) -> "BridgeClaim":
        """Close an active claim without asserting that the tool ran."""
        if self.status is not BridgeClaimStatus.ACTIVE:
            raise ValueError("only an active bridge claim can release")
        if at < self.claimed_at:
            raise ValueError("bridge claim cannot close before claimed_at")
        return self.model_copy(update={"status": BridgeClaimStatus.RELEASED, "closed_at": at})


class ToolResult(DomainModel):
    """The terminal, bounded observation of one ToolCall."""

    call_id: ToolCallId
    status: ToolCallStatus
    result: dict[str, JsonValue] | None = None
    output: Annotated[str, StringConstraints(max_length=MAX_TOOL_OUTPUT_BYTES)] = ""
    truncated: bool = False
    changed_paths: tuple[RelativePath, ...] = ()
    exit_code: int | None = None
    error: ErrorInfo | None = None
    completed_at: UtcTimestamp

    @model_validator(mode="after")
    def _result_is_terminal_and_safe(self) -> "ToolResult":
        if not self.status.is_terminal:
            raise ValueError("tool result status must be terminal")
        if self.status in (
            ToolCallStatus.FAILED,
            ToolCallStatus.REJECTED,
            ToolCallStatus.INTERRUPTED,
            ToolCallStatus.UNKNOWN,
        ):
            if self.error is None:
                raise ValueError("failed or rejected tool result requires error")
        elif self.error is not None:
            raise ValueError("successful or cancelled tool result must not carry error")
        if self.result is not None:
            _assert_no_forbidden_keys(self.result)
            if _json_size(self.result) > MAX_TOOL_OUTPUT_BYTES:
                raise ValueError("tool result exceeds the size limit")
        if len(self.changed_paths) > MAX_CHANGED_PATHS:
            raise ValueError("too many changed paths")
        for path in self.changed_paths:
            _assert_relative_path(path)
        return self

    @classmethod
    def from_call(
        cls,
        call: ToolCall,
        *,
        status: ToolCallStatus,
        completed_at: UtcTimestamp,
        result: dict[str, JsonValue] | None = None,
        output: str = "",
        truncated: bool = False,
        changed_paths: tuple[RelativePath, ...] = (),
        exit_code: int | None = None,
        error: ErrorInfo | None = None,
    ) -> "ToolResult":
        """Create a result only after the call has entered ``RUNNING``."""
        if call.status is not ToolCallStatus.RUNNING:
            raise ValueError("tool result requires a running or approved call")
        return cls(
            call_id=call.call_id,
            status=status,
            result=result,
            output=output,
            truncated=truncated,
            changed_paths=changed_paths,
            exit_code=exit_code,
            error=error,
            completed_at=completed_at,
        )

    @classmethod
    def from_rejected_call(
        cls,
        call: ToolCall,
        *,
        reason: str,
        completed_at: UtcTimestamp,
    ) -> "ToolResult":
        """Create a rejection directly from a pre-execution call."""
        if call.status not in (
            ToolCallStatus.REQUESTED,
            ToolCallStatus.AWAITING_APPROVAL,
        ):
            raise ValueError("tool rejection requires a pre-execution call")
        safe_reason = validate_tool_rejection_reason(reason)
        return cls(
            call_id=call.call_id,
            status=ToolCallStatus.REJECTED,
            result={"error": safe_reason},
            output="tool request rejected",
            error=ErrorInfo(
                category=ErrorCategory.INVALID_REQUEST,
                message=safe_reason,
            ),
            completed_at=completed_at,
        )
