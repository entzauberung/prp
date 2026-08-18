"""Domain identifiers and base value objects.

Identifiers carry a type prefix, so a work unit id can never be accepted where a
run id is required. Value objects are frozen and reject unknown fields.
"""

import re
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints

from prp_runtime.domain.enums import ResourceAccess

__all__ = [
    "APPROVAL_REQUEST_ID_PREFIX",
    "ATTEMPT_ID_PREFIX",
    "ApprovalRequestId",
    "AttemptId",
    "BRIDGE_CLAIM_ID_PREFIX",
    "BridgeClaimId",
    "LEASE_ID_PREFIX",
    "MERGE_ID_PREFIX",
    "PRINCIPAL_ID_PREFIX",
    "SESSION_ID_PREFIX",
    "LeaseId",
    "MergeId",
    "PrincipalId",
    "RESERVATION_ID_PREFIX",
    "ReservationId",
    "SNAPSHOT_ID_PREFIX",
    "TOOL_CALL_ID_PREFIX",
    "ModelRef",
    "ResourceClaim",
    "RUN_ID_PREFIX",
    "RunId",
    "SnapshotId",
    "SessionId",
    "ToolCallId",
    "UtcTimestamp",
    "WORK_UNIT_ID_PREFIX",
    "WorkUnitId",
    "WORKSPACE_ID_PREFIX",
    "new_bridge_claim_id",
    "new_attempt_id",
    "new_approval_request_id",
    "new_lease_id",
    "new_merge_id",
    "new_principal_id",
    "new_reservation_id",
    "new_snapshot_id",
    "new_session_id",
    "new_tool_call_id",
    "new_run_id",
    "new_work_unit_id",
    "new_workspace_id",
    "utc_now",
    "validate_attempt_id",
    "validate_bridge_claim_id",
    "validate_approval_request_id",
    "validate_lease_id",
    "validate_merge_id",
    "validate_principal_id",
    "validate_reservation_id",
    "validate_run_id",
    "validate_snapshot_id",
    "validate_session_id",
    "validate_tool_call_id",
    "validate_work_unit_id",
    "validate_workspace_id",
]

RUN_ID_PREFIX = "run_"
WORK_UNIT_ID_PREFIX = "wu_"
ATTEMPT_ID_PREFIX = "att_"
APPROVAL_REQUEST_ID_PREFIX = "apr_"
BRIDGE_CLAIM_ID_PREFIX = "claim_"
LEASE_ID_PREFIX = "lease_"
MERGE_ID_PREFIX = "merge_"
PRINCIPAL_ID_PREFIX = "prn_"
SESSION_ID_PREFIX = "ses_"
RESERVATION_ID_PREFIX = "res_"
WORKSPACE_ID_PREFIX = "ws_"
SNAPSHOT_ID_PREFIX = "snap_"
TOOL_CALL_ID_PREFIX = "tc_"

_ID_TAIL = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"

RunId = Annotated[str, StringConstraints(pattern=rf"^{RUN_ID_PREFIX}{_ID_TAIL}$")]
WorkUnitId = Annotated[str, StringConstraints(pattern=rf"^{WORK_UNIT_ID_PREFIX}{_ID_TAIL}$")]
AttemptId = Annotated[str, StringConstraints(pattern=rf"^{ATTEMPT_ID_PREFIX}{_ID_TAIL}$")]
ApprovalRequestId = Annotated[
    str, StringConstraints(pattern=rf"^{APPROVAL_REQUEST_ID_PREFIX}{_ID_TAIL}$")
]
BridgeClaimId = Annotated[
    str, StringConstraints(pattern=rf"^{BRIDGE_CLAIM_ID_PREFIX}{_ID_TAIL}$")
]
LeaseId = Annotated[str, StringConstraints(pattern=rf"^{LEASE_ID_PREFIX}{_ID_TAIL}$")]
MergeId = Annotated[str, StringConstraints(pattern=rf"^{MERGE_ID_PREFIX}{_ID_TAIL}$")]
PrincipalId = Annotated[str, StringConstraints(pattern=rf"^{PRINCIPAL_ID_PREFIX}{_ID_TAIL}$")]
SessionId = Annotated[str, StringConstraints(pattern=rf"^{SESSION_ID_PREFIX}{_ID_TAIL}$")]
ReservationId = Annotated[
    str, StringConstraints(pattern=rf"^{RESERVATION_ID_PREFIX}{_ID_TAIL}$")
]
WorkspaceId = Annotated[str, StringConstraints(pattern=rf"^{WORKSPACE_ID_PREFIX}{_ID_TAIL}$")]
SnapshotId = Annotated[str, StringConstraints(pattern=rf"^{SNAPSHOT_ID_PREFIX}{_ID_TAIL}$")]
ToolCallId = Annotated[str, StringConstraints(pattern=rf"^{TOOL_CALL_ID_PREFIX}{_ID_TAIL}$")]

_RUN_ID_RE = re.compile(rf"^{RUN_ID_PREFIX}{_ID_TAIL}$")
_WORK_UNIT_ID_RE = re.compile(rf"^{WORK_UNIT_ID_PREFIX}{_ID_TAIL}$")
_ATTEMPT_ID_RE = re.compile(rf"^{ATTEMPT_ID_PREFIX}{_ID_TAIL}$")
_APPROVAL_REQUEST_ID_RE = re.compile(rf"^{APPROVAL_REQUEST_ID_PREFIX}{_ID_TAIL}$")
_BRIDGE_CLAIM_ID_RE = re.compile(rf"^{BRIDGE_CLAIM_ID_PREFIX}{_ID_TAIL}$")
_LEASE_ID_RE = re.compile(rf"^{LEASE_ID_PREFIX}{_ID_TAIL}$")
_MERGE_ID_RE = re.compile(rf"^{MERGE_ID_PREFIX}{_ID_TAIL}$")
_PRINCIPAL_ID_RE = re.compile(rf"^{PRINCIPAL_ID_PREFIX}{_ID_TAIL}$")
_SESSION_ID_RE = re.compile(rf"^{SESSION_ID_PREFIX}{_ID_TAIL}$")
_RESERVATION_ID_RE = re.compile(rf"^{RESERVATION_ID_PREFIX}{_ID_TAIL}$")
_WORKSPACE_ID_RE = re.compile(rf"^{WORKSPACE_ID_PREFIX}{_ID_TAIL}$")
_SNAPSHOT_ID_RE = re.compile(rf"^{SNAPSHOT_ID_PREFIX}{_ID_TAIL}$")
_TOOL_CALL_ID_RE = re.compile(rf"^{TOOL_CALL_ID_PREFIX}{_ID_TAIL}$")

NonEmptyName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
]
ResourceKey = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]


def new_run_id() -> str:
    """Generate a fresh run id."""
    return f"{RUN_ID_PREFIX}{uuid4().hex}"


def new_work_unit_id() -> str:
    """Generate a fresh work unit id."""
    return f"{WORK_UNIT_ID_PREFIX}{uuid4().hex}"


def new_attempt_id() -> str:
    """Generate a fresh attempt id."""
    return f"{ATTEMPT_ID_PREFIX}{uuid4().hex}"


def new_approval_request_id() -> str:
    """Generate a fresh approval request id."""
    return f"{APPROVAL_REQUEST_ID_PREFIX}{uuid4().hex}"


def new_bridge_claim_id() -> str:
    """Generate a fresh Native Bridge claim id."""
    return f"{BRIDGE_CLAIM_ID_PREFIX}{uuid4().hex}"


def new_lease_id() -> str:
    """Generate a fresh capability lease id."""
    return f"{LEASE_ID_PREFIX}{uuid4().hex}"


def new_merge_id() -> str:
    """Generate a fresh durable merge lifecycle id."""
    return f"{MERGE_ID_PREFIX}{uuid4().hex}"


def new_principal_id() -> str:
    """Generate a non-secret principal identity."""
    return f"{PRINCIPAL_ID_PREFIX}{uuid4().hex}"


def new_reservation_id() -> str:
    """Generate a fresh reservation id."""
    return f"{RESERVATION_ID_PREFIX}{uuid4().hex}"


def new_workspace_id() -> str:
    """Generate a fresh workspace id."""
    return f"{WORKSPACE_ID_PREFIX}{uuid4().hex}"


def new_snapshot_id() -> str:
    """Generate a fresh snapshot id."""
    return f"{SNAPSHOT_ID_PREFIX}{uuid4().hex}"


def new_session_id() -> str:
    """Generate a fresh session identity."""
    return f"{SESSION_ID_PREFIX}{uuid4().hex}"


def new_tool_call_id() -> str:
    """Generate a fresh tool call id."""
    return f"{TOOL_CALL_ID_PREFIX}{uuid4().hex}"


def _validate_identifier(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def validate_run_id(value: str) -> str:
    """Return the value if it is a well formed run id, otherwise raise."""
    return _validate_identifier(value, _RUN_ID_RE, "run id")


def validate_work_unit_id(value: str) -> str:
    """Return the value if it is a well formed work unit id, otherwise raise."""
    return _validate_identifier(value, _WORK_UNIT_ID_RE, "work unit id")


def validate_attempt_id(value: str) -> str:
    """Return the value if it is a well formed attempt id, otherwise raise."""
    return _validate_identifier(value, _ATTEMPT_ID_RE, "attempt id")


def validate_approval_request_id(value: str) -> str:
    """Return the value if it is a well formed approval request id."""
    return _validate_identifier(value, _APPROVAL_REQUEST_ID_RE, "approval request id")


def validate_bridge_claim_id(value: str) -> str:
    """Return the value if it is a well formed Bridge claim id."""
    return _validate_identifier(value, _BRIDGE_CLAIM_ID_RE, "bridge claim id")


def validate_lease_id(value: str) -> str:
    """Return the value if it is a well formed lease id."""
    return _validate_identifier(value, _LEASE_ID_RE, "lease id")


def validate_merge_id(value: str) -> str:
    """Return the value if it is a well formed merge id."""
    return _validate_identifier(value, _MERGE_ID_RE, "merge id")


def validate_principal_id(value: str) -> str:
    """Return the value if it is a well formed principal id."""
    return _validate_identifier(value, _PRINCIPAL_ID_RE, "principal id")


def validate_reservation_id(value: str) -> str:
    """Return the value if it is a well formed reservation id, otherwise raise."""
    return _validate_identifier(value, _RESERVATION_ID_RE, "reservation id")


def validate_workspace_id(value: str) -> str:
    """Return the value if it is a well formed workspace id, otherwise raise."""
    return _validate_identifier(value, _WORKSPACE_ID_RE, "workspace id")


def validate_snapshot_id(value: str) -> str:
    """Return the value if it is a well formed snapshot id, otherwise raise."""
    return _validate_identifier(value, _SNAPSHOT_ID_RE, "snapshot id")


def validate_session_id(value: str) -> str:
    """Return the value if it is a well formed session id."""
    return _validate_identifier(value, _SESSION_ID_RE, "session id")


def validate_tool_call_id(value: str) -> str:
    """Return the value if it is a well formed tool call id, otherwise raise."""
    return _validate_identifier(value, _TOOL_CALL_ID_RE, "tool call id")


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC timestamp."""
    return datetime.now(tz=UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


UtcTimestamp = Annotated[datetime, AfterValidator(_ensure_utc)]


class ResourceClaim(BaseModel):
    """A declared claim on one logical resource."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: ResourceKey
    access: ResourceAccess

    def conflicts_with(self, other: "ResourceClaim") -> bool:
        """Two claims conflict when they share a resource and either one writes."""
        if self.resource != other.resource:
            return False
        return ResourceAccess.WRITE in (self.access, other.access)


class ModelRef(BaseModel):
    """A provider-qualified model reference.

    The provider name selects a server-side configured endpoint. A request can
    never supply a URL.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: NonEmptyName
    model: NonEmptyName

    @property
    def identifier(self) -> str:
        return f"{self.provider}/{self.model}"
