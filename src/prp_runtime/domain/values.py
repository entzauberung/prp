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
    "ATTEMPT_ID_PREFIX",
    "AttemptId",
    "ModelRef",
    "ResourceClaim",
    "RUN_ID_PREFIX",
    "RunId",
    "UtcTimestamp",
    "WORK_UNIT_ID_PREFIX",
    "WorkUnitId",
    "new_attempt_id",
    "new_run_id",
    "new_work_unit_id",
    "utc_now",
    "validate_attempt_id",
    "validate_run_id",
    "validate_work_unit_id",
]

RUN_ID_PREFIX = "run_"
WORK_UNIT_ID_PREFIX = "wu_"
ATTEMPT_ID_PREFIX = "att_"

_ID_TAIL = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"

RunId = Annotated[str, StringConstraints(pattern=rf"^{RUN_ID_PREFIX}{_ID_TAIL}$")]
WorkUnitId = Annotated[str, StringConstraints(pattern=rf"^{WORK_UNIT_ID_PREFIX}{_ID_TAIL}$")]
AttemptId = Annotated[str, StringConstraints(pattern=rf"^{ATTEMPT_ID_PREFIX}{_ID_TAIL}$")]

_RUN_ID_RE = re.compile(rf"^{RUN_ID_PREFIX}{_ID_TAIL}$")
_WORK_UNIT_ID_RE = re.compile(rf"^{WORK_UNIT_ID_PREFIX}{_ID_TAIL}$")
_ATTEMPT_ID_RE = re.compile(rf"^{ATTEMPT_ID_PREFIX}{_ID_TAIL}$")

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
