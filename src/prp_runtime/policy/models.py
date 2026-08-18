"""Closed approval and capability lease contracts.

Policy data is deliberately more structured than a free-form permission note:
every grant names its workspace, tools, effects, paths, command classes, budget,
and expiry. A model or external provider never supplies an issuer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, StrictBool, StringConstraints, field_validator, model_validator

from prp_runtime.domain.enums import ExecutionLocation, IsolationMode, ToolEffect
from prp_runtime.domain.models import DomainModel, Evidence
from prp_runtime.domain.values import (
    ApprovalRequestId,
    LeaseId,
    PrincipalId,
    RunId,
    ToolCallId,
    UtcTimestamp,
    WorkspaceId,
    new_approval_request_id,
    new_lease_id,
)
from prp_runtime.tools.models import MAX_TOOL_OUTPUT_BYTES

if TYPE_CHECKING:
    from prp_runtime.tools.models import ToolCall

__all__ = [
    "ApprovalDecision",
    "ApprovalIssuer",
    "ApprovalOutcome",
    "ApprovalRequest",
    "CapabilityBudget",
    "CapabilityScope",
    "CommandClass",
    "DevEvidenceMetadata",
    "DevExecutionMode",
    "DevScope",
    "Lease",
    "LeaseStatus",
    "guard_dev_scope",
    "serialize_dev_evidence",
]

ToolName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
Reason = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
PathScope = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
]


@unique
class ApprovalIssuer(StrEnum):
    """The trusted principal that issued a decision or lease."""

    USER = "USER"
    SERVER = "SERVER"


@unique
class ApprovalOutcome(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"


@unique
class CommandClass(StrEnum):
    """Finite command categories; raw shell text is never a capability."""

    READ_ONLY = "READ_ONLY"
    TEST = "TEST"
    LINT = "LINT"
    FORMAT = "FORMAT"
    BUILD = "BUILD"
    PACKAGE = "PACKAGE"


@unique
class DevExecutionMode(StrEnum):
    """The only execution shapes that may carry DEV evidence."""

    HOST = "HOST"
    BRIDGE = "BRIDGE"
    TEXT_ONLY = "TEXT_ONLY"


@unique
class LeaseStatus(StrEnum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class CapabilityBudget(DomainModel):
    """Finite limits attached to one capability scope."""

    max_calls: int = Field(gt=0, le=10_000)
    max_output_bytes: int = Field(gt=0, le=MAX_TOOL_OUTPUT_BYTES)
    max_wall_clock_ms: int = Field(gt=0, le=86_400_000)


class DevScope(DomainModel):
    """The single source of truth for a non-production execution scope.

    DEV never represents a sandbox capability.  Its public form contains
    identity and execution facts only; temporary host roots stay in the
    runtime context that owns them.
    """

    principal_id: PrincipalId
    workspace_id: WorkspaceId
    mode: DevExecutionMode
    isolation_mode: IsolationMode
    execution_location: ExecutionLocation
    text_only: StrictBool = False
    dev_only: Literal[True]
    temporary_workspace: Literal[True] = True

    @model_validator(mode="after")
    def _dev_scope_is_non_sandboxed_and_consistent(self) -> DevScope:
        if self.isolation_mode is IsolationMode.SANDBOXED:
            raise ValueError("DEV scope cannot request SANDBOXED isolation")
        if self.isolation_mode is not IsolationMode.HOST:
            raise ValueError("DEV scope requires HOST isolation")
        if self.mode is DevExecutionMode.BRIDGE:
            if self.execution_location is not ExecutionLocation.BRIDGE:
                raise ValueError("BRIDGE DEV scope requires BRIDGE execution location")
        elif self.mode is DevExecutionMode.HOST:
            if self.text_only:
                raise ValueError("HOST DEV scope cannot be text-only")
        elif not self.text_only:
            raise ValueError("TEXT_ONLY DEV scope requires text_only=true")
        if self.mode is not DevExecutionMode.TEXT_ONLY and self.text_only:
            raise ValueError("text_only is only valid for TEXT_ONLY DEV scope")
        return self

    def assert_owner(
        self,
        *,
        principal_id: PrincipalId,
        workspace_id: WorkspaceId,
    ) -> None:
        """Reject evidence or runtime use outside this scope's owner."""
        if principal_id != self.principal_id or workspace_id != self.workspace_id:
            raise ValueError("DEV scope owner or workspace mismatch")

    def evidence_metadata(
        self,
        *,
        principal_id: PrincipalId | None = None,
        workspace_id: WorkspaceId | None = None,
    ) -> DevEvidenceMetadata:
        """Derive the public DEV label from this scope, never from caller text."""
        return DevEvidenceMetadata.from_scope(
            self,
            principal_id=principal_id,
            workspace_id=workspace_id,
        )


class DevEvidenceMetadata(DomainModel):
    """Public evidence facts that make DEV results non-production proof."""

    dev_only: Literal[True]
    principal_id: PrincipalId
    workspace_id: WorkspaceId
    mode: DevExecutionMode
    isolation_mode: IsolationMode
    execution_location: ExecutionLocation

    @classmethod
    def from_scope(
        cls,
        scope: DevScope,
        *,
        principal_id: PrincipalId | None = None,
        workspace_id: WorkspaceId | None = None,
    ) -> DevEvidenceMetadata:
        actual_principal = (
            principal_id if principal_id is not None else scope.principal_id
        )
        actual_workspace = (
            workspace_id if workspace_id is not None else scope.workspace_id
        )
        scope.assert_owner(
            principal_id=actual_principal,
            workspace_id=actual_workspace,
        )
        return cls(
            principal_id=scope.principal_id,
            workspace_id=scope.workspace_id,
            mode=scope.mode,
            isolation_mode=scope.isolation_mode,
            execution_location=scope.execution_location,
            dev_only=True,
        )


def guard_dev_scope(
    *,
    principal_id: PrincipalId,
    workspace_id: WorkspaceId,
    mode: DevExecutionMode,
    isolation_mode: IsolationMode,
    execution_location: ExecutionLocation,
    text_only: bool = False,
    owner_id: PrincipalId | None = None,
    owner_workspace_id: WorkspaceId | None = None,
    production_capability: bool = False,
) -> DevScope:
    """Build a DEV scope and fail closed on production or owner crossings."""
    if production_capability:
        raise ValueError("DEV scope cannot write production capability")
    scope = DevScope(
        principal_id=principal_id,
        workspace_id=workspace_id,
        mode=mode,
        isolation_mode=isolation_mode,
        execution_location=execution_location,
        text_only=text_only,
        dev_only=True,
    )
    if owner_id is not None or owner_workspace_id is not None:
        scope.assert_owner(
            principal_id=owner_id if owner_id is not None else principal_id,
            workspace_id=(
                owner_workspace_id
                if owner_workspace_id is not None
                else workspace_id
            ),
        )
    return scope


def serialize_dev_evidence(
    evidence: Evidence | Mapping[str, object],
    *,
    scope: DevScope,
    principal_id: PrincipalId | None = None,
    workspace_id: WorkspaceId | None = None,
) -> dict[str, object]:
    """Serialize evidence with a scope-derived ``dev_only`` label.

    Caller-supplied scope labels and host-root fields are rejected so a DEV
    payload cannot masquerade as production evidence or leak its temp root.
    """
    metadata = scope.evidence_metadata(
        principal_id=principal_id,
        workspace_id=workspace_id,
    )
    if isinstance(evidence, Evidence):
        evidence_payload = evidence.model_dump(mode="json")
    else:
        evidence_payload = dict(evidence)
        reserved = {
            "dev_only",
            "metadata",
            "host_path",
            "host_root",
            "production_capability",
            "workspace_root",
        }
        if reserved.intersection(evidence_payload):
            raise ValueError("DEV evidence owns its metadata and cannot carry host facts")
    return {
        "dev_only": True,
        "evidence": evidence_payload,
        "metadata": metadata.model_dump(mode="json"),
    }


def _unique_nonempty(value: tuple[object, ...], field_name: str) -> tuple[object, ...]:
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must not contain duplicates")
    return value


def _validate_path_scope(value: str) -> str:
    if (
        value.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", value)
        or "\\" in value
    ):
        raise ValueError("path scope must be relative POSIX syntax")
    base = value[:-3] if value.endswith("/**") else value
    if not base or "*" in base:
        raise ValueError("path scope only permits a trailing /** prefix")
    if any(part in {"", ".", ".."} for part in base.split("/")):
        raise ValueError("path scope contains an unsafe path component")
    return value


class CapabilityScope(DomainModel):
    """A finite capability set for one workspace."""

    tools: tuple[ToolName, ...]
    effects: tuple[ToolEffect, ...]
    workspace_id: WorkspaceId
    paths: tuple[PathScope, ...]
    command_classes: tuple[CommandClass, ...] = ()
    budget: CapabilityBudget
    expires_at: UtcTimestamp

    @field_validator("tools", "effects", "paths")
    @classmethod
    def _collections_are_finite_and_unique(
        cls, value: tuple[object, ...], info: object
    ) -> tuple[object, ...]:
        field_name = getattr(info, "field_name", "scope values")
        return _unique_nonempty(value, str(field_name))

    @field_validator("command_classes")
    @classmethod
    def _command_classes_are_unique(
        cls, value: tuple[CommandClass, ...]
    ) -> tuple[CommandClass, ...]:
        if len(set(value)) != len(value):
            raise ValueError("command_classes must not contain duplicates")
        return value

    @field_validator("paths")
    @classmethod
    def _paths_are_relative(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_validate_path_scope(path) for path in value)

    @model_validator(mode="after")
    def _command_classes_match_effects(self) -> CapabilityScope:
        has_command_effect = ToolEffect.COMMAND in self.effects
        if has_command_effect and not self.command_classes:
            raise ValueError("COMMAND scope requires command_classes")
        if not has_command_effect and self.command_classes:
            raise ValueError("command_classes require the COMMAND effect")
        return self

    def is_subset_of(self, approved: CapabilityScope) -> bool:
        """Whether this scope is no broader than an approved scope."""
        if self.workspace_id != approved.workspace_id:
            return False
        if not set(self.tools) <= set(approved.tools):
            return False
        if not set(self.effects) <= set(approved.effects):
            return False
        if not set(self.command_classes) <= set(approved.command_classes):
            return False
        if any(
            not any(_path_is_covered(path, approved_path) for approved_path in approved.paths)
            for path in self.paths
        ):
            return False
        if self.budget.max_calls > approved.budget.max_calls:
            return False
        if self.budget.max_output_bytes > approved.budget.max_output_bytes:
            return False
        if self.budget.max_wall_clock_ms > approved.budget.max_wall_clock_ms:
            return False
        return self.expires_at <= approved.expires_at


class ApprovalRequest(DomainModel):
    """A request for a bounded capability decision."""

    request_id: ApprovalRequestId = Field(default_factory=new_approval_request_id)
    call_id: ToolCallId
    run_id: RunId
    workspace_id: WorkspaceId
    tool_name: ToolName
    effect: ToolEffect
    scope: CapabilityScope
    reason: Reason
    issuer: ApprovalIssuer = ApprovalIssuer.SERVER
    requested_at: UtcTimestamp

    @classmethod
    def from_tool_call(
        cls,
        call: ToolCall,
        *,
        workspace_id: WorkspaceId,
        scope: CapabilityScope,
        reason: str,
        requested_at: UtcTimestamp,
    ) -> ApprovalRequest:
        """Build a server-issued request with a stable call/scope identity."""
        if scope.workspace_id != workspace_id:
            raise ValueError("approval scope workspace does not match the request")
        identity = json.dumps(
            {
                "call_id": call.call_id,
                "run_id": call.run_id,
                "scope": scope.model_dump(mode="json"),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        request_id = "apr_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return cls(
            request_id=request_id,
            call_id=call.call_id,
            run_id=call.run_id,
            workspace_id=workspace_id,
            tool_name=call.tool_name,
            effect=call.effect,
            scope=scope,
            reason=reason,
            issuer=ApprovalIssuer.SERVER,
            requested_at=requested_at,
        )

    @model_validator(mode="after")
    def _requested_capability_is_in_scope(self) -> ApprovalRequest:
        if self.scope.workspace_id != self.workspace_id:
            raise ValueError("approval workspace does not match capability scope")
        if self.tool_name not in self.scope.tools:
            raise ValueError("requested tool is outside capability scope")
        if self.effect not in self.scope.effects:
            raise ValueError("requested effect is outside capability scope")
        return self


class ApprovalDecision(DomainModel):
    """An auditable allow or deny decision for one request."""

    approval_request_id: ApprovalRequestId
    outcome: ApprovalOutcome
    issuer: ApprovalIssuer
    reason: Reason | None = None
    decided_at: UtcTimestamp

    @model_validator(mode="after")
    def _deny_requires_reason(self) -> ApprovalDecision:
        if self.outcome is ApprovalOutcome.DENY and self.reason is None:
            raise ValueError("DENY decision requires a reason")
        return self


class Lease(DomainModel):
    """A time-bounded, revocable capability grant."""

    lease_id: LeaseId = Field(default_factory=new_lease_id)
    approval_request_id: ApprovalRequestId
    call_id: ToolCallId
    scope: CapabilityScope
    issuer: ApprovalIssuer
    issued_at: UtcTimestamp
    expires_at: UtcTimestamp
    status: LeaseStatus = LeaseStatus.ACTIVE
    closed_at: UtcTimestamp | None = None
    close_reason: Reason | None = None

    @model_validator(mode="after")
    def _lease_window_and_state_are_closed(self) -> Lease:
        if self.expires_at <= self.issued_at:
            raise ValueError("lease expiry must be after issuance")
        if self.expires_at > self.scope.expires_at:
            raise ValueError("lease cannot outlive its capability scope")
        if self.status is LeaseStatus.ACTIVE:
            if self.closed_at is not None or self.close_reason is not None:
                raise ValueError("active lease cannot have close facts")
        elif self.closed_at is None:
            raise ValueError("closed lease requires closed_at")
        if self.status is LeaseStatus.REVOKED and self.close_reason is None:
            raise ValueError("revoked lease requires a reason")
        return self

    def revoke(self, *, at: datetime, reason: str) -> Lease:
        """Return a revoked lease; terminal lease states cannot be revived."""
        if self.status is not LeaseStatus.ACTIVE:
            raise ValueError("only an active lease can be revoked")
        data = self.model_dump(mode="python")
        data.update(
            {
                "status": LeaseStatus.REVOKED,
                "closed_at": at,
                "close_reason": reason,
            }
        )
        return Lease.model_validate(data)

    def is_active_at(self, at: datetime) -> bool:
        """Return whether this lease can authorize a call at an injected time."""
        if at.tzinfo is None:
            raise ValueError("lease check time must be timezone-aware")
        return (
            self.status is LeaseStatus.ACTIVE
            and self.issued_at <= at < self.expires_at
        )

    def expire(self, *, at: datetime) -> Lease:
        """Close a lease only at or after its declared expiry."""
        if self.status is not LeaseStatus.ACTIVE:
            raise ValueError("only an active lease can expire")
        if at < self.expires_at:
            raise ValueError("lease cannot expire before expires_at")
        data = self.model_dump(mode="python")
        data.update({"status": LeaseStatus.EXPIRED, "closed_at": at})
        return Lease.model_validate(data)


def _path_is_covered(requested: str, approved: str) -> bool:
    if approved.endswith("/**"):
        prefix = approved[:-3]
        return requested == prefix or requested.startswith(prefix + "/")
    return requested == approved
