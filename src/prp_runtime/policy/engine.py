"""Deterministic authorization matrix for tool calls."""

from collections.abc import Collection
from datetime import datetime
from enum import StrEnum, unique

from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    IsolationMode,
    ToolEffect,
)
from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.values import utc_now
from prp_runtime.policy.models import CommandClass, Lease
from prp_runtime.settings import Settings
from prp_runtime.tools.models import ToolCall

__all__ = [
    "DEFAULT_KNOWN_TOOLS",
    "PolicyDecision",
    "PolicyOutcome",
    "PolicyReasonCode",
    "decide_tool_call",
    "evaluate_tool_call",
]

DEFAULT_KNOWN_TOOLS: frozenset[str] = frozenset(
    {
        "list_files",
        "read_file",
        "search_text",
        "apply_patch",
        "run_targeted_test",
        "get_diff",
        "get_status",
    }
)


@unique
class PolicyOutcome(StrEnum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


@unique
class PolicyReasonCode(StrEnum):
    READ_ALLOWED = "READ_ALLOWED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    AUTO_LOW_RISK_COMMAND = "AUTO_LOW_RISK_COMMAND"
    PLAN_SIDE_EFFECT = "PLAN_SIDE_EFFECT"
    YOLO_ALLOWED = "YOLO_ALLOWED"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    NETWORK_DISABLED = "NETWORK_DISABLED"
    LEASE_REQUIRED = "LEASE_REQUIRED"
    LEASE_INACTIVE = "LEASE_INACTIVE"
    LEASE_SCOPE_MISMATCH = "LEASE_SCOPE_MISMATCH"
    PATH_UNRESOLVED = "PATH_UNRESOLVED"
    UNSAFE_RESOLVED_PATH = "UNSAFE_RESOLVED_PATH"
    UNKNOWN_COMMAND_CLASS = "UNKNOWN_COMMAND_CLASS"
    HOST_YOLO_DISABLED = "HOST_YOLO_DISABLED"


class PolicyDecision(DomainModel):
    """One deterministic, auditable decision for one tool request."""

    call_id: str
    tool_name: str
    effect: ToolEffect
    mode: AgentMode
    outcome: PolicyOutcome
    reason_code: PolicyReasonCode


def decide_tool_call(
    call: ToolCall,
    mode: AgentMode,
    *,
    known_tools: Collection[str] = DEFAULT_KNOWN_TOOLS,
    lease: Lease | None = None,
    workspace_id: str | None = None,
    resolved_paths: Collection[str] | None = None,
    command_class: CommandClass | None = None,
    isolation_mode: IsolationMode = IsolationMode.SANDBOXED,
    execution_location: ExecutionLocation = ExecutionLocation.CLOUD,
    user_explicit_host_yolo: bool = False,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> PolicyDecision:
    """Decide without inspecting model text, arguments or self-reported risk."""
    if call.tool_name not in known_tools:
        outcome = PolicyOutcome.DENY
        reason = PolicyReasonCode.UNKNOWN_TOOL
    elif call.effect is ToolEffect.NETWORK:
        outcome = PolicyOutcome.DENY
        reason = PolicyReasonCode.NETWORK_DISABLED
    elif call.effect is ToolEffect.COMMAND and not isinstance(command_class, CommandClass):
        outcome = PolicyOutcome.DENY
        reason = PolicyReasonCode.UNKNOWN_COMMAND_CLASS
    elif mode is AgentMode.PLAN and call.effect is not ToolEffect.READ:
        outcome = PolicyOutcome.DENY
        reason = PolicyReasonCode.PLAN_SIDE_EFFECT
    elif lease is not None:
        lease_reason = _lease_reason(
            call,
            lease,
            workspace_id=workspace_id,
            resolved_paths=resolved_paths,
            command_class=command_class,
            now=now or utc_now(),
        )
        if lease_reason is not None:
            outcome = PolicyOutcome.DENY
            reason = lease_reason
        elif mode is AgentMode.YOLO and isolation_mode is IsolationMode.HOST:
            if not user_explicit_host_yolo or not (settings and settings.allow_host_yolo):
                outcome = PolicyOutcome.DENY
                reason = PolicyReasonCode.HOST_YOLO_DISABLED
            else:
                outcome = PolicyOutcome.ALLOW
                reason = PolicyReasonCode.YOLO_ALLOWED
        else:
            outcome = PolicyOutcome.ALLOW
            reason = PolicyReasonCode.READ_ALLOWED
    elif mode is AgentMode.YOLO and isolation_mode is IsolationMode.HOST:
        if not user_explicit_host_yolo or not (settings and settings.allow_host_yolo):
            outcome = PolicyOutcome.DENY
            reason = PolicyReasonCode.HOST_YOLO_DISABLED
        elif execution_location not in (ExecutionLocation.CLOUD, ExecutionLocation.BRIDGE):
            outcome = PolicyOutcome.DENY
            reason = PolicyReasonCode.HOST_YOLO_DISABLED
        else:
            outcome = PolicyOutcome.ALLOW
            reason = PolicyReasonCode.YOLO_ALLOWED
    elif call.effect is ToolEffect.READ:
        outcome = PolicyOutcome.ALLOW
        reason = PolicyReasonCode.READ_ALLOWED
    elif mode is AgentMode.YOLO:
        outcome = PolicyOutcome.ALLOW
        reason = PolicyReasonCode.YOLO_ALLOWED
    elif (
        mode is AgentMode.AUTO
        and call.effect is ToolEffect.COMMAND
        and command_class in {
            CommandClass.READ_ONLY,
            CommandClass.TEST,
            CommandClass.LINT,
        }
    ):
        outcome = PolicyOutcome.ALLOW
        reason = PolicyReasonCode.AUTO_LOW_RISK_COMMAND
    else:
        outcome = PolicyOutcome.ASK
        reason = PolicyReasonCode.APPROVAL_REQUIRED
    return PolicyDecision(
        call_id=call.call_id,
        tool_name=call.tool_name,
        effect=call.effect,
        mode=mode,
        outcome=outcome,
        reason_code=reason,
    )


def _lease_reason(
    call: ToolCall,
    lease: Lease,
    *,
    workspace_id: str | None,
    resolved_paths: Collection[str] | None,
    command_class: CommandClass | None,
    now: datetime,
) -> PolicyReasonCode | None:
    if workspace_id is None or workspace_id != lease.scope.workspace_id:
        return PolicyReasonCode.LEASE_SCOPE_MISMATCH
    if not lease.is_active_at(now):
        return PolicyReasonCode.LEASE_INACTIVE
    if call.tool_name not in lease.scope.tools or call.effect not in lease.scope.effects:
        return PolicyReasonCode.LEASE_SCOPE_MISMATCH
    if call.effect is ToolEffect.COMMAND:
        if not isinstance(command_class, CommandClass):
            return PolicyReasonCode.UNKNOWN_COMMAND_CLASS
        if command_class not in lease.scope.command_classes:
            return PolicyReasonCode.LEASE_SCOPE_MISMATCH
    if not resolved_paths:
        return PolicyReasonCode.PATH_UNRESOLVED
    for path in resolved_paths:
        if (
            path.startswith(("/", "\\"))
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            return PolicyReasonCode.UNSAFE_RESOLVED_PATH
        if not any(_path_is_covered(path, scope_path) for scope_path in lease.scope.paths):
            return PolicyReasonCode.LEASE_SCOPE_MISMATCH
    return None


def _path_is_covered(requested: str, approved: str) -> bool:
    if approved.endswith("/**"):
        prefix = approved[:-3]
        return requested == prefix or requested.startswith(prefix + "/")
    return requested == approved


def evaluate_tool_call(
    call: ToolCall,
    mode: AgentMode,
    *,
    known_tools: Collection[str] = DEFAULT_KNOWN_TOOLS,
    lease: Lease | None = None,
    workspace_id: str | None = None,
    resolved_paths: Collection[str] | None = None,
    command_class: CommandClass | None = None,
    isolation_mode: IsolationMode = IsolationMode.SANDBOXED,
    execution_location: ExecutionLocation = ExecutionLocation.CLOUD,
    user_explicit_host_yolo: bool = False,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> PolicyDecision:
    """Explicit-name alias for the pure policy decision function."""
    return decide_tool_call(
        call,
        mode,
        known_tools=known_tools,
        lease=lease,
        workspace_id=workspace_id,
        resolved_paths=resolved_paths,
        command_class=command_class,
        isolation_mode=isolation_mode,
        execution_location=execution_location,
        user_explicit_host_yolo=user_explicit_host_yolo,
        settings=settings,
        now=now,
    )
