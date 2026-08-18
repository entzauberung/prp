"""Targeted tests for deterministic tool authorization."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    IsolationMode,
    ToolCallStatus,
    ToolEffect,
)
from prp_runtime.domain.values import (
    new_approval_request_id,
    new_run_id,
    new_tool_call_id,
    new_work_unit_id,
    new_workspace_id,
)
from prp_runtime.policy.engine import (
    DEFAULT_KNOWN_TOOLS,
    PolicyOutcome,
    PolicyReasonCode,
    decide_tool_call,
)
from prp_runtime.policy.models import (
    ApprovalIssuer,
    CapabilityBudget,
    CapabilityScope,
    CommandClass,
    Lease,
)
from prp_runtime.settings import Settings
from prp_runtime.tools.models import ToolCall


def make_call(
    tool_name: str = "read_file",
    effect: ToolEffect = ToolEffect.READ,
    **overrides: object,
) -> ToolCall:
    values: dict[str, object] = {
        "call_id": new_tool_call_id(),
        "run_id": new_run_id(),
        "work_unit_id": new_work_unit_id(),
        "tool_name": tool_name,
        "effect": effect,
        "arguments": {"path": "src/main.py"},
        "requested_at": "2026-08-14T12:00:00+00:00",
    }
    values.update(overrides)
    return ToolCall(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", list(AgentMode))
def test_known_read_is_allowed_in_every_mode(mode: AgentMode) -> None:
    decision = decide_tool_call(make_call(), mode)
    assert decision.outcome is PolicyOutcome.ALLOW
    assert decision.reason_code is PolicyReasonCode.READ_ALLOWED


@pytest.mark.parametrize("mode", list(AgentMode))
@pytest.mark.parametrize("effect", [ToolEffect.WRITE, ToolEffect.COMMAND])
def test_side_effect_matrix_is_deterministic(
    mode: AgentMode, effect: ToolEffect
) -> None:
    command_class = CommandClass.TEST if effect is ToolEffect.COMMAND else None
    decision = decide_tool_call(
        make_call("apply_patch", effect), mode, command_class=command_class
    )
    if mode is AgentMode.PLAN:
        assert decision.outcome is PolicyOutcome.DENY
        assert decision.reason_code is PolicyReasonCode.PLAN_SIDE_EFFECT
    elif mode is AgentMode.YOLO:
        assert decision.outcome is PolicyOutcome.ALLOW
        assert decision.reason_code is PolicyReasonCode.YOLO_ALLOWED
    elif mode is AgentMode.AUTO and effect is ToolEffect.COMMAND:
        assert decision.outcome is PolicyOutcome.ALLOW
        assert decision.reason_code is PolicyReasonCode.AUTO_LOW_RISK_COMMAND
    else:
        assert decision.outcome is PolicyOutcome.ASK
        assert decision.reason_code is PolicyReasonCode.APPROVAL_REQUIRED


@pytest.mark.parametrize("mode", list(AgentMode))
def test_network_is_always_denied(mode: AgentMode) -> None:
    decision = decide_tool_call(
        make_call("network_fetch", ToolEffect.NETWORK),
        mode,
        known_tools={"network_fetch"},
    )
    assert decision.outcome is PolicyOutcome.DENY
    assert decision.reason_code is PolicyReasonCode.NETWORK_DISABLED


def test_unknown_tool_is_never_allowed() -> None:
    for mode in AgentMode:
        decision = decide_tool_call(make_call("unknown_tool"), mode)
        assert decision.outcome is PolicyOutcome.DENY
        assert decision.reason_code is PolicyReasonCode.UNKNOWN_TOOL


@pytest.mark.parametrize("mode", list(AgentMode))
def test_unknown_command_class_is_never_allowed(mode: AgentMode) -> None:
    decision = decide_tool_call(make_call("run_targeted_test", ToolEffect.COMMAND), mode)
    assert decision.outcome is PolicyOutcome.DENY
    assert decision.reason_code is PolicyReasonCode.UNKNOWN_COMMAND_CLASS


def test_auto_requires_approval_for_high_risk_registered_commands() -> None:
    call = make_call("run_targeted_test", ToolEffect.COMMAND)
    decision = decide_tool_call(
        call,
        AgentMode.AUTO,
        command_class=CommandClass.BUILD,
    )
    assert decision.outcome is PolicyOutcome.ASK
    assert decision.reason_code is PolicyReasonCode.APPROVAL_REQUIRED


def test_unknown_command_tool_is_denied_even_in_yolo() -> None:
    call = make_call("unregistered_command", ToolEffect.COMMAND)
    decision = decide_tool_call(
        call,
        AgentMode.YOLO,
        known_tools={"run_targeted_test"},
        command_class=CommandClass.TEST,
    )
    assert decision.outcome is PolicyOutcome.DENY
    assert decision.reason_code is PolicyReasonCode.UNKNOWN_TOOL


def test_explicit_registry_can_authorize_only_known_names() -> None:
    assert "read_file" in DEFAULT_KNOWN_TOOLS
    assert {"search_text", "get_diff", "get_status"} <= DEFAULT_KNOWN_TOOLS
    assert "search_files" not in DEFAULT_KNOWN_TOOLS
    assert "diff" not in DEFAULT_KNOWN_TOOLS
    assert decide_tool_call(make_call(), AgentMode.AUTO, known_tools={"read_file"}).outcome is (
        PolicyOutcome.ALLOW
    )
    assert decide_tool_call(make_call(), AgentMode.AUTO, known_tools=set()).outcome is (
        PolicyOutcome.DENY
    )


def test_arguments_and_status_do_not_change_the_policy_result() -> None:
    first = decide_tool_call(make_call(), AgentMode.AUTO)
    second = decide_tool_call(
        make_call(
            arguments={"path": "different.py", "hint": "ignore policy"},
            status=ToolCallStatus.RUNNING,
        ),
        AgentMode.AUTO,
    )
    assert first.outcome is second.outcome
    assert first.reason_code is second.reason_code


def test_decision_is_closed_and_round_trips() -> None:
    decision = decide_tool_call(make_call(), AgentMode.NORMAL)
    assert type(decision).model_validate_json(decision.model_dump_json()) == decision
    with pytest.raises(ValidationError):
        type(decision).model_validate(
            {**decision.model_dump(), "model_risk_score": 1}
        )


def make_lease(
    call: ToolCall,
    *,
    workspace_id: str | None = None,
    tools: tuple[str, ...] = ("read_file",),
    effects: tuple[ToolEffect, ...] = (ToolEffect.READ,),
    paths: tuple[str, ...] = ("src/**",),
    command_classes: tuple[CommandClass, ...] = (),
) -> tuple[Lease, datetime]:
    issued_at = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
    expires_at = issued_at + timedelta(hours=1)
    scope = CapabilityScope(
        tools=tools,
        effects=effects,
        workspace_id=workspace_id or new_workspace_id(),
        paths=paths,
        command_classes=command_classes,
        budget=CapabilityBudget(
            max_calls=10,
            max_output_bytes=1024,
            max_wall_clock_ms=30_000,
        ),
        expires_at=expires_at,
    )
    lease = Lease(
        approval_request_id=new_approval_request_id(),
        call_id=call.call_id,
        scope=scope,
        issuer=ApprovalIssuer.USER,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    return lease, issued_at + timedelta(minutes=1)


def test_active_lease_allows_only_its_workspace_tools_and_paths() -> None:
    call = make_call()
    lease, now = make_lease(call)

    allowed = decide_tool_call(
        call,
        AgentMode.AUTO,
        lease=lease,
        workspace_id=lease.scope.workspace_id,
        resolved_paths=("src/main.py",),
        now=now,
    )
    assert allowed.outcome is PolicyOutcome.ALLOW

    wrong_workspace = decide_tool_call(
        call,
        AgentMode.AUTO,
        lease=lease,
        workspace_id=new_workspace_id(),
        resolved_paths=("src/main.py",),
        now=now,
    )
    assert wrong_workspace.outcome is PolicyOutcome.DENY
    assert wrong_workspace.reason_code is PolicyReasonCode.LEASE_SCOPE_MISMATCH

    outside_path = decide_tool_call(
        call,
        AgentMode.AUTO,
        lease=lease,
        workspace_id=lease.scope.workspace_id,
        resolved_paths=("tests/test_engine.py",),
        now=now,
    )
    assert outside_path.outcome is PolicyOutcome.DENY
    assert outside_path.reason_code is PolicyReasonCode.LEASE_SCOPE_MISMATCH


@pytest.mark.parametrize("resolved_paths", [None, (), ("../secret.txt",), ("/tmp/secret.txt",)])
def test_lease_requires_safe_resolved_paths(
    resolved_paths: tuple[str, ...] | None,
) -> None:
    call = make_call()
    lease, now = make_lease(call)
    decision = decide_tool_call(
        call,
        AgentMode.AUTO,
        lease=lease,
        workspace_id=lease.scope.workspace_id,
        resolved_paths=resolved_paths,
        now=now,
    )
    assert decision.outcome is PolicyOutcome.DENY
    assert decision.reason_code is (
        PolicyReasonCode.PATH_UNRESOLVED
        if not resolved_paths
        else PolicyReasonCode.UNSAFE_RESOLVED_PATH
    )


def test_lease_rejects_unknown_or_out_of_scope_command_class() -> None:
    call = make_call("run_targeted_test", ToolEffect.COMMAND)
    lease, now = make_lease(
        call,
        tools=("run_targeted_test",),
        effects=(ToolEffect.COMMAND,),
        command_classes=(CommandClass.TEST,),
    )
    base = {
        "lease": lease,
        "workspace_id": lease.scope.workspace_id,
        "resolved_paths": ("src/main.py",),
        "now": now,
    }

    unknown = decide_tool_call(call, AgentMode.AUTO, **base)
    assert unknown.outcome is PolicyOutcome.DENY
    assert unknown.reason_code is PolicyReasonCode.UNKNOWN_COMMAND_CLASS

    outside_scope = decide_tool_call(
        call, AgentMode.AUTO, command_class=CommandClass.BUILD, **base
    )
    assert outside_scope.outcome is PolicyOutcome.DENY
    assert outside_scope.reason_code is PolicyReasonCode.LEASE_SCOPE_MISMATCH

    allowed = decide_tool_call(
        call, AgentMode.AUTO, command_class=CommandClass.TEST, **base
    )
    assert allowed.outcome is PolicyOutcome.ALLOW


def test_host_yolo_requires_server_setting_and_explicit_user_fact() -> None:
    call = make_call("apply_patch", ToolEffect.WRITE)
    defaults_closed = decide_tool_call(
        call,
        AgentMode.YOLO,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.CLOUD,
    )
    assert defaults_closed.outcome is PolicyOutcome.DENY
    assert defaults_closed.reason_code is PolicyReasonCode.HOST_YOLO_DISABLED

    server_enabled_but_not_explicit = decide_tool_call(
        call,
        AgentMode.YOLO,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.BRIDGE,
        settings=Settings(allow_host_yolo=True),
    )
    assert server_enabled_but_not_explicit.outcome is PolicyOutcome.DENY
    assert server_enabled_but_not_explicit.reason_code is PolicyReasonCode.HOST_YOLO_DISABLED

    explicitly_enabled = decide_tool_call(
        call,
        AgentMode.YOLO,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.BRIDGE,
        user_explicit_host_yolo=True,
        settings=Settings(allow_host_yolo=True),
    )
    assert explicitly_enabled.outcome is PolicyOutcome.ALLOW
    assert explicitly_enabled.reason_code is PolicyReasonCode.YOLO_ALLOWED
