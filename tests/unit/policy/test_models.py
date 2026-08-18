"""Targeted tests for bounded approval and capability contracts."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import ExecutionLocation, IsolationMode, ToolEffect
from prp_runtime.domain.values import (
    new_principal_id,
    new_run_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
    new_workspace_id,
)
from prp_runtime.policy.models import (
    ApprovalDecision,
    ApprovalIssuer,
    ApprovalOutcome,
    ApprovalRequest,
    CapabilityBudget,
    CapabilityScope,
    CommandClass,
    DevExecutionMode,
    DevScope,
    Lease,
    LeaseStatus,
    guard_dev_scope,
    serialize_dev_evidence,
)
from prp_runtime.tools.models import ToolCall

T0 = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def make_scope(**overrides: object) -> CapabilityScope:
    values: dict[str, object] = {
        "tools": ("read_file",),
        "effects": (ToolEffect.READ,),
        "workspace_id": new_workspace_id(),
        "paths": ("src/**",),
        "budget": CapabilityBudget(
            max_calls=10,
            max_output_bytes=4096,
            max_wall_clock_ms=30_000,
        ),
        "expires_at": T0 + timedelta(hours=1),
    }
    values.update(overrides)
    return CapabilityScope(**values)  # type: ignore[arg-type]


def make_request(scope: CapabilityScope | None = None, **overrides: object) -> ApprovalRequest:
    actual_scope = scope or make_scope()
    values: dict[str, object] = {
        "call_id": new_tool_call_id(),
        "run_id": new_run_id(),
        "workspace_id": actual_scope.workspace_id,
        "tool_name": "read_file",
        "effect": ToolEffect.READ,
        "scope": actual_scope,
        "reason": "read the requested source file",
        "requested_at": T0,
    }
    values.update(overrides)
    return ApprovalRequest(**values)  # type: ignore[arg-type]


def make_lease(scope: CapabilityScope | None = None, **overrides: object) -> Lease:
    request = make_request(scope)
    values: dict[str, object] = {
        "approval_request_id": request.request_id,
        "call_id": request.call_id,
        "scope": request.scope,
        "issuer": ApprovalIssuer.USER,
        "issued_at": T0,
        "expires_at": T0 + timedelta(minutes=10),
    }
    values.update(overrides)
    return Lease(**values)  # type: ignore[arg-type]


def test_approval_and_lease_contracts_round_trip_as_closed_json() -> None:
    request = make_request()
    decision = ApprovalDecision(
        approval_request_id=request.request_id,
        outcome=ApprovalOutcome.ALLOW,
        issuer=ApprovalIssuer.USER,
        decided_at=T0,
    )
    lease = make_lease(request.scope)
    assert request.request_id.startswith("apr_")
    assert lease.lease_id.startswith("lease_")
    assert ApprovalRequest.model_validate_json(request.model_dump_json()) == request
    assert ApprovalDecision.model_validate_json(decision.model_dump_json()) == decision
    assert Lease.model_validate_json(lease.model_dump_json()) == lease
    with pytest.raises(ValidationError):
        make_request(extra_field=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tools": ()},
        {"effects": ()},
        {"paths": ()},
        {"paths": ("/etc/passwd",)},
        {"paths": ("../secret",)},
        {"paths": ("src\\main.py",)},
        {"paths": ("*",)},
        {"budget": {"max_calls": 0, "max_output_bytes": 1, "max_wall_clock_ms": 1}},
        {
            "budget": {
                "max_calls": 10,
                "max_output_bytes": 4096,
                "max_wall_clock_ms": 86_400_001,
            }
        },
    ],
)
def test_empty_infinite_or_unsafe_scope_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_scope(**overrides)


def test_command_scope_requires_a_finite_command_class_set() -> None:
    with pytest.raises(ValidationError, match="command_classes"):
        make_scope(effects=(ToolEffect.COMMAND,))
    with pytest.raises(ValidationError, match="COMMAND"):
        make_scope(command_classes=(CommandClass.TEST,))
    command_scope = make_scope(
        tools=("run_targeted_test",),
        effects=(ToolEffect.COMMAND,),
        command_classes=(CommandClass.TEST,),
    )
    assert command_scope.command_classes == (CommandClass.TEST,)


def test_lease_scope_can_only_shrink_an_approved_scope() -> None:
    approved = make_scope()
    narrowed = make_scope(
        workspace_id=approved.workspace_id,
        paths=("src/main.py",),
        budget={"max_calls": 2, "max_output_bytes": 1024, "max_wall_clock_ms": 5_000},
        expires_at=T0 + timedelta(minutes=10),
    )
    assert narrowed.is_subset_of(approved) is True
    assert approved.is_subset_of(narrowed) is False
    broader_paths = make_scope(
        workspace_id=approved.workspace_id,
        paths=("src/**", "README.md"),
    )
    assert broader_paths.is_subset_of(approved) is False


def test_lease_active_check_uses_an_injected_clock_and_terminal_state() -> None:
    lease = make_lease()
    assert lease.is_active_at(T0 + timedelta(minutes=1)) is True
    assert lease.is_active_at(T0 + timedelta(minutes=10)) is False
    assert lease.revoke(at=T0 + timedelta(minutes=1), reason="revoked").is_active_at(
        T0 + timedelta(minutes=1)
    ) is False


def test_dev_scope_accepts_only_host_bridge_or_text_only_shapes() -> None:
    principal_id = new_principal_id()
    workspace_id = new_workspace_id()
    host = guard_dev_scope(
        principal_id=principal_id,
        workspace_id=workspace_id,
        mode=DevExecutionMode.HOST,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.CLOUD,
    )
    bridge = guard_dev_scope(
        principal_id=principal_id,
        workspace_id=workspace_id,
        mode=DevExecutionMode.BRIDGE,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.BRIDGE,
    )
    text_only = guard_dev_scope(
        principal_id=principal_id,
        workspace_id=workspace_id,
        mode=DevExecutionMode.TEXT_ONLY,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.CLOUD,
        text_only=True,
    )
    assert {host.mode, bridge.mode, text_only.mode} == set(DevExecutionMode)
    assert host.dev_only is True
    assert host.temporary_workspace is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"isolation_mode": IsolationMode.SANDBOXED},
        {"mode": DevExecutionMode.BRIDGE, "execution_location": ExecutionLocation.CLOUD},
        {"mode": DevExecutionMode.TEXT_ONLY},
        {"mode": DevExecutionMode.HOST, "text_only": True},
    ],
)
def test_dev_scope_rejects_sandbox_and_inconsistent_shapes(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "principal_id": new_principal_id(),
        "workspace_id": new_workspace_id(),
        "mode": DevExecutionMode.HOST,
        "isolation_mode": IsolationMode.HOST,
        "execution_location": ExecutionLocation.CLOUD,
    }
    values.update(overrides)
    with pytest.raises(ValidationError):
        DevScope(**values)  # type: ignore[arg-type]


def test_dev_scope_owner_and_production_capability_guards_fail_closed() -> None:
    principal_id = new_principal_id()
    workspace_id = new_workspace_id()
    with pytest.raises(ValueError, match="owner"):
        guard_dev_scope(
            principal_id=principal_id,
            workspace_id=workspace_id,
            owner_id=new_principal_id(),
            mode=DevExecutionMode.HOST,
            isolation_mode=IsolationMode.HOST,
            execution_location=ExecutionLocation.CLOUD,
        )
    with pytest.raises(ValueError, match="production capability"):
        guard_dev_scope(
            principal_id=principal_id,
            workspace_id=workspace_id,
            mode=DevExecutionMode.HOST,
            isolation_mode=IsolationMode.HOST,
            execution_location=ExecutionLocation.CLOUD,
            production_capability=True,
        )


def test_dev_evidence_metadata_is_scope_derived_and_cannot_leak_host_facts() -> None:
    scope = guard_dev_scope(
        principal_id=new_principal_id(),
        workspace_id=new_workspace_id(),
        mode=DevExecutionMode.TEXT_ONLY,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.CLOUD,
        text_only=True,
    )
    payload = serialize_dev_evidence({"result": "PASS"}, scope=scope)
    assert payload["metadata"] == {
        "dev_only": True,
        "principal_id": scope.principal_id,
        "workspace_id": scope.workspace_id,
        "mode": "TEXT_ONLY",
        "isolation_mode": "HOST",
        "execution_location": "CLOUD",
    }
    with pytest.raises(ValueError, match="host facts"):
        serialize_dev_evidence({"host_root": "/tmp/private"}, scope=scope)
    with pytest.raises(ValueError, match="owner"):
        serialize_dev_evidence(
            {"result": "PASS"},
            scope=scope,
            principal_id=new_principal_id(),
        )


def test_approval_request_cannot_escape_its_scope() -> None:
    scope = make_scope()
    with pytest.raises(ValidationError, match="workspace"):
        make_request(scope, workspace_id=new_workspace_id())
    with pytest.raises(ValidationError, match="tool"):
        make_request(scope, tool_name="write_file")
    with pytest.raises(ValidationError, match="effect"):
        make_request(scope, effect=ToolEffect.WRITE)


def test_server_approval_factory_has_stable_call_scope_identity() -> None:
    scope = make_scope()
    call = ToolCall(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        tool_name="read_file",
        effect=ToolEffect.READ,
        arguments={"path": "src/main.py"},
        snapshot_id=new_snapshot_id(),
        requested_at=T0,
    )
    first = ApprovalRequest.from_tool_call(
        call,
        workspace_id=scope.workspace_id,
        scope=scope,
        reason="APPROVAL_REQUIRED",
        requested_at=T0,
    )
    replay = ApprovalRequest.from_tool_call(
        call,
        workspace_id=scope.workspace_id,
        scope=scope,
        reason="APPROVAL_REQUIRED",
        requested_at=T0,
    )
    assert first.request_id == replay.request_id
    assert first.issuer is ApprovalIssuer.SERVER
    assert first.scope == scope
    with pytest.raises(ValueError, match="workspace"):
        ApprovalRequest.from_tool_call(
            call,
            workspace_id=new_workspace_id(),
            scope=scope,
            reason="APPROVAL_REQUIRED",
            requested_at=T0,
        )


def test_deny_requires_a_nonblank_reason_and_unknown_fields_are_rejected() -> None:
    request = make_request()
    with pytest.raises(ValidationError, match="reason"):
        ApprovalDecision(
            approval_request_id=request.request_id,
            outcome=ApprovalOutcome.DENY,
            issuer=ApprovalIssuer.SERVER,
            decided_at=T0,
        )
    with pytest.raises(ValidationError):
        ApprovalDecision(
            approval_request_id=request.request_id,
            outcome=ApprovalOutcome.DENY,
            issuer=ApprovalIssuer.SERVER,
            reason="   ",
            decided_at=T0,
        )


def test_lease_is_bounded_and_revocation_is_terminal() -> None:
    lease = make_lease()
    revoked = lease.revoke(at=T0 + timedelta(minutes=1), reason="user denied write")
    assert revoked.status is LeaseStatus.REVOKED
    assert revoked.close_reason == "user denied write"
    with pytest.raises(ValueError, match="active lease"):
        revoked.revoke(at=T0 + timedelta(minutes=2), reason="again")

    expired = make_lease().expire(at=T0 + timedelta(minutes=10))
    assert expired.status is LeaseStatus.EXPIRED
    with pytest.raises(ValueError, match="before"):
        make_lease().expire(at=T0 + timedelta(minutes=9))
    with pytest.raises(ValidationError, match="outlive"):
        make_lease(expires_at=T0 + timedelta(hours=2))


def test_terminal_lease_cannot_be_reintroduced_with_active_close_facts() -> None:
    lease = make_lease()
    with pytest.raises(ValidationError, match="active lease"):
        Lease.model_validate(
            {
                **lease.model_dump(),
                "closed_at": T0 + timedelta(minutes=1),
            }
        )
