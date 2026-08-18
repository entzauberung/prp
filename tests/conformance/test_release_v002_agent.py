"""Release conformance for Agent mode, DEV scope, and safety boundaries."""

import pytest

from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    IsolationMode,
    ToolEffect,
)
from prp_runtime.domain.values import (
    new_run_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
    utc_now,
)
from prp_runtime.policy.engine import PolicyOutcome, PolicyReasonCode, decide_tool_call
from prp_runtime.policy.models import (
    CommandClass,
    DevExecutionMode,
    guard_dev_scope,
    serialize_dev_evidence,
)
from prp_runtime.tools.models import ToolCall


def _call(tool_name: str, effect: ToolEffect) -> ToolCall:
    return ToolCall(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        tool_name=tool_name,
        effect=effect,
        snapshot_id=new_snapshot_id(),
        requested_at=utc_now(),
    )


def test_release_agent_mode_policy_matrix_has_no_permission_escalation() -> None:
    patch = _call("apply_patch", ToolEffect.WRITE)
    command = _call("run_targeted_test", ToolEffect.COMMAND)

    plan = decide_tool_call(
        patch,
        AgentMode.PLAN,
        known_tools=("apply_patch", "run_targeted_test"),
        isolation_mode=IsolationMode.SANDBOXED,
        execution_location=ExecutionLocation.CLOUD,
    )
    normal = decide_tool_call(
        patch,
        AgentMode.NORMAL,
        known_tools=("apply_patch",),
    )
    auto_write = decide_tool_call(
        patch,
        AgentMode.AUTO,
        known_tools=("apply_patch",),
    )
    auto_test = decide_tool_call(
        command,
        AgentMode.AUTO,
        known_tools=("run_targeted_test",),
        command_class=CommandClass.TEST,
    )
    sandbox_yolo = decide_tool_call(
        patch,
        AgentMode.YOLO,
        known_tools=("apply_patch",),
        isolation_mode=IsolationMode.SANDBOXED,
        execution_location=ExecutionLocation.CLOUD,
    )
    host_yolo = decide_tool_call(
        patch,
        AgentMode.YOLO,
        known_tools=("apply_patch",),
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.CLOUD,
        user_explicit_host_yolo=False,
    )

    assert (plan.outcome, plan.reason_code) == (
        PolicyOutcome.DENY,
        PolicyReasonCode.PLAN_SIDE_EFFECT,
    )
    assert (normal.outcome, normal.reason_code) == (
        PolicyOutcome.ASK,
        PolicyReasonCode.APPROVAL_REQUIRED,
    )
    assert (auto_write.outcome, auto_write.reason_code) == (
        PolicyOutcome.ASK,
        PolicyReasonCode.APPROVAL_REQUIRED,
    )
    assert (auto_test.outcome, auto_test.reason_code) == (
        PolicyOutcome.ALLOW,
        PolicyReasonCode.AUTO_LOW_RISK_COMMAND,
    )
    assert (sandbox_yolo.outcome, sandbox_yolo.reason_code) == (
        PolicyOutcome.ALLOW,
        PolicyReasonCode.YOLO_ALLOWED,
    )
    assert (host_yolo.outcome, host_yolo.reason_code) == (
        PolicyOutcome.DENY,
        PolicyReasonCode.HOST_YOLO_DISABLED,
    )


@pytest.mark.parametrize(
    ("mode", "location", "text_only"),
    [
        (DevExecutionMode.HOST, ExecutionLocation.CLOUD, False),
        (DevExecutionMode.BRIDGE, ExecutionLocation.BRIDGE, False),
        (DevExecutionMode.TEXT_ONLY, ExecutionLocation.CLOUD, True),
    ],
)
def test_dev_matrix_is_explicitly_non_sandboxed_and_labeled(
    mode: DevExecutionMode,
    location: ExecutionLocation,
    text_only: bool,
) -> None:
    scope = guard_dev_scope(
        principal_id="prn_dev_conformance",
        workspace_id="ws_dev_conformance",
        mode=mode,
        isolation_mode=IsolationMode.HOST,
        execution_location=location,
        text_only=text_only,
    )

    assert scope.dev_only is True
    assert scope.temporary_workspace is True
    assert scope.isolation_mode is IsolationMode.HOST
    payload = serialize_dev_evidence({"fact": "temporary candidate"}, scope=scope)
    assert payload["dev_only"] is True
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["dev_only"] is True
    assert metadata["isolation_mode"] == IsolationMode.HOST.value


def test_dev_scope_and_evidence_fail_closed_at_production_boundaries() -> None:
    common = {
        "principal_id": "prn_dev_conformance",
        "workspace_id": "ws_dev_conformance",
        "mode": DevExecutionMode.HOST,
        "execution_location": ExecutionLocation.CLOUD,
    }
    with pytest.raises(ValueError, match="SANDBOXED"):
        guard_dev_scope(**common, isolation_mode=IsolationMode.SANDBOXED)
    with pytest.raises(ValueError, match="production capability"):
        guard_dev_scope(
            **common,
            isolation_mode=IsolationMode.HOST,
            production_capability=True,
        )

    scope = guard_dev_scope(
        **common,
        isolation_mode=IsolationMode.HOST,
    )
    with pytest.raises(ValueError, match="host facts"):
        serialize_dev_evidence({"host_path": "/tmp/dev-root"}, scope=scope)
    with pytest.raises(ValueError, match="mismatch"):
        scope.evidence_metadata(principal_id="prn_other_owner")
