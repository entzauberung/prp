"""Targeted tests for the protocol-independent tool contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import ToolCallStatus, ToolEffect
from prp_runtime.domain.models import ErrorCategory, ErrorInfo
from prp_runtime.domain.values import (
    new_run_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
)
from prp_runtime.tools.models import (
    MAX_TOOL_ARGUMENT_BYTES,
    MAX_TOOL_OUTPUT_BYTES,
    ToolCall,
    ToolResult,
)

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def make_call(**overrides: object) -> ToolCall:
    values: dict[str, object] = {
        "call_id": new_tool_call_id(),
        "run_id": new_run_id(),
        "work_unit_id": new_work_unit_id(),
        "tool_name": "read_file",
        "effect": ToolEffect.READ,
        "arguments": {"path": "src/main.py"},
        "requested_at": T0,
    }
    values.update(overrides)
    return ToolCall(**values)  # type: ignore[arg-type]


def make_result(call_id: str, **overrides: object) -> ToolResult:
    values: dict[str, object] = {
        "call_id": call_id,
        "status": ToolCallStatus.SUCCEEDED,
        "result": {"content": "hello"},
        "output": "hello",
        "completed_at": T0,
    }
    values.update(overrides)
    return ToolResult(**values)  # type: ignore[arg-type]


def test_tool_call_round_trips_as_a_closed_immutable_contract() -> None:
    call = make_call(snapshot_id=new_snapshot_id())
    assert call.call_id.startswith("tc_")
    assert ToolCall.model_validate_json(call.model_dump_json()) == call
    with pytest.raises(ValidationError):
        make_call(extra_field=True)
    with pytest.raises(ValidationError):
        call.tool_name = "write_file"


@pytest.mark.parametrize("effect", list(ToolEffect))
def test_all_tool_effects_are_closed(effect: ToolEffect) -> None:
    assert make_call(effect=effect).effect is effect
    with pytest.raises(ValidationError):
        make_call(effect="UNKNOWN")


def test_tool_arguments_are_bounded_json_without_secret_or_runtime_objects() -> None:
    with pytest.raises(ValidationError):
        make_call(arguments={"secret": "do-not-store"})
    with pytest.raises(ValidationError):
        make_call(arguments={"env": {"TOKEN": "redacted"}})
    with pytest.raises(ValidationError):
        make_call(arguments={"payload": "x" * MAX_TOOL_ARGUMENT_BYTES})
    with pytest.raises(ValidationError):
        make_call(arguments={"callable": object()})
    with pytest.raises(ValidationError):
        make_call(arguments={"nested": [{"PASSWORD": "redacted"}]})


def test_tool_result_requires_terminal_shape_and_preserves_explicit_truncation() -> None:
    call = make_call()
    success = make_result(
        call.call_id,
        truncated=True,
        changed_paths=("src/main.py", "README.md"),
        exit_code=0,
    )
    assert success.truncated is True
    assert success.changed_paths == ("src/main.py", "README.md")
    assert ToolResult.model_validate_json(success.model_dump_json()) == success
    with pytest.raises(ValidationError):
        make_result(call.call_id, status=ToolCallStatus.RUNNING)
    with pytest.raises(ValidationError):
        make_result(call.call_id, status=ToolCallStatus.FAILED)
    with pytest.raises(ValidationError):
        make_result(
            call.call_id,
            error=ErrorInfo(category=ErrorCategory.UNKNOWN, message="failed"),
        )


def test_tool_result_rejects_absolute_paths_and_oversized_outputs() -> None:
    call_id = make_call().call_id
    for path in ("/etc/passwd", "../secret", "C:/secret", "src\\main.py"):
        with pytest.raises(ValidationError):
            make_result(call_id, changed_paths=(path,))
    with pytest.raises(ValidationError):
        make_result(call_id, changed_paths=("src//main.py",))
    with pytest.raises(ValidationError):
        make_result(call_id, output="x" * (MAX_TOOL_OUTPUT_BYTES + 1))
    with pytest.raises(ValidationError):
        make_result(call_id, result={"output": "x" * MAX_TOOL_OUTPUT_BYTES})
    with pytest.raises(ValidationError):
        make_result(call_id, result={"nested": {"token": "redacted"}})


def test_tool_lifecycle_requires_approval_and_keeps_terminal_states_immutable() -> None:
    call = make_call()
    rejected = call.transition(ToolCallStatus.REJECTED)
    assert rejected.status is ToolCallStatus.REJECTED
    awaiting = call.transition(ToolCallStatus.AWAITING_APPROVAL)
    with pytest.raises(ValueError, match="without approval"):
        awaiting.transition(ToolCallStatus.RUNNING)
    running = awaiting.transition(ToolCallStatus.RUNNING, approved=True)
    assert running.status is ToolCallStatus.RUNNING
    succeeded = running.transition(ToolCallStatus.SUCCEEDED)
    with pytest.raises(ValueError):
        succeeded.transition(ToolCallStatus.RUNNING)
    assert (
        succeeded.transition(
            ToolCallStatus.SUCCEEDED,
            idempotent_terminal=True,
        ).status
        is ToolCallStatus.SUCCEEDED
    )


def test_rejected_result_is_created_without_a_running_transition() -> None:
    call = make_call()
    result = ToolResult.from_rejected_call(
        call,
        reason="grant_denied",
        completed_at=T0,
    )
    assert result.status is ToolCallStatus.REJECTED
    assert result.result == {"error": "grant_denied"}
    assert result.error is not None
    assert result.error.category is ErrorCategory.INVALID_REQUEST
    with pytest.raises(ValueError, match="safe reason code"):
        ToolResult.from_rejected_call(call, reason="not allowed", completed_at=T0)
    with pytest.raises(ValueError, match="pre-execution"):
        ToolResult.from_rejected_call(
            call.transition(ToolCallStatus.RUNNING),
            reason="grant_denied",
            completed_at=T0,
        )


@pytest.mark.parametrize(
    "status",
    [ToolCallStatus.INTERRUPTED, ToolCallStatus.UNKNOWN],
)
def test_unknown_and_interrupted_are_first_class_terminal_results(
    status: ToolCallStatus,
) -> None:
    call = make_call().transition(ToolCallStatus.RUNNING)
    error = ErrorInfo(category=ErrorCategory.UNKNOWN, message="outcome is unconfirmed")
    result = ToolResult.from_call(
        call,
        status=status,
        error=error,
        completed_at=T0,
    )
    assert result.status is status
    assert result.error == error


def test_result_cannot_be_created_before_a_running_call() -> None:
    with pytest.raises(ValueError, match="running"):
        ToolResult.from_call(
            make_call(),
            status=ToolCallStatus.SUCCEEDED,
            completed_at=T0,
        )
