"""Targeted tests for the policy-gated tool executor."""

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from prp_runtime.domain.enums import AgentMode, ToolCallStatus, ToolEffect
from prp_runtime.domain.models import ErrorCategory
from prp_runtime.domain.values import (
    new_run_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
)
from prp_runtime.policy.engine import PolicyOutcome, PolicyReasonCode
from prp_runtime.policy.models import CommandClass
from prp_runtime.tools import (
    ExecutionContext,
    ToolDefinition,
    ToolExecutionOutcome,
    ToolExecutor,
    ToolRegistry,
)
from prp_runtime.tools.command import (
    CommandInvocation,
    CommandResult,
    build_targeted_test_definition,
)
from prp_runtime.tools.models import ToolCall, ToolResult


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str


class FakeStore:
    def __init__(self) -> None:
        self.calls: dict[str, ToolCall] = {}
        self.results: dict[str, ToolResult] = {}
        self.idempotency: dict[str, str] = {}
        self.transitions: list[ToolCallStatus] = []

    async def create_tool_call(
        self,
        call: ToolCall,
        *,
        workspace_id: str,
        idempotency_key: str,
    ) -> ToolCall:
        del workspace_id
        existing_id = self.idempotency.get(idempotency_key)
        if existing_id is not None:
            return self.calls[existing_id]
        self.idempotency[idempotency_key] = call.call_id
        self.calls[call.call_id] = call
        self.transitions.append(call.status)
        return call

    async def await_tool_call(
        self,
        call_id: str,
        *,
        reason: str = "approval required",
        timestamp: Any = None,
    ) -> ToolCall:
        del reason, timestamp
        current = self.calls[call_id]
        updated = current.transition(ToolCallStatus.AWAITING_APPROVAL)
        self.calls[call_id] = updated
        self.transitions.append(updated.status)
        return updated

    async def start_tool_call(
        self,
        call_id: str,
        *,
        approved: bool | None = None,
        started_at: Any = None,
    ) -> ToolCall:
        del started_at
        current = self.calls[call_id]
        updated = current.transition(ToolCallStatus.RUNNING, approved=approved)
        self.calls[call_id] = updated
        self.transitions.append(updated.status)
        return updated

    async def complete_tool_call(self, result: ToolResult) -> ToolResult:
        existing = self.results.get(result.call_id)
        if existing is not None:
            return existing
        current = self.calls[result.call_id]
        self.calls[result.call_id] = current.transition(result.status)
        self.results[result.call_id] = result
        self.transitions.append(result.status)
        return result

    async def reject_tool_call(
        self,
        call_id: str,
        *,
        reason: str,
        completed_at: Any = None,
    ) -> ToolResult:
        existing = self.results.get(call_id)
        if existing is not None:
            return existing
        current = self.calls[call_id]
        result = ToolResult.from_rejected_call(
            current,
            reason=reason,
            completed_at=completed_at or "2026-08-14T12:00:00+00:00",
        )
        self.calls[call_id] = current.transition(ToolCallStatus.REJECTED)
        self.results[call_id] = result
        self.transitions.append(ToolCallStatus.REJECTED)
        return result

    async def get_tool_result(self, call_id: str) -> ToolResult:
        return self.results[call_id]

    async def mark_tool_call_unknown(
        self,
        call_id: str,
        *,
        completed_at: Any = None,
        message: str = "tool outcome is unconfirmed after restart",
    ) -> ToolResult:
        current = self.calls[call_id]
        result = ToolResult.from_call(
            current,
            status=ToolCallStatus.UNKNOWN,
            error={"category": ErrorCategory.UNKNOWN, "message": message},
            completed_at=completed_at,
        )
        return await self.complete_tool_call(result)


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
        "snapshot_id": new_snapshot_id(),
        "requested_at": "2026-08-14T12:00:00+00:00",
    }
    values.update(overrides)
    return ToolCall(**values)  # type: ignore[arg-type]


def definition(handler: Any, *, effect: ToolEffect = ToolEffect.READ) -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        effect=effect,
        argument_model=Arguments,
        handler=handler,
        max_output_bytes=8,
    )


@pytest.mark.asyncio
async def test_deny_and_ask_never_invoke_handler() -> None:
    calls = 0

    async def handler(context: BaseModel) -> Mapping[str, object]:
        del context
        nonlocal calls
        calls += 1
        return {"ok": True}

    store = FakeStore()
    executor = ToolExecutor(ToolRegistry((definition(handler),)), store)

    denied = await executor.execute(
        make_call("not_registered"), AgentMode.AUTO, workspace_id="ws-test"
    )
    assert denied.decision.outcome is PolicyOutcome.DENY
    assert denied.decision.reason_code is PolicyReasonCode.UNKNOWN_TOOL
    assert denied.result is not None
    assert denied.result.status is ToolCallStatus.REJECTED
    assert denied.call.status is ToolCallStatus.REJECTED
    assert calls == 0
    assert len(store.calls) == 1
    assert store.transitions == [ToolCallStatus.REQUESTED, ToolCallStatus.REJECTED]

    asking_executor = ToolExecutor(
        ToolRegistry((definition(handler, effect=ToolEffect.WRITE),)), store
    )
    asking = await asking_executor.execute(
        make_call(effect=ToolEffect.WRITE),
        AgentMode.NORMAL,
        workspace_id="ws-test",
    )
    assert asking.decision.outcome is PolicyOutcome.ASK
    assert asking.call.status is ToolCallStatus.AWAITING_APPROVAL
    assert calls == 0


@pytest.mark.asyncio
async def test_allow_validates_arguments_runs_handler_and_replays_idempotently() -> None:
    calls = 0

    async def handler(context: ExecutionContext) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        assert context.arguments.path == "src/main.py"
        return {"ok": True}

    store = FakeStore()
    executor = ToolExecutor(ToolRegistry((definition(handler),)), store)
    first = await executor.execute(
        make_call(),
        AgentMode.AUTO,
        workspace_id="ws-test",
        idempotency_key="same-call",
    )
    replay = await executor.execute(
        make_call(),
        AgentMode.AUTO,
        workspace_id="ws-test",
        idempotency_key="same-call",
    )

    assert isinstance(first, ToolExecutionOutcome)
    assert first.result is not None
    assert first.result.status is ToolCallStatus.SUCCEEDED
    assert replay.result == first.result
    assert calls == 1
    assert store.transitions == [
        ToolCallStatus.REQUESTED,
        ToolCallStatus.RUNNING,
        ToolCallStatus.SUCCEEDED,
    ]


@pytest.mark.asyncio
async def test_handler_exception_and_timeout_are_failed_results() -> None:
    async def raises(context: BaseModel) -> None:
        del context
        raise RuntimeError("private handler detail")

    store = FakeStore()
    failed = await ToolExecutor(ToolRegistry((definition(raises),)), store).execute(
        make_call(), AgentMode.AUTO, workspace_id="ws-test"
    )
    assert failed.result is not None
    assert failed.result.status is ToolCallStatus.FAILED
    assert failed.result.error is not None
    assert failed.result.error.category is ErrorCategory.INTERNAL
    assert failed.result.error.message == "tool handler failed"

    async def hangs(context: BaseModel) -> None:
        del context
        await asyncio.Future()

    timed_out = await ToolExecutor(
        ToolRegistry((definition(hangs),)), FakeStore(), timeout_seconds=0.01
    ).execute(make_call(), AgentMode.AUTO, workspace_id="ws-test")
    assert timed_out.result is not None
    assert timed_out.result.status is ToolCallStatus.FAILED
    assert timed_out.result.error is not None
    assert timed_out.result.error.category is ErrorCategory.TIMEOUT


@pytest.mark.asyncio
async def test_cancellation_records_unknown_and_propagates_cancel() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(context: BaseModel) -> None:
        del context
        started.set()
        await release.wait()

    store = FakeStore()
    task = asyncio.create_task(
        ToolExecutor(ToolRegistry((definition(handler),)), store).execute(
            make_call(), AgentMode.AUTO, workspace_id="ws-test"
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(store.results) == 1
    assert next(iter(store.results.values())).status is ToolCallStatus.UNKNOWN


@pytest.mark.asyncio
async def test_result_limit_is_explicitly_truncated_and_effect_mismatch_is_rejected() -> None:
    async def long_result(context: BaseModel) -> str:
        del context
        return "0123456789"

    store = FakeStore()
    executor = ToolExecutor(ToolRegistry((definition(long_result),)), store)
    outcome = await executor.execute(make_call(), AgentMode.AUTO, workspace_id="ws-test")
    assert outcome.result is not None
    assert outcome.result.output == "01234567"
    assert outcome.result.truncated is True

    with pytest.raises(RuntimeError, match="does not match registered effect"):
        await executor.execute(
            make_call(effect=ToolEffect.WRITE),
            AgentMode.YOLO,
            workspace_id="ws-test",
        )


@pytest.mark.asyncio
async def test_plan_denies_targeted_command_before_runner_spawn() -> None:
    class RecordingRunner:
        calls = 0

        async def run(self, invocation: CommandInvocation) -> CommandResult:
            del invocation
            self.calls += 1
            return CommandResult(exit_code=0, duration_ms=1)

    runner = RecordingRunner()
    definition = build_targeted_test_definition(runner)  # type: ignore[arg-type]
    store = FakeStore()
    outcome = await ToolExecutor(ToolRegistry((definition,)), store).execute(
        make_call(
            "run_targeted_test",
            ToolEffect.COMMAND,
            arguments={"spec_name": "fixture", "parameters": {"mode": "success"}},
        ),
        AgentMode.PLAN,
        workspace_id="ws-test",
        command_class=CommandClass.TEST,
    )
    assert outcome.decision.outcome is PolicyOutcome.DENY
    assert outcome.decision.reason_code is PolicyReasonCode.PLAN_SIDE_EFFECT
    assert runner.calls == 0
    assert outcome.result is not None
    assert outcome.result.status is ToolCallStatus.REJECTED
    assert store.transitions == [ToolCallStatus.REQUESTED, ToolCallStatus.REJECTED]
