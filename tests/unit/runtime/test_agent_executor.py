"""R0 contract tests for the production Agent tool adapter."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from prp_runtime.domain.enums import (
    AgentMode,
    AttemptStatus,
    ModelRole,
    ResourceAccess,
    ToolCallStatus,
    ToolEffect,
)
from prp_runtime.domain.models import (
    MAX_AGENT_RESULT_BYTES,
    AgentToolCall,
    Attempt,
    ExecutionScope,
    WorkspaceGrant,
)
from prp_runtime.domain.values import (
    ModelRef,
    new_attempt_id,
    new_principal_id,
    new_run_id,
    new_session_id,
    new_snapshot_id,
    new_work_unit_id,
    new_workspace_id,
)
from prp_runtime.policy.engine import PolicyDecision, PolicyOutcome, PolicyReasonCode
from prp_runtime.runtime.agent_executor import AgentToolExecutor
from prp_runtime.runtime.agent_loop import AgentToolContext
from prp_runtime.runtime.worker import ResumeAction, ResumeState
from prp_runtime.tools.executor import ToolExecutionOutcome
from prp_runtime.tools.models import ToolCall, ToolResult
from prp_runtime.tools.registry import ToolDefinition, ToolRegistry

T0 = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str


class FakeWorker:
    def __init__(
        self,
        *,
        result: ToolResult | None = None,
        outcome: PolicyOutcome = PolicyOutcome.ALLOW,
        raises: bool = False,
    ) -> None:
        self.result = result
        self.outcome = outcome
        self.raises = raises
        self.calls: list[ToolCall] = []
        self.kwargs: list[dict[str, Any]] = []

    async def execute(
        self, call: ToolCall, mode: AgentMode, **kwargs: object
    ) -> ToolExecutionOutcome:
        if self.raises:
            raise RuntimeError("private path=/srv/workspace secret=hidden")
        self.calls.append(call)
        self.kwargs.append({"mode": mode, **kwargs})
        decision = PolicyDecision(
            call_id=call.call_id,
            tool_name=call.tool_name,
            effect=call.effect,
            mode=mode,
            outcome=self.outcome,
            reason_code=(
                PolicyReasonCode.APPROVAL_REQUIRED
                if self.outcome is PolicyOutcome.ASK
                else PolicyReasonCode.READ_ALLOWED
            ),
        )
        result = self.result
        if result is None and self.outcome is PolicyOutcome.ALLOW:
            result = ToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.SUCCEEDED,
                result={"content": "safe"},
                output="safe",
                completed_at=T0,
            )
        return ToolExecutionOutcome(decision=decision, call=call, result=result)


class PrivateStoreWorker(FakeWorker):
    """A regression fixture for the removed private dependency discovery."""

    def __init__(self) -> None:
        super().__init__(outcome=PolicyOutcome.ASK)
        self._executor = type("PrivateExecutor", (), {"_store": object()})()


def make_scope() -> ExecutionScope:
    principal_id = new_principal_id()
    workspace_id = new_workspace_id()
    return ExecutionScope(
        run_id=new_run_id(),
        session_id=new_session_id(),
        principal_id=principal_id,
        workspace_id=workspace_id,
        grant=WorkspaceGrant(
            principal_id=principal_id,
            workspace_id=workspace_id,
        ),
    )


def make_registry() -> ToolRegistry:
    async def read_file(arguments: BaseModel) -> dict[str, object]:
        del arguments
        return {"content": "safe"}

    return ToolRegistry(
        (
            ToolDefinition(
                name="read_file",
                effect=ToolEffect.READ,
                argument_model=Arguments,
                handler=read_file,
            ),
        )
    )


def make_call() -> AgentToolCall:
    return AgentToolCall(
        call_id="provider-call/1",
        tool_name="read_file",
        arguments={"path": "src/main.py"},
    )


def make_context(scope: ExecutionScope) -> AgentToolContext:
    return AgentToolContext(
        run_id=scope.run_id,
        work_unit_id=new_work_unit_id(),
        mode=scope.agent_options.agent_mode,
        round_index=1,
        attempt_index=1,
    )


@pytest.mark.asyncio
async def test_public_call_maps_to_scoped_stable_internal_call() -> None:
    scope = make_scope()
    worker = FakeWorker()
    adapter = AgentToolExecutor(
        worker,
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )
    call = make_call()
    context = make_context(scope)

    first = await adapter.execute(call, context=context)
    second = await adapter.execute(call, context=context)

    assert first.result is not None
    assert second.result is not None
    assert first.result.call_id == call.call_id
    assert first.result.output == "safe"
    assert len(worker.calls) == 2
    assert worker.calls[0].call_id.startswith("tc_")
    assert worker.calls[0].call_id != call.call_id
    assert worker.calls[0].call_id == worker.calls[1].call_id
    assert worker.calls[0].effect is ToolEffect.READ
    assert worker.calls[0].run_id == scope.run_id
    assert worker.calls[0].work_unit_id == context.work_unit_id
    assert worker.calls[0].snapshot_id == adapter.snapshot_id
    assert worker.kwargs[0]["workspace_id"] == scope.workspace_id


def test_approval_factory_narrows_grant_and_is_stable() -> None:
    base_scope = make_scope()
    scope = base_scope.model_copy(
        update={
            "grant": WorkspaceGrant(
                principal_id=base_scope.principal_id,
                workspace_id=base_scope.workspace_id,
                access=(ResourceAccess.READ, ResourceAccess.WRITE),
                expires_at=T0 + timedelta(hours=1),
            )
        }
    )
    adapter = AgentToolExecutor(
        FakeWorker(),
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )
    call = ToolCall(
        call_id="tc_approval",
        run_id=scope.run_id,
        work_unit_id=new_work_unit_id(),
        tool_name="apply_patch",
        effect=ToolEffect.WRITE,
        arguments={"path": "src/main.py"},
        snapshot_id=adapter.snapshot_id,
        requested_at=T0,
    )
    first = adapter.build_approval_request(
        call,
        reason=PolicyReasonCode.APPROVAL_REQUIRED,
        resolved_paths=("src/main.py",),
        requested_at=T0,
    )
    replay = adapter.build_approval_request(
        call,
        reason=PolicyReasonCode.APPROVAL_REQUIRED,
        resolved_paths=("src/main.py",),
        requested_at=T0,
    )
    assert first.request_id == replay.request_id
    assert first.issuer.value == "SERVER"
    assert first.scope.tools == ("apply_patch",)
    assert first.scope.effects == (ToolEffect.WRITE,)
    assert first.scope.paths == ("src/main.py",)
    assert first.scope.budget.max_calls == 1
    with pytest.raises(ValueError, match="resolved paths"):
        adapter.build_approval_request(
            call,
            reason=PolicyReasonCode.APPROVAL_REQUIRED,
            resolved_paths=(),
            requested_at=T0,
        )


@pytest.mark.asyncio
async def test_unknown_or_invalid_public_calls_are_rejected_without_worker() -> None:
    scope = make_scope()
    worker = FakeWorker()
    adapter = AgentToolExecutor(
        worker,
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )
    context = make_context(scope)

    unknown = await adapter.execute(
        AgentToolCall(call_id="unknown", tool_name="unknown_tool"),
        context=context,
    )
    invalid = await adapter.execute(
        AgentToolCall(call_id="invalid", tool_name="read_file", arguments={"other": 1}),
        context=context,
    )

    assert unknown.result is not None
    assert unknown.result.status is ToolCallStatus.REJECTED
    assert invalid.result is not None
    assert invalid.result.status is ToolCallStatus.REJECTED
    assert worker.calls == []


@pytest.mark.asyncio
async def test_scope_mismatch_is_rejected_before_mapping() -> None:
    scope = make_scope()
    worker = FakeWorker()
    adapter = AgentToolExecutor(
        worker,
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )
    context = make_context(scope).model_copy(update={"run_id": new_run_id()})

    execution = await adapter.execute(make_call(), context=context)

    assert execution.result is not None
    assert execution.result.status is ToolCallStatus.REJECTED
    assert worker.calls == []


@pytest.mark.asyncio
async def test_ask_pauses_and_deny_returns_a_public_rejection() -> None:
    scope = make_scope()
    context = make_context(scope)
    paused = AgentToolExecutor(
        FakeWorker(outcome=PolicyOutcome.ASK),
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )
    denied = AgentToolExecutor(
        FakeWorker(outcome=PolicyOutcome.DENY),
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )

    pause_execution = await paused.execute(make_call(), context=context)
    deny_execution = await denied.execute(make_call(), context=context)

    assert pause_execution.awaiting_approval is True
    assert pause_execution.reason == PolicyReasonCode.APPROVAL_REQUIRED.value
    assert pause_execution.result is None
    assert deny_execution.result is not None
    assert deny_execution.result.status is ToolCallStatus.REJECTED


@pytest.mark.asyncio
async def test_ask_does_not_discover_a_store_through_private_worker_fields() -> None:
    scope = make_scope()
    worker = PrivateStoreWorker()
    adapter = AgentToolExecutor(
        worker,
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )

    execution = await adapter.execute(make_call(), context=make_context(scope))

    assert execution.awaiting_approval is True
    assert execution.result is None


@pytest.mark.asyncio
async def test_internal_worker_error_is_a_safe_failed_public_result() -> None:
    scope = make_scope()
    adapter = AgentToolExecutor(
        FakeWorker(raises=True),
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )

    execution = await adapter.execute(make_call(), context=make_context(scope))

    assert execution.result is not None
    assert execution.result.status is ToolCallStatus.FAILED
    assert "srv" not in execution.result.output
    assert "secret" not in execution.result.output
    assert execution.result.result == {"error": "internal_tool_failure"}


@pytest.mark.asyncio
async def test_public_result_is_bounded_and_marked_truncated() -> None:
    scope = make_scope()
    adapter = AgentToolExecutor(
        FakeWorker(
            result=ToolResult(
                call_id="tc_result",
                status=ToolCallStatus.SUCCEEDED,
                result={"content": "y" * MAX_AGENT_RESULT_BYTES},
                output="x" * (MAX_AGENT_RESULT_BYTES + 100),
                completed_at=T0,
            )
        ),
        make_registry(),
        scope,
        snapshot_id=new_snapshot_id(),
    )

    execution = await adapter.execute(make_call(), context=make_context(scope))

    assert execution.result is not None
    assert execution.result.status is ToolCallStatus.SUCCEEDED
    assert execution.result.truncated is True
    assert execution.result.result is None
    assert len(execution.result.output) <= MAX_AGENT_RESULT_BYTES


def test_resume_state_requires_a_closed_action_shape() -> None:
    scope = make_scope()
    attempt = Attempt(
        attempt_id=new_attempt_id(),
        run_id=scope.run_id,
        work_unit_id=new_work_unit_id(),
        role=ModelRole.WORKER,
        model=ModelRef(provider="fake", model="model"),
        status=AttemptStatus.SUCCEEDED,
        created_at=T0,
        started_at=T0,
        completed_at=T0,
    )

    state = ResumeState(
        action=ResumeAction.CONTINUE,
        attempt=attempt,
    )
    assert state.action is ResumeAction.CONTINUE
    with pytest.raises(ValueError, match="blocked resume state"):
        ResumeState(action=ResumeAction.BLOCK, attempt=attempt)
    with pytest.raises(ValueError, match="pending resume action"):
        ResumeState(action=ResumeAction.WAIT, attempt=attempt)
