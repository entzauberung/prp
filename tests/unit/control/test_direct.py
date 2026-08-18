"""Targeted tests for the DIRECT control loop. No network, fake provider only."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from prp_runtime.control.controller import RunController
from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, ProviderError, StateError
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import (
    AgentToolCall,
    AgentToolResult,
    ArtifactKind,
    Budget,
    ErrorCategory,
    NativeRunRequest,
    OutputRequirement,
    Session,
    Usage,
    WorkspaceGrant,
)
from prp_runtime.domain.values import new_session_id, new_workspace_id
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.runtime.agent_loop import AgentToolContext, AgentToolExecution
from prp_runtime.runtime.assembler import assemble_run_result
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.workspace.models import Workspace, WorkspaceSource, WorkspaceSourceType

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="openai_compatible",
    model="weak-model",
    role=ModelRole.WORKER,
    base_url="https://models.internal/v1",
    context_window_tokens=32_000,
    max_output_tokens=4_000,
)

LEADER_PROFILE = ModelProfile(
    alias="leader",
    provider="openai_compatible",
    model="strong-model",
    role=ModelRole.PLANNER,
    base_url="https://models.internal/v1",
    context_window_tokens=128_000,
    max_output_tokens=8_000,
    supports_structured_output=True,
)


class FakeAdapter:
    """Returns queued outcomes and records every request it received."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes: list[object] = list(outcomes)
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        outcome = self._outcomes.pop(0) if self._outcomes else _text_response("the answer")
        if callable(outcome):
            outcome = await outcome(request)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, ProviderResponse)
        return outcome


class FakeToolExecutor:
    def __init__(self, *, pause: bool = False) -> None:
        self.pause = pause
        self.calls: list[AgentToolCall] = []

    async def execute(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        del context
        self.calls.append(call)
        if self.pause:
            return AgentToolExecution(
                call=call,
                awaiting_approval=True,
                reason="approval required",
            )
        return AgentToolExecution(
            call=call,
            result=AgentToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.SUCCEEDED,
                result={"content": "tool output"},
                output="tool output",
            ),
        )


def _text_response(text: str, usage: Usage | None = None) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        usage=Usage(input_tokens=11, output_tokens=7, elapsed_ms=12) if usage is None else usage,
        finish_reason=FinishReason.STOP,
    )


def _tool_response(call: AgentToolCall) -> ProviderResponse:
    return ProviderResponse(
        tool_calls=(call,),
        usage=Usage(input_tokens=3, output_tokens=2, elapsed_ms=1),
        finish_reason=FinishReason.TOOL_CALLS,
    )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "direct.db") as opened:
        yield opened


def build_controller(
    store: SqliteStore,
    adapter: FakeAdapter,
    *,
    leader: FakeAdapter | None = None,
    tool_executor: FakeToolExecutor | None = None,
) -> RunController:
    settings = Settings(leader_profile=LEADER_PROFILE, worker_profile=WORKER_PROFILE)
    adapters: dict[str, object] = {"worker": adapter}
    if leader is not None:
        adapters["leader"] = leader
    return RunController(
        store,
        settings,
        adapters,  # type: ignore[arg-type]
        tool_executor=tool_executor,
    )


async def event_types(store: SqliteStore, run_id: str) -> list[EventType]:
    ledger = await store.list_events(run_id)
    assert assert_sequence_chain(ledger) is None
    return [event.event_type for event in ledger]


# --- success ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_run_completes_with_one_attempt(store: SqliteStore) -> None:
    adapter = FakeAdapter()
    controller = build_controller(store, adapter)

    created = await controller.create_run(
        NativeRunRequest(input="summarise the report", instructions="be terse")
    )
    assert created.status is RunStatus.PENDING
    assert created.strategy is None

    finished = await controller.execute(created.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.strategy is ExecutionStrategy.DIRECT
    assert finished.error is None
    assert finished.final_work_unit_id is None
    assert finished.started_at is not None
    assert finished.completed_at is not None

    units = await store.list_work_units(created.run_id)
    assert len(units) == 1
    assert units[0].status is WorkUnitStatus.SUCCEEDED
    assert units[0].name == "direct"
    assert units[0].instruction == "summarise the report"

    attempts = await store.list_attempts(units[0].work_unit_id)
    assert len(attempts) == 1
    assert attempts[0].status is AttemptStatus.SUCCEEDED
    assert attempts[0].attempt_index == 1
    assert attempts[0].role is ModelRole.WORKER
    assert attempts[0].model == WORKER_PROFILE.model_ref
    assert attempts[0].usage == Usage(input_tokens=11, output_tokens=7, elapsed_ms=12)

    artifacts = await store.list_artifacts(units[0].work_unit_id)
    assert len(artifacts) == 1
    assert artifacts[0].content == "the answer"
    assert artifacts[0].name == "answer"

    assert (await store.get_run_usage(created.run_id)) == Usage(
        input_tokens=11, output_tokens=7, elapsed_ms=12
    )
    assert adapter.call_count == 1
    assert adapter.requests[0].alias == "worker"
    assert adapter.requests[0].input == "summarise the report"
    assert adapter.requests[0].instructions == "be terse"
    assert adapter.requests[0].json_schema is None


@pytest.mark.asyncio
async def test_direct_worker_reuses_bounded_agent_loop_for_tool_calls(
    store: SqliteStore,
) -> None:
    call = AgentToolCall(
        call_id="call-1",
        tool_name="read_file",
        arguments={"path": "src/main.py"},
    )
    adapter = FakeAdapter(_tool_response(call), _text_response("final answer"))
    executor = FakeToolExecutor()
    controller = build_controller(store, adapter, tool_executor=executor)
    run = await controller.create_run(NativeRunRequest(input="inspect the file"))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert adapter.call_count == 2
    assert executor.calls == [call]
    assert adapter.requests[1].history[0].kind == "turn"
    assert adapter.requests[1].history[1].kind == "tool_result"
    units = await store.list_work_units(run.run_id)
    artifacts = await store.list_artifacts(units[0].work_unit_id)
    assert artifacts[0].content == "final answer"


@pytest.mark.asyncio
async def test_direct_approval_pause_blocks_unit_without_failing_run(
    store: SqliteStore,
) -> None:
    call = AgentToolCall(call_id="call-1", tool_name="apply_patch")
    adapter = FakeAdapter(_tool_response(call))
    controller = build_controller(
        store,
        adapter,
        tool_executor=FakeToolExecutor(pause=True),
    )
    run = await controller.create_run(NativeRunRequest(input="make the change"))

    paused = await controller.execute(run.run_id)

    assert paused.status is RunStatus.RUNNING
    units = await store.list_work_units(run.run_id)
    assert units[0].status is WorkUnitStatus.RUNNING
    assert paused.error is None


@pytest.mark.asyncio
async def test_direct_success_event_order(store: SqliteStore) -> None:
    controller = build_controller(store, FakeAdapter())
    run = await controller.create_run(NativeRunRequest(input="hello"))
    await controller.execute(run.run_id)
    types = await event_types(store, run.run_id)
    assert types == [
        EventType.RUN_CREATED,
        EventType.STRATEGY_SELECTED,
        EventType.CONTROLLER_DECISION,
        EventType.RUN_STARTED,
        EventType.WORK_UNIT_CREATED,
        EventType.WORK_UNIT_READY,
        EventType.WORK_UNIT_STARTED,
        EventType.RESERVATION_CREATED,
        EventType.RESERVATION_HELD,
        EventType.ATTEMPT_STARTED,
        EventType.AGENT_HISTORY_RECORDED,
        EventType.ATTEMPT_SUCCEEDED,
        EventType.ARTIFACT_PRODUCED,
        EventType.USAGE_UPDATED,
        EventType.RESERVATION_SETTLED,
        EventType.EVIDENCE_RECORDED,
        EventType.EVIDENCE_RECORDED,
        EventType.CONTROLLER_DECISION,
        EventType.WORK_UNIT_SUCCEEDED,
        EventType.RUN_SUCCEEDED,
    ]


@pytest.mark.asyncio
async def test_assembled_result_reflects_persisted_facts(store: SqliteStore) -> None:
    controller = build_controller(store, FakeAdapter())
    run = await controller.create_run(NativeRunRequest(input="hello"))
    await controller.execute(run.run_id)
    result = await assemble_run_result(store, run.run_id)
    assert result.status is RunStatus.SUCCEEDED
    assert result.strategy is ExecutionStrategy.DIRECT
    assert result.output_text == "the answer"
    assert result.output_kind is ArtifactKind.TEXT
    assert result.usage.total_tokens == 18
    assert result.error is None


@pytest.mark.asyncio
async def test_expired_session_scope_stops_before_provider_dispatch(
    store: SqliteStore,
) -> None:
    owner_id = "prn_tenant_owner"
    created_at = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    workspace = Workspace(
        workspace_id=new_workspace_id(),
        owner_id=owner_id,
        alias="project-main",
        source=WorkspaceSource(
            source_type=WorkspaceSourceType.SERVER_ALIAS,
            server_alias="repo-main",
        ),
        created_at=created_at,
    )
    await store.create_workspace(workspace)
    session = Session(
        session_id=new_session_id(),
        principal_id=owner_id,
        workspace_id=workspace.workspace_id,
        grant=WorkspaceGrant(
            principal_id=owner_id,
            workspace_id=workspace.workspace_id,
            access=(ResourceAccess.READ,),
            expires_at=created_at + timedelta(seconds=1),
        ),
        created_at=created_at,
        expires_at=created_at + timedelta(seconds=1),
    )
    await store.create_session(session)
    adapter = FakeAdapter()
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="inspect"))
    await store.attach_run_to_session(session.session_id, run.run_id, principal_id=owner_id)

    with pytest.raises(StateError, match="expired"):
        await controller.execute(run.run_id, principal_id=owner_id)
    assert adapter.call_count == 0


@pytest.mark.asyncio
async def test_structured_output_requirement_reaches_the_provider(
    store: SqliteStore,
) -> None:
    adapter = FakeAdapter(_text_response('{"ok": true}'))
    settings = Settings(leader_profile=LEADER_PROFILE, worker_profile=LEADER_PROFILE.model_copy(
        update={"alias": "worker", "role": ModelRole.WORKER}
    ))
    controller = RunController(store, settings, {"worker": adapter})  # type: ignore[arg-type]
    run = await controller.create_run(
        NativeRunRequest(
            input="emit json",
            output=OutputRequirement(kind=ArtifactKind.JSON, json_schema='{"type":"object"}'),
        )
    )
    finished = await controller.execute(run.run_id)
    assert finished.status is RunStatus.SUCCEEDED
    assert adapter.requests[0].json_schema == '{"type":"object"}'
    assert adapter.requests[0].instructions is not None
    result = await assemble_run_result(store, run.run_id)
    assert result.output_kind is ArtifactKind.JSON
    assert result.output_text == '{"ok": true}'


@pytest.mark.asyncio
async def test_missing_upstream_usage_is_not_invented(store: SqliteStore) -> None:
    adapter = FakeAdapter(
        ProviderResponse(text="answer", usage=None, finish_reason=FinishReason.STOP)
    )
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="hello"))
    await controller.execute(run.run_id)
    units = await store.list_work_units(run.run_id)
    assert (await store.list_attempts(units[0].work_unit_id))[0].usage is None
    assert await store.get_run_usage(run.run_id) == Usage()
    assert EventType.USAGE_UPDATED not in await event_types(store, run.run_id)


# --- failure ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_failure_fails_the_run(store: SqliteStore) -> None:
    adapter = FakeAdapter(
        ProviderError("upstream timed out", code=ErrorCode.PROVIDER_TIMEOUT)
    )
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="hello"))
    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.TIMEOUT
    units = await store.list_work_units(run.run_id)
    assert units[0].status is WorkUnitStatus.FAILED
    attempts = await store.list_attempts(units[0].work_unit_id)
    assert attempts[0].status is AttemptStatus.FAILED
    assert attempts[0].error is not None
    assert await store.list_artifacts(units[0].work_unit_id) == ()
    assert await event_types(store, run.run_id) == [
        EventType.RUN_CREATED,
        EventType.STRATEGY_SELECTED,
        EventType.CONTROLLER_DECISION,
        EventType.RUN_STARTED,
        EventType.WORK_UNIT_CREATED,
        EventType.WORK_UNIT_READY,
        EventType.WORK_UNIT_STARTED,
        EventType.RESERVATION_CREATED,
        EventType.RESERVATION_HELD,
        EventType.ATTEMPT_STARTED,
        EventType.ATTEMPT_FAILED,
        EventType.RESERVATION_SETTLED,
        EventType.WORK_UNIT_FAILED,
        EventType.RUN_FAILED,
    ]


@pytest.mark.asyncio
async def test_blank_completion_is_a_failure_not_an_artifact(store: SqliteStore) -> None:
    adapter = FakeAdapter(
        ProviderError("blank completion", code=ErrorCode.PROVIDER_INVALID_RESPONSE)
    )
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="hello"))
    finished = await controller.execute(run.run_id)
    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.PROVIDER_ERROR
    units = await store.list_work_units(run.run_id)
    assert await store.list_artifacts(units[0].work_unit_id) == ()


@pytest.mark.asyncio
async def test_a_failed_run_has_no_second_attempt(store: SqliteStore) -> None:
    adapter = FakeAdapter(
        ProviderError("boom", code=ErrorCode.PROVIDER_INVALID_RESPONSE),
        _text_response("recovered"),
    )
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="hello"))
    await controller.execute(run.run_id)
    assert adapter.call_count == 1
    units = await store.list_work_units(run.run_id)
    assert len(await store.list_attempts(units[0].work_unit_id)) == 1


# --- cancellation ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_before_execute_prevents_any_provider_call(
    store: SqliteStore,
) -> None:
    adapter = FakeAdapter()
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="hello"))

    cancelled = await controller.cancel(run.run_id)
    assert cancelled.status is RunStatus.CANCELLED

    same = await controller.execute(run.run_id)
    assert same.status is RunStatus.CANCELLED
    assert adapter.call_count == 0
    assert await store.list_work_units(run.run_id) == ()
    assert await store.list_run_attempts(run.run_id) == ()
    assert await event_types(store, run.run_id) == [
        EventType.RUN_CREATED,
        EventType.RUN_CANCELLED,
    ]


@pytest.mark.asyncio
async def test_cancel_is_idempotent(store: SqliteStore) -> None:
    controller = build_controller(store, FakeAdapter())
    run = await controller.create_run(NativeRunRequest(input="hello"))
    first = await controller.cancel(run.run_id)
    second = await controller.cancel(run.run_id)
    assert first.status is second.status is RunStatus.CANCELLED
    assert await event_types(store, run.run_id) == [
        EventType.RUN_CREATED,
        EventType.RUN_CANCELLED,
    ]


@pytest.mark.asyncio
async def test_cancel_during_an_attempt_wins_over_success(store: SqliteStore) -> None:
    controller: RunController | None = None
    run_id = ""

    async def cancel_midflight(request: ProviderRequest) -> ProviderResponse:
        assert controller is not None
        await controller.cancel(run_id)
        return _text_response("late answer")

    adapter = FakeAdapter(cancel_midflight)
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="hello"))
    run_id = run.run_id

    finished = await controller.execute(run_id)

    assert finished.status is RunStatus.CANCELLED
    assert finished.error is None
    units = await store.list_work_units(run_id)
    assert units[0].status is WorkUnitStatus.SUCCEEDED
    attempts = await store.list_attempts(units[0].work_unit_id)
    assert attempts[0].status is AttemptStatus.SUCCEEDED
    assert len(await store.list_artifacts(units[0].work_unit_id)) == 1
    types = await event_types(store, run_id)
    assert types[-1] is EventType.RUN_CANCELLED
    assert EventType.RUN_CANCELLING in types
    assert EventType.RUN_SUCCEEDED not in types


@pytest.mark.asyncio
async def test_task_cancellation_records_an_unconfirmed_attempt(
    store: SqliteStore,
) -> None:
    release = asyncio.Event()

    async def never_answers(request: ProviderRequest) -> ProviderResponse:
        await release.wait()
        return _text_response("unreachable")

    adapter = FakeAdapter(never_answers)
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="hello"))

    task = asyncio.create_task(controller.execute(run.run_id))
    # Wait for the attempt to actually be dispatched: the store runs on a worker
    # thread, so yielding alone does not guarantee progress.
    deadline = asyncio.get_running_loop().time() + 5.0
    while not adapter.requests:
        assert asyncio.get_running_loop().time() < deadline, "the attempt never started"
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    attempts = await store.list_run_attempts(run.run_id)
    assert len(attempts) == 1
    assert attempts[0].status is AttemptStatus.UNKNOWN
    assert attempts[0].completed_at is not None
    # The run is left running on purpose: nothing proved what happened upstream.
    assert (await store.get_run(run.run_id)).status is RunStatus.RUNNING
    assert EventType.ATTEMPT_UNKNOWN in await event_types(store, run.run_id)


# --- strategy boundaries --------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_never_calls_the_planner_model(store: SqliteStore) -> None:
    worker = FakeAdapter()
    leader = FakeAdapter()
    controller = build_controller(store, worker, leader=leader)
    run = await controller.create_run(NativeRunRequest(input="hello"))
    await controller.execute(run.run_id)
    assert leader.call_count == 0
    assert worker.call_count == 1
    attempts = await store.list_run_attempts(run.run_id)
    assert [attempt.role for attempt in attempts] == [ModelRole.WORKER]
    types = await event_types(store, run.run_id)
    assert EventType.PLAN_PROPOSED not in types
    # EVIDENCE_RECORDED is now emitted by the verifier integration


@pytest.mark.asyncio
async def test_auto_routing_selects_direct_with_a_rationale(store: SqliteStore) -> None:
    controller = build_controller(store, FakeAdapter())
    run = await controller.create_run(NativeRunRequest(input="hello"))
    await controller.execute(run.run_id)
    ledger = await store.list_events(run.run_id)
    selected = next(
        event for event in ledger if event.event_type is EventType.STRATEGY_SELECTED
    )
    assert selected.payload["strategy"] == "DIRECT"
    assert selected.payload["routing_policy"] == "AUTO"
    assert isinstance(selected.payload["rationale"], str)


@pytest.mark.asyncio
async def test_manual_direct_is_honoured(store: SqliteStore) -> None:
    controller = build_controller(store, FakeAdapter())
    run = await controller.create_run(
        NativeRunRequest(
            input="hello",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.DIRECT,
        )
    )
    finished = await controller.execute(run.run_id)
    assert finished.status is RunStatus.SUCCEEDED
    assert finished.strategy is ExecutionStrategy.DIRECT


@pytest.mark.asyncio
async def test_executing_a_finished_run_changes_nothing(store: SqliteStore) -> None:
    adapter = FakeAdapter()
    controller = build_controller(store, adapter)
    run = await controller.create_run(NativeRunRequest(input="hello"))
    first = await controller.execute(run.run_id)
    ledger_length = len(await store.list_events(run.run_id))
    second = await controller.execute(run.run_id)
    assert second.status is first.status
    assert adapter.call_count == 1
    assert len(await store.list_events(run.run_id)) == ledger_length


@pytest.mark.asyncio
async def test_a_missing_adapter_is_reported_as_configuration(store: SqliteStore) -> None:
    settings = Settings(worker_profile=WORKER_PROFILE)
    controller = RunController(store, settings, {})
    run = await controller.create_run(NativeRunRequest(input="hello"))
    with pytest.raises(ProviderError) as excinfo:
        await controller.execute(run.run_id)
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


@pytest.mark.asyncio
async def test_worker_context_carries_no_graph_history(store: SqliteStore) -> None:
    adapter = FakeAdapter()
    controller = build_controller(store, adapter)
    run = await controller.create_run(
        NativeRunRequest(input="the only task", instructions="system rules")
    )
    await controller.execute(run.run_id)
    sent: Sequence[ProviderRequest] = adapter.requests
    assert sent[0].input == "the only task"
    assert run.run_id not in sent[0].input
    assert "RUN_CREATED" not in sent[0].input


# --- budget enforcement -------------------------------------------------------


@pytest.mark.asyncio
async def test_deadline_exceeded_before_dispatch_prevents_any_attempt(
    store: SqliteStore,
) -> None:
    past = datetime(2026, 1, 1, tzinfo=UTC)
    adapter = FakeAdapter()
    controller = build_controller(store, adapter)
    run = await controller.create_run(
        NativeRunRequest(input="hello", budget=Budget(deadline=past))
    )
    finished = await controller.execute(run.run_id)
    assert finished.status is RunStatus.CANCELLED
    assert finished.error is None
    assert adapter.call_count == 0
    types = await event_types(store, run.run_id)
    assert EventType.BUDGET_EXHAUSTED in types
    assert EventType.ATTEMPT_STARTED not in types


@pytest.mark.asyncio
async def test_token_budget_exceeded_after_attempt_fails_the_run(
    store: SqliteStore,
) -> None:
    adapter = FakeAdapter(
        _text_response("the answer", usage=Usage(input_tokens=10, output_tokens=5, elapsed_ms=1))
    )
    controller = build_controller(store, adapter)
    run = await controller.create_run(
        NativeRunRequest(input="hello", budget=Budget(max_total_tokens=5))
    )
    finished = await controller.execute(run.run_id)
    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert adapter.call_count == 1
    types = await event_types(store, run.run_id)
    assert EventType.BUDGET_EXHAUSTED in types
    assert EventType.ATTEMPT_SUCCEEDED in types


@pytest.mark.asyncio
async def test_exact_token_ceilings_accept_the_current_artifact(
    store: SqliteStore,
) -> None:
    adapter = FakeAdapter(
        _text_response(
            "the answer",
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
                strong_model_tokens=5,
                elapsed_ms=1,
            ),
        )
    )
    controller = build_controller(store, adapter)
    run = await controller.create_run(
        NativeRunRequest(
            input="hello",
            budget=Budget(max_total_tokens=15, max_strong_model_tokens=5),
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert adapter.call_count == 1
    assert len(await store.list_run_attempts(run.run_id)) == 1
    types = await event_types(store, run.run_id)
    assert EventType.EVIDENCE_RECORDED in types
    assert EventType.BUDGET_EXHAUSTED not in types


@pytest.mark.asyncio
async def test_zero_token_ceiling_blocks_initial_dispatch(store: SqliteStore) -> None:
    adapter = FakeAdapter()
    controller = build_controller(store, adapter)
    run = await controller.create_run(
        NativeRunRequest(input="hello", budget=Budget(max_total_tokens=0))
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.CANCELLED
    assert adapter.call_count == 0
    assert await store.list_run_attempts(run.run_id) == ()
    assert EventType.BUDGET_EXHAUSTED in await event_types(store, run.run_id)
