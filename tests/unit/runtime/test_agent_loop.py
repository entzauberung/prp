"""Contract tests for provider-neutral text/tool turns and bounded history."""

import json

import httpx
import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    IsolationMode,
    ModelRole,
    ToolCallStatus,
)
from prp_runtime.domain.models import (
    AgentToolCall,
    AgentToolResult,
    AgentTurn,
    ErrorCategory,
    ProviderToolDescriptor,
    Usage,
)
from prp_runtime.domain.values import new_principal_id, new_workspace_id
from prp_runtime.tools.executor import ToolExecutor
from prp_runtime.policy.models import DevExecutionMode, guard_dev_scope
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.providers.openai_compatible import OpenAICompatibleProvider
from prp_runtime.runtime.agent_loop import (
    REMOTE_ASSIGNMENT_PENDING,
    AgentLoop,
    AgentLoopStatus,
    AgentToolContext,
    AgentToolExecution,
)

PROFILE = ModelProfile(
    alias="worker",
    provider="fake",
    model="bounded-model",
    role=ModelRole.WORKER,
    base_url="https://models.internal/v1",
    context_window_tokens=32_000,
    max_output_tokens=4_000,
)


class FakeAdapter:
    def __init__(self, *responses: ProviderResponse) -> None:
        self.responses = list(responses)
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class FakeToolExecutor:
    def __init__(
        self, *, pause: bool = False, reject: bool = False, remote: bool = False
    ) -> None:
        self.pause = pause
        self.reject = reject
        self.remote = remote
        self.calls: list[tuple[AgentToolCall, AgentToolContext]] = []
        self.results: dict[str, AgentToolResult] = {}

    async def execute(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        self.calls.append((call, context))
        stored = self.results.get(call.call_id)
        if stored is not None:
            return AgentToolExecution(call=call, result=stored)
        if self.pause:
            return AgentToolExecution(
                call=call,
                awaiting_approval=True,
                reason="approval required",
            )
        if self.remote:
            return AgentToolExecution(
                call=call,
                awaiting_remote_result=True,
                reason=REMOTE_ASSIGNMENT_PENDING,
            )
        status = ToolCallStatus.REJECTED if self.reject else ToolCallStatus.SUCCEEDED
        return AgentToolExecution(
            call=call,
            result=AgentToolResult(
                call_id=call.call_id,
                status=status,
                result={"reason": "policy denied"} if self.reject else {"content": "file contents"},
                output="policy denied" if self.reject else "file contents",
            ),
        )


def tool_response(call: AgentToolCall) -> ProviderResponse:
    return ProviderResponse(
        tool_calls=(call,),
        usage=Usage(input_tokens=3, output_tokens=2, elapsed_ms=1),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def text_response(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        usage=Usage(input_tokens=4, output_tokens=3, elapsed_ms=1),
        finish_reason=FinishReason.STOP,
    )


def test_agent_turn_is_text_or_unique_tool_calls() -> None:
    text = AgentTurn(text="final answer")
    tool = AgentToolCall(call_id="call-1", tool_name="read_file", arguments={"path": "a.py"})
    calls = AgentTurn(tool_calls=(tool,))

    assert text.text == "final answer"
    assert calls.tool_calls == (tool,)
    with pytest.raises(ValidationError, match="exclusively"):
        AgentTurn(text="answer", tool_calls=(tool,))
    with pytest.raises(ValidationError, match="exclusively"):
        AgentTurn()
    with pytest.raises(ValidationError, match="unique"):
        AgentTurn(tool_calls=(tool, tool))


def test_provider_response_preserves_text_compatibility_and_maps_tool_turn() -> None:
    text_response = ProviderResponse(text="done", finish_reason=FinishReason.STOP)
    tool_call = AgentToolCall(call_id="call-1", tool_name="read_file")
    tool_response = ProviderResponse(
        tool_calls=(tool_call,),
        finish_reason=FinishReason.TOOL_CALLS,
    )

    assert text_response.turn == AgentTurn(text="done")
    assert tool_response.turn == AgentTurn(tool_calls=(tool_call,))
    with pytest.raises(ValidationError, match="exclusively"):
        ProviderResponse(text="done", tool_calls=(tool_call,), finish_reason=FinishReason.OTHER)
    with pytest.raises(ValidationError, match="text or tool_calls"):
        ProviderResponse(finish_reason=FinishReason.OTHER)


def test_history_accepts_only_structured_bounded_tool_results() -> None:
    tool_call = AgentToolCall(call_id="call-1", tool_name="read_file")
    result = AgentToolResult(
        call_id="call-1",
        status=ToolCallStatus.SUCCEEDED,
        result={"content": "safe"},
    )
    request = ProviderRequest(
        alias="worker",
        model="model",
        input="continue",
        max_output_tokens=10,
        timeout_seconds=5,
        history=(AgentTurn(tool_calls=(tool_call,)), result),
    )

    assert request.history[1] == result
    with pytest.raises(ValidationError, match="terminal"):
        AgentToolResult(call_id="call-1", status=ToolCallStatus.RUNNING)
    with pytest.raises(ValidationError):
        ProviderRequest(
            alias="worker",
            model="model",
            input="continue",
            max_output_tokens=10,
            timeout_seconds=5,
            history=({"kind": "tool_result", "call_id": "call-1", "status": "RUNNING"},),
        )


@pytest.mark.asyncio
async def test_agent_loop_executes_tools_then_returns_final_text() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="read_file", arguments={"path": "a.py"})
    adapter = FakeAdapter(tool_response(call), text_response("final answer"))
    executor = FakeToolExecutor()

    result = await AgentLoop(adapter, PROFILE, tool_executor=executor).execute(
        input="inspect the file",
        run_id="run-1",
        work_unit_id="wu-1",
        mode=AgentMode.AUTO,
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.text == "final answer"
    assert result.attempts == 2
    assert result.tool_rounds == 1
    assert result.usage == Usage(input_tokens=7, output_tokens=5, elapsed_ms=2)
    assert len(executor.calls) == 1
    assert executor.calls[0][1].mode is AgentMode.AUTO
    assert adapter.requests[1].history == (
        AgentTurn(tool_calls=(call,)),
        AgentToolResult(
            call_id="call-1",
            status=ToolCallStatus.SUCCEEDED,
            result={"content": "file contents"},
            output="file contents",
        ),
    )


@pytest.mark.asyncio
async def test_agent_loop_sends_only_the_public_tool_catalog() -> None:
    adapter = FakeAdapter(text_response("final answer"))
    catalog = (ProviderToolDescriptor(name="read_file", description="Read a file."),)

    result = await AgentLoop(adapter, PROFILE, tool_catalog=catalog).execute(input="inspect")

    assert result.status is AgentLoopStatus.COMPLETED
    assert adapter.requests[0].tools == catalog


@pytest.mark.asyncio
async def test_agent_loop_rejects_orphaned_history_before_provider_dispatch() -> None:
    orphan = AgentToolResult(
        call_id="call-orphan",
        status=ToolCallStatus.SUCCEEDED,
        result={"ok": True},
    )
    adapter = FakeAdapter(text_response("must not run"))

    result = await AgentLoop(adapter, PROFILE).execute(input="resume", history=(orphan,))

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.error is not None
    assert result.error.category is ErrorCategory.INVALID_REQUEST
    assert adapter.requests == []


@pytest.mark.asyncio
async def test_agent_loop_rejects_blank_input_without_provider_dispatch() -> None:
    adapter = FakeAdapter(text_response("must not run"))

    result = await AgentLoop(adapter, PROFILE).execute(input=" \n")

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.error is not None
    assert result.error.category is ErrorCategory.INVALID_REQUEST
    assert adapter.requests == []


@pytest.mark.asyncio
async def test_agent_loop_result_serializes_as_dev_only_from_one_scope() -> None:
    adapter = FakeAdapter(text_response("final answer"))
    result = await AgentLoop(adapter, PROFILE).execute(input="text only")
    scope = guard_dev_scope(
        principal_id=new_principal_id(),
        workspace_id=new_workspace_id(),
        mode=DevExecutionMode.TEXT_ONLY,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.CLOUD,
        text_only=True,
    )

    payload = result.serialize_dev_evidence(scope=scope)

    assert payload["dev_only"] is True
    assert payload["metadata"]["dev_only"] is True  # type: ignore[index]
    assert payload["evidence"]["text"] == "final answer"  # type: ignore[index]


@pytest.mark.asyncio
async def test_agent_loop_history_writer_receives_turns_and_results_in_order() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="read_file")
    adapter = FakeAdapter(tool_response(call), text_response("final answer"))
    entries: list[tuple[int, object]] = []

    async def write_history(sequence: int, item: object) -> None:
        entries.append((sequence, item))

    result = await AgentLoop(
        adapter,
        PROFILE,
        tool_executor=FakeToolExecutor(),
    ).execute(
        input="inspect the file",
        run_id="run-1",
        work_unit_id="wu-1",
        history_writer=write_history,
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert [sequence for sequence, _ in entries] == [1, 2, 3]
    assert entries[0][1] == AgentTurn(tool_calls=(call,))
    assert entries[1][1] == AgentToolResult(
        call_id="call-1",
        status=ToolCallStatus.SUCCEEDED,
        result={"content": "file contents"},
        output="file contents",
    )
    assert entries[2][1] == AgentTurn(text="final answer")


@pytest.mark.asyncio
async def test_real_openai_adapter_completes_multiple_tool_rounds() -> None:
    calls = [
        ("call-one", "read_file", '{"path":"README.md"}'),
        ("call-two", "search_text", '{"pattern":"Provider"}'),
    ]
    responses = [
        {
            "id": "provider-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": calls[0][0],
                                "type": "function",
                                "function": {
                                    "name": calls[0][1],
                                    "arguments": calls[0][2],
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        {
            "id": "provider-2",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": calls[1][0],
                                "type": "function",
                                "function": {
                                    "name": calls[1][1],
                                    "arguments": calls[1][2],
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4},
        },
        {
            "id": "provider-3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "final answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 6},
        },
    ]
    sent: list[dict[str, object]] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0), request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        trust_env=False,
    ) as client:
        adapter = OpenAICompatibleProvider(PROFILE, client=client)
        executor = FakeToolExecutor()
        result = await AgentLoop(adapter, PROFILE, tool_executor=executor).execute(
            input="inspect the workspace",
            mode=AgentMode.AUTO,
        )

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.text == "final answer"
    assert result.attempts == 3
    assert result.tool_rounds == 2
    assert result.provider_request_id == "provider-3"
    assert result.usage is not None
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 12
    assert result.usage.elapsed_ms >= 0
    assert [call.call_id for call, _ in executor.calls] == ["call-one", "call-two"]
    assert len(sent) == 3
    assert sent[1]["messages"] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-one",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-one",
            "content": "file contents",
        },
        {"role": "user", "content": "inspect the workspace"},
    ]
    assert sent[2]["messages"][-3:] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-two",
                    "type": "function",
                    "function": {
                        "name": "search_text",
                        "arguments": '{"pattern":"Provider"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-two",
            "content": "file contents",
        },
        {"role": "user", "content": "inspect the workspace"},
    ]


@pytest.mark.asyncio
async def test_duplicate_tool_call_id_replays_result_without_reexecution() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="read_file")
    adapter = FakeAdapter(
        tool_response(call),
        tool_response(call),
        text_response("done"),
    )
    executor = FakeToolExecutor()

    result = await AgentLoop(adapter, PROFILE, tool_executor=executor).execute(
        input="inspect",
        run_id="run-1",
        work_unit_id="wu-1",
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert len(executor.calls) == 1
    assert len(result.history) == 5


@pytest.mark.asyncio
async def test_agent_loop_pauses_for_approval_without_failure() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="apply_patch")
    adapter = FakeAdapter(tool_response(call))
    executor = FakeToolExecutor(pause=True)

    result = await AgentLoop(adapter, PROFILE, tool_executor=executor).execute(
        input="make the change",
        run_id="run-1",
        work_unit_id="wu-1",
    )

    assert result.status is AgentLoopStatus.PAUSED
    assert result.error is None
    assert result.pending_call_ids == ("call-1",)


@pytest.mark.asyncio
async def test_agent_loop_can_resume_from_public_pause_history() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="apply_patch")
    adapter = FakeAdapter(tool_response(call))
    executor = FakeToolExecutor(pause=True)
    loop = AgentLoop(adapter, PROFILE, tool_executor=executor)

    paused = await loop.execute(input="make the change")
    adapter.responses.append(text_response("approved result"))
    resumed = await loop.execute(input="make the change", history=paused.history)

    assert paused.status is AgentLoopStatus.PAUSED
    assert resumed.status is AgentLoopStatus.COMPLETED
    assert resumed.text == "approved result"
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_unknown_tool_result_is_not_replayed_or_followed_by_provider_dispatch() -> None:
    call = AgentToolCall(call_id="call-unknown", tool_name="apply_patch")
    unknown = AgentToolResult(
        call_id=call.call_id,
        status=ToolCallStatus.UNKNOWN,
        result={"error": "unconfirmed"},
        output="outcome is unconfirmed",
    )
    adapter = FakeAdapter(text_response("must not run"))
    executor = FakeToolExecutor()

    result = await AgentLoop(adapter, PROFILE, tool_executor=executor).execute(
        input="resume",
        history=(AgentTurn(tool_calls=(call,)), unknown),
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.error is not None
    assert result.error.category is ErrorCategory.UNKNOWN
    assert adapter.requests == []
    assert executor.calls == []


@pytest.mark.asyncio
async def test_replay_result_must_match_a_known_tool_call() -> None:
    unknown_call = AgentToolCall(call_id="call-unknown", tool_name="read_file")
    result = await AgentLoop(FakeAdapter(text_response("must not run")), PROFILE).execute(
        input="resume",
        replay_results={
            unknown_call.call_id: AgentToolResult(
                call_id=unknown_call.call_id,
                status=ToolCallStatus.REJECTED,
                result={"error": "denied"},
            )
        },
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.error is not None
    assert result.error.category is ErrorCategory.INVALID_REQUEST


@pytest.mark.asyncio
async def test_agent_loop_resumes_pending_call_before_provider_dispatch() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="apply_patch")
    adapter = FakeAdapter(text_response("approved result"))
    executor = FakeToolExecutor()
    history = (AgentTurn(tool_calls=(call,)),)

    resumed = await AgentLoop(adapter, PROFILE, tool_executor=executor).execute(
        input="make the change",
        history=history,
        resume_pending_call=call,
    )

    assert resumed.status is AgentLoopStatus.COMPLETED
    assert resumed.text == "approved result"
    assert [called.call_id for called, _ in executor.calls] == ["call-1"]
    assert adapter.requests[0].history == (
        AgentTurn(tool_calls=(call,)),
        AgentToolResult(
            call_id="call-1",
            status=ToolCallStatus.SUCCEEDED,
            result={"content": "file contents"},
            output="file contents",
        ),
    )


@pytest.mark.asyncio
async def test_agent_loop_passes_policy_deny_result_back_to_provider() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="apply_patch")
    adapter = FakeAdapter(tool_response(call), text_response("I cannot apply it"))
    executor = FakeToolExecutor(reject=True)

    result = await AgentLoop(adapter, PROFILE, tool_executor=executor).execute(
        input="make the change"
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert adapter.requests[1].history[-1].status is ToolCallStatus.REJECTED


@pytest.mark.asyncio
async def test_agent_loop_honours_deadline_before_provider_dispatch() -> None:
    from datetime import UTC, datetime

    adapter = FakeAdapter(text_response("must not run"))
    result = await AgentLoop(adapter, PROFILE).execute(
        input="expired",
        deadline=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.error is not None
    assert result.error.category.name == "DEADLINE_EXCEEDED"
    assert adapter.requests == []


@pytest.mark.asyncio
async def test_agent_loop_propagates_provider_cancellation() -> None:
    import asyncio

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingAdapter(FakeAdapter):
        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            self.requests.append(request)
            started.set()
            await release.wait()
            return text_response("unreachable")

    adapter = BlockingAdapter()
    task = asyncio.create_task(AgentLoop(adapter, PROFILE).execute(input="cancel"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_agent_loop_exhaustion_is_structured_and_bounded() -> None:
    first = AgentToolCall(call_id="call-1", tool_name="read_file")
    second = AgentToolCall(call_id="call-2", tool_name="read_file")
    adapter = FakeAdapter(tool_response(first), tool_response(second))
    executor = FakeToolExecutor()

    result = await AgentLoop(
        adapter,
        PROFILE,
        tool_executor=executor,
        max_tool_rounds=1,
    ).execute(input="inspect")

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.error is not None
    assert result.error.category.name == "BUDGET_EXCEEDED"
    assert len(adapter.requests) == 2
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_agent_loop_waits_for_remote_result_without_approval_or_provider() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="read_file")
    adapter = FakeAdapter(tool_response(call), text_response("must not run"))
    executor = FakeToolExecutor(remote=True)
    loop = AgentLoop(adapter, PROFILE, tool_executor=executor)

    paused = await loop.execute(input="inspect", run_id="run-1", work_unit_id="wu-1")
    still_waiting = await loop.execute(
        input="inspect",
        history=paused.history,
        resume_pending_call=call,
        run_id="run-1",
        work_unit_id="wu-1",
    )

    assert paused.status is AgentLoopStatus.WAITING_REMOTE
    assert paused.status is not AgentLoopStatus.PAUSED
    assert paused.error is None
    assert paused.pending_call_ids == ("call-1",)
    assert still_waiting.status is AgentLoopStatus.WAITING_REMOTE
    assert still_waiting.history == paused.history
    assert len(adapter.requests) == 1
    assert len(executor.calls) == 2


@pytest.mark.asyncio
async def test_remote_result_replays_once_without_duplicate_history_or_usage() -> None:
    call = AgentToolCall(call_id="call-1", tool_name="read_file")
    durable = AgentToolResult(
        call_id=call.call_id,
        status=ToolCallStatus.SUCCEEDED,
        result={"content": "file contents"},
        output="file contents",
    )
    adapter = FakeAdapter(text_response("done"), text_response("must not duplicate"))
    executor = FakeToolExecutor(remote=True)
    written: list[object] = []

    async def history_writer(sequence: int, item: object) -> None:
        written.append((sequence, item))

    loop = AgentLoop(adapter, PROFILE, tool_executor=executor)
    first = await loop.execute(
        input="inspect",
        history=(AgentTurn(tool_calls=(call,)),),
        resume_pending_call=call,
        replay_results={call.call_id: durable},
        history_writer=history_writer,
        run_id="run-1",
        work_unit_id="wu-1",
    )
    second = await loop.execute(
        input="inspect",
        history=(AgentTurn(tool_calls=(call,)), durable),
        resume_pending_call=call,
        replay_results={call.call_id: durable},
        history_writer=history_writer,
        run_id="run-1",
        work_unit_id="wu-1",
    )

    assert first.status is AgentLoopStatus.COMPLETED
    assert second.status is AgentLoopStatus.COMPLETED
    assert [item for item in first.history if isinstance(item, AgentToolResult)] == [durable]
    assert [item for item in second.history if isinstance(item, AgentToolResult)] == [durable]
    assert sum(isinstance(item, AgentToolResult) for _, item in written) == 1
    assert executor.calls == []
    assert first.usage is not None
    assert second.usage == first.usage


@pytest.mark.asyncio
async def test_agent_loop_injects_bounded_facts_and_redacts_private_context() -> None:
    adapter = FakeAdapter(text_response("done"))
    loop = AgentLoop(adapter, PROFILE)
    result = await loop.execute(
        input="inspect /tmp/project/src/main.py with api_key=secret-value",
        static_facts=(
            {
                "key": "function:run",
                "kind": "ast",
                "summary": "FUNCTION run L1",
                "work_unit_id": "wu-1",
            },
            {
                "key": "function:other",
                "kind": "ast",
                "summary": "FUNCTION other L1",
                "work_unit_id": "wu-2",
            },
            {
                "key": "function:run",
                "kind": "ast",
                "summary": "FUNCTION run duplicate",
                "work_unit_id": "wu-1",
            },
        ),
        run_id="run-1",
        work_unit_id="wu-1",
    )

    assert result.status is AgentLoopStatus.COMPLETED
    sent = adapter.requests[0].input
    assert "FUNCTION run L1" in sent
    assert "FUNCTION other" not in sent
    assert sent.count("function:run") == 1
    assert "/tmp/project" not in sent
    assert "secret-value" not in sent
    assert "[redacted-root]" in sent
    assert "api_key=[redacted]" in sent


@pytest.mark.asyncio
async def test_agent_loop_does_not_duplicate_already_rendered_facts() -> None:
    adapter = FakeAdapter(text_response("done"))
    rendered = "### Static facts\n- [ast] function:run: FUNCTION run L1"
    result = await AgentLoop(adapter, PROFILE).execute(
        input=f"continue\n\n{rendered}",
        static_facts=(
            {
                "key": "function:run",
                "kind": "ast",
                "summary": "FUNCTION run L1",
                "work_unit_id": "wu-1",
            },
        ),
        work_unit_id="wu-1",
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert adapter.requests[0].input.count("function:run") == 1
