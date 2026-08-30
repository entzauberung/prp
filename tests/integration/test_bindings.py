"""ASGI integration for all inbound bindings with fake providers."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from prp_runtime.app import create_app
from prp_runtime.client import cli
from prp_runtime.client.serve import build_serve_app
from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    IsolationMode,
    ModelRole,
    ToolCallStatus,
    ToolEffect,
)
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.models import (
    AgentRequestOptions,
    AgentToolCall,
    ErrorCategory,
    ExecutionScope,
    Usage,
    WorkspaceGrant,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.domain.values import (
    new_principal_id,
    new_run_id,
    new_session_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
    new_workspace_id,
)
from prp_runtime.runtime.agent_executor import AgentToolExecutor
from prp_runtime.runtime.agent_loop import AgentToolContext
from prp_runtime.runtime.tool_worker import ToolWorker
from prp_runtime.settings import Settings
from prp_runtime.tools.executor import ToolExecutor, uses_in_process_tool_settlement
from prp_runtime.tools.models import ToolCall, ToolResult
from prp_runtime.tools.registry import ToolDefinition, ToolRegistry
from prp_runtime.workspace.sandbox import SandboxCapabilities

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="fake",
    model="fake-worker",
    role=ModelRole.WORKER,
    base_url="https://models.invalid/v1",
    context_window_tokens=16_000,
    max_output_tokens=2_000,
)


class FakeAdapter:
    def __init__(self, text: str = "bound answer") -> None:
        self.text = text
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "binding-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=self.text,
            usage=Usage(input_tokens=3, output_tokens=2, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


def _app(tmp_path: Path, adapter: FakeAdapter) -> FastAPI:
    app = create_app(
        Settings(database_path=tmp_path / "bindings.db", worker_profile=WORKER_PROFILE),
        adapters={"worker": adapter},  # type: ignore[arg-type]
    )
    return app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(_app(tmp_path, FakeAdapter())) as opened:
        yield opened


def wait_for_response(client: TestClient, run_id: str) -> dict[str, object]:
    for _ in range(200):
        body = client.get(f"/v1/responses/{run_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
    raise AssertionError("Responses run did not reach a terminal state")


def test_responses_create_query_and_cancel_share_one_envelope(
    client: TestClient,
) -> None:
    created = client.post(
        "/v1/responses",
        json={"input": "hello", "instructions": "be terse"},
    )
    assert created.status_code == 202
    created_body = created.json()
    assert created_body["status"] in {"pending", "in_progress", "completed"}
    body = wait_for_response(client, created_body["id"])
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output_text"] == "bound answer"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert "strategy" not in body
    assert "graph_version" not in body

    run_id = body["id"]
    assert client.get(f"/v1/responses/{run_id}").json() == body
    events = client.get(f"/v1/responses/{run_id}/events")
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "response.run_created" in events.text
    cancelled = client.post(f"/v1/responses/{run_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "completed"


def test_chat_create_maps_system_and_user_text(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "bound answer",
    }
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_anthropic_create_query_and_cancel(client: TestClient) -> None:
    created = client.post(
        "/v1/messages",
        json={
            "system": "be concise",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["status"] == "completed"
    assert body["content"] == [{"type": "text", "text": "bound answer"}]
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert "strategy" not in body
    run_id = body["id"]
    assert client.get(f"/v1/messages/{run_id}").json() == body
    assert client.post(f"/v1/messages/{run_id}/cancel").json()["status"] == "completed"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/responses",
            {"input": "hello", "routing": {"requires_cascade": True}},
        ),
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "routing": {"requires_cascade": True},
            },
        ),
        (
            "/v1/messages",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "routing": {"requires_cascade": True},
            },
        ),
    ],
)
def test_external_routing_intent_reaches_controller(
    client: TestClient,
    path: str,
    payload: dict[str, object],
) -> None:
    created = client.post(path, json=payload)

    if path == "/v1/responses":
        assert created.status_code == 202
        body = wait_for_response(client, created.json()["id"])
    else:
        assert created.status_code == 200
        body = created.json()
    native = client.get(f"/v1/runs/{body['id']}")
    assert native.status_code == 200
    assert native.json()["strategy"] == "CASCADE"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/responses",
            {"input": "hello", "routing": {"desired_parallelism": 0}},
        ),
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "routing": {"desired_parallelism": 0},
            },
        ),
        (
            "/v1/messages",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "routing": {"desired_parallelism": 0},
            },
        ),
    ],
)
def test_external_invalid_routing_value_is_structured(
    client: TestClient,
    path: str,
    payload: dict[str, object],
) -> None:
    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value
    assert response.json()["error"]["field"] == "routing"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            ErrorCode.UNSUPPORTED_STREAM_MODE,
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": "private"}],
                    }
                ]
            },
            ErrorCode.UNSUPPORTED_MODALITY,
        ),
    ],
)
def test_anthropic_errors_are_structured(
    client: TestClient,
    payload: dict[str, object],
    code: ErrorCode,
) -> None:
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code.value
    assert "private" not in response.text


@pytest.mark.parametrize(
    ("path", "payload", "code"),
    [
        (
            "/v1/responses",
            {"input": "hello", "stream": True},
            ErrorCode.UNSUPPORTED_STREAM_MODE,
        ),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}], "tools": []},
            ErrorCode.UNSUPPORTED_TOOLS,
        ),
        (
            "/v1/responses",
            {"input": "hello", "base_url": "https://private.invalid"},
            ErrorCode.UNSUPPORTED_FIELD,
        ),
    ],
)
def test_binding_errors_are_structured_and_redacted(
    client: TestClient,
    path: str,
    payload: dict[str, object],
    code: ErrorCode,
) -> None:
    response = client.post(path, json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code.value
    assert "private.invalid" not in response.text


class _ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str


class _ClaimRecordingStore:
    def __init__(self) -> None:
        self.calls: dict[str, ToolCall] = {}
        self.results: dict[str, ToolResult] = {}
        self.idempotency: dict[str, str] = {}
        self.bridge_calls: list[str] = []

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
        return updated

    async def complete_tool_call(self, result: ToolResult) -> ToolResult:
        existing = self.results.get(result.call_id)
        if existing is not None:
            return existing
        current = self.calls[result.call_id]
        self.calls[result.call_id] = current.transition(result.status)
        self.results[result.call_id] = result
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

    async def create_bridge_claim(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.bridge_calls.append("create_bridge_claim")
        raise AssertionError("LOCAL must not create a Bridge claim")

    async def claim_bridge_call(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.bridge_calls.append("claim_bridge_call")
        raise AssertionError("LOCAL must not claim a Bridge call")

    async def settle_bridge_claim(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.bridge_calls.append("settle_bridge_claim")
        raise AssertionError("LOCAL must not settle a Bridge claim")

    async def submit_bridge_tool_result(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.bridge_calls.append("submit_bridge_tool_result")
        raise AssertionError("LOCAL must not submit a Bridge tool result")


def _local_registry(handler: Any) -> ToolRegistry:
    return ToolRegistry(
        (
            ToolDefinition(
                name="read_file",
                effect=ToolEffect.READ,
                argument_model=_ReadArgs,
                handler=handler,
            ),
        )
    )


@pytest.mark.asyncio
async def test_local_tool_execution_never_uses_bridge_claims() -> None:
    assert uses_in_process_tool_settlement(ExecutionLocation.LOCAL) is True
    assert uses_in_process_tool_settlement(ExecutionLocation.BRIDGE) is False

    handler_calls: list[str] = []

    async def read_file(context: BaseModel) -> Mapping[str, object]:
        del context
        handler_calls.append("read_file")
        return {"content": "local-ok"}

    store = _ClaimRecordingStore()
    executor = ToolExecutor(_local_registry(read_file), store)
    call = ToolCall(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        tool_name="read_file",
        effect=ToolEffect.READ,
        arguments={"path": "src/main.py"},
        snapshot_id=new_snapshot_id(),
        requested_at="2026-08-14T12:00:00+00:00",
    )
    outcome = await executor.execute(
        call,
        AgentMode.NORMAL,
        workspace_id=new_workspace_id(),
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.LOCAL,
    )
    assert outcome.call.status is ToolCallStatus.SUCCEEDED
    assert outcome.result is not None
    assert outcome.result.output
    assert "local-ok" in outcome.result.output
    assert handler_calls == ["read_file"]
    assert store.bridge_calls == []

    principal_id = new_principal_id()
    workspace_id = new_workspace_id()
    scope = ExecutionScope(
        run_id=new_run_id(),
        session_id=new_session_id(),
        principal_id=principal_id,
        workspace_id=workspace_id,
        grant=WorkspaceGrant(principal_id=principal_id, workspace_id=workspace_id),
        agent_options=AgentRequestOptions(
            isolation_mode=IsolationMode.HOST,
            execution_location=ExecutionLocation.LOCAL,
        ),
    )
    adapter = AgentToolExecutor(
        ToolWorker(executor),
        _local_registry(read_file),
        scope,
        snapshot_id=new_snapshot_id(),
    )
    execution = await adapter.execute(
        AgentToolCall(
            call_id="provider-call/local",
            tool_name="read_file",
            arguments={"path": "src/main.py"},
        ),
        context=AgentToolContext(
            run_id=scope.run_id,
            work_unit_id=new_work_unit_id(),
            mode=AgentMode.NORMAL,
            round_index=1,
            attempt_index=1,
        ),
    )
    assert execution.result is not None
    assert execution.result.status is ToolCallStatus.SUCCEEDED
    assert handler_calls == ["read_file", "read_file"]
    assert store.bridge_calls == []



def test_serve_handoff_reuses_create_app_without_a_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factories: list[object] = []
    real_create_app = create_app

    def wrapped(*args: object, **kwargs: object):
        factories.append((args, kwargs))
        return real_create_app(*args, **kwargs)

    monkeypatch.setattr("prp_runtime.client.serve.create_app", wrapped)
    adapter = FakeAdapter()
    settings = Settings(
        database_path=tmp_path / "serve-handoff.db",
        worker_profile=WORKER_PROFILE,
    )
    app = build_serve_app(settings, adapters={"worker": adapter})
    assert len(factories) == 1
    parsed = cli.build_parser().parse_args(["serve"])
    assert parsed.host == "127.0.0.1"
    assert parsed.port == 8000
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        created = client.post("/v1/responses", json={"input": "hello"})
        assert created.status_code == 202
        body = wait_for_response(client, created.json()["id"])
        assert body["status"] == "completed"
        assert body["output_text"] == "bound answer"


LEADER_PROFILE = ModelProfile(
    alias="leader",
    provider="fake",
    model="fake-leader",
    role=ModelRole.PLANNER,
    base_url="https://models.invalid/v1",
    context_window_tokens=16_000,
    max_output_tokens=2_000,
)

UNAVAILABLE_SANDBOX = SandboxCapabilities(
    backend="unavailable",
    available=False,
    reason="injected sandbox is unavailable",
)


def _ready_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "ready-mode.db",
        leader_profile=LEADER_PROFILE,
        worker_profile=WORKER_PROFILE,
    )


def test_host_local_ready_does_not_require_bwrap(tmp_path: Path) -> None:
    app = create_app(
        _ready_settings(tmp_path),
        adapters={"leader": FakeAdapter(), "worker": FakeAdapter()},
        execution_location=ExecutionLocation.LOCAL,
        isolation_mode=IsolationMode.HOST,
        sandbox_capabilities=UNAVAILABLE_SANDBOX,
    )
    with TestClient(app) as client:
        response = client.get("/ready")
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ready"
    assert payload["sandbox_ready"] is False
    assert payload["sandbox_required"] is False
    assert payload["execution_location"] == ExecutionLocation.LOCAL.value
    assert payload["isolation_mode"] == IsolationMode.HOST.value
    assert payload["path_boundary_ready"] is True
    dumped = response.text
    assert "secret" not in dumped
    assert str(tmp_path) not in dumped


def test_sandboxed_ready_still_requires_bwrap(tmp_path: Path) -> None:
    app = create_app(
        _ready_settings(tmp_path / "sandbox"),
        adapters={"leader": FakeAdapter(), "worker": FakeAdapter()},
        execution_location=ExecutionLocation.CLOUD,
        isolation_mode=IsolationMode.SANDBOXED,
        sandbox_capabilities=UNAVAILABLE_SANDBOX,
    )
    with TestClient(app) as client:
        response = client.get("/ready")
    payload = response.json()
    assert response.status_code == 503
    assert payload["status"] == "not_ready"
    assert payload["sandbox_ready"] is False
    assert payload["sandbox_required"] is True
    assert payload["isolation_mode"] == IsolationMode.SANDBOXED.value


def test_readiness_matrix_covers_host_sandbox_and_missing_components(
    tmp_path: Path,
) -> None:
    unavailable = UNAVAILABLE_SANDBOX
    adapters = {"leader": FakeAdapter(), "worker": FakeAdapter()}
    settings = _ready_settings(tmp_path)

    host_cloud = create_app(
        settings,
        adapters=adapters,
        execution_location=ExecutionLocation.CLOUD,
        isolation_mode=IsolationMode.HOST,
        sandbox_capabilities=unavailable,
    )
    with TestClient(host_cloud) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["sandbox_required"] is False
    assert str(tmp_path) not in response.text

    local_sandboxed = create_app(
        Settings(
            database_path=tmp_path / "local-sandboxed.db",
            leader_profile=LEADER_PROFILE,
            worker_profile=WORKER_PROFILE,
        ),
        adapters=adapters,
        execution_location=ExecutionLocation.LOCAL,
        isolation_mode=IsolationMode.SANDBOXED,
        sandbox_capabilities=unavailable,
    )
    with TestClient(local_sandboxed) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["sandbox_required"] is True
    assert response.json()["isolation_mode"] == IsolationMode.SANDBOXED.value

    missing_profile = create_app(
        Settings(database_path=tmp_path / "missing-profile.db"),
        adapters={},
        execution_location=ExecutionLocation.LOCAL,
        isolation_mode=IsolationMode.HOST,
        sandbox_capabilities=unavailable,
    )
    with TestClient(missing_profile) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["profiles_configured"] is False

    missing_adapter = create_app(
        Settings(
            database_path=tmp_path / "missing-adapter.db",
            leader_profile=LEADER_PROFILE,
            worker_profile=WORKER_PROFILE,
        ),
        adapters={"worker": FakeAdapter()},
        execution_location=ExecutionLocation.LOCAL,
        isolation_mode=IsolationMode.HOST,
        sandbox_capabilities=unavailable,
    )
    with TestClient(missing_adapter) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["adapters_ready"] is False
    assert "token" not in response.text.lower()
    assert "api_key" not in response.text.lower()
