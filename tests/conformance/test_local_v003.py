"""Focused local v0.0.3 conformance: in-process DIRECT, no HTTP, no Docker."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import prp_runtime
from prp_runtime.app import create_app
from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    ExecutionStrategy,
    IsolationMode,
    ModelRole,
    RunStatus,
    ToolCallStatus,
)
from prp_runtime.domain.errors import BudgetError, DomainValidationError, ErrorCode
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import AgentRequestOptions, AgentToolCall, Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.runtime.local import LocalRuntime
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import MissingEntityError
from prp_runtime.workspace.isolation import (
    ExecutionCopyMode,
    IsolationCapacityError,
    LocalIsolationBackend,
    select_execution_copy_mode,
)
from prp_runtime.workspace.sandbox import SandboxCapabilities

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="openai_compatible",
    model="weak-model",
    role=ModelRole.WORKER,
    base_url="https://models.invalid/v1",
    context_window_tokens=1_024,
    max_output_tokens=128,
)
LEADER_PROFILE = ModelProfile(
    alias="leader",
    provider="openai_compatible",
    model="planner-model",
    role=ModelRole.PLANNER,
    base_url="https://models.invalid/v1",
    context_window_tokens=1_024,
    max_output_tokens=128,
)

ORIGINAL_MAIN = 'def answer():\n    return "needle"\n'
PATCH_DIFF = (
    "--- a/src/main.py\n"
    "+++ b/src/main.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def answer():\n"
    '-    return "needle"\n'
    '+    return "patched"\n'
)
BRIDGE_CLAIM_EVENTS = {
    EventType.BRIDGE_CLAIM_CREATED,
    EventType.BRIDGE_CLAIM_EXPIRED,
    EventType.BRIDGE_CLAIM_SETTLED,
    EventType.BRIDGE_CLAIM_RELEASED,
}
UNAVAILABLE_SANDBOX = SandboxCapabilities(
    backend="unavailable",
    available=False,
    reason="injected sandbox is unavailable",
)


class FakeAdapter:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "local-v003-fake"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text="the local answer",
            usage=Usage(input_tokens=4, output_tokens=3, elapsed_ms=2),
            finish_reason=FinishReason.STOP,
        )


class PatchAskAdapter:
    def __init__(self, runtime_holder: list[LocalRuntime]) -> None:
        self.requests: list[ProviderRequest] = []
        self._runtime_holder = runtime_holder

    @property
    def name(self) -> str:
        return "local-v003-ask"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            snapshot_id = self._runtime_holder[0].last_snapshot_id
            assert snapshot_id is not None
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-patch",
                        tool_name="apply_patch",
                        arguments={
                            "patch": {
                                "base_snapshot_id": snapshot_id,
                                "unified_diff": PATCH_DIFF,
                            }
                        },
                    ),
                ),
                usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        return ProviderResponse(
            text="patched the file",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


def _settings(tmp_path: Path, name: str) -> Settings:
    return Settings(database_path=tmp_path / name, worker_profile=WORKER_PROFILE)


def test_package_identity_is_003() -> None:
    assert prp_runtime.__version__ == "0.0.3"
    assert prp_runtime.package_info()["version"] == "0.0.3"


@pytest.mark.asyncio
async def test_local_direct_completes_in_process_without_http_or_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_calls: list[object] = []

    def forbidden_copytree(*args: object, **kwargs: object) -> object:
        copy_calls.append((args, kwargs))
        raise AssertionError("sequential local DIRECT must not copy the workspace tree")

    monkeypatch.setattr(shutil, "copytree", forbidden_copytree)
    monkeypatch.setattr("prp_runtime.workspace.isolation.shutil.copytree", forbidden_copytree)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello local\n", encoding="utf-8")
    adapter = FakeAdapter()
    async with LocalRuntime(_settings(tmp_path, "direct.db"), adapters={"worker": adapter}) as runtime:
        assert runtime.defaults.execution_location is ExecutionLocation.LOCAL
        assert runtime.defaults.isolation_mode is IsolationMode.HOST
        facts = runtime.public_facts()
        assert facts["execution_location"] == "LOCAL"
        assert facts["isolation_mode"] == "HOST"
        assert facts["max_concurrency"] == 1
        assert runtime.composition.execution_location is ExecutionLocation.LOCAL
        assert runtime.composition.isolation_mode is IsolationMode.HOST
        assert runtime.composition.public_facts()["execution_location"] == "LOCAL"
        assert runtime.composition.public_facts()["isolation_mode"] == "HOST"
        result = await runtime.run("summarise the report", workspace=workspace)
        events = await runtime.store.list_events(result.run_id)
        dumped = "".join(event.model_dump_json() for event in events)
        assert result.status is RunStatus.SUCCEEDED
        assert result.strategy is ExecutionStrategy.DIRECT
        assert result.output_text == "the local answer"
        assert runtime.last_copy_mode is ExecutionCopyMode.IN_PLACE
        assert runtime.held_capacity()["max_concurrency"] == 0
        assert runtime.composition.event_bus is not None
        assert str(workspace) not in dumped
        assert not any(event.event_type in BRIDGE_CLAIM_EVENTS for event in events)
    assert adapter.requests
    assert copy_calls == []


@pytest.mark.asyncio
async def test_local_ask_has_no_bridge_claim_and_structured_limits(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "patch"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text(ORIGINAL_MAIN, encoding="utf-8")
    holder: list[LocalRuntime] = []
    adapter = PatchAskAdapter(holder)
    async with LocalRuntime(_settings(tmp_path, "ask.db"), adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        paused = await runtime.run("patch the file", workspace=workspace)
        assert paused.status is RunStatus.RUNNING
        pending = await runtime.store.list_tool_calls(
            paused.run_id, statuses=[ToolCallStatus.AWAITING_APPROVAL]
        )
        assert len(pending) == 1
        approvals = await runtime.pending_approvals(run_id=paused.run_id)
        assert len(approvals) == 1
        events = await runtime.store.list_events(paused.run_id)
        assert [event.event_type for event in events if event.event_type in BRIDGE_CLAIM_EVENTS] == []
        approved = await runtime.approve(approvals[0].request_id)
        assert approved.status is RunStatus.SUCCEEDED
        assert approved.run_id == paused.run_id
        assert runtime.held_capacity()["max_concurrency"] == 0
        with pytest.raises(DomainValidationError) as raised:
            runtime.admit(concurrency=2)
        assert raised.value.code is ErrorCode.INVALID_BUDGET
        lease = runtime.admit(concurrency=1)
        with pytest.raises(BudgetError) as exhausted:
            runtime.admit(concurrency=1)
        assert exhausted.value.code is ErrorCode.RESOURCE_BUDGET_EXCEEDED
        runtime.release(lease)

    backend = LocalIsolationBackend(tmp_path / "slots")
    snapshot = backend.create_base_snapshot(workspace, "ws_test")
    first = backend.create_slot(snapshot.snapshot_id, "wu_one", "owner_a")
    second = backend.create_slot(snapshot.snapshot_id, "wu_two", "owner_a")
    with pytest.raises(IsolationCapacityError) as capacity:
        backend.create_slot(snapshot.snapshot_id, "wu_three", "owner_a")
    assert capacity.value.detail.code is ErrorCode.RESOURCE_BUDGET_EXCEEDED
    assert capacity.value.detail.field == "max_slots"
    backend.cleanup(first.slot_id, owner_id="owner_a")
    backend.cleanup(second.slot_id, owner_id="owner_a")


def test_local_ready_uses_in_process_client_without_bwrap(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            database_path=tmp_path / "ready.db",
            leader_profile=LEADER_PROFILE,
            worker_profile=WORKER_PROFILE,
        ),
        adapters={"leader": FakeAdapter(), "worker": FakeAdapter()},
        execution_location=ExecutionLocation.LOCAL,
        isolation_mode=IsolationMode.HOST,
        sandbox_capabilities=UNAVAILABLE_SANDBOX,
    )
    with TestClient(app) as client:
        response = client.get("/ready")
        health = client.get("/health")
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ready"
    assert payload["execution_location"] == "LOCAL"
    assert payload["isolation_mode"] == "HOST"
    assert payload["sandbox_required"] is False
    assert payload["sandbox_ready"] is False
    assert payload["path_boundary_ready"] is True
    assert health.status_code == 200
    assert health.json()["version"] == "0.0.3"
    assert "secret" not in response.text
    assert str(tmp_path) not in response.text
    assert str(tmp_path) not in health.text


def test_execution_location_is_closed_and_json_stable() -> None:
    assert [item.value for item in ExecutionLocation] == ["CLOUD", "BRIDGE", "LOCAL"]
    assert ExecutionLocation.LOCAL.value == "LOCAL"


def test_host_yolo_requires_explicit_user_fact() -> None:
    with pytest.raises(ValidationError):
        AgentRequestOptions(
            agent_mode=AgentMode.YOLO,
            isolation_mode=IsolationMode.HOST,
            execution_location=ExecutionLocation.LOCAL,
        )
    options = AgentRequestOptions(
        agent_mode=AgentMode.YOLO,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.LOCAL,
        user_explicit=True,
    )
    assert options.execution_location is ExecutionLocation.LOCAL
    assert options.isolation_mode is IsolationMode.HOST


def test_sequential_local_direct_copy_mode_is_in_place() -> None:
    assert (
        select_execution_copy_mode(
            execution_location=ExecutionLocation.LOCAL,
            isolation_mode=IsolationMode.HOST,
            strategy=ExecutionStrategy.DIRECT,
            concurrency=1,
        )
        is ExecutionCopyMode.IN_PLACE
    )
    assert (
        select_execution_copy_mode(
            execution_location=ExecutionLocation.LOCAL,
            isolation_mode=IsolationMode.HOST,
            strategy=ExecutionStrategy.CASCADE,
            concurrency=1,
        )
        is ExecutionCopyMode.COPY_BACKED
    )
    assert (
        select_execution_copy_mode(
            execution_location=ExecutionLocation.LOCAL,
            isolation_mode=IsolationMode.SANDBOXED,
            strategy=ExecutionStrategy.DIRECT,
            concurrency=1,
        )
        is ExecutionCopyMode.COPY_BACKED
    )


@pytest.mark.asyncio
async def test_local_copy_backed_strategy_is_explicit_rejection(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "copied"
    workspace.mkdir()
    adapter = FakeAdapter()
    async with LocalRuntime(
        _settings(tmp_path, "copied.db"), adapters={"worker": adapter}
    ) as runtime:
        with pytest.raises(DomainValidationError) as raised:
            await runtime.run(
                "summarise the report",
                workspace=workspace,
                strategy=ExecutionStrategy.CASCADE,
            )
        assert raised.value.code is ErrorCode.INVALID_AGENT_OPTIONS
        assert raised.value.detail.field == "execution_copy_mode"
        assert runtime.last_copy_mode is ExecutionCopyMode.COPY_BACKED
        assert runtime.held_capacity()["max_concurrency"] == 0
        with pytest.raises(DomainValidationError) as raised:
            await runtime.run(
                "summarise the report",
                workspace=workspace,
                isolation_mode=IsolationMode.SANDBOXED,
            )
        assert raised.value.code is ErrorCode.INVALID_AGENT_OPTIONS
        assert raised.value.detail.field == "execution_copy_mode"
        assert "copy-backed" in str(raised.value)
    assert adapter.requests == []


class SlowAdapter:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    @property
    def name(self) -> str:
        return "local-v003-slow"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        del request
        await asyncio.sleep(self.delay)
        return ProviderResponse(
            text="slow answer",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


@pytest.mark.asyncio
async def test_local_wait_is_not_a_fixed_five_second_abandonment(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "slow"
    workspace.mkdir()
    settings = Settings(
        database_path=tmp_path / "slow.db",
        worker_profile=WORKER_PROFILE,
        local_wait_seconds=1,
    )
    async with LocalRuntime(
        settings, adapters={"worker": SlowAdapter(0.2)}
    ) as runtime:
        result = await runtime.run("hello", workspace=workspace)
        assert result.status is RunStatus.SUCCEEDED
        assert result.output_text == "slow answer"
        assert runtime.held_capacity()["max_concurrency"] == 0


@pytest.mark.asyncio
async def test_local_run_enforces_and_releases_actual_token_envelope(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "envelope"
    workspace.mkdir()
    settings = Settings(
        database_path=tmp_path / "envelope.db",
        worker_profile=WORKER_PROFILE,
        process_max_total_tokens=1,
    )
    async with LocalRuntime(settings, adapters={"worker": FakeAdapter()}) as runtime:
        with pytest.raises(BudgetError) as raised:
            await runtime.run("hello", workspace=workspace)
        assert raised.value.code is ErrorCode.TOKEN_BUDGET_EXCEEDED
        assert raised.value.detail.field == "max_total_tokens"
        assert runtime.held_capacity()["max_total_tokens"] == 0
        assert runtime.held_capacity()["max_concurrency"] == 0
        assert runtime.last_run_id is not None
        persisted = await runtime.store.get_run(runtime.last_run_id)
        assert persisted.status is not RunStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_fresh_runtime_approval_requires_bound_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "fresh"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text(ORIGINAL_MAIN, encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    settings = _settings(tmp_path, "fresh.db")
    holder: list[LocalRuntime] = []
    adapter = PatchAskAdapter(holder)
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        paused = await runtime.run("patch the file", workspace=workspace)
        request_id = (await runtime.pending_approvals(run_id=paused.run_id))[0].request_id
        workspace_id = runtime.last_workspace_id
        assert workspace_id is not None
        assert str(workspace) not in workspace_id
    holder.clear()
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        supervisor = runtime.composition.supervisor
        assert supervisor is not None
        assert paused.run_id in supervisor.held_run_ids
        assert paused.run_id not in supervisor._queued
        discovered = await supervisor.scan()
        assert paused.run_id not in discovered
        assert paused.run_id not in supervisor._queued
        assert paused.run_id not in supervisor.active_run_ids
        with pytest.raises(RuntimeError, match="not bound"):
            await runtime.approve(request_id)
        runtime.bind_workspace(other)
        with pytest.raises(MissingEntityError):
            await runtime.approve(request_id, workspace_id=runtime.last_workspace_id)
        runtime.bind_workspace(workspace)
        approved = await runtime.approve(request_id, workspace_id=workspace_id)
        assert approved.status is RunStatus.SUCCEEDED
        replayed = await runtime.approve(request_id, workspace_id=workspace_id)
        assert replayed.status is RunStatus.SUCCEEDED
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8").count("patched") == 1
        assert runtime.held_capacity()["max_concurrency"] == 0


def test_serve_dispatch_is_synchronous_without_a_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prp_runtime.client import cli

    called: dict[str, object] = {}

    def fake_serve_app(*, host: str, port: int, **kwargs: object) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            nested = False
        else:
            nested = True
        called["host"] = host
        called["port"] = port
        called["nested"] = nested

    monkeypatch.setattr("prp_runtime.client.serve.serve_app", fake_serve_app)
    assert cli.main(["serve"]) == 0
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8000
    assert called["nested"] is False
