"""Authenticated Native Agent API integration coverage."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from prp_runtime.api.native_agent import BridgeToolResultRequest
from prp_runtime.app import create_app
from prp_runtime.client.bridge import Bridge, BridgeTransportError
from prp_runtime.client.executor import BridgeExecutor
from prp_runtime.control.controller import RunController
from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    ExecutionStrategy,
    IsolationMode,
    ModelRole,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    ToolEffect,
)
from prp_runtime.domain.models import (
    AgentRequestOptions,
    AgentToolCall,
    AgentToolResult,
    Budget,
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
    Session,
    Usage,
    WorkspaceGrant,
    WorkUnit,
)
from prp_runtime.domain.values import (
    new_session_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
    utc_now,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderAdapter,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.tools import build_filesystem_registry
from prp_runtime.tools.models import ToolCall
from prp_runtime.workspace import WorkspaceBackend
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotStatus,
    Workspace,
    WorkspaceSource,
    WorkspaceSourceType,
)

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="fake",
    model="fake-model",
    role=ModelRole.WORKER,
    base_url="https://models.internal/v1",
    context_window_tokens=32_000,
    max_output_tokens=4_000,
)


class FakeAdapter:
    name = "fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            text="agent result",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )

    async def aclose(self) -> None:
        return None


class ProductionReadAdapter:
    """Provider fixture used only through the production app composition."""

    name = "production-read-fixture"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        usage = Usage(input_tokens=1, output_tokens=1, elapsed_ms=1)
        if len(self.requests) == 1:
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=new_tool_call_id(),
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )
        previous = request.history[-1]
        assert isinstance(previous, AgentToolResult)
        assert previous.status is ToolCallStatus.SUCCEEDED
        assert previous.result is not None
        assert previous.result["content"] == "production read\n"
        return ProviderResponse(
            text="production read completed",
            usage=usage,
            finish_reason=FinishReason.STOP,
        )

    async def aclose(self) -> None:
        return None


class ProductionApprovalAdapter:
    """Provider fixture that stops after the production approval request."""

    name = "production-approval-fixture"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        assert len(self.requests) == 1
        return ProviderResponse(
            tool_calls=(
                AgentToolCall(
                    call_id=new_tool_call_id(),
                    tool_name="apply_patch",
                    arguments={
                        "patch": {
                            "base_snapshot_id": new_snapshot_id(),
                            "unified_diff": (
                                "--- a/README.md\n"
                                "+++ b/README.md\n"
                                "@@ -1 +1 @@\n"
                                "-production read\n"
                                "+production patched\n"
                            ),
                        }
                    },
                ),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.TOOL_CALLS,
        )

    async def aclose(self) -> None:
        return None


def seed_workspace(
    database_path: Path,
    owner_id: str,
    workspace_root: Path | None = None,
) -> None:
    async def seed() -> None:
        async with SqliteStore(database_path) as store:
            created_at = utc_now()
            await store.create_workspace(
                Workspace(
                    workspace_id="ws_project",
                    owner_id=owner_id,
                    alias="project",
                    source=WorkspaceSource(
                        source_type=WorkspaceSourceType.SERVER_ALIAS,
                        server_alias="project-main",
                    ),
                    created_at=created_at,
                )
            )
            if workspace_root is not None:
                with WorkspaceBackend(workspace_root) as backend:
                    manifest = backend.snapshot_manifest()
                    await store.create_snapshot(
                        Snapshot(
                            snapshot_id=new_snapshot_id(),
                            workspace_id="ws_project",
                            status=SnapshotStatus.READY,
                            created_at=created_at,
                            completed_at=created_at,
                            file_count=len(manifest.entries),
                            total_size=manifest.total_size,
                        ),
                        manifest,
                        owner_id=owner_id,
                    )

    asyncio.run(seed())


def build_app(
    tmp_path: Path,
    *,
    owner_id: str = "prn_operator",
    workspace_root: Path | None = None,
    adapter: ProviderAdapter | None = None,
) -> FastAPI:
    database_path = tmp_path / "agent.db"
    seed_workspace(database_path, owner_id, workspace_root)
    settings = Settings(
        database_path=database_path,
        worker_profile=WORKER_PROFILE,
        service_token=SecretStr("agent-secret"),
        service_principal="prn_operator",
        workspace_roots=(
            {} if workspace_root is None else {"project-main": str(workspace_root)}
        ),
    )
    return create_app(settings, adapters={"worker": adapter or FakeAdapter()})


def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer agent-secret"}


@asynccontextmanager
async def app_client(
    app: FastAPI,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        active_transport = transport or httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=active_transport,
            base_url="http://bridge.test",
        ) as client:
            yield client


class DropCommittedResponseTransport(httpx.AsyncBaseTransport):
    """Drop selected responses after ASGI has committed the production route."""

    def __init__(self, app: FastAPI, paths: set[str]) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self._paths = set(paths)
        self.requests: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        path = str(request.url.path)
        self.requests.append(path)
        if path in self._paths:
            self._paths.remove(path)
            await response.aread()
            await response.aclose()
            raise httpx.ReadError("simulated committed response disconnect", request=request)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def test_bridge_result_request_is_closed_and_canonical(tmp_path: Path) -> None:
    request = BridgeToolResultRequest(
        dev_only=True,
        status=ToolCallStatus.SUCCEEDED,
        result={"content": "done"},
        output="done",
        changed_paths=("src/main.py",),
        exit_code=0,
    )
    candidate = request.to_tool_result("tc_result", completed_at=utc_now())
    assert candidate.call_id == "tc_result"
    assert request.fingerprint("tc_result") == request.fingerprint("tc_result")
    assert request.fingerprint("tc_result") != request.fingerprint("tc_other")
    with pytest.raises(ValidationError):
        BridgeToolResultRequest(status=ToolCallStatus.RUNNING)
    with pytest.raises(ValidationError):
        BridgeToolResultRequest(
            status=ToolCallStatus.FAILED,
            error=ErrorInfo(category=ErrorCategory.INTERNAL, message="failed"),
            changed_paths=("../secret",),
        )
    with pytest.raises(ValidationError):
        BridgeToolResultRequest(
            status=ToolCallStatus.SUCCEEDED,
            result={"value": float("nan")},
        )
    with pytest.raises(ValidationError):
        BridgeToolResultRequest(
            status=ToolCallStatus.SUCCEEDED,
            workspace_id="/host/root",
        )


def wait_for_terminal(client: TestClient, session_id: str, run_id: str) -> dict[str, object]:
    for _ in range(200):
        response = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}", headers=auth_headers()
        )
        body = response.json()
        if body["status"] in {status.value for status in RunStatus if status.is_terminal}:
            return body
    raise AssertionError("run did not reach a terminal state")


def test_health_is_public_but_agent_api_requires_auth(tmp_path: Path) -> None:
    with TestClient(build_app(tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/sessions", json={"workspace_id": "ws_project"}).status_code == 401


def test_create_app_composition_exposes_l1_runtime_without_claiming_l3_ready(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)

    with TestClient(app) as client:
        assert isinstance(app.state.controller, RunController)
        assert app.state.tool_runtime_provider is not None
        readiness = client.get("/ready")
        body = readiness.json()
        assert body["store_open"] is True
        assert body["controller_present"] is True
        assert body["adapters_ready"] is True
        if body["sandbox_ready"] is False:
            assert readiness.status_code == 503
            assert body["status"] == "not_ready"


def test_session_run_is_async_and_principal_scoped(tmp_path: Path) -> None:
    with TestClient(build_app(tmp_path)) as client:
        created = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        )
        assert created.status_code == 201
        session = created.json()
        session_id = session["session_id"]

        run_response = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        )
        assert run_response.status_code == 202
        run_id = run_response.json()["run_id"]
        assert wait_for_terminal(client, session_id, run_id)["output_text"] == "agent result"

        assert client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}", headers=auth_headers()
        ).status_code == 200


def test_production_app_read_path_uses_scoped_runtime(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "read-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("production read\n", encoding="utf-8")
    adapter = ProductionReadAdapter()

    with TestClient(
        build_app(tmp_path, workspace_root=workspace_root, adapter=adapter)
    ) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        session_id = session["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "read README.md"},
        ).json()["run_id"]

        finished = wait_for_terminal(client, session_id, run_id)
        assert finished["status"] == RunStatus.SUCCEEDED.value
        assert finished["output_text"] == "production read completed"
        calls = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls",
            headers=auth_headers(),
        ).json()
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "read_file"
        assert calls[0]["status"] == ToolCallStatus.SUCCEEDED.value
        assert len(adapter.requests) == 2


def test_current_planner_spawn_agent_workspace_lifecycle_is_session_scoped(
    tmp_path: Path,
) -> None:
    """Reproduce current workspace timing without creating an eager child workspace."""
    workspace_root = tmp_path / "current-lifecycle-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("production read\n", encoding="utf-8")
    adapter = ProductionReadAdapter()
    app = build_app(tmp_path, workspace_root=workspace_root, adapter=adapter)

    with TestClient(app) as client:
        assert client.post(
            "/v1/sessions",
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).status_code == 401
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        session_id = session["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "read README.md"},
        ).json()["run_id"]
        time.sleep(0.02)
        finished = wait_for_terminal(client, session_id, run_id)
        assert finished["status"] == RunStatus.SUCCEEDED.value

    async def inspect() -> tuple[int, int, tuple[ToolCallStatus, ...]]:
        async with SqliteStore(tmp_path / "agent.db") as store:
            workspaces = await store.list_workspaces(owner_id="prn_operator")
            snapshots = await store.list_snapshots(
                "ws_project", owner_id="prn_operator"
            )
            calls = await store.list_tool_calls(run_id)
            return len(workspaces), len(snapshots), tuple(call.status for call in calls)

    workspace_count, snapshot_count, call_statuses = asyncio.run(inspect())
    assert workspace_count == 1
    assert snapshot_count == 1
    assert call_statuses == (ToolCallStatus.SUCCEEDED,)
    assert len(adapter.requests) == 2


def test_production_app_write_path_pauses_for_approval(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "approval-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("production read\n", encoding="utf-8")
    adapter = ProductionApprovalAdapter()

    with TestClient(
        build_app(tmp_path, workspace_root=workspace_root, adapter=adapter)
    ) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={
                "workspace_id": "ws_project",
                "access": ["READ", "WRITE"],
                "agent_options": {"agent_mode": "AUTO"},
            },
        ).json()
        session_id = session["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "update README.md"},
        ).json()["run_id"]

        approval: dict[str, object] | None = None
        for _ in range(200):
            approvals = client.get(
                f"/v1/sessions/{session_id}/approvals",
                headers=auth_headers(),
            ).json()
            if approvals:
                approval = approvals[0]
                break
        assert approval is not None
        assert approval["tool_name"] == "apply_patch"
        assert approval["effect"] == ToolEffect.WRITE.value
        assert approval["workspace_id"] == "ws_project"
        assert len(adapter.requests) == 1
        assert (
            workspace_root / "README.md"
        ).read_text(encoding="utf-8") == "production read\n"
        run_view = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}",
            headers=auth_headers(),
        ).json()
        assert run_view["status"] == RunStatus.RUNNING.value


def test_events_support_cursor_replay(tmp_path: Path) -> None:
    with TestClient(build_app(tmp_path)) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project"},
        ).json()
        session_id = session["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        wait_for_terminal(client, session_id, run_id)

        all_events = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/events",
            headers=auth_headers(),
        )
        assert all_events.status_code == 200
        assert all_events.headers["content-type"].startswith("text/event-stream")
        records = [block for block in all_events.text.strip().split("\n\n") if block]
        assert records
        first_id = records[0].splitlines()[0].removeprefix("id: ")
        resumed = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/events",
            headers={**auth_headers(), "Last-Event-ID": first_id},
        )
        assert resumed.status_code == 200
        assert first_id not in resumed.text.split("\n", 1)[0]


def test_workspace_owner_is_injected_from_authentication(tmp_path: Path) -> None:
    with TestClient(build_app(tmp_path, owner_id="prn_other")) as client:
        response = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project"},
        )
        assert response.status_code == 404


def test_bridge_claim_and_read_result_use_production_routes_and_resume(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "bridge-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("fixture read\n", encoding="utf-8")
    app = build_app(tmp_path)
    database_path = tmp_path / "agent.db"
    owner_id = "prn_operator"

    async def seed_running_call() -> tuple[str, str, str]:
        async with SqliteStore(database_path) as store:
            now = utc_now()
            session = Session(
                session_id=new_session_id(),
                principal_id=owner_id,
                workspace_id="ws_project",
                grant=WorkspaceGrant(
                    principal_id=owner_id,
                    workspace_id="ws_project",
                ),
                agent_options=AgentRequestOptions(
                    agent_mode=AgentMode.AUTO,
                    isolation_mode=IsolationMode.HOST,
                    execution_location=ExecutionLocation.BRIDGE,
                ),
                created_at=now,
            )
            await store.create_session(session)
            controller = RunController(
                store,
                Settings(worker_profile=WORKER_PROFILE),
                {"worker": FakeAdapter()},
            )
            run = await controller.create_run(
                NativeRunRequest(
                    input="read the fixture README",
                    routing_policy=RoutingPolicy.MANUAL,
                    strategy=ExecutionStrategy.DIRECT,
                    agent_options=session.agent_options,
                    budget=Budget(max_attempts=2, max_total_tokens=8),
                )
            )
            await store.attach_run_to_session(
                session.session_id,
                run.run_id,
                principal_id=owner_id,
            )
            run = run.model_copy(
                update={
                    "status": RunStatus.SUCCEEDED,
                    "started_at": run.created_at,
                    "completed_at": run.created_at,
                }
            )
            await store.update_run(run)
            with WorkspaceBackend(workspace_root) as backend:
                manifest = backend.snapshot_manifest()
                snapshot_id = new_snapshot_id()
                await store.create_snapshot(
                    Snapshot(
                        snapshot_id=snapshot_id,
                        workspace_id="ws_project",
                        status=SnapshotStatus.READY,
                        created_at=now,
                        completed_at=now,
                        file_count=len(manifest.entries),
                        total_size=manifest.total_size,
                    ),
                    manifest,
                    owner_id=owner_id,
                )
            work_unit_id = new_work_unit_id()
            await store.create_work_unit(
                WorkUnit(
                    work_unit_id=work_unit_id,
                    run_id=run.run_id,
                    name="bridge-read",
                    instruction="read the fixture README",
                    created_at=now,
                )
            )
            call = ToolCall(
                call_id="tc_bridge_read",
                run_id=run.run_id,
                work_unit_id=work_unit_id,
                tool_name="read_file",
                effect=ToolEffect.READ,
                arguments={"path": "README.md"},
                snapshot_id=snapshot_id,
                requested_at=now,
            )
            await store.create_tool_call(
                call,
                workspace_id="ws_project",
                idempotency_key="bridge-read-call",
            )
            return session.session_id, run.run_id, call.call_id

    session_id, run_id, call_id = asyncio.run(seed_running_call())
    state_path = tmp_path / "bridge-state.json"
    claim_path = "/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim"
    result_path = "/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/result"
    transport = DropCommittedResponseTransport(
        app,
        {
            claim_path.format(session_id=session_id, run_id=run_id, call_id=call_id),
            result_path.format(session_id=session_id, run_id=run_id, call_id=call_id),
        },
    )

    async def exercise_routes() -> tuple[dict[str, object], str]:
        async with app_client(app, transport=transport) as http_client:
            store = app.state.store
            assert isinstance(store, SqliteStore)
            await store.start_tool_call(call_id)
            bridge = Bridge(
                "http://bridge.test",
                "agent-secret",
                state_path=state_path,
                workspace_root=workspace_root,
                client=http_client,
            )
            bridge.state.session_id = session_id
            bridge.state.run_id = run_id
            bridge._save_state()
            with pytest.raises(BridgeTransportError):
                await bridge.claim_tool_call(call_id)
            claim = await bridge.claim_tool_call(call_id)
            call_view = await bridge._request(
                "GET",
                f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}",
            )
            dispatch_claim = dict(claim)
            dispatch_claim.update(
                {
                    "tool_name": call_view["tool_name"],
                    "effect": call_view["effect"],
                    "arguments": call_view["arguments"],
                    "scope": {"paths": ["**"]},
                }
            )
            with WorkspaceBackend(workspace_root) as backend:
                executor = BridgeExecutor(
                    build_filesystem_registry(backend), workspace_root
                )
                with pytest.raises(BridgeTransportError):
                    await bridge.execute_claim(dispatch_claim, executor)
                resumed = Bridge(
                    "http://bridge.test",
                    "agent-secret",
                    state_path=state_path,
                    workspace_root=workspace_root,
                    client=http_client,
                )
                resumed.state.session_id = session_id
                resumed.state.run_id = run_id
                submitted = await resumed.execute_claim(dispatch_claim, executor)
                replay = await resumed.execute_claim(dispatch_claim, executor)
            events = await http_client.get(
                f"/v1/sessions/{session_id}/runs/{run_id}/events",
                headers=auth_headers(),
            )
            run_view = await bridge.get_run()
            assert events.status_code == 200
            assert "TOOL_CALL_SUCCEEDED" in events.text
            assert run_view["run_id"] == run_id
            assert replay == submitted
            assert transport.requests.count(
                claim_path.format(session_id=session_id, run_id=run_id, call_id=call_id)
            ) == 2
            assert transport.requests.count(
                result_path.format(session_id=session_id, run_id=run_id, call_id=call_id)
            ) == 2
            return submitted, state_path.read_text(encoding="utf-8")

    submitted, state_text = asyncio.run(exercise_routes())
    assert submitted["status"] == ToolCallStatus.SUCCEEDED.value
    assert submitted["result"]["path"] == "README.md"
    assert str(workspace_root) not in state_text
    assert "agent-secret" not in state_text


def test_bridge_patch_test_diff_claim_submit_builds_change_set_in_tmp_root(
    tmp_path: Path,
) -> None:
    import sys

    from prp_runtime.domain.values import new_snapshot_id, new_tool_call_id
    from prp_runtime.policy.models import CommandClass
    from prp_runtime.tools import ToolRegistry
    from prp_runtime.tools.command import (
        CommandParameter,
        CommandRegistry,
        CommandRunner,
        CommandSpec,
        build_targeted_test_definition,
    )
    from prp_runtime.tools.diff import DiffToolRunner, build_diff_definitions
    from prp_runtime.tools.patch import PatchRunner, build_patch_definition
    from prp_runtime.workspace.changes import ChangeSet
    from prp_runtime.workspace.models import Snapshot, SnapshotStatus

    workspace_root = tmp_path / "bridge-code-workspace"
    source_root = workspace_root / "src"
    source_root.mkdir(parents=True)
    target = source_root / "main.py"
    target.write_text('def answer():\n    return "old"\n', encoding="utf-8")
    app = build_app(tmp_path)
    database_path = tmp_path / "agent.db"
    owner_id = "prn_operator"
    base_snapshot_id = new_snapshot_id()
    patch_call_id = new_tool_call_id()
    test_call_id = new_tool_call_id()
    diff_call_id = new_tool_call_id()

    async def seed_calls() -> tuple[str, str]:
        async with SqliteStore(database_path) as store:
            now = utc_now()
            session = Session(
                session_id=new_session_id(),
                principal_id=owner_id,
                workspace_id="ws_project",
                grant=WorkspaceGrant(
                    principal_id=owner_id,
                    workspace_id="ws_project",
                ),
                agent_options=AgentRequestOptions(
                    agent_mode=AgentMode.AUTO,
                    isolation_mode=IsolationMode.HOST,
                    execution_location=ExecutionLocation.BRIDGE,
                ),
                created_at=now,
            )
            await store.create_session(session)
            controller = RunController(
                store,
                Settings(worker_profile=WORKER_PROFILE),
                {"worker": FakeAdapter()},
            )
            run = await controller.create_run(
                NativeRunRequest(
                    input="patch, test and inspect the fixture",
                    routing_policy=RoutingPolicy.MANUAL,
                    strategy=ExecutionStrategy.DIRECT,
                    agent_options=session.agent_options,
                    budget=Budget(max_attempts=4, max_total_tokens=16),
                )
            )
            await store.attach_run_to_session(
                session.session_id,
                run.run_id,
                principal_id=owner_id,
            )
            run = run.model_copy(
                update={
                    "status": RunStatus.SUCCEEDED,
                    "started_at": run.created_at,
                    "completed_at": run.created_at,
                }
            )
            await store.update_run(run)
            work_unit_ids = tuple(new_work_unit_id() for _ in range(3))
            await store.create_work_units(
                tuple(
                    WorkUnit(
                        work_unit_id=work_unit_id,
                        run_id=run.run_id,
                        name=name,
                        instruction="patch, test and inspect the fixture",
                        created_at=now,
                    )
                    for work_unit_id, name in zip(
                        work_unit_ids,
                        ("bridge-patch", "bridge-test", "bridge-diff"),
                    )
                )
            )
            with WorkspaceBackend(workspace_root) as backend:
                manifest = backend.snapshot_manifest()
                await store.create_snapshot(
                    Snapshot(
                        snapshot_id=base_snapshot_id,
                        workspace_id="ws_project",
                        status=SnapshotStatus.READY,
                        created_at=now,
                        completed_at=now,
                        file_count=len(manifest.entries),
                        total_size=manifest.total_size,
                    ),
                    manifest,
                    owner_id=owner_id,
                )
            calls = (
                ToolCall(
                    call_id=patch_call_id,
                    run_id=run.run_id,
                    work_unit_id=work_unit_ids[0],
                    tool_name="apply_patch",
                    effect=ToolEffect.WRITE,
                    arguments={
                        "patch": {
                            "base_snapshot_id": base_snapshot_id,
                            "unified_diff": (
                                "--- a/src/main.py\n"
                                "+++ b/src/main.py\n"
                                "@@ -1,2 +1,2 @@\n"
                                " def answer():\n"
                                '-    return "old"\n'
                                '+    return "patched"\n'
                            ),
                        }
                    },
                    snapshot_id=base_snapshot_id,
                    requested_at=now,
                ),
                ToolCall(
                    call_id=test_call_id,
                    run_id=run.run_id,
                    work_unit_id=work_unit_ids[1],
                    tool_name="run_targeted_test",
                    effect=ToolEffect.COMMAND,
                    arguments={
                        "spec_name": "fixture_patch_test",
                        "parameters": {"mode": "verify_patch"},
                    },
                    snapshot_id=base_snapshot_id,
                    requested_at=now,
                ),
                ToolCall(
                    call_id=diff_call_id,
                    run_id=run.run_id,
                    work_unit_id=work_unit_ids[2],
                    tool_name="get_diff",
                    effect=ToolEffect.READ,
                    arguments={},
                    snapshot_id=base_snapshot_id,
                    requested_at=now,
                ),
            )
            for call in calls:
                await store.create_tool_call(
                    call,
                    workspace_id="ws_project",
                    idempotency_key=f"bridge-{call.call_id}",
                )
            return session.session_id, run.run_id

    session_id, run_id = asyncio.run(seed_calls())
    state_path = tmp_path / "bridge-code-state.json"

    async def execute_calls() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        async with app_client(app) as http_client:
            store = app.state.store
            assert isinstance(store, SqliteStore)
            for call_id in (patch_call_id, test_call_id, diff_call_id):
                await store.start_tool_call(call_id)
            bridge = Bridge(
                "http://bridge.test",
                "agent-secret",
                state_path=state_path,
                workspace_root=workspace_root,
                client=http_client,
            )
            bridge.state.session_id = session_id
            bridge.state.run_id = run_id
            bridge._save_state()
            async with SqliteStore(database_path) as store:
                with WorkspaceBackend(workspace_root) as backend:
                    manifest = backend.snapshot_manifest()
                    snapshot = await store.get_snapshot(
                        base_snapshot_id,
                        owner_id=owner_id,
                    )
                    patch_runner = PatchRunner(
                        backend,
                        store,
                        owner_id=owner_id,
                        base_snapshot=snapshot,
                        base_manifest=manifest,
                    )
                    fixture = Path(__file__).parents[1] / "fixtures" / "command_fixture.py"
                    command_spec = CommandSpec(
                        name="fixture_patch_test",
                        executable=sys.executable,
                        argv_template=(str(fixture), "{mode}"),
                        parameters=(CommandParameter(name="mode"),),
                        command_class=CommandClass.TEST,
                        timeout_seconds=15,
                        max_output_bytes=32 * 1024,
                    )
                    command_runner = CommandRunner(
                        CommandRegistry((command_spec,)),
                        workspace_cwd=workspace_root,
                        test_only=True,
                        isolation_mode=IsolationMode.HOST,
                    )

                    class DeferredDiff:
                        def __init__(self) -> None:
                            self.runner: DiffToolRunner | None = None

                        def bind(self, change_set: ChangeSet) -> None:
                            self.runner = DiffToolRunner(
                                change_set,
                                base_manifest=manifest,
                                manifest_provider=backend.snapshot_manifest,
                            )

                        def get_diff(self):
                            assert self.runner is not None
                            return self.runner.get_diff()

                        def get_status(self):
                            assert self.runner is not None
                            return self.runner.get_status()

                    deferred = DeferredDiff()
                    filesystem = build_filesystem_registry(backend)
                    registry = ToolRegistry(
                        (
                            *filesystem.definitions,
                            build_patch_definition(patch_runner),
                            build_targeted_test_definition(command_runner),
                            *build_diff_definitions(deferred),
                        )
                    )
                    executor = BridgeExecutor(registry, workspace_root)

                    async def run_call(call_id: str) -> dict[str, object]:
                        claim = await bridge.claim_tool_call(call_id)
                        view = await bridge._request(
                            "GET",
                            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}",
                        )
                        dispatch = dict(claim)
                        dispatch.update(
                            {
                                "tool_name": view["tool_name"],
                                "effect": view["effect"],
                                "arguments": view["arguments"],
                                "scope": {"paths": ["**"]},
                                "snapshot_id": view["snapshot_id"],
                            }
                        )
                        return await bridge.execute_claim(dispatch, executor)

                    patch_result = await run_call(patch_call_id)
                    assert target.read_text(encoding="utf-8") == (
                        'def answer():\n    return "patched"\n'
                    )
                    assert patch_result["result"] is not None
                    change_set_id = patch_result["result"]["change_set_id"]
                    deferred.bind(await store.get_change_set(change_set_id))
                    test_result = await run_call(test_call_id)
                    diff_result = await run_call(diff_call_id)
                    return patch_result, test_result, diff_result

    patch_result, test_result, diff_result = asyncio.run(execute_calls())
    assert patch_result["status"] == ToolCallStatus.SUCCEEDED.value
    assert patch_result["changed_paths"] == ["src/main.py"]
    assert test_result["status"] == ToolCallStatus.SUCCEEDED.value
    assert test_result["exit_code"] == 0
    assert diff_result["status"] == ToolCallStatus.SUCCEEDED.value
    assert diff_result["result"]["entries"][0]["path"] == "src/main.py"
