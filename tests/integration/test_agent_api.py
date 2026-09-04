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
    AttemptStatus,
    ExecutionLocation,
    ExecutionStrategy,
    IsolationMode,
    ModelRole,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    ToolEffect,
)
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import (
    AgentRequestOptions,
    AgentToolCall,
    AgentToolResult,
    Attempt,
    Budget,
    ClientCapabilityDescriptor,
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
    Session,
    Usage,
    WorkspaceGrant,
    WorkUnit,
    fingerprint_client_capabilities,
    new_client_id,
)
from prp_runtime.domain.values import (
    ModelRef,
    new_attempt_id,
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
    SnapshotEntry,
    SnapshotEntryType,
    SnapshotManifest,
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
                    manifest, file_contents = backend.capture_snapshot()
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
                        file_contents=file_contents,
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



def _handshake_body(*tools: str, effects: tuple[str, ...] = ("READ",)) -> dict[str, object]:
    capabilities = ClientCapabilityDescriptor(
        tools=tuple(sorted(tools)),
        effects=tuple(sorted((ToolEffect(item) for item in set(effects)), key=lambda item: item.value)),
    )
    return {
        "client_id": new_client_id(),
        "protocol_version": "0.0.4",
        "capabilities": capabilities.model_dump(mode="json"),
        "fingerprint": fingerprint_client_capabilities(capabilities),
        "workspace_id": "ws_project",
    }


def _registration_body(payload: dict[str, object]) -> dict[str, object]:
    return {
        "client_id": payload["client_id"],
        "workspace_id": payload["workspace_id"],
        "fingerprint": payload["fingerprint"],
        "capabilities": payload["capabilities"],
    }


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
        client_id=new_client_id(),
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
    last: dict[str, object] | None = None
    for _ in range(200):
        response = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}", headers=auth_headers()
        )
        body = response.json()
        last = body
        if body["status"] in {status.value for status in RunStatus if status.is_terminal}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run did not reach a terminal state: {last}")


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
            handshake_payload = _handshake_body("list_files", "read_file")
            await bridge._request(
                "POST", "/v1/bridge/clients", body=_registration_body(handshake_payload)
            )
            await bridge.handshake(handshake_payload)
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
            handshake_payload = _handshake_body(
                "apply_patch",
                "get_diff",
                "get_status",
                "list_files",
                "read_file",
                "run_targeted_test",
                effects=("COMMAND", "READ", "WRITE"),
            )
            await bridge._request(
                "POST", "/v1/bridge/clients", body=_registration_body(handshake_payload)
            )
            await bridge.handshake(handshake_payload)
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



def test_bridge_handshake_accepts_compatible_client_and_rejects_incompatible(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    capabilities = ClientCapabilityDescriptor(
        tools=("list_files", "read_file"),
        effects=(ToolEffect.READ,),
    )
    fingerprint = fingerprint_client_capabilities(capabilities)
    client_id = new_client_id()
    payload = {
        "client_id": client_id,
        "protocol_version": "0.0.4",
        "capabilities": capabilities.model_dump(mode="json"),
        "fingerprint": fingerprint,
        "workspace_id": "ws_project",
    }
    with TestClient(app) as client:
        assert client.post("/v1/bridge/handshake", json=payload).status_code == 401
        registered = client.post(
            "/v1/bridge/clients",
            json=_registration_body(payload),
            headers=auth_headers(),
        )
        assert registered.status_code == 201
        accepted = client.post(
            "/v1/bridge/handshake", json=payload, headers=auth_headers()
        )
        assert accepted.status_code == 200
        body = accepted.json()
        assert body["accepted"] is True
        assert body["client_id"] == client_id
        assert body["principal_id"] == "prn_operator"
        assert body["workspace_id"] == "ws_project"
        assert body["fingerprint"] == fingerprint
        assert "strategy" not in body
        unknown_version = dict(payload)
        unknown_version["protocol_version"] = "1.2.3"
        rejected = client.post(
            "/v1/bridge/handshake", json=unknown_version, headers=auth_headers()
        )
        assert rejected.status_code in {400, 422}
        unknown_tool = dict(payload)
        unknown_tool["capabilities"] = {
            "tools": ["list_files", "unknown_shell"],
            "effects": ["READ"],
        }
        unknown_tool["fingerprint"] = "b" * 64
        denied = client.post(
            "/v1/bridge/handshake", json=unknown_tool, headers=auth_headers()
        )
        assert denied.status_code in {400, 422}
        authority = dict(payload)
        authority["strategy"] = "PROGRESSIVE"
        forbidden = client.post(
            "/v1/bridge/handshake", json=authority, headers=auth_headers()
        )
        assert forbidden.status_code in {400, 422}



def test_bridge_handshake_is_fail_closed_for_invalid_and_duplicate_clients(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    payload = _handshake_body("list_files", "read_file")
    with TestClient(app) as client:
        registered = client.post(
            "/v1/bridge/clients",
            json=_registration_body(payload),
            headers=auth_headers(),
        )
        assert registered.status_code == 201
        first = client.post(
            "/v1/bridge/handshake", json=payload, headers=auth_headers()
        )
        assert first.status_code == 200
        original = first.json()
        duplicate = _handshake_body("read_file")
        duplicate["client_id"] = payload["client_id"]
        conflict = client.post(
            "/v1/bridge/handshake", json=duplicate, headers=auth_headers()
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "bridge_client_conflict"
        replay = client.post(
            "/v1/bridge/handshake", json=payload, headers=auth_headers()
        )
        assert replay.status_code == 200
        assert replay.json() == original

        missing = dict(payload)
        missing.pop("capabilities")
        missing_response = client.post(
            "/v1/bridge/handshake", json=missing, headers=auth_headers()
        )
        assert missing_response.status_code in {400, 422}

        bad_fingerprint = dict(payload)
        bad_fingerprint["fingerprint"] = "c" * 64
        fingerprint_response = client.post(
            "/v1/bridge/handshake", json=bad_fingerprint, headers=auth_headers()
        )
        assert fingerprint_response.status_code in {400, 422}

        oversized = dict(payload)
        oversized["capabilities"] = {
            **payload["capabilities"],
            "max_argument_bytes": 10_000_000,
        }
        limit_response = client.post(
            "/v1/bridge/handshake", json=oversized, headers=auth_headers()
        )
        assert limit_response.status_code in {400, 422}

        session = client.post(
            "/v1/sessions",
            json={"workspace_id": "ws_project"},
            headers=auth_headers(),
        )
        assert session.status_code == 201
        claim = client.post(
            f"/v1/sessions/{session.json()['session_id']}/runs/run_missing/tool-calls/tc_missing/claim",
            json={"call_id": "tc_missing", "client_id": payload["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-1"},
        )
        assert claim.status_code in {403, 404}

        other = tmp_path / "other"
        other.mkdir()
        stranger_app = build_app(other)
        with TestClient(stranger_app) as stranger:
            session = stranger.post(
                "/v1/sessions",
                json={"workspace_id": "ws_project"},
                headers=auth_headers(),
            )
            assert session.status_code == 201
            stranger_payload = _handshake_body("list_files", "read_file")
            registered = stranger.post(
                "/v1/bridge/clients",
                json=_registration_body(stranger_payload),
                headers=auth_headers(),
            )
            assert registered.status_code == 201
            unhandshaken = stranger.post(
                f"/v1/sessions/{session.json()['session_id']}/runs/run_missing/tool-calls/tc_missing/claim",
                json={"call_id": "tc_missing", "client_id": stranger_payload["client_id"]},
                headers={**auth_headers(), "Idempotency-Key": "claim-2"},
            )
            assert unhandshaken.status_code == 403
            assert unhandshaken.json()["detail"]["code"] == "bridge_handshake_required"



def test_bridge_client_registration_is_owner_scoped_and_gates_claims(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    payload = _handshake_body("list_files", "read_file")
    with TestClient(app) as client:
        created = client.post(
            "/v1/bridge/clients",
            json=_registration_body(payload),
            headers=auth_headers(),
        )
        assert created.status_code == 201
        body = created.json()
        assert body["client_id"] == payload["client_id"]
        assert body["principal_id"] == "prn_operator"
        assert body["workspace_id"] == "ws_project"
        assert body["status"] == "ACTIVE"
        assert "token" not in body
        assert "workspace_root" not in body
        listed = client.get("/v1/bridge/clients", headers=auth_headers())
        assert listed.status_code == 200
        assert listed.json()[0]["client_id"] == payload["client_id"]
        handshake = client.post(
            "/v1/bridge/handshake", json=payload, headers=auth_headers()
        )
        assert handshake.status_code == 200
        disabled = client.post(
            f"/v1/bridge/clients/{payload['client_id']}/disable",
            headers=auth_headers(),
        )
        assert disabled.status_code == 200
        assert disabled.json()["status"] == "DISABLED"
        blocked = client.post(
            "/v1/bridge/handshake", json=payload, headers=auth_headers()
        )
        assert blocked.status_code == 409
        session = client.post(
            "/v1/sessions",
            json={"workspace_id": "ws_project"},
            headers=auth_headers(),
        )
        assert session.status_code == 201
        claim = client.post(
            f"/v1/sessions/{session.json()['session_id']}/runs/run_missing/tool-calls/tc_missing/claim",
            json={"call_id": "tc_missing", "client_id": payload["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-disabled"},
        )
        assert claim.status_code == 403
        missing = client.get(
            "/v1/bridge/clients/cli_missingclient",
            headers=auth_headers(),
        )
        assert missing.status_code == 404



def test_bridge_heartbeat_marks_live_and_rejects_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    payload = _handshake_body("list_files", "read_file")
    with TestClient(app) as client:
        created = client.post(
            "/v1/bridge/clients",
            json=_registration_body(payload),
            headers=auth_headers(),
        )
        assert created.status_code == 201
        beat = client.post(
            f"/v1/bridge/clients/{payload['client_id']}/heartbeat",
            json={"fingerprint": payload["fingerprint"]},
            headers=auth_headers(),
        )
        assert beat.status_code == 200
        body = beat.json()
        assert body["liveness"] == "LIVE"
        assert body["client_id"] == payload["client_id"]
        assert "token" not in body
        assert "workspace_root" not in body
        mismatch = client.post(
            f"/v1/bridge/clients/{payload['client_id']}/heartbeat",
            json={"fingerprint": "b" * 64},
            headers=auth_headers(),
        )
        assert mismatch.status_code == 409
        session = client.get("/v1/sessions/" + "missing", headers=auth_headers())
        assert session.status_code == 404



def test_stale_client_can_reconnect_only_with_matching_capability(tmp_path: Path) -> None:
    app = build_app(tmp_path)
    payload = _handshake_body("list_files", "read_file")
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/bridge/clients",
                json=_registration_body(payload),
                headers=auth_headers(),
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/v1/bridge/handshake", json=payload, headers=auth_headers()
            ).status_code
            == 200
        )
        supervisor = app.state.supervisor
        supervisor._heartbeat_ttl = __import__("datetime").timedelta(seconds=0)
        session = client.post(
            "/v1/sessions",
            json={"workspace_id": "ws_project"},
            headers=auth_headers(),
        )
        assert session.status_code == 201
        stale_claim = client.post(
            f"/v1/sessions/{session.json()['session_id']}/runs/run_missing/tool-calls/tc_missing/claim",
            json={"call_id": "tc_missing", "client_id": payload["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "stale-claim"},
        )
        assert stale_claim.status_code == 403
        supervisor._heartbeat_ttl = __import__("datetime").timedelta(seconds=30)
        replay = client.post(
            f"/v1/bridge/clients/{payload['client_id']}/heartbeat",
            json={"fingerprint": payload["fingerprint"]},
            headers=auth_headers(),
        )
        assert replay.status_code == 200
        assert replay.json()["liveness"] == "LIVE"
        other = _handshake_body("read_file")
        other["client_id"] = payload["client_id"]
        crossed = client.post(
            "/v1/bridge/handshake", json=other, headers=auth_headers()
        )
        assert crossed.status_code == 409


def test_same_principal_clients_do_not_union_capabilities(tmp_path: Path) -> None:
    workspace_root = tmp_path / "bridge-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("fixture read\n", encoding="utf-8")
    app = build_app(tmp_path, workspace_root=workspace_root)
    database_path = tmp_path / "agent.db"
    call_ids = seed_running_bridge_call(
        database_path, workspace_root=workspace_root, call_id="tc_bridge_exact"
    )
    lister = _handshake_body("list_files")
    reader = _handshake_body("read_file")
    result_body = {
        "dev_only": True,
        "status": ToolCallStatus.SUCCEEDED.value,
        "result": {"path": "README.md"},
        "output": "fixture read\n",
        "changed_paths": [],
        "exit_code": 0,
        "client_id": lister["client_id"],
    }
    with TestClient(app) as client:
        for payload in (lister, reader):
            assert (
                client.post(
                    "/v1/bridge/clients",
                    json=_registration_body(payload),
                    headers=auth_headers(),
                ).status_code
                == 201
            )
            assert (
                client.post(
                    "/v1/bridge/handshake", json=payload, headers=auth_headers()
                ).status_code
                == 200
            )
        store = app.state.store
        assert isinstance(store, SqliteStore)
        asyncio.run(store.start_tool_call(call_ids[2]))
        missing_identity = client.post(
            f"/v1/sessions/{call_ids[0]}/runs/{call_ids[1]}/tool-calls/{call_ids[2]}/claim",
            json={"call_id": call_ids[2]},
            headers={**auth_headers(), "Idempotency-Key": "claim-missing-client"},
        )
        assert missing_identity.status_code in {400, 422}
        denied = client.post(
            f"/v1/sessions/{call_ids[0]}/runs/{call_ids[1]}/tool-calls/{call_ids[2]}/claim",
            json={"call_id": call_ids[2], "client_id": lister["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-lister"},
        )
        assert denied.status_code == 403
        assert denied.json()["detail"]["code"] == "bridge_capability_mismatch"
        claimed = client.post(
            f"/v1/sessions/{call_ids[0]}/runs/{call_ids[1]}/tool-calls/{call_ids[2]}/claim",
            json={"call_id": call_ids[2], "client_id": reader["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-reader"},
        )
        assert claimed.status_code == 201
        assert claimed.json()["client_id"] == reader["client_id"]
        assert "token" not in claimed.json()
        assert "workspace_root" not in claimed.json()
        renewed = client.post(
            f"/v1/sessions/{call_ids[0]}/runs/{call_ids[1]}/tool-calls/{call_ids[2]}/claim/renew",
            json={"client_id": lister["client_id"]},
            headers=auth_headers(),
        )
        assert renewed.status_code == 403
        assert renewed.json()["detail"]["code"] == "bridge_client_mismatch"
        released = client.post(
            f"/v1/sessions/{call_ids[0]}/runs/{call_ids[1]}/tool-calls/{call_ids[2]}/claim/release",
            json={"client_id": lister["client_id"]},
            headers=auth_headers(),
        )
        assert released.status_code == 403
        assert released.json()["detail"]["code"] == "bridge_client_mismatch"
        result = client.post(
            f"/v1/sessions/{call_ids[0]}/runs/{call_ids[1]}/tool-calls/{call_ids[2]}/result",
            json=result_body,
            headers={**auth_headers(), "Idempotency-Key": "result-lister"},
        )
        assert result.status_code == 403
        assert result.json()["detail"]["code"] == "bridge_client_mismatch"


def test_bridge_manifest_publication_is_client_bound_and_content_free(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    payload = _handshake_body("list_files", "read_file")
    entries = (
        {
            "path": "README.md",
            "sha256": "a" * 64,
            "size": 12,
            "entry_type": SnapshotEntryType.FILE.value,
        },
    )
    manifest = SnapshotManifest(
        entries=(
            SnapshotEntry(
                path="README.md",
                sha256="a" * 64,
                size=12,
                entry_type=SnapshotEntryType.FILE,
            ),
        )
    )
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/bridge/clients",
                json=_registration_body(payload),
                headers=auth_headers(),
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/v1/bridge/handshake", json=payload, headers=auth_headers()
            ).status_code
            == 200
        )
        published = client.post(
            "/v1/bridge/manifests",
            json={
                "snapshot_id": new_snapshot_id(),
                "client_id": payload["client_id"],
                "workspace_id": payload["workspace_id"],
                "entries": list(entries),
                "manifest_hash": manifest.manifest_hash,
            },
            headers=auth_headers(),
        )
        assert published.status_code == 201
        body = published.json()
        assert body["client_id"] == payload["client_id"]
        assert body["workspace_id"] == "ws_project"
        assert body["manifest_hash"] == manifest.manifest_hash
        assert body["file_count"] == 1
        assert "root" not in body
        assert "content" not in body
        assert "workspace_root" not in body
        forbidden = client.post(
            "/v1/bridge/manifests",
            json={
                "snapshot_id": new_snapshot_id(),
                "client_id": payload["client_id"],
                "workspace_id": payload["workspace_id"],
                "entries": list(entries),
                "workspace_root": "/tmp/project",
            },
            headers=auth_headers(),
        )
        assert forbidden.status_code in {400, 422}


def seed_running_bridge_call(
    database_path: Path,
    *,
    workspace_root: Path,
    call_id: str,
    owner_id: str = "prn_operator",
    tool_name: str = "read_file",
    effect: ToolEffect = ToolEffect.READ,
) -> tuple[str, str, str, str]:
    async def seed() -> tuple[str, str, str]:
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
                persisted_snapshot = await store.create_snapshot(
                    Snapshot(
                        snapshot_id=new_snapshot_id(),
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
                snapshot_id = persisted_snapshot.snapshot_id
            work_unit_id = new_work_unit_id()
            await store.create_work_unit(
                WorkUnit(
                    work_unit_id=work_unit_id,
                    run_id=run.run_id,
                    name="bridge-lease",
                    instruction="read the fixture README",
                    created_at=now,
                )
            )
            await store.create_attempt(
                Attempt(
                    attempt_id=new_attempt_id(),
                    run_id=run.run_id,
                    work_unit_id=work_unit_id,
                    role=ModelRole.WORKER,
                    model=ModelRef(provider="fake", model="fake-model"),
                    status=AttemptStatus.RUNNING,
                    created_at=now,
                    started_at=now,
                )
            )
            call = ToolCall(
                call_id=call_id,
                run_id=run.run_id,
                work_unit_id=work_unit_id,
                tool_name=tool_name,
                effect=effect,
                arguments={"path": "README.md"},
                snapshot_id=snapshot_id,
                requested_at=now,
            )
            await store.create_tool_call(
                call,
                workspace_id="ws_project",
                idempotency_key=f"{call_id}-call",
            )
            return session.session_id, run.run_id, call.call_id, snapshot_id

    return asyncio.run(seed())


def test_expired_claim_rejects_late_and_conflicting_results(tmp_path: Path) -> None:
    workspace_root = tmp_path / "bridge-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("fixture read\n", encoding="utf-8")
    app = build_app(tmp_path, workspace_root=workspace_root)
    database_path = tmp_path / "agent.db"
    expired_ids = seed_running_bridge_call(
        database_path, workspace_root=workspace_root, call_id="tc_bridge_expired"
    )
    replay_ids = seed_running_bridge_call(
        database_path, workspace_root=workspace_root, call_id="tc_bridge_replay"
    )
    released_ids = seed_running_bridge_call(
        database_path, workspace_root=workspace_root, call_id="tc_bridge_released"
    )
    result_body = {
        "dev_only": True,
        "status": ToolCallStatus.SUCCEEDED.value,
        "result": {"path": "README.md"},
        "output": "fixture read\n",
        "changed_paths": [],
        "exit_code": 0,
    }

    with TestClient(app) as client:
        payload = _handshake_body("list_files", "read_file")
        assert (
            client.post(
                "/v1/bridge/clients",
                json=_registration_body(payload),
                headers=auth_headers(),
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/v1/bridge/handshake", json=payload, headers=auth_headers()
            ).status_code
            == 200
        )
        result_body["client_id"] = payload["client_id"]
        store = app.state.store
        assert isinstance(store, SqliteStore)
        asyncio.run(store.start_tool_call(expired_ids[2]))
        asyncio.run(store.start_tool_call(replay_ids[2]))
        asyncio.run(store.start_tool_call(released_ids[2]))

        expired_claim = client.post(
            f"/v1/sessions/{expired_ids[0]}/runs/{expired_ids[1]}/tool-calls/{expired_ids[2]}/claim",
            json={"call_id": expired_ids[2], "client_id": payload["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-expired"},
        )
        assert expired_claim.status_code == 201
        expired_body = expired_claim.json()
        expire_at = __import__("datetime").datetime.fromisoformat(expired_body["expires_at"])
        asyncio.run(
            store.expire_bridge_claim(
                expired_body["claim_id"],
                principal_id="prn_operator",
                at=expire_at,
            )
        )
        late = client.post(
            f"/v1/sessions/{expired_ids[0]}/runs/{expired_ids[1]}/tool-calls/{expired_ids[2]}/result",
            json=result_body,
            headers={**auth_headers(), "Idempotency-Key": "result-expired"},
        )
        assert late.status_code == 409

        replay_claim = client.post(
            f"/v1/sessions/{replay_ids[0]}/runs/{replay_ids[1]}/tool-calls/{replay_ids[2]}/claim",
            json={"call_id": replay_ids[2], "client_id": payload["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-replay"},
        )
        assert replay_claim.status_code == 201
        first = client.post(
            f"/v1/sessions/{replay_ids[0]}/runs/{replay_ids[1]}/tool-calls/{replay_ids[2]}/result",
            json=result_body,
            headers={**auth_headers(), "Idempotency-Key": "result-replay"},
        )
        assert first.status_code == 200
        replay = client.post(
            f"/v1/sessions/{replay_ids[0]}/runs/{replay_ids[1]}/tool-calls/{replay_ids[2]}/result",
            json=result_body,
            headers={**auth_headers(), "Idempotency-Key": "result-replay-2"},
        )
        assert replay.status_code == 200
        assert replay.json()["status"] == first.json()["status"]
        conflict = client.post(
            f"/v1/sessions/{replay_ids[0]}/runs/{replay_ids[1]}/tool-calls/{replay_ids[2]}/result",
            json={**result_body, "output": "other"},
            headers={**auth_headers(), "Idempotency-Key": "result-conflict"},
        )
        assert conflict.status_code == 409

        released_claim = client.post(
            f"/v1/sessions/{released_ids[0]}/runs/{released_ids[1]}/tool-calls/{released_ids[2]}/claim",
            json={"call_id": released_ids[2], "client_id": payload["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-released"},
        )
        assert released_claim.status_code == 201
        released = client.post(
            f"/v1/sessions/{released_ids[0]}/runs/{released_ids[1]}/tool-calls/{released_ids[2]}/claim/release",
            json={"client_id": payload["client_id"]},
            headers=auth_headers(),
        )
        assert released.status_code == 200
        assert released.json()["status"] == "RELEASED"
        after_release = client.post(
            f"/v1/sessions/{released_ids[0]}/runs/{released_ids[1]}/tool-calls/{released_ids[2]}/result",
            json=result_body,
            headers={**auth_headers(), "Idempotency-Key": "result-released"},
        )
        assert after_release.status_code == 409


def test_mutating_bridge_claim_requires_named_ready_snapshot(tmp_path: Path) -> None:
    workspace_root = tmp_path / "bridge-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("fixture read\n", encoding="utf-8")
    app = build_app(tmp_path, workspace_root=workspace_root)
    database_path = tmp_path / "agent.db"
    session_id, run_id, call_id, snapshot_id = seed_running_bridge_call(
        database_path,
        workspace_root=workspace_root,
        call_id="tc_bridge_write",
        tool_name="apply_patch",
        effect=ToolEffect.WRITE,
    )
    payload = _handshake_body("apply_patch", effects=("WRITE",))
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/bridge/clients",
                json=_registration_body(payload),
                headers=auth_headers(),
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/v1/bridge/handshake", json=payload, headers=auth_headers()
            ).status_code
            == 200
        )
        store = app.state.store
        assert isinstance(store, SqliteStore)
        asyncio.run(store.start_tool_call(call_id))
        stale = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
            json={
                "call_id": call_id,
                "client_id": payload["client_id"],
                "snapshot_id": new_snapshot_id(),
            },
            headers={**auth_headers(), "Idempotency-Key": "claim-write-stale"},
        )
        assert stale.status_code == 409
        claimed = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
            json={
                "call_id": call_id,
                "client_id": payload["client_id"],
                "snapshot_id": snapshot_id,
            },
            headers={**auth_headers(), "Idempotency-Key": "claim-write-ready"},
        )
        assert claimed.status_code == 201
        assert claimed.json()["snapshot_id"] == snapshot_id


def test_event_cursor_rejects_malformed_and_negative_values(tmp_path: Path) -> None:
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
        malformed = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/events",
            headers={**auth_headers(), "Last-Event-ID": "not-a-cursor"},
        )
        assert malformed.status_code in {400, 422}
        negative = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/events",
            params={"after": -1},
            headers=auth_headers(),
        )
        assert negative.status_code in {400, 422}
        other_session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project"},
        ).json()["session_id"]
        crossed = client.get(
            f"/v1/sessions/{other_session}/runs/{run_id}/events",
            headers=auth_headers(),
        )
        assert crossed.status_code in {403, 404}


def test_public_session_events_are_monotonic_redacted_and_scope_closed(tmp_path: Path) -> None:
    import json

    workspace_root = tmp_path / "event-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("ok\n", encoding="utf-8")
    with TestClient(build_app(tmp_path, workspace_root=workspace_root)) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        session_id = session["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        wait_for_terminal(client, session_id, run_id)
        events = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/events",
            headers=auth_headers(),
        )
        assert events.status_code == 200
        records = [
            json.loads(line.removeprefix("data: "))
            for line in events.text.splitlines()
            if line.startswith("data: ")
        ]
        sequences = [record["sequence"] for record in records]
        assert sequences == list(range(1, len(sequences) + 1))
        assert all(record["run_id"] == run_id for record in records)
        forbidden = {
            "api_key",
            "apikey",
            "token",
            "secret",
            "reasoning",
            "root",
            "credential",
            "authorization",
            "raw_response",
            "provider_body",
        }

        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    assert key.lower() not in forbidden
                    walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    walk(nested)

        for record in records:
            walk(record.get("payload") or {})
        serialized = events.text
        assert "agent-secret" not in serialized
        assert str(workspace_root) not in serialized
        assert "https://models.internal" not in serialized
        other_session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project"},
        ).json()["session_id"]
        crossed = client.get(
            f"/v1/sessions/{other_session}/runs/{run_id}/events",
            headers=auth_headers(),
        )
        assert crossed.status_code in {403, 404}
        assert client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/events"
        ).status_code == 401


def test_cloud_bundle_exports_authorized_verified_files(tmp_path: Path) -> None:
    workspace_root = tmp_path / "cloud-workspace"
    workspace_root.mkdir()
    (workspace_root / "result.txt").write_text("cloud-ok\n", encoding="utf-8")
    (workspace_root / ".env").write_text("SECRET=nope\n", encoding="utf-8")
    git_dir = workspace_root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git-internal\n", encoding="utf-8")
    before = (workspace_root / "result.txt").read_bytes()

    with TestClient(build_app(tmp_path, workspace_root=workspace_root)) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        session_id = session["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        finished = wait_for_terminal(client, session_id, run_id)
        assert finished["status"] == RunStatus.SUCCEEDED.value

        unauthorized = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/bundle"
        )
        assert unauthorized.status_code == 401

        bundle = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/bundle",
            headers=auth_headers(),
        )
        assert bundle.status_code == 200, bundle.text
        body = bundle.json()
        paths = [item["path"] for item in body["files"]]
        assert "result.txt" in paths
        assert ".env" not in paths
        assert all(".git" not in path for path in paths)
        assert "SECRET=nope" not in bundle.text
        assert "git-internal" not in bundle.text
        assert str(workspace_root) not in bundle.text
        assert body["workspace_id"] == "ws_project"
        assert body["file_count"] == len(body["files"])

    assert (workspace_root / "result.txt").read_bytes() == before
    assert (workspace_root / ".env").read_text(encoding="utf-8") == "SECRET=nope\n"


def test_cloud_bundle_rejects_unverified_and_oversize_export(tmp_path: Path) -> None:
    with TestClient(build_app(tmp_path)) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        session_id = session["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        wait_for_terminal(client, session_id, run_id)
        missing = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/bundle",
            headers=auth_headers(),
        )
        assert missing.status_code == 409
        assert missing.json()["detail"]["code"] == "bundle_unverified"

    workspace_root = tmp_path / "oversize-workspace"
    workspace_root.mkdir()
    (workspace_root / "huge.txt").write_text("x" * 64, encoding="utf-8")
    database_path = tmp_path / "oversize.db"
    seed_workspace(database_path, "prn_operator", workspace_root)
    settings = Settings(
        database_path=database_path,
        worker_profile=WORKER_PROFILE,
        service_token=SecretStr("agent-secret"),
        service_principal="prn_operator",
        workspace_roots={"project-main": str(workspace_root)},
        export_max_bytes=8,
    )
    with TestClient(create_app(settings, adapters={"worker": FakeAdapter()})) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        session_id = session["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        wait_for_terminal(client, session_id, run_id)
        oversize = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/bundle",
            headers=auth_headers(),
        )
        assert oversize.status_code == 409
        assert oversize.json()["detail"]["code"] == "bundle_oversize"


def test_cloud_bundle_rejects_cross_session_and_keeps_bridge_separate(tmp_path: Path) -> None:
    workspace_root = tmp_path / "scope-workspace"
    workspace_root.mkdir()
    (workspace_root / "result.txt").write_text("cloud-ok\n", encoding="utf-8")
    app = build_app(tmp_path, workspace_root=workspace_root)
    database_path = tmp_path / "agent.db"

    with TestClient(app) as client:
        first = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        second = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        first_run = client.post(
            f"/v1/sessions/{first['session_id']}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        second_run = client.post(
            f"/v1/sessions/{second['session_id']}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        wait_for_terminal(client, first["session_id"], first_run)
        wait_for_terminal(client, second["session_id"], second_run)

        crossed = client.get(
            f"/v1/sessions/{second['session_id']}/runs/{first_run}/bundle",
            headers=auth_headers(),
        )
        assert crossed.status_code in {403, 404}
        assert "cloud-ok" not in crossed.text
        assert "files" not in crossed.json().get("detail", {})

        missing = client.get(
            f"/v1/sessions/{first['session_id']}/runs/run_missing/bundle",
            headers=auth_headers(),
        )
        assert missing.status_code == 404

        allowed = client.get(
            f"/v1/sessions/{first['session_id']}/runs/{first_run}/bundle",
            headers=auth_headers(),
        )
        assert allowed.status_code == 200, allowed.text
        assert allowed.json()["file_count"] >= 1

        handshake = client.post("/v1/bridge/handshake", json={"workspace_id": "ws_project"})
        assert handshake.status_code == 401
        assert client.get("/v1/bridge/bundle", headers=auth_headers()).status_code == 404

    async def snapshot_count() -> int:
        async with SqliteStore(database_path) as store:
            snapshots = await store.list_snapshots("ws_project", owner_id="prn_operator")
            return len(snapshots)

    assert asyncio.run(snapshot_count()) >= 1
    assert (workspace_root / "result.txt").read_text(encoding="utf-8") == "cloud-ok\n"


def test_cloud_bundle_rejects_later_edits_and_other_run_snapshot(tmp_path: Path) -> None:
    workspace_root = tmp_path / "cloud-workspace"
    workspace_root.mkdir()
    (workspace_root / "result.txt").write_text("cloud-ok\n", encoding="utf-8")
    (workspace_root / "keep.txt").write_text("frozen\n", encoding="utf-8")
    app = build_app(tmp_path, workspace_root=workspace_root)
    database_path = tmp_path / "agent.db"

    async def ready_snapshots():
        async with SqliteStore(database_path) as store:
            return await store.list_snapshots("ws_project", owner_id="prn_operator")

    with TestClient(app) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={"workspace_id": "ws_project", "access": ["READ"]},
        ).json()
        session_id = session["session_id"]
        first_run = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        finished = wait_for_terminal(client, session_id, first_run)
        assert finished["status"] == RunStatus.SUCCEEDED.value
        original = next(
            item for item in asyncio.run(ready_snapshots()) if item.status is SnapshotStatus.READY
        )

        (workspace_root / "result.txt").write_text("later-edit\n", encoding="utf-8")
        (workspace_root / "later.txt").write_text("from-another-run\n", encoding="utf-8")
        later_at = utc_now()

        async def persist_later_snapshot() -> str:
            async with SqliteStore(database_path) as store:
                with WorkspaceBackend(workspace_root) as backend:
                    manifest, file_contents = backend.capture_snapshot()
                    persisted = await store.create_snapshot(
                        Snapshot(
                            snapshot_id=new_snapshot_id(),
                            workspace_id="ws_project",
                            status=SnapshotStatus.READY,
                            created_at=later_at,
                            completed_at=later_at,
                            file_count=len(manifest.entries),
                            total_size=manifest.total_size,
                        ),
                        manifest,
                        owner_id="prn_operator",
                        file_contents=file_contents,
                    )
                    return persisted.snapshot_id

        later_snapshot_id = asyncio.run(persist_later_snapshot())
        keep_before = (workspace_root / "keep.txt").read_bytes()
        result_before = (workspace_root / "result.txt").read_bytes()
        later_before = (workspace_root / "later.txt").read_bytes()
        second_run = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello again"},
        ).json()["run_id"]
        second_finished = wait_for_terminal(client, session_id, second_run)
        assert second_finished["status"] == RunStatus.SUCCEEDED.value
        assert later_snapshot_id != original.snapshot_id

        bundle = client.get(
            f"/v1/sessions/{session_id}/runs/{first_run}/bundle",
            headers=auth_headers(),
        )
        assert bundle.status_code == 200, bundle.text
        body = bundle.json()
        paths = [item["path"] for item in body["files"]]
        contents = "".join(item["content"] for item in body["files"])
        assert body["run_id"] == first_run
        assert body["snapshot_id"] == original.snapshot_id
        assert body["snapshot_id"] != later_snapshot_id
        assert "keep.txt" in paths
        assert "result.txt" in paths
        assert "frozen\n" in contents
        assert "cloud-ok\n" in contents
        assert "later.txt" not in paths
        assert "later-edit" not in bundle.text
        assert "from-another-run" not in bundle.text
        assert (workspace_root / "keep.txt").read_bytes() == keep_before
        assert (workspace_root / "result.txt").read_bytes() == result_before
        assert (workspace_root / "later.txt").read_bytes() == later_before


def test_cloud_bundle_rejects_local_export_and_keeps_bridge_route_absent(tmp_path: Path) -> None:
    workspace_root = tmp_path / "local-workspace"
    workspace_root.mkdir()
    (workspace_root / "result.txt").write_text("local-secret\n", encoding="utf-8")
    with TestClient(build_app(tmp_path, workspace_root=workspace_root)) as client:
        session = client.post(
            "/v1/sessions",
            headers=auth_headers(),
            json={
                "workspace_id": "ws_project",
                "access": ["READ"],
                "agent_options": {"execution_location": "LOCAL"},
            },
        )
        assert session.status_code == 201, session.text
        session_id = session.json()["session_id"]
        run_id = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=auth_headers(),
            json={"input": "hello"},
        ).json()["run_id"]
        rejected = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/bundle",
            headers=auth_headers(),
        )
        assert rejected.status_code == 409
        assert rejected.json()["detail"]["code"] == "bundle_not_cloud"
        assert "local-secret" not in rejected.text
        assert client.get("/v1/bridge/bundle", headers=auth_headers()).status_code == 404


def test_bridge_lease_limits_are_server_owned(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    from prp_runtime.domain.models import (
        BRIDGE_HEARTBEAT_CADENCE_SECONDS,
        MAX_BRIDGE_HEARTBEAT_TTL_SECONDS,
        MAX_BRIDGE_LEASE_RENEWS,
        MAX_BRIDGE_LEASE_SECONDS,
        MAX_BRIDGE_LEASE_TOTAL_SECONDS,
    )

    workspace_root = tmp_path / "bridge-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("fixture read\n", encoding="utf-8")
    app = build_app(tmp_path, workspace_root=workspace_root)
    database_path = tmp_path / "agent.db"
    ids = seed_running_bridge_call(
        database_path, workspace_root=workspace_root, call_id="tc_bridge_keepalive"
    )
    start = utc_now() + timedelta(seconds=1)
    current = {"now": start}

    def fake_now() -> datetime:
        return current["now"]

    with TestClient(app) as client, patch(
        "prp_runtime.api.native_agent.utc_now", fake_now
    ):
        payload = _handshake_body("list_files", "read_file")
        assert (
            client.post(
                "/v1/bridge/clients",
                json=_registration_body(payload),
                headers=auth_headers(),
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/v1/bridge/handshake", json=payload, headers=auth_headers()
            ).status_code
            == 200
        )
        beat = client.post(
            f"/v1/bridge/clients/{payload['client_id']}/heartbeat",
            json={"fingerprint": payload["fingerprint"], "ttl_seconds": 9_999},
            headers=auth_headers(),
        )
        assert beat.status_code in {400, 422}
        beat = client.post(
            f"/v1/bridge/clients/{payload['client_id']}/heartbeat",
            json={"fingerprint": payload["fingerprint"]},
            headers=auth_headers(),
        )
        assert beat.status_code == 200
        body = beat.json()
        assert body["liveness"] == "LIVE"
        assert body["ttl_seconds"] == MAX_BRIDGE_HEARTBEAT_TTL_SECONDS
        assert body["cadence_seconds"] == BRIDGE_HEARTBEAT_CADENCE_SECONDS
        assert body["lease_seconds"] == MAX_BRIDGE_LEASE_SECONDS
        assert body["lease_total_seconds"] == MAX_BRIDGE_LEASE_TOTAL_SECONDS
        assert body["max_renews"] == MAX_BRIDGE_LEASE_RENEWS
        assert "token" not in body
        store = app.state.store
        assert isinstance(store, SqliteStore)
        asyncio.run(store.start_tool_call(ids[2]))
        claimed = client.post(
            f"/v1/sessions/{ids[0]}/runs/{ids[1]}/tool-calls/{ids[2]}/claim",
            json={"call_id": ids[2], "client_id": payload["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-keepalive"},
        )
        assert claimed.status_code == 201, claimed.text
        claimed_body = claimed.json()
        claimed_at = datetime.fromisoformat(claimed_body["claimed_at"])
        expires_at = datetime.fromisoformat(claimed_body["expires_at"])
        assert expires_at - claimed_at <= timedelta(seconds=MAX_BRIDGE_LEASE_SECONDS)
        too_soon = client.post(
            f"/v1/sessions/{ids[0]}/runs/{ids[1]}/tool-calls/{ids[2]}/claim/renew",
            json={"client_id": payload["client_id"], "extend_seconds": 9_999},
            headers=auth_headers(),
        )
        assert too_soon.status_code in {400, 409, 422}
        current["now"] = start + timedelta(seconds=MAX_BRIDGE_LEASE_SECONDS - 1)
        renews = 0
        last_status = None
        for index in range(MAX_BRIDGE_LEASE_RENEWS + 1):
            current["now"] = start + timedelta(
                seconds=MAX_BRIDGE_LEASE_SECONDS - 1 + (index * 30)
            )
            renewed = client.post(
                f"/v1/sessions/{ids[0]}/runs/{ids[1]}/tool-calls/{ids[2]}/claim/renew",
                json={"client_id": payload["client_id"]},
                headers=auth_headers(),
            )
            last_status = renewed.status_code
            if renewed.status_code == 200:
                renews += 1
                new_expiry = datetime.fromisoformat(renewed.json()["expires_at"])
                assert new_expiry - claimed_at <= timedelta(
                    seconds=MAX_BRIDGE_LEASE_TOTAL_SECONDS
                )
            else:
                break
        assert renews == MAX_BRIDGE_LEASE_RENEWS
        assert last_status == 409

def test_bridge_result_wakes_owning_run_once(tmp_path: Path) -> None:
    workspace_root = tmp_path / "bridge-workspace"
    workspace_root.mkdir()
    (workspace_root / "README.md").write_text("fixture read\n", encoding="utf-8")
    app = build_app(tmp_path, workspace_root=workspace_root)
    database_path = tmp_path / "agent.db"
    session_id, run_id, call_id, _snapshot_id = seed_running_bridge_call(
        database_path, workspace_root=workspace_root, call_id="tc_bridge_wake"
    )
    result_body = {
        "dev_only": True,
        "status": ToolCallStatus.SUCCEEDED.value,
        "result": {"path": "README.md"},
        "output": "fixture read\n",
        "changed_paths": [],
        "exit_code": 0,
    }
    with TestClient(app) as client:
        payload = _handshake_body("list_files", "read_file")
        assert (
            client.post(
                "/v1/bridge/clients",
                json=_registration_body(payload),
                headers=auth_headers(),
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/v1/bridge/handshake", json=payload, headers=auth_headers()
            ).status_code
            == 200
        )
        result_body["client_id"] = payload["client_id"]
        store = app.state.store
        assert isinstance(store, SqliteStore)
        asyncio.run(store.start_tool_call(call_id))
        claimed = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
            json={"call_id": call_id, "client_id": payload["client_id"]},
            headers={**auth_headers(), "Idempotency-Key": "claim-wake"},
        )
        assert claimed.status_code == 201, claimed.text
        first = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/result",
            json=result_body,
            headers={**auth_headers(), "Idempotency-Key": "result-wake"},
        )
        assert first.status_code == 200, first.text
        enqueued_after_first = app.state.supervisor.state.enqueued
        second = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/result",
            json=result_body,
            headers={**auth_headers(), "Idempotency-Key": "result-wake-2"},
        )
        assert second.status_code == 200, second.text
        assert app.state.supervisor.state.enqueued == enqueued_after_first
        events = asyncio.run(store.list_events(run_id))
        wakes = [event for event in events if event.event_type is EventType.BRIDGE_RESULT_WAKE]
        assert len(wakes) == 1
        assert wakes[0].payload["call_id"] == call_id
        assert wakes[0].payload["status"] == ToolCallStatus.SUCCEEDED.value
        call = asyncio.run(store.get_tool_call(call_id))
        artifacts = asyncio.run(store.list_artifacts(call.work_unit_id))
        evidence = asyncio.run(store.list_evidence(call.work_unit_id))
        assert len(artifacts) == 1
        assert artifacts[0].name == "bridge-result"
        assert evidence[0].rule == "bridge.result.observation"
        assert evidence[0].artifact_id == artifacts[0].artifact_id
