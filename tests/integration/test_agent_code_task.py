"""Minimal cloud code-agent read path through the real ToolExecutor."""

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic import SecretStr

from prp_runtime.app import create_app
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
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import (
    AgentRequestOptions,
    AgentToolCall,
    AgentToolResult,
    Budget,
    NativeRunRequest,
    Session,
    Usage,
    WorkspaceGrant,
)
from prp_runtime.domain.values import (
    new_session_id,
    new_snapshot_id,
    new_tool_call_id,
    new_workspace_id,
    utc_now,
)
from prp_runtime.policy.engine import PolicyOutcome, PolicyReasonCode
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.runtime.agent_loop import (
    AgentLoop,
    AgentLoopStatus,
    AgentToolContext,
    AgentToolExecution,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.tools import ToolExecutor, ToolRegistry
from prp_runtime.tools.command import (
    CommandClass,
    CommandParameter,
    CommandRegistry,
    CommandRunner,
    CommandSpec,
    build_targeted_test_definition,
)
from prp_runtime.tools.diff import DiffToolRunner, build_diff_definitions
from prp_runtime.tools.filesystem import build_filesystem_registry
from prp_runtime.tools.models import ToolCall
from prp_runtime.tools.patch import PatchRunner, build_patch_definition
from prp_runtime.tools.search import (
    SearchRunner,
    build_search_definition,
    require_rg,
)
from prp_runtime.workspace import WorkspaceBackend
from prp_runtime.workspace.changes import ChangeSet
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotManifest,
    SnapshotStatus,
    Workspace,
    WorkspaceSource,
    WorkspaceSourceType,
)


def _profile() -> ModelProfile:
    return ModelProfile(
        alias="worker",
        provider="fixture",
        model="read-only-fixture",
        role=ModelRole.WORKER,
        base_url="https://fixture.invalid/v1",
        context_window_tokens=8_000,
        max_output_tokens=1_000,
    )


class ReadPathAdapter:
    """A provider fixture that only observes public tool results in history."""

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "read-path-fixture"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        call_id = new_tool_call_id()
        usage = Usage(input_tokens=1, output_tokens=1, elapsed_ms=1)
        if len(self.requests) == 1:
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=call_id,
                        tool_name="list_files",
                        arguments={"path": ""},
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )
        previous = request.history[-1]
        assert isinstance(previous, AgentToolResult)
        assert previous.status is ToolCallStatus.SUCCEEDED
        if len(self.requests) == 2:
            assert previous.result is not None
            assert any(
                entry["path"] == "src"
                for entry in previous.result["entries"]  # type: ignore[index]
            )
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=call_id,
                        tool_name="search_text",
                        arguments={
                            "pattern": "needle",
                            "root": "src",
                            "glob": "*.py",
                        },
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )
        if len(self.requests) == 3:
            assert previous.result is not None
            assert previous.result["matches"][0]["path"] == "main.py"  # type: ignore[index]
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=call_id,
                        tool_name="read_file",
                        arguments={"path": "src/main.py"},
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )
        assert len(self.requests) == 4
        assert previous.result is not None
        assert previous.result["content"] == 'def answer():\n    return "needle"\n'  # type: ignore[index]
        return ProviderResponse(
            text="read src/main.py and found the requested implementation",
            usage=usage,
            finish_reason=FinishReason.STOP,
        )


class PatchTestAdapter:
    """A fixture provider that only advances from public tool results."""

    def __init__(self, *, base_snapshot_id: str) -> None:
        self._base_snapshot_id = base_snapshot_id
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "patch-test-fixture"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        call_id = new_tool_call_id()
        usage = Usage(input_tokens=1, output_tokens=1, elapsed_ms=1)
        if len(self.requests) == 1:
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=call_id,
                        tool_name="apply_patch",
                        arguments={
                            "patch": {
                                "base_snapshot_id": self._base_snapshot_id,
                                "unified_diff": (
                                    "--- a/src/main.py\n"
                                    "+++ b/src/main.py\n"
                                    "@@ -1,2 +1,2 @@\n"
                                    " def answer():\n"
                                    '-    return "needle"\n'
                                    '+    return "patched"\n'
                                ),
                            }
                        },
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )

        previous = request.history[-1]
        assert isinstance(previous, AgentToolResult)
        assert previous.status is ToolCallStatus.SUCCEEDED
        assert previous.result is not None
        if len(self.requests) == 2:
            assert previous.result["changed_paths"] == ["src/main.py"]  # type: ignore[index]
            new_snapshot_id = previous.result["new_snapshot_id"]  # type: ignore[index]
            assert isinstance(new_snapshot_id, str)
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=call_id,
                        tool_name="run_targeted_test",
                        arguments={
                            "spec_name": "fixture_patch_test",
                            "parameters": {"mode": "verify_patch"},
                        },
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )

        if len(self.requests) == 3:
            assert previous.result["exit_code"] == 0  # type: ignore[index]
            assert "targeted test passed" in previous.result["stdout"]  # type: ignore[index]
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=call_id,
                        tool_name="get_diff",
                        arguments={},
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )

        assert len(self.requests) == 4
        assert previous.result["entries"][0]["path"] == "src/main.py"  # type: ignore[index]
        assert previous.result["entries"][0]["status"] == "MODIFIED"  # type: ignore[index]
        return ProviderResponse(
            text="patched src/main.py, verified it, and reviewed the diff",
            usage=usage,
            finish_reason=FinishReason.STOP,
        )


class ProductionPatchTestAdapter:
    """Drive the production app through approval, patch, pytest and diff."""

    def __init__(self, *, base_snapshot_id: str) -> None:
        self._base_snapshot_id = base_snapshot_id
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "production-patch-test-fixture"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        usage = Usage(input_tokens=1, output_tokens=1, elapsed_ms=1)
        if len(self.requests) == 1:
            assert "get_diff" in {tool.name for tool in request.tools}
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=new_tool_call_id(),
                        tool_name="apply_patch",
                        arguments={
                            "patch": {
                                "base_snapshot_id": self._base_snapshot_id,
                                "unified_diff": (
                                    "--- a/src/main.py\n"
                                    "+++ b/src/main.py\n"
                                    "@@ -1,2 +1,2 @@\n"
                                    " def answer():\n"
                                    '-    return "needle"\n'
                                    '+    return "patched"\n'
                                ),
                            }
                        },
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )

        previous = request.history[-1]
        assert isinstance(previous, AgentToolResult)
        assert previous.status is ToolCallStatus.SUCCEEDED
        assert previous.result is not None
        if len(self.requests) == 2:
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=new_tool_call_id(),
                        tool_name="run_targeted_test",
                        arguments={
                            "spec_name": "pytest",
                            "parameters": {"targets": ["test_fixture.py"]},
                        },
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )
        if len(self.requests) == 3:
            assert previous.result["exit_code"] == 0  # type: ignore[index]
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=new_tool_call_id(),
                        tool_name="get_diff",
                        arguments={},
                    ),
                ),
                usage=usage,
                finish_reason=FinishReason.TOOL_CALLS,
            )
        assert len(self.requests) == 4
        assert previous.result["entries"][0]["path"] == "src/main.py"  # type: ignore[index]
        return ProviderResponse(
            text="production patch, test and diff completed",
            usage=usage,
            finish_reason=FinishReason.STOP,
        )


class DeferredDiffRunner:
    """Bind the persisted ChangeSet after the preceding patch tool succeeds."""

    def __init__(
        self,
        base_manifest: SnapshotManifest,
        manifest_provider: Callable[[], SnapshotManifest],
    ) -> None:
        self._base_manifest = base_manifest
        self._manifest_provider = manifest_provider
        self._runner: DiffToolRunner | None = None

    def bind(self, change_set: ChangeSet) -> None:
        self._runner = DiffToolRunner(
            change_set,
            base_manifest=self._base_manifest,
            manifest_provider=self._manifest_provider,
        )

    def get_diff(self):
        assert self._runner is not None
        return self._runner.get_diff()

    def get_status(self):
        assert self._runner is not None
        return self._runner.get_status()


class FailedTargetedTestAdapter:
    """A provider fixture that tries to finalize after a failed test result."""

    @property
    def name(self) -> str:
        return "failed-test-fixture"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        if not request.history:
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=new_tool_call_id(),
                        tool_name="run_targeted_test",
                        arguments={"spec_name": "fixture_patch_test"},
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        return ProviderResponse(
            text="the targeted test passed",
            finish_reason=FinishReason.STOP,
        )


class FailedTargetedTestExecutor:
    async def execute(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        del context
        return AgentToolExecution(
            call=call,
            result=AgentToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.FAILED,
                result={"exit_code": 1},
                output="targeted test failed",
            ),
        )


class WorkspaceAgentExecutor:
    """Adapt AgentLoop calls to the persisted, policy-gated ToolExecutor."""

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        workspace_id: str,
        snapshot_id: str,
        approve_writes: bool = False,
        store: SqliteStore | None = None,
        diff_runner: DeferredDiffRunner | None = None,
    ) -> None:
        self._executor = executor
        self._workspace_id = workspace_id
        self._snapshot_id = snapshot_id
        self._approve_writes = approve_writes
        self._store = store
        self._diff_runner = diff_runner
        self.calls: list[ToolCall] = []
        self.decisions = []

    async def execute(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        effect = {
            "apply_patch": ToolEffect.WRITE,
            "run_targeted_test": ToolEffect.COMMAND,
        }.get(call.tool_name, ToolEffect.READ)
        persisted_call = ToolCall(
            call_id=call.call_id,
            run_id=context.run_id,
            work_unit_id=context.work_unit_id,
            tool_name=call.tool_name,
            effect=effect,
            arguments=call.arguments,
            snapshot_id=self._snapshot_id,
            requested_at=utc_now(),
        )
        self.calls.append(persisted_call)
        outcome = await self._executor.execute(
            persisted_call,
            context.mode,
            workspace_id=self._workspace_id,
            idempotency_key=call.call_id,
            command_class=(
                CommandClass.TEST if effect is ToolEffect.COMMAND else None
            ),
            approved=(
                True
                if effect is ToolEffect.WRITE and self._approve_writes
                else None
            ),
        )
        self.decisions.append(outcome.decision)
        if outcome.result is None:
            assert outcome.call.status is ToolCallStatus.AWAITING_APPROVAL
            return AgentToolExecution(
                call=call,
                awaiting_approval=True,
                reason=outcome.decision.reason_code.value,
            )
        result = outcome.result
        if call.tool_name == "apply_patch":
            assert result.result is not None
            new_snapshot_id = result.result.get("new_snapshot_id")
            assert isinstance(new_snapshot_id, str)
            self._snapshot_id = new_snapshot_id
            change_set_id = result.result.get("change_set_id")
            if self._store is not None and self._diff_runner is not None:
                assert isinstance(change_set_id, str)
                self._diff_runner.bind(
                    await self._store.get_change_set(change_set_id)
                )
        return AgentToolExecution(
            call=call,
            result=AgentToolResult(
                call_id=call.call_id,
                status=result.status,
                result=result.result,
                output=result.output,
                truncated=result.truncated,
            ),
        )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "agent-code-task.db") as opened:
        yield opened


@pytest.mark.asyncio
async def test_read_search_agent_path_uses_workspace_tools_and_audited_session_run(
    store: SqliteStore,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "src").mkdir(parents=True)
    (workspace_root / "README.md").write_text("fixture workspace\n", encoding="utf-8")
    source = 'def answer():\n    return "needle"\n'
    (workspace_root / "src" / "main.py").write_text(source, encoding="utf-8")

    owner_id = "prn_agent_e2e"
    workspace_id = new_workspace_id()
    snapshot_id = new_snapshot_id()
    now = utc_now()
    workspace = Workspace(
        workspace_id=workspace_id,
        owner_id=owner_id,
        alias="agent-e2e",
        source=WorkspaceSource(
            source_type=WorkspaceSourceType.SERVER_ALIAS,
            server_alias="agent-fixture",
        ),
        created_at=now,
    )
    await store.create_workspace(workspace)
    with WorkspaceBackend(workspace_root) as backend:
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            workspace_id=workspace_id,
            status=SnapshotStatus.READY,
            created_at=now,
            completed_at=now,
        )
        await store.create_snapshot(
            snapshot,
            backend.snapshot_manifest(),
            owner_id=owner_id,
        )
        search = SearchRunner(
            backend,
            workspace_cwd=workspace_root,
            rg_path=require_rg(),
        )
        filesystem = build_filesystem_registry(backend)
        registry = ToolRegistry(
            (*filesystem.definitions, build_search_definition(search))
        )
        tool_executor = ToolExecutor(registry, store)
        agent_executor = WorkspaceAgentExecutor(
            tool_executor,
            workspace_id=workspace_id,
            snapshot_id=snapshot_id,
        )
        adapter = ReadPathAdapter()
        settings = Settings(worker_profile=_profile())
        controller = RunController(
            store,
            settings,
            {"worker": adapter},
            tool_executor=agent_executor,
        )
        session = Session(
            session_id=new_session_id(),
            principal_id=owner_id,
            workspace_id=workspace_id,
            grant=WorkspaceGrant(
                principal_id=owner_id,
                workspace_id=workspace_id,
            ),
            agent_options=AgentRequestOptions(
                agent_mode=AgentMode.PLAN,
                isolation_mode=IsolationMode.SANDBOXED,
                execution_location=ExecutionLocation.CLOUD,
            ),
            created_at=now,
        )
        await store.create_session(session)
        run = await controller.create_run(
            NativeRunRequest(
                input="inspect the fixture implementation",
                routing_policy=RoutingPolicy.MANUAL,
                strategy=ExecutionStrategy.DIRECT,
                agent_options=session.agent_options,
                budget=Budget(max_attempts=8, max_total_tokens=16),
            )
        )
        await store.attach_run_to_session(
            session.session_id,
            run.run_id,
            principal_id=owner_id,
        )
        finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED, finished.error
    scoped = await store.get_run_for_session(
        session.session_id,
        run.run_id,
        principal_id=owner_id,
    )
    assert scoped.status is RunStatus.SUCCEEDED
    assert [call.tool_name for call in agent_executor.calls] == [
        "list_files",
        "search_text",
        "read_file",
    ]
    assert len(adapter.requests) == 4

    tool_calls = await store.list_tool_calls(run.run_id)
    assert [call.tool_name for call in tool_calls] == [
        "list_files",
        "search_text",
        "read_file",
    ]
    assert all(call.status is ToolCallStatus.SUCCEEDED for call in tool_calls)
    results = [await store.get_tool_result(call.call_id) for call in tool_calls]
    assert all(result.status is ToolCallStatus.SUCCEEDED for result in results)
    assert results[1].result is not None
    assert results[1].result["matches"][0]["path"] == "main.py"  # type: ignore[index]

    events = await store.list_events(run.run_id)
    assert assert_sequence_chain(events) is None
    event_types = [event.event_type for event in events]
    assert event_types.count(EventType.TOOL_CALL_REQUESTED) == 3
    assert event_types.count(EventType.TOOL_CALL_SUCCEEDED) == 3
    assert EventType.ARTIFACT_PRODUCED in event_types
    assert str(workspace_root) not in json.dumps(
        [event.payload for event in events],
        ensure_ascii=True,
    )


def test_production_patch_test_agent_path_binds_diff_after_approval(
    tmp_path: Path,
) -> None:
    """Keep the production catalog stable while binding diff after the patch."""
    from tests.integration.test_agent_api import seed_workspace

    database_path = tmp_path / "production-agent.db"
    workspace_root = tmp_path / "production-workspace"
    (workspace_root / "src").mkdir(parents=True)
    target = workspace_root / "src" / "main.py"
    target.write_text('def answer():\n    return "needle"\n', encoding="utf-8")
    (workspace_root / "test_fixture.py").write_text(
        'from src.main import answer\n\n\ndef test_answer():\n    assert answer() == "patched"\n',
        encoding="utf-8",
    )
    owner_id = "prn_production_patch"
    seed_workspace(database_path, owner_id, workspace_root)

    async def initial_snapshot_id() -> str:
        async with SqliteStore(database_path) as store:
            snapshots = await store.list_snapshots("ws_project", owner_id=owner_id)
            assert len(snapshots) == 1
            return snapshots[0].snapshot_id

    base_snapshot_id = asyncio.run(initial_snapshot_id())
    adapter = ProductionPatchTestAdapter(base_snapshot_id=base_snapshot_id)
    profile = _profile()
    app = create_app(
        Settings(
            database_path=database_path,
            worker_profile=profile,
            service_token=SecretStr("production-agent-token"),
            service_principal=owner_id,
            workspace_roots={"project-main": str(workspace_root)},
        ),
        adapters={profile.alias: adapter},
    )
    headers = {"Authorization": "Bearer production-agent-token"}

    with TestClient(app) as client:
        session_response = client.post(
            "/v1/sessions",
            headers=headers,
            json={
                "workspace_id": "ws_project",
                "access": ["READ", "WRITE"],
                "agent_options": {
                    "agent_mode": "AUTO",
                    "isolation_mode": "HOST",
                    "execution_location": "CLOUD",
                },
            },
        )
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]
        run_response = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=headers,
            json={
                "input": "patch the fixture, run its targeted test, then inspect the diff",
                "routing_policy": "MANUAL",
                "strategy": "DIRECT",
                "budget": {"max_attempts": 8, "max_total_tokens": 32},
            },
        )
        assert run_response.status_code == 202
        run_id = run_response.json()["run_id"]

        approval = None
        for _ in range(500):
            approvals_response = client.get(
                f"/v1/sessions/{session_id}/approvals",
                headers=headers,
            )
            assert approvals_response.status_code == 200
            approvals = approvals_response.json()
            if approvals:
                assert len(approvals) == 1
                approval = approvals[0]
                break
            time.sleep(0.01)
        assert approval is not None
        assert approval["tool_name"] == "apply_patch"
        assert approval["effect"] == ToolEffect.WRITE.value
        assert approval["scope"]["paths"] == ["src/main.py"]
        assert target.read_text(encoding="utf-8").endswith('return "needle"\n')

        decision = client.post(
            f"/v1/sessions/{session_id}/approvals/{approval['request_id']}/decision",
            headers=headers,
            json={"outcome": "ALLOW"},
        )
        assert decision.status_code == 200
        assert decision.json()["outcome"] == "ALLOW"

        terminal = None
        for _ in range(1_000):
            run_view = client.get(
                f"/v1/sessions/{session_id}/runs/{run_id}",
                headers=headers,
            ).json()
            if run_view["status"] in {
                status.value for status in RunStatus if status.is_terminal
            }:
                terminal = run_view
                break
            time.sleep(0.01)
        assert terminal is not None
        assert terminal["status"] == RunStatus.SUCCEEDED.value, terminal.get("error")
        assert terminal["output_text"] == "production patch, test and diff completed"

    async def persisted_facts() -> tuple[object, ...]:
        async with SqliteStore(database_path) as store:
            calls = await store.list_tool_calls(run_id)
            results = []
            for call in calls:
                results.append(await store.get_tool_result(call.call_id))
            approvals = await store.list_approvals(owner_id=owner_id, run_id=run_id)
            decisions = []
            for item in approvals:
                decisions.append(
                    await store.get_approval_decision(
                        item.request_id,
                        owner_id=owner_id,
                    )
                )
            change_sets = await store.list_change_sets(run_id=run_id)
            snapshots = await store.list_snapshots("ws_project", owner_id=owner_id)
            return (
                calls,
                tuple(results),
                approvals,
                tuple(decisions),
                change_sets,
                snapshots,
            )

    calls, results, approvals, decisions, change_sets, snapshots = asyncio.run(
        persisted_facts()
    )
    assert [call.tool_name for call in calls] == [
        "apply_patch",
        "run_targeted_test",
        "get_diff",
    ]
    assert all(call.status is ToolCallStatus.SUCCEEDED for call in calls)
    assert all(result.status is ToolCallStatus.SUCCEEDED for result in results)
    assert len(approvals) == len(decisions) == len(change_sets) == 1
    assert decisions[0].outcome.value == "ALLOW"
    assert len(snapshots) == 2
    assert target.read_text(encoding="utf-8").endswith('return "patched"\n')
    assert len(adapter.requests) == 4


@pytest.mark.asyncio
async def test_failed_targeted_test_cannot_become_a_final_agent_result() -> None:
    result = await AgentLoop(
        FailedTargetedTestAdapter(),
        _profile(),
        tool_executor=FailedTargetedTestExecutor(),
    ).execute(input="run the targeted test")

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.text is None
    assert result.error is not None
    assert result.error.category.value == "VERIFICATION_FAILED"


@pytest.mark.asyncio
async def test_patch_test_agent_path_requires_write_approval_and_persists_changeset(
    store: SqliteStore,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "src").mkdir(parents=True)
    source = 'def answer():\n    return "needle"\n'
    target = workspace_root / "src" / "main.py"
    target.write_text(source, encoding="utf-8")

    owner_id = "prn_agent_patch_e2e"
    workspace_id = new_workspace_id()
    snapshot_id = new_snapshot_id()
    now = utc_now()
    workspace = Workspace(
        workspace_id=workspace_id,
        owner_id=owner_id,
        alias="agent-patch-e2e",
        source=WorkspaceSource(
            source_type=WorkspaceSourceType.SERVER_ALIAS,
            server_alias="agent-fixture",
        ),
        created_at=now,
    )
    await store.create_workspace(workspace)
    with WorkspaceBackend(workspace_root) as backend:
        base_manifest = backend.snapshot_manifest()
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            workspace_id=workspace_id,
            status=SnapshotStatus.READY,
            created_at=now,
            completed_at=now,
            file_count=len(base_manifest.entries),
            total_size=base_manifest.total_size,
        )
        await store.create_snapshot(snapshot, base_manifest, owner_id=owner_id)
        patch_runner = PatchRunner(
            backend,
            store,
            owner_id=owner_id,
            base_snapshot=snapshot,
            base_manifest=base_manifest,
        )
        command_spec = CommandSpec(
            name="fixture_patch_test",
            executable=sys.executable,
            argv_template=(
                str(Path(__file__).parents[1] / "fixtures" / "command_fixture.py"),
                "{mode}",
            ),
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
        diff_runner = DeferredDiffRunner(base_manifest, backend.snapshot_manifest)
        filesystem = build_filesystem_registry(backend)
        registry = ToolRegistry(
            (
                *filesystem.definitions,
                build_patch_definition(patch_runner),
                build_targeted_test_definition(command_runner),
                *build_diff_definitions(diff_runner),
            )
        )
        tool_executor = ToolExecutor(registry, store)
        agent_executor = WorkspaceAgentExecutor(
            tool_executor,
            workspace_id=workspace_id,
            snapshot_id=snapshot_id,
            approve_writes=True,
            store=store,
            diff_runner=diff_runner,
        )
        adapter = PatchTestAdapter(base_snapshot_id=snapshot_id)
        settings = Settings(worker_profile=_profile())
        controller = RunController(
            store,
            settings,
            {"worker": adapter},
            tool_executor=agent_executor,
        )
        session = Session(
            session_id=new_session_id(),
            principal_id=owner_id,
            workspace_id=workspace_id,
            grant=WorkspaceGrant(
                principal_id=owner_id,
                workspace_id=workspace_id,
            ),
            agent_options=AgentRequestOptions(
                agent_mode=AgentMode.AUTO,
                isolation_mode=IsolationMode.HOST,
                execution_location=ExecutionLocation.CLOUD,
            ),
            created_at=now,
        )
        await store.create_session(session)
        run = await controller.create_run(
            NativeRunRequest(
                input="update the fixture and run its targeted test",
                routing_policy=RoutingPolicy.MANUAL,
                strategy=ExecutionStrategy.DIRECT,
                agent_options=session.agent_options,
                budget=Budget(max_attempts=8, max_total_tokens=16),
            )
        )
        await store.attach_run_to_session(
            session.session_id,
            run.run_id,
            principal_id=owner_id,
        )
        finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED, finished.error
    assert finished.usage is not None
    assert finished.usage.total_tokens == 8
    assert target.read_text(encoding="utf-8") == 'def answer():\n    return "patched"\n'
    assert [decision.outcome for decision in agent_executor.decisions] == [
        PolicyOutcome.ASK,
        PolicyOutcome.ALLOW,
        PolicyOutcome.ALLOW,
    ]
    assert agent_executor.decisions[0].reason_code is PolicyReasonCode.APPROVAL_REQUIRED
    assert (
        agent_executor.decisions[1].reason_code
        is PolicyReasonCode.AUTO_LOW_RISK_COMMAND
    )
    assert agent_executor.decisions[2].reason_code is PolicyReasonCode.READ_ALLOWED
    assert len(adapter.requests) == 4

    tool_calls = await store.list_tool_calls(run.run_id)
    assert [call.tool_name for call in tool_calls] == [
        "apply_patch",
        "run_targeted_test",
        "get_diff",
    ]
    assert all(call.status is ToolCallStatus.SUCCEEDED for call in tool_calls)
    patch_result = await store.get_tool_result(tool_calls[0].call_id)
    test_result = await store.get_tool_result(tool_calls[1].call_id)
    diff_result = await store.get_tool_result(tool_calls[2].call_id)
    assert patch_result.status is ToolCallStatus.SUCCEEDED
    assert patch_result.changed_paths == ("src/main.py",)
    assert patch_result.result is not None
    produced_snapshot_id = patch_result.result["new_snapshot_id"]
    assert isinstance(produced_snapshot_id, str)
    assert test_result.status is ToolCallStatus.SUCCEEDED
    assert test_result.exit_code == 0
    assert test_result.result is not None
    assert test_result.result["timed_out"] is False
    assert diff_result.status is ToolCallStatus.SUCCEEDED
    assert diff_result.result is not None
    assert diff_result.result["entries"][0]["path"] == "src/main.py"  # type: ignore[index]
    assert diff_result.result["entries"][0]["status"] == "MODIFIED"  # type: ignore[index]

    stored_base = await store.get_snapshot(snapshot_id, owner_id=owner_id)
    stored_base_manifest = await store.get_snapshot_manifest(
        snapshot_id, owner_id=owner_id
    )
    new_manifest = await store.get_snapshot_manifest(
        produced_snapshot_id, owner_id=owner_id
    )
    assert stored_base.file_count == len(base_manifest.entries)
    assert stored_base_manifest.manifest_hash == base_manifest.manifest_hash
    assert new_manifest.manifest_hash != base_manifest.manifest_hash
    assert next(
        entry for entry in new_manifest.entries if entry.path == "src/main.py"
    ).sha256 != next(
        entry for entry in base_manifest.entries if entry.path == "src/main.py"
    ).sha256

    change_sets = await store.list_change_sets(run_id=run.run_id)
    assert len(change_sets) == 1
    assert change_sets[0].base_snapshot_id == snapshot_id
    assert change_sets[0].new_snapshot_id == produced_snapshot_id
    assert change_sets[0].files[0].path == "src/main.py"

    events = await store.list_events(run.run_id)
    assert assert_sequence_chain(events) is None
    event_types = [event.event_type for event in events]
    assert event_types.count(EventType.TOOL_CALL_REQUESTED) == 3
    assert event_types.count(EventType.TOOL_CALL_SUCCEEDED) == 3
    assert EventType.ARTIFACT_PRODUCED in event_types
    assert str(workspace_root) not in json.dumps(
        [event.payload for event in events],
        ensure_ascii=True,
    )
