"""In-process local DIRECT execution with a fake provider."""

import shutil
from pathlib import Path


import pytest

from prp_runtime.domain.enums import (
    AgentMode,
    AttemptStatus,
    ExecutionLocation,
    ExecutionStrategy,
    IsolationMode,
    ModelRole,
    RunStatus,
    ToolCallStatus,
    ToolEffect,
)
from prp_runtime.domain.errors import DomainValidationError, ErrorCode, ProviderError
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import AgentToolCall, ErrorCategory, Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.runtime.assembler import assemble_run_result
from prp_runtime.runtime.local import LocalRuntime
from prp_runtime.workspace.isolation import (
    ExecutionCopyMode,
    LocalIsolationBackend,
    select_execution_copy_mode,
)
from prp_runtime.settings import Settings
from prp_runtime.policy.models import ApprovalOutcome
from prp_runtime.storage.sqlite import MissingEntityError, SqliteStore


WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="openai_compatible",
    model="weak-model",
    role=ModelRole.WORKER,
    base_url="https://models.invalid/v1",
    context_window_tokens=32_000,
    max_output_tokens=4_000,
)



ORIGINAL_MAIN = 'def answer():\n    return "needle"\n'
PATCHED_MAIN = 'def answer():\n    return "patched"\n'
PATCH_DIFF = (
    "--- a/src/main.py\n"
    "+++ b/src/main.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def answer():\n"
    '-    return "needle"\n'
    '+    return "patched"\n'
)


class PatchAdapter:
    def __init__(self, runtime_holder: list[LocalRuntime]) -> None:
        self.requests: list[ProviderRequest] = []
        self._runtime_holder = runtime_holder

    @property
    def name(self) -> str:
        return "local-patch-fake"

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


def _prepare_patch_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "patch"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text(ORIGINAL_MAIN, encoding="utf-8")
    return workspace


def _bridge_claim_events(events) -> list[EventType]:
    return [
        event.event_type
        for event in events
        if event.event_type
        in {
            EventType.BRIDGE_CLAIM_CREATED,
            EventType.BRIDGE_CLAIM_EXPIRED,
            EventType.BRIDGE_CLAIM_SETTLED,
            EventType.BRIDGE_CLAIM_RELEASED,
        }
    ]


class FakeAdapter:
    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[ProviderRequest] = []
        self.close_calls = 0

    @property
    def name(self) -> str:
        return "local-direct-fake"

    async def aclose(self) -> None:
        self.close_calls += 1

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, ProviderResponse):
            return outcome
        return ProviderResponse(
            text="the local answer",
            usage=Usage(input_tokens=4, output_tokens=3, elapsed_ms=2),
            finish_reason=FinishReason.STOP,
        )


@pytest.mark.asyncio
async def test_local_direct_run_completes_without_http(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = Settings(
        database_path=tmp_path / "local-direct.db",
        worker_profile=WORKER_PROFILE,
    )
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        result = await runtime.run("summarise the report", workspace=workspace)

    assert result.status is RunStatus.SUCCEEDED
    assert result.strategy is ExecutionStrategy.DIRECT
    assert result.output_text == "the local answer"
    assert result.error is None
    assert len(adapter.requests) == 1
    assert adapter.close_calls == 0
    assert runtime.defaults.execution_location is ExecutionLocation.LOCAL
    assert runtime.defaults.isolation_mode is IsolationMode.HOST


@pytest.mark.asyncio
async def test_local_direct_facts_survive_close_and_provider_failure(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database_path = tmp_path / "local-facts.db"
    settings = Settings(database_path=database_path, worker_profile=WORKER_PROFILE)
    adapter = FakeAdapter()
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        result = await runtime.run("summarise the report", workspace=workspace)
        run_id = result.run_id
        units = await runtime.store.list_work_units(run_id)
        attempts = await runtime.store.list_attempts(units[0].work_unit_id)
        events = await runtime.store.list_events(run_id)
        assert_sequence_chain(events)
        dumped = "".join(event.model_dump_json() for event in events)
        assert str(workspace) not in dumped
        assert "secret" not in dumped
        assert events[0].event_type is EventType.RUN_CREATED
        assert EventType.RUN_SUCCEEDED in {event.event_type for event in events}
        assert attempts[0].status is AttemptStatus.SUCCEEDED
        assert attempts[0].usage == Usage(input_tokens=4, output_tokens=3, elapsed_ms=2)
        artifacts = await runtime.store.list_artifacts(units[0].work_unit_id)
        assert artifacts[-1].content == "the local answer"

    async with SqliteStore(database_path) as store:
        restored = await assemble_run_result(store, run_id)
        assert restored.status is RunStatus.SUCCEEDED
        assert restored.output_text == "the local answer"
        assert restored.usage.total_tokens == 7

    failing = FakeAdapter(ProviderError("upstream timed out", code=ErrorCode.PROVIDER_TIMEOUT))
    async with LocalRuntime(settings, adapters={"worker": failing}) as runtime:
        failed = await runtime.run("hello", workspace=workspace)
        assert failed.status is RunStatus.FAILED
        assert failed.error is not None
        assert failed.error.category is ErrorCategory.TIMEOUT
        assert failed.output_text is None
        units = await runtime.store.list_work_units(failed.run_id)
        attempts = await runtime.store.list_attempts(units[0].work_unit_id)
        assert attempts[0].status is AttemptStatus.FAILED
        assert attempts[0].status is not AttemptStatus.SUCCEEDED


class SequentialAdapter:
    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "local-tool-fake"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, ProviderResponse)
        return outcome


@pytest.mark.asyncio
async def test_local_read_tool_and_unknown_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "tools"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello local\n", encoding="utf-8")
    settings = Settings(
        database_path=tmp_path / "local-tools.db",
        worker_profile=WORKER_PROFILE,
    )
    reader = SequentialAdapter(
        ProviderResponse(
            tool_calls=(
                AgentToolCall(
                    call_id="call-read",
                    tool_name="read_file",
                    arguments={"path": "README.md"},
                ),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        ProviderResponse(
            text="read README.md",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        ),
    )
    async with LocalRuntime(settings, adapters={"worker": reader}) as runtime:
        result = await runtime.run("read the file", workspace=workspace)
        assert result.status is RunStatus.SUCCEEDED
        assert result.output_text == "read README.md"
        assert len(reader.requests) == 2

    unknown = SequentialAdapter(
        ProviderResponse(
            tool_calls=(
                AgentToolCall(
                    call_id="call-unknown",
                    tool_name="network_fetch",
                    arguments={"url": "https://example.invalid"},
                ),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        ProviderResponse(
            text="stopped after denial",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        ),
    )
    async with LocalRuntime(settings, adapters={"worker": unknown}) as runtime:
        result = await runtime.run("fetch the network", workspace=workspace)
        assert result.status is RunStatus.FAILED
        assert result.error is not None
        assert result.error.category is ErrorCategory.INVALID_REQUEST
        assert "outside the public catalog" in result.error.message


@pytest.mark.asyncio
async def test_local_approved_patch_updates_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_calls: list[object] = []

    def forbidden_copytree(*args: object, **kwargs: object) -> object:
        copy_calls.append((args, kwargs))
        raise AssertionError("sequential local DIRECT must not copy the workspace tree")

    monkeypatch.setattr(shutil, "copytree", forbidden_copytree)
    monkeypatch.setattr(
        "prp_runtime.workspace.isolation.shutil.copytree",
        forbidden_copytree,
    )
    workspace = _prepare_patch_workspace(tmp_path)
    settings = Settings(
        database_path=tmp_path / "local-patch.db",
        worker_profile=WORKER_PROFILE,
        allow_host_yolo=True,
    )
    holder: list[LocalRuntime] = []
    adapter = PatchAdapter(holder)
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        result = await runtime.run(
            "patch the file",
            workspace=workspace,
            agent_mode=AgentMode.YOLO,
            user_explicit=True,
        )
        assert result.status is RunStatus.SUCCEEDED
        assert runtime.last_copy_mode is ExecutionCopyMode.IN_PLACE
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == PATCHED_MAIN
        change_sets = await runtime.store.list_change_sets(run_id=result.run_id)
        assert len(change_sets) == 1
        assert change_sets[0].files[0].path == "src/main.py"
    assert copy_calls == []



@pytest.mark.asyncio
async def test_local_ask_approve_deny_replay_and_no_bridge_claim(tmp_path: Path) -> None:
    workspace = _prepare_patch_workspace(tmp_path)
    settings = Settings(
        database_path=tmp_path / "local-ask.db",
        worker_profile=WORKER_PROFILE,
    )
    holder: list[LocalRuntime] = []
    adapter = PatchAdapter(holder)
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        paused = await runtime.run("patch the file", workspace=workspace)
        assert paused.status is RunStatus.RUNNING
        assert paused.error is None
        assert not paused.status.is_terminal
        assert runtime.held_capacity()["max_concurrency"] == 1
        assert runtime.held_capacity()["max_attempts"] == 1
        assert str(workspace) not in str(runtime.public_facts())
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == ORIGINAL_MAIN
        pending_calls = await runtime.store.list_tool_calls(
            paused.run_id,
            statuses=[ToolCallStatus.AWAITING_APPROVAL],
        )
        assert len(pending_calls) == 1
        assert pending_calls[0].tool_name == "apply_patch"
        assert pending_calls[0].effect is ToolEffect.WRITE
        approvals = await runtime.pending_approvals(run_id=paused.run_id)
        assert len(approvals) == 1
        approval = approvals[0]
        assert approval.tool_name == "apply_patch"
        assert approval.run_id == paused.run_id
        assert await runtime.store.list_change_sets(run_id=paused.run_id) == ()
        events = await runtime.store.list_events(paused.run_id)
        assert _bridge_claim_events(events) == []

        with pytest.raises(MissingEntityError):
            await runtime.approve(approval.request_id, principal_id="prn_other")
        with pytest.raises(MissingEntityError):
            await runtime.approve(approval.request_id, run_id="run_other")
        with pytest.raises(MissingEntityError):
            await runtime.approve(approval.request_id, workspace_id="ws_other")
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == ORIGINAL_MAIN
        assert await runtime.store.list_change_sets(run_id=paused.run_id) == ()

        approved = await runtime.approve(approval.request_id)
        assert approved.status is RunStatus.SUCCEEDED
        assert approved.run_id == paused.run_id
        assert runtime.held_capacity()["max_concurrency"] == 0
        assert runtime.held_capacity()["max_attempts"] == 0
        assert runtime.held_capacity()["max_total_tokens"] == 0
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == PATCHED_MAIN
        change_sets = await runtime.store.list_change_sets(run_id=paused.run_id)
        assert len(change_sets) == 1
        decision = await runtime.store.get_approval_decision(
            approval.request_id, owner_id=settings.service_principal
        )
        assert decision.outcome is ApprovalOutcome.ALLOW
        replayed = await runtime.replay(approval.request_id)
        assert replayed.status is RunStatus.SUCCEEDED
        again = await runtime.approve(approval.request_id)
        assert again.status is RunStatus.SUCCEEDED
        assert await runtime.store.list_change_sets(run_id=paused.run_id) == change_sets
        events = await runtime.store.list_events(paused.run_id)
        assert _bridge_claim_events(events) == []

    deny_workspace = tmp_path / "deny"
    deny_workspace.mkdir()
    (deny_workspace / "src").mkdir()
    (deny_workspace / "src" / "main.py").write_text(ORIGINAL_MAIN, encoding="utf-8")
    deny_settings = Settings(
        database_path=tmp_path / "local-deny.db",
        worker_profile=WORKER_PROFILE,
    )
    deny_holder: list[LocalRuntime] = []
    deny_adapter = PatchAdapter(deny_holder)
    async with LocalRuntime(deny_settings, adapters={"worker": deny_adapter}) as runtime:
        deny_holder.append(runtime)
        paused = await runtime.run("patch the file", workspace=deny_workspace)
        assert paused.status is RunStatus.RUNNING
        approvals = await runtime.pending_approvals(run_id=paused.run_id)
        assert len(approvals) == 1
        denied = await runtime.deny(approvals[0].request_id)
        assert denied.status.is_terminal
        assert runtime.held_capacity()["max_concurrency"] == 0
        assert runtime.held_capacity()["max_attempts"] == 0
        assert (deny_workspace / "src" / "main.py").read_text(encoding="utf-8") == ORIGINAL_MAIN
        assert await runtime.store.list_change_sets(run_id=paused.run_id) == ()
        decision = await runtime.store.get_approval_decision(
            approvals[0].request_id, owner_id=deny_settings.service_principal
        )
        assert decision.outcome is ApprovalOutcome.DENY
        calls = await runtime.store.list_tool_calls(paused.run_id)
        assert calls[0].status is not ToolCallStatus.SUCCEEDED
        events = await runtime.store.list_events(paused.run_id)
        assert _bridge_claim_events(events) == []


def test_sequential_local_direct_selector_is_in_place() -> None:
    in_place = dict(
        execution_location=ExecutionLocation.LOCAL,
        isolation_mode=IsolationMode.HOST,
        strategy=ExecutionStrategy.DIRECT,
        concurrency=1,
    )
    assert select_execution_copy_mode(**in_place) is ExecutionCopyMode.IN_PLACE
    variants = (
        {"execution_location": ExecutionLocation.CLOUD},
        {"execution_location": ExecutionLocation.BRIDGE},
        {"isolation_mode": IsolationMode.SANDBOXED},
        {"strategy": ExecutionStrategy.PLANNED},
        {"strategy": ExecutionStrategy.PROGRESSIVE},
        {"concurrency": 2},
    )
    for update in variants:
        selected = dict(in_place)
        selected.update(update)
        assert select_execution_copy_mode(**selected) is ExecutionCopyMode.COPY_BACKED


@pytest.mark.asyncio
async def test_local_direct_does_not_copy_the_workspace_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_calls: list[tuple[object, object]] = []

    def forbidden_copytree(*args: object, **kwargs: object) -> object:
        copy_calls.append((args, kwargs))
        raise AssertionError("sequential local DIRECT must not copy the workspace tree")

    monkeypatch.setattr(shutil, "copytree", forbidden_copytree)
    monkeypatch.setattr(
        "prp_runtime.workspace.isolation.shutil.copytree",
        forbidden_copytree,
    )

    def forbidden_snapshot(*args: object, **kwargs: object) -> object:
        raise AssertionError("sequential local DIRECT must not create a copied snapshot")

    def forbidden_slot(*args: object, **kwargs: object) -> object:
        raise AssertionError("sequential local DIRECT must not create a copied slot")

    monkeypatch.setattr(LocalIsolationBackend, "create_base_snapshot", forbidden_snapshot)
    monkeypatch.setattr(LocalIsolationBackend, "create_slot", forbidden_slot)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello local\n", encoding="utf-8")
    settings = Settings(
        database_path=tmp_path / "local-nocopy.db",
        worker_profile=WORKER_PROFILE,
    )
    adapter = FakeAdapter()
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        with pytest.raises(DomainValidationError):
            await runtime.run("hello", workspace=workspace, concurrency=2)
        with pytest.raises(DomainValidationError) as raised:
            await runtime.run(
                "hello",
                workspace=workspace,
                strategy=ExecutionStrategy.PLANNED,
            )
        assert raised.value.code is ErrorCode.INVALID_AGENT_OPTIONS
        assert raised.value.detail.field == "execution_copy_mode"
        assert "copy-backed" in str(raised.value)
        result = await runtime.run("summarise the report", workspace=workspace)
        assert result.status is RunStatus.SUCCEEDED
        assert runtime.last_copy_mode is ExecutionCopyMode.IN_PLACE
    assert copy_calls == []


@pytest.mark.asyncio
async def test_fresh_process_approve_binds_root_and_rejects_wrong_workspace(
    tmp_path: Path,
) -> None:
    workspace = _prepare_patch_workspace(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    settings = Settings(
        database_path=tmp_path / "fresh-approve.db",
        worker_profile=WORKER_PROFILE,
    )
    holder: list[LocalRuntime] = []
    adapter = PatchAdapter(holder)
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        paused = await runtime.run("patch the file", workspace=workspace)
        approvals = await runtime.pending_approvals(run_id=paused.run_id)
        request_id = approvals[0].request_id
        run_id = paused.run_id
        workspace_id = runtime.last_workspace_id
        assert workspace_id is not None
        assert str(workspace) not in workspace_id
    holder.clear()
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        supervisor = runtime.composition.supervisor
        assert supervisor is not None
        assert run_id in supervisor.held_run_ids
        discovered = await supervisor.scan()
        assert run_id not in discovered
        assert run_id not in supervisor._queued
        assert run_id not in supervisor.active_run_ids
        with pytest.raises(RuntimeError, match="not bound"):
            await runtime.approve(request_id)
        runtime.bind_workspace(other)
        with pytest.raises(MissingEntityError):
            await runtime.approve(request_id, workspace_id=runtime.last_workspace_id)
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == ORIGINAL_MAIN
        bound = runtime.bind_workspace(workspace)
        assert bound == workspace_id
        approved = await runtime.approve(request_id, workspace_id=workspace_id)
        assert approved.status is RunStatus.SUCCEEDED
        assert approved.run_id == run_id
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == PATCHED_MAIN
        replayed = await runtime.replay(request_id, workspace_id=workspace_id)
        assert replayed.status is RunStatus.SUCCEEDED
        assert replayed.run_id == run_id
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == PATCHED_MAIN
        events = await runtime.store.list_events(run_id)
        assert _bridge_claim_events(events) == []
        assert str(workspace) not in str(runtime.public_facts())
        assert str(workspace) not in workspace_id
        for request in adapter.requests:
            assert str(workspace) not in request.model_dump_json()


@pytest.mark.asyncio
async def test_fresh_process_does_not_auto_resume_before_bind(
    tmp_path: Path,
) -> None:
    workspace = _prepare_patch_workspace(tmp_path)
    settings = Settings(
        database_path=tmp_path / "fresh-no-auto.db",
        worker_profile=WORKER_PROFILE,
    )
    holder: list[LocalRuntime] = []
    adapter = PatchAdapter(holder)
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        paused = await runtime.run("patch the file", workspace=workspace)
        run_id = paused.run_id
        assert paused.status is RunStatus.RUNNING
    holder.clear()
    async with LocalRuntime(settings, adapters={"worker": adapter}) as runtime:
        holder.append(runtime)
        supervisor = runtime.composition.supervisor
        assert supervisor is not None
        assert run_id in supervisor.held_run_ids
        assert run_id not in supervisor._queued
        assert run_id not in supervisor.active_run_ids
        discovered = await supervisor.scan()
        assert run_id not in discovered
        assert run_id not in supervisor._queued
        assert run_id not in supervisor.active_run_ids
        persisted = await runtime.store.get_run(run_id)
        assert persisted.status is RunStatus.RUNNING
        assert (workspace / "src" / "main.py").read_text(encoding="utf-8") == ORIGINAL_MAIN
        assert str(workspace) not in str(runtime.public_facts())
