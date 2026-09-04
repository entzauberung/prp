"""Progressive integration: evidence-gated revision and bounded termination."""

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio

from prp_runtime.control.controller import RunController
from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import Budget, ErrorCategory, NativeRunRequest, Usage
from prp_runtime.planning.models import (
    PlanProposal,
    PlanRevision,
    PlanRevisionReason,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

RESULT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"ok": {"const": True}},
        "required": ["ok"],
        "additionalProperties": False,
    }
)


def _profile(alias: str, role: ModelRole) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="fake",
        model=f"{alias}-model",
        role=role,
        base_url="https://models.invalid/v1",
        supports_structured_output=True,
        context_window_tokens=8_000,
        max_output_tokens=1_000,
    )


def _settings() -> Settings:
    return Settings(
        leader_profile=_profile("planner", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
    )


def _proposal(key: str) -> PlanProposal:
    return PlanProposal(
        summary=f"plan {key}",
        final_node=key,
        nodes=(
            {
                "key": key,
                "name": key.title(),
                "instruction": f"produce {key}",
                "output": {"kind": "JSON", "json_schema": RESULT_SCHEMA},
            },
        ),
    )


def _chain_proposal(*, changed_tail: bool = False) -> PlanProposal:
    return PlanProposal(
        summary="chain revision",
        final_node="tail",
        nodes=(
            {
                "key": "root",
                "lineage_key": "root-lineage",
                "name": "Root",
                "instruction": "produce root",
                "output": {"kind": "JSON", "json_schema": RESULT_SCHEMA},
            },
            {
                "key": "tail",
                "lineage_key": "tail-lineage",
                "name": "Tail",
                "instruction": (
                    "produce changed tail" if changed_tail else "produce tail"
                ),
                "depends_on": ["root"],
                "output": {"kind": "JSON", "json_schema": RESULT_SCHEMA},
            },
        ),
    )


class RevisionPlannerAdapter:
    def __init__(
        self,
        initial: PlanProposal,
        revision: PlanRevision | BaseException | None,
    ) -> None:
        self.initial = initial
        self.revision = revision
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "progressive-planner"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            text = self.initial.model_dump_json()
        else:
            assert self.revision is not None
            if isinstance(self.revision, BaseException):
                raise self.revision
            text = self.revision.model_dump_json()
        return ProviderResponse(
            text=text,
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


class RevisionWorkerAdapter:
    def __init__(self, responses: tuple[str, ...]) -> None:
        self.responses = responses
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "progressive-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        response = self.responses[len(self.requests) - 1]
        return ProviderResponse(
            text=response,
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    async with SqliteStore(tmp_path / "progressive-integration.db") as opened:
        yield opened


def _run_request(
    *,
    max_revisions: int | None = 1,
    max_attempts: int = 4,
    max_total_tokens: int | None = None,
    max_strong_model_tokens: int | None = None,
) -> NativeRunRequest:
    return NativeRunRequest(
        input="produce a verified JSON result",
        routing_policy=RoutingPolicy.MANUAL,
        strategy=ExecutionStrategy.PROGRESSIVE,
        budget=Budget(
            max_plan_revisions=max_revisions,
            max_attempts=max_attempts,
            max_total_tokens=max_total_tokens,
            max_strong_model_tokens=max_strong_model_tokens,
        ),
    )


@pytest.mark.asyncio
async def test_fail_evidence_revises_once_and_isolates_old_graph(
    store: SqliteStore,
) -> None:
    initial = _proposal("initial")
    revised = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="replace failed output graph",
        proposal=_proposal("revised"),
    )
    planner = RevisionPlannerAdapter(initial, revised)
    worker = RevisionWorkerAdapter(("{\"ok\":false}", "{\"ok\":true}"))
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_run_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED, finished.error
    assert finished.strategy is ExecutionStrategy.PROGRESSIVE
    assert finished.graph_version == 3
    assert finished.usage == Usage(
        input_tokens=4,
        output_tokens=4,
        strong_model_tokens=4,
    )
    assert len(planner.requests) == 2
    assert len(worker.requests) == 2
    old_units = await store.list_work_units(run.run_id, graph_version=2)
    new_units = await store.list_work_units(run.run_id, graph_version=3)
    persisted = await store.get_run(run.run_id)
    assert persisted.final_work_unit_id == new_units[0].work_unit_id
    assert persisted.final_work_unit_id != old_units[0].work_unit_id
    assert old_units[0].status is WorkUnitStatus.FAILED
    assert new_units[0].status is WorkUnitStatus.SUCCEEDED
    assert len(await store.list_evidence(old_units[0].work_unit_id)) == 4
    assert len(await store.list_evidence(new_units[0].work_unit_id)) == 4
    rounds = await store.list_rounds(run.run_id)
    assert [round_fact.round_index for round_fact in rounds] == [0, 1]
    assert [round_fact.graph_version for round_fact in rounds] == [2, 3]
    assert [round_fact.status.value for round_fact in rounds] == ["FAILED", "VERIFIED"]
    assert rounds[0].merged_snapshot_id is None
    assert rounds[1].merged_snapshot_id is not None
    assert rounds[0].base_snapshot_id == rounds[1].base_snapshot_id
    assert rounds[1].revision_of_round_id == rounds[0].round_id
    assert rounds[1].evidence_ids
    attempts = await store.list_run_attempts(run.run_id)
    planner_attempts = tuple(
        attempt for attempt in attempts if attempt.role is ModelRole.PLANNER
    )
    assert len(attempts) == 4
    assert [attempt.attempt_index for attempt in planner_attempts] == [1, 2]
    assert [attempt.status for attempt in planner_attempts] == [
        AttemptStatus.SUCCEEDED,
        AttemptStatus.SUCCEEDED,
    ]
    assert all(
        attempt.usage is not None
        and attempt.usage.strong_model_tokens == 2
        for attempt in planner_attempts
    )
    planning_units = await store.list_work_units(run.run_id, graph_version=1)
    assert len(planning_units) == 2
    assert all(unit.status is WorkUnitStatus.SUCCEEDED for unit in planning_units)
    events = await store.list_events(run.run_id)
    assert EventType.PLAN_REVISED in {event.event_type for event in events}
    global_reports = [
        event
        for event in events
        if event.event_type is EventType.CONTROLLER_DECISION
        and "global_report" in event.payload
    ]
    assert [event.payload["global_report"]["result"] for event in global_reports] == [
        "FAIL",
        "PASS",
    ]
    assert global_reports[1].payload["comparison"]["outcome"] == "IMPROVED"
    assert events[-1].event_type is EventType.RUN_SUCCEEDED
    assert assert_sequence_chain(events) is None


@pytest.mark.asyncio
async def test_progressive_pass_never_calls_revision_planner(store: SqliteStore) -> None:
    planner = RevisionPlannerAdapter(_proposal("success"), None)
    worker = RevisionWorkerAdapter(("{\"ok\":true}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.graph_version == 2
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    rounds = await store.list_rounds(run.run_id)
    assert len(rounds) == 1
    assert rounds[0].status.value == "VERIFIED"
    assert rounds[0].graph_version == 2
    assert rounds[0].merged_snapshot_id is not None
    assert rounds[0].evidence_ids
    assert EventType.PLAN_REVISED not in {
        event.event_type for event in await store.list_events(run.run_id)
    }
    reports = [
        event
        for event in await store.list_events(run.run_id)
        if event.event_type is EventType.CONTROLLER_DECISION
        and "global_report" in event.payload
    ]
    assert len(reports) == 1
    assert reports[0].payload["global_report"]["result"] == "PASS"
    assert reports[0].payload["comparison"]["outcome"] == "BASELINE"
    global_report = reports[0].payload["global_report"]
    assert global_report["round_id"] == rounds[0].round_id
    candidate_checks = [
        check for check in global_report["checks"] if check["kind"] == "CANDIDATE"
    ]
    assert len(candidate_checks) == 1
    assert rounds[0].merged_snapshot_id in candidate_checks[0]["fact_ids"]
    assert set(rounds[0].evidence_ids).issubset(candidate_checks[0]["fact_ids"])


@pytest.mark.asyncio
async def test_revision_reuses_unchanged_root_and_recomputes_changed_tail(
    store: SqliteStore,
) -> None:
    initial = _chain_proposal()
    revised = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="change only the tail",
        proposal=_chain_proposal(changed_tail=True),
    )
    planner = RevisionPlannerAdapter(initial, revised)
    worker = RevisionWorkerAdapter(
        ('{"ok":true}', '{"ok":false}', '{"ok":true}')
    )
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_run_request(max_attempts=8))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED, finished.error
    assert len(worker.requests) == 3
    old_units = await store.list_work_units(run.run_id, graph_version=2)
    new_units = await store.list_work_units(run.run_id, graph_version=3)
    old_by_lineage = {unit.lineage_key: unit for unit in old_units}
    new_by_lineage = {unit.lineage_key: unit for unit in new_units}
    old_root_artifacts = await store.list_artifacts(old_by_lineage["root-lineage"].work_unit_id)
    old_tail_artifacts = await store.list_artifacts(old_by_lineage["tail-lineage"].work_unit_id)
    new_root_artifacts = await store.list_artifacts(new_by_lineage["root-lineage"].work_unit_id)
    new_tail_artifacts = await store.list_artifacts(new_by_lineage["tail-lineage"].work_unit_id)

    assert old_by_lineage["root-lineage"].status is WorkUnitStatus.SUCCEEDED
    assert old_by_lineage["tail-lineage"].status is WorkUnitStatus.FAILED
    assert new_by_lineage["root-lineage"].status is WorkUnitStatus.SUCCEEDED
    assert new_by_lineage["tail-lineage"].status is WorkUnitStatus.SUCCEEDED
    assert [artifact.content for artifact in old_root_artifacts] == ["{\"ok\":true}"]
    assert [artifact.content for artifact in old_tail_artifacts] == ["{\"ok\":false}"]
    assert [artifact.content for artifact in new_root_artifacts] == ["{\"ok\":true}"]
    assert [artifact.content for artifact in new_tail_artifacts] == ["{\"ok\":true}"]
    assert old_root_artifacts[0].artifact_id != new_root_artifacts[0].artifact_id
    assert old_tail_artifacts[0].artifact_id != new_tail_artifacts[0].artifact_id

    events = await store.list_events(run.run_id)
    reused = [event for event in events if event.event_type is EventType.WORK_UNIT_REUSED]
    invalidated = [
        event for event in events if event.event_type is EventType.WORK_UNIT_INVALIDATED
    ]
    assert len(reused) == 1
    assert reused[0].payload["source_work_unit_id"] == old_by_lineage["root-lineage"].work_unit_id
    assert reused[0].payload["work_unit_id"] == new_by_lineage["root-lineage"].work_unit_id
    assert any(
        event.payload["work_unit_id"] == old_by_lineage["tail-lineage"].work_unit_id
        for event in invalidated
    )


@pytest.mark.asyncio
async def test_revision_budget_zero_stops_after_first_failed_graph(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("budget"), None)
    worker = RevisionWorkerAdapter(("{\"ok\":false}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_revisions=0))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category.value == "VERIFICATION_FAILED"
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert finished.graph_version == 2
    rounds = await store.list_rounds(run.run_id)
    assert len(rounds) == 1
    assert rounds[0].status.value == "FAILED"
    assert rounds[0].merged_snapshot_id is None
    assert rounds[0].evidence_ids == ()


@pytest.mark.asyncio
async def test_attempt_budget_stops_before_revision_planner_call(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("attempt-budget"), None)
    worker = RevisionWorkerAdapter(("{\"ok\":false}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_attempts=2))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert len(await store.list_run_attempts(run.run_id)) == 2
    assert await store.list_work_units(run.run_id, graph_version=3) == ()


@pytest.mark.asyncio
async def test_strong_token_budget_records_revision_but_prevents_graph_commit(
    store: SqliteStore,
) -> None:
    revision = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="replace failed output graph",
        proposal=_proposal("must-not-commit"),
    )
    planner = RevisionPlannerAdapter(_proposal("strong-budget"), revision)
    worker = RevisionWorkerAdapter(("{\"ok\":false}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(
        _run_request(max_strong_model_tokens=3)
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert finished.graph_version == 2
    assert finished.usage.strong_model_tokens == 4
    assert len(planner.requests) == 2
    assert len(worker.requests) == 1
    assert await store.list_work_units(run.run_id, graph_version=3) == ()


@pytest.mark.asyncio
async def test_progressive_exact_total_ceiling_accepts_verified_current_artifact(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("success"), None)
    worker = RevisionWorkerAdapter(('{"ok":true}',))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_total_tokens=4))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.usage.total_tokens == 4
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert len(await store.list_run_attempts(run.run_id)) == 2
    event_types = [event.event_type for event in await store.list_events(run.run_id)]
    assert EventType.EVIDENCE_RECORDED in event_types
    assert EventType.BUDGET_EXHAUSTED not in event_types


@pytest.mark.asyncio
async def test_progressive_exact_total_ceiling_blocks_revision_call(
    store: SqliteStore,
) -> None:
    revision = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="must not be requested",
        proposal=_proposal("unused"),
    )
    planner = RevisionPlannerAdapter(_proposal("exact-budget"), revision)
    worker = RevisionWorkerAdapter(('{"ok":false}',))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_total_tokens=4))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert len(await store.list_run_attempts(run.run_id)) == 2
    assert await store.list_work_units(run.run_id, graph_version=3) == ()
    assert any(
        event.event_type is EventType.EVIDENCE_RECORDED
        for event in await store.list_events(run.run_id)
    )


@pytest.mark.asyncio
async def test_progressive_over_total_ceiling_rejects_before_verification(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("over-budget"), None)
    worker = RevisionWorkerAdapter(('{"ok":true}',))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_total_tokens=3))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert finished.usage.total_tokens == 4
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert len(await store.list_run_attempts(run.run_id)) == 2
    assert all(
        event.event_type is not EventType.EVIDENCE_RECORDED
        for event in await store.list_events(run.run_id)
    )


@pytest.mark.asyncio
async def test_revision_planner_failure_records_failed_attempt_without_new_graph(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(
        _proposal("provider-failure"),
        ProviderError("private upstream detail", code=ErrorCode.PROVIDER_TIMEOUT),
    )
    worker = RevisionWorkerAdapter(("{\"ok\":false}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.TIMEOUT
    assert finished.graph_version == 2
    assert await store.list_work_units(run.run_id, graph_version=3) == ()
    planner_attempts = tuple(
        attempt
        for attempt in await store.list_run_attempts(run.run_id)
        if attempt.role is ModelRole.PLANNER
    )
    assert [attempt.status for attempt in planner_attempts] == [
        AttemptStatus.SUCCEEDED,
        AttemptStatus.FAILED,
    ]
    assert planner_attempts[-1].usage is None
    assert planner_attempts[-1].error is not None
    assert planner_attempts[-1].error.category is ErrorCategory.TIMEOUT


@pytest.mark.asyncio
async def test_cancel_during_progressive_worker_skips_revision_call(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("cancel"), None)
    cancel_started = asyncio.Event()
    controller: RunController | None = None
    run_id = ""

    class CancellingAdapter(RevisionWorkerAdapter):
        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            assert controller is not None
            self.requests.append(request)
            cancel_started.set()
            await controller.cancel(run_id)
            return ProviderResponse(
                text='{"ok":true}',
                usage=Usage(input_tokens=1, output_tokens=1),
                finish_reason=FinishReason.STOP,
            )

    worker = CancellingAdapter(())
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request())
    run_id = run.run_id

    finished = await controller.execute(run.run_id)

    assert cancel_started.is_set()
    assert finished.status is RunStatus.CANCELLED
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    rounds = await store.list_rounds(run.run_id)
    assert len(rounds) == 1
    assert rounds[0].status.value == "CANCELLED"
    assert rounds[0].merged_snapshot_id is None
    assert rounds[0].failure_reason
    events = await store.list_events(run.run_id)
    assert events[-1].event_type is EventType.RUN_CANCELLED
    assert sum(event.event_type is EventType.ROUND_FAILED for event in events) == 1


class _BridgeToolWorkerAdapter:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "progressive-bridge-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        from prp_runtime.domain.models import AgentToolCall

        return ProviderResponse(
            tool_calls=(
                AgentToolCall(
                    call_id="provider-call/1",
                    tool_name="read_file",
                    arguments={"path": "src/main.py"},
                ),
            ),
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.TOOL_CALLS,
        )


class _ForbiddenBridgeFactory:
    def build(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("BRIDGE must not create a server WorkspaceToolRuntime")


@pytest.mark.asyncio
async def test_bridge_progressive_execute_assigns_one_published_snapshot_claim(
    store: SqliteStore,
) -> None:
    from prp_runtime.domain.enums import ExecutionLocation, ToolCallStatus, ToolEffect
    from prp_runtime.domain.models import (
        AgentRequestOptions,
        ClientCapabilityDescriptor,
        RegisteredBridgeClient,
        Session,
        WorkspaceGrant,
        fingerprint_client_capabilities,
        new_client_id,
    )
    from prp_runtime.domain.values import new_principal_id, new_session_id, utc_now
    from prp_runtime.runtime.tooling import ScopeToolRuntimeProvider
    from prp_runtime.workspace.models import Workspace, WorkspaceSource, WorkspaceSourceType
    from tests.unit.storage.test_store import make_manifest, make_snapshot

    principal_id = new_principal_id()
    now = utc_now()
    workspace = Workspace(
        workspace_id="ws_" + "a" * 32,
        owner_id=principal_id,
        alias="client-project",
        source=WorkspaceSource(
            source_type=WorkspaceSourceType.SERVER_ALIAS,
            server_alias="repo-main",
        ),
        created_at=now,
    )
    await store.create_workspace(workspace)
    snapshot = make_snapshot(workspace.workspace_id)
    published = await store.create_snapshot(
        snapshot, make_manifest(suffix="-published"), owner_id=principal_id
    )
    agent_options = AgentRequestOptions(execution_location=ExecutionLocation.BRIDGE)
    session = Session(
        session_id=new_session_id(),
        principal_id=principal_id,
        workspace_id=workspace.workspace_id,
        grant=WorkspaceGrant(
            principal_id=principal_id,
            workspace_id=workspace.workspace_id,
        ),
        agent_options=agent_options,
        created_at=now,
    )
    await store.create_session(session)
    capabilities = ClientCapabilityDescriptor(
        tools=("read_file",),
        effects=(ToolEffect.READ,),
    )
    client = RegisteredBridgeClient(
        client_id=new_client_id(),
        principal_id=principal_id,
        workspace_id=workspace.workspace_id,
        capabilities=capabilities,
        capability_fingerprint=fingerprint_client_capabilities(capabilities),
        created_at=now,
        last_seen_at=now,
    )
    await store.register_bridge_client(client)
    await store.record_bridge_heartbeat(
        client.client_id,
        principal_id=principal_id,
        fingerprint=client.capability_fingerprint,
        at=utc_now(),
    )

    planner = RevisionPlannerAdapter(_proposal("inspect"), None)
    worker = _BridgeToolWorkerAdapter()
    settings = _settings()
    provider = ScopeToolRuntimeProvider(
        store,
        settings,
        factory=_ForbiddenBridgeFactory(),  # type: ignore[arg-type]
        enable_server_resolver=False,
    )
    controller = RunController(
        store,
        settings,
        {"planner": planner, "worker": worker},
        tool_executor_provider=provider.executor_for,
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="inspect the published client snapshot",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.PROGRESSIVE,
            agent_options=agent_options,
            budget=Budget(max_plan_revisions=0, max_attempts=4),
        )
    )
    await store.attach_run_to_session(
        session.session_id, run.run_id, principal_id=principal_id
    )

    finished = await controller.execute(run.run_id, principal_id=principal_id)

    assert finished.status is RunStatus.RUNNING, finished.error
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    owner_workspaces = await store.list_workspaces(owner_id=principal_id)
    assert [item.workspace_id for item in owner_workspaces] == [workspace.workspace_id]
    service_workspaces = await store.list_workspaces(owner_id=settings.service_principal)
    assert all(not item.alias.startswith("progressive-") for item in service_workspaces)
    snapshots = await store.list_snapshots(
        workspace.workspace_id, owner_id=principal_id
    )
    assert [item.snapshot_id for item in snapshots] == [published.snapshot_id]
    units = await store.list_work_units(
        finished.run_id, graph_version=finished.graph_version
    )
    running_units = [unit for unit in units if unit.status is WorkUnitStatus.RUNNING]
    assert len(running_units) == 1
    calls = await store.list_tool_calls(
        finished.run_id,
        work_unit_id=running_units[0].work_unit_id,
        statuses=[ToolCallStatus.RUNNING],
    )
    assert len(calls) == 1
    assert calls[0].snapshot_id == published.snapshot_id
    claim_ids = await store.list_active_bridge_call_ids(
        principal_id=principal_id, workspace_id=workspace.workspace_id
    )
    assert claim_ids == (calls[0].call_id,)
    claim = await store.get_bridge_claim_for_session(
        session.session_id,
        finished.run_id,
        calls[0].call_id,
        principal_id=principal_id,
    )
    assert claim.client_id == client.client_id
    assert claim.run_id == finished.run_id
    assert claim.workspace_id == workspace.workspace_id
    assert claim.snapshot_id == published.snapshot_id
    events = await store.list_events(finished.run_id)
    assert EventType.BRIDGE_CLAIM_CREATED in {event.event_type for event in events}
    assert events[-1].event_type is not EventType.RUN_SUCCEEDED
    assert events[-1].event_type is not EventType.RUN_FAILED


class _BridgeRootToolThenJsonAdapter:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []
        self._root_tool_emitted = False

    @property
    def name(self) -> str:
        return "progressive-bridge-resume-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        from prp_runtime.domain.models import AgentToolCall

        if "produce root" in request.input and not self._root_tool_emitted:
            self._root_tool_emitted = True
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="provider-call/1",
                        tool_name="read_file",
                        arguments={"path": "src/main.py"},
                    ),
                ),
                usage=Usage(input_tokens=1, output_tokens=1),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        return ProviderResponse(
            text='{"ok":true}',
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


@pytest.mark.asyncio
async def test_bridge_progressive_resume_continues_same_attempt_after_result(
    store: SqliteStore,
) -> None:
    from prp_runtime.domain.enums import (
        AttemptStatus,
        ExecutionLocation,
        ToolCallStatus,
        ToolEffect,
    )
    from prp_runtime.domain.models import (
        AgentRequestOptions,
        ClientCapabilityDescriptor,
        RegisteredBridgeClient,
        Session,
        WorkspaceGrant,
        fingerprint_client_capabilities,
        new_client_id,
    )
    from prp_runtime.domain.values import new_principal_id, new_session_id, utc_now
    from prp_runtime.runtime.tooling import ScopeToolRuntimeProvider
    from prp_runtime.tools.models import ToolResult
    from prp_runtime.workspace.models import Workspace, WorkspaceSource, WorkspaceSourceType
    from tests.unit.storage.test_store import make_manifest, make_snapshot

    principal_id = new_principal_id()
    now = utc_now()
    workspace = Workspace(
        workspace_id="ws_" + "b" * 32,
        owner_id=principal_id,
        alias="client-project-resume",
        source=WorkspaceSource(
            source_type=WorkspaceSourceType.SERVER_ALIAS,
            server_alias="repo-main",
        ),
        created_at=now,
    )
    await store.create_workspace(workspace)
    snapshot = make_snapshot(workspace.workspace_id)
    published = await store.create_snapshot(
        snapshot, make_manifest(suffix="-resume"), owner_id=principal_id
    )
    agent_options = AgentRequestOptions(execution_location=ExecutionLocation.BRIDGE)
    session = Session(
        session_id=new_session_id(),
        principal_id=principal_id,
        workspace_id=workspace.workspace_id,
        grant=WorkspaceGrant(
            principal_id=principal_id,
            workspace_id=workspace.workspace_id,
        ),
        agent_options=agent_options,
        created_at=now,
    )
    await store.create_session(session)
    capabilities = ClientCapabilityDescriptor(
        tools=("read_file",),
        effects=(ToolEffect.READ,),
    )
    client = RegisteredBridgeClient(
        client_id=new_client_id(),
        principal_id=principal_id,
        workspace_id=workspace.workspace_id,
        capabilities=capabilities,
        capability_fingerprint=fingerprint_client_capabilities(capabilities),
        created_at=now,
        last_seen_at=now,
    )
    await store.register_bridge_client(client)
    await store.record_bridge_heartbeat(
        client.client_id,
        principal_id=principal_id,
        fingerprint=client.capability_fingerprint,
        at=utc_now(),
    )

    planner = RevisionPlannerAdapter(_chain_proposal(), None)
    worker = _BridgeRootToolThenJsonAdapter()
    settings = _settings()
    provider = ScopeToolRuntimeProvider(
        store,
        settings,
        factory=_ForbiddenBridgeFactory(),  # type: ignore[arg-type]
        enable_server_resolver=False,
    )
    controller = RunController(
        store,
        settings,
        {"planner": planner, "worker": worker},
        tool_executor_provider=provider.executor_for,
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="inspect then produce the tail",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.PROGRESSIVE,
            agent_options=agent_options,
            budget=Budget(max_plan_revisions=0, max_attempts=8),
        )
    )
    await store.attach_run_to_session(
        session.session_id, run.run_id, principal_id=principal_id
    )

    waiting = await controller.execute(run.run_id, principal_id=principal_id)
    assert waiting.status is RunStatus.RUNNING, waiting.error
    first_worker_calls = len(worker.requests)
    assert first_worker_calls == 1
    graph = await store.list_work_units(
        waiting.run_id, graph_version=waiting.graph_version
    )
    by_lineage = {unit.lineage_key: unit for unit in graph}
    root = by_lineage["root-lineage"]
    tail = by_lineage["tail-lineage"]
    assert root.status is WorkUnitStatus.RUNNING
    assert tail.status is not WorkUnitStatus.RUNNING
    assert tail.status is not WorkUnitStatus.SUCCEEDED
    root_attempts = await store.list_attempts(root.work_unit_id)
    assert len(root_attempts) == 1
    attempt_id = root_attempts[0].attempt_id
    assert root_attempts[0].status is AttemptStatus.RUNNING
    calls = await store.list_tool_calls(
        waiting.run_id, work_unit_id=root.work_unit_id, statuses=[ToolCallStatus.RUNNING]
    )
    assert len(calls) == 1

    still_waiting = await controller.execute(waiting.run_id, principal_id=principal_id)
    assert still_waiting.status is RunStatus.RUNNING
    assert len(worker.requests) == first_worker_calls
    assert len(await store.list_attempts(root.work_unit_id)) == 1

    await store.submit_bridge_tool_result(
        session.session_id,
        waiting.run_id,
        calls[0].call_id,
        ToolResult(
            call_id=calls[0].call_id,
            status=ToolCallStatus.SUCCEEDED,
            result={"content": "safe"},
            output="safe",
            completed_at=utc_now(),
        ),
        principal_id=principal_id,
        client_id=client.client_id,
    )
    facts_before_wave = await store.list_evidence(root.work_unit_id)
    assert any(item.rule == "bridge.result.observation" for item in facts_before_wave)
    assert len(worker.requests) == first_worker_calls
    assert (await store.get_work_unit(tail.work_unit_id)).status is not WorkUnitStatus.RUNNING
    assert (await store.get_work_unit(tail.work_unit_id)).status is not WorkUnitStatus.SUCCEEDED
    finished = await controller.execute(waiting.run_id, principal_id=principal_id)
    assert finished.status is RunStatus.SUCCEEDED, finished.error
    assert len(worker.requests) == 3
    resumed_attempts = await store.list_attempts(root.work_unit_id)
    assert [item.attempt_id for item in resumed_attempts] == [attempt_id]
    assert resumed_attempts[0].status is AttemptStatus.SUCCEEDED
    history = await store.list_agent_history(attempt_id)
    result_items = [
        record.item for record in history if getattr(record.item, "kind", None) == "tool_result"
    ]
    assert len(result_items) == 1
    tail_after = await store.get_work_unit(tail.work_unit_id)
    assert tail_after.status is WorkUnitStatus.SUCCEEDED
    replayed = await controller.execute(finished.run_id, principal_id=principal_id)
    assert replayed.status is RunStatus.SUCCEEDED
    assert len(worker.requests) == 3
    assert len(await store.list_attempts(root.work_unit_id)) == 1
    assert published.snapshot_id == calls[0].snapshot_id


def test_ready_wave_assigns_one_scoped_bridge_claim() -> None:
    from prp_runtime.runtime.scheduler import BridgeClientCandidate, assign_bridge_wave_claims

    live = BridgeClientCandidate(
        client_id="cli_live",
        workspace_id="ws_project",
        tools=("read_file",),
        liveness="LIVE",
    )
    foreign = BridgeClientCandidate(
        client_id="cli_other",
        workspace_id="ws_other",
        tools=("read_file",),
        liveness="LIVE",
    )
    claims, skipped = assign_bridge_wave_claims(
        ("wu_ready", "wu_second"),
        run_id="run_progressive01",
        graph_version=2,
        snapshot_id="snap_base01",
        workspace_id="ws_project",
        tool_name="read_file",
        candidates=(foreign, live),
        claimed_call_ids=("tc_ready",),
        call_ids={"wu_ready": "tc_ready"},
    )
    assert [claim.work_unit_id for claim in claims] == ["wu_second"]
    assert claims[0].client_id == "cli_live"
    assert claims[0].graph_version == 2
    assert claims[0].snapshot_id == "snap_base01"
    assert "workspace scope mismatch" in {reason for _client, reason in skipped}
    assert "duplicate active claim" in {reason for _client, reason in skipped}


def test_bridge_wave_replay_does_not_duplicate_claim() -> None:
    from prp_runtime.runtime.scheduler import BridgeClientCandidate, assign_bridge_wave_claims

    client = BridgeClientCandidate(
        client_id="cli_live",
        workspace_id="ws_project",
        tools=("read_file",),
        liveness="LIVE",
        max_active_claims=1,
    )
    first, _skipped = assign_bridge_wave_claims(
        ("wu_ready",),
        run_id="run_progressive01",
        graph_version=1,
        snapshot_id="snap_base01",
        workspace_id="ws_project",
        tool_name="read_file",
        candidates=(client,),
        call_ids={"wu_ready": "tc_ready"},
    )
    assert len(first) == 1
    replay, skipped = assign_bridge_wave_claims(
        ("wu_ready",),
        run_id="run_progressive01",
        graph_version=1,
        snapshot_id="snap_base01",
        workspace_id="ws_project",
        tool_name="read_file",
        candidates=(client,),
        claimed_call_ids=("tc_ready",),
        call_ids={"wu_ready": "tc_ready"},
    )
    assert replay == ()
    assert skipped == (("cli_live", "duplicate active claim"),)



@pytest.mark.asyncio
async def test_running_bridge_tool_resumes_as_remote_wait_and_replays_once(
    store: SqliteStore,
) -> None:
    from datetime import UTC, datetime

    from prp_runtime.domain.enums import ExecutionLocation, ToolCallStatus, ToolEffect
    from prp_runtime.domain.events import payload_from_remote_wait
    from prp_runtime.domain.models import (
        AgentHistoryRecord,
        AgentRequestOptions,
        AgentToolCall,
        AgentTurn,
        ExecutionScope,
        RemoteWaitFacts,
        WorkspaceGrant,
    )
    from prp_runtime.domain.values import new_principal_id, new_session_id
    from prp_runtime.runtime.agent_executor import AgentToolExecutor
    from prp_runtime.runtime.worker import ResumeAction, Worker
    from prp_runtime.tools.models import ToolCall, ToolResult
    from tests.unit.storage.test_store import (
        T0,
        make_attempt,
        make_manifest,
        make_run,
        make_snapshot,
        make_work_unit,
        make_workspace,
    )

    principal_id = new_principal_id()
    run = make_run()
    await store.create_run(run)
    unit = make_work_unit(run.run_id)
    await store.create_work_unit(unit)
    workspace = make_workspace(owner_id=principal_id)
    await store.create_workspace(workspace)
    snapshot = make_snapshot(workspace.workspace_id)
    await store.create_snapshot(snapshot, make_manifest(), owner_id=principal_id)
    attempt = make_attempt(
        run.run_id,
        unit.work_unit_id,
        status=AttemptStatus.SUCCEEDED,
        started_at=T0,
        completed_at=T0,
    )
    await store.create_attempt(attempt)
    public = AgentToolCall(
        call_id="provider-call/1",
        tool_name="read_file",
        arguments={"path": "src/main.py"},
    )
    internal_id = AgentToolExecutor._internal_call_id(
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        snapshot_id=snapshot.snapshot_id,
        provider_call_id=public.call_id,
        tool_name=public.tool_name,
    )
    persisted = ToolCall(
        call_id=internal_id,
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        tool_name="read_file",
        effect=ToolEffect.READ,
        arguments={"path": "src/main.py"},
        snapshot_id=snapshot.snapshot_id,
        requested_at=T0,
    )
    await store.create_tool_call(
        persisted, workspace_id=workspace.workspace_id, idempotency_key=internal_id
    )
    await store.start_tool_call(internal_id, started_at=T0)
    await store.append_agent_history(
        AgentHistoryRecord(
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            sequence=1,
            idempotency_key=f"{attempt.attempt_id}:1",
            item=AgentTurn(tool_calls=(public,)),
            created_at=T0,
        )
    )
    scope = ExecutionScope(
        run_id=run.run_id,
        session_id=new_session_id(),
        principal_id=principal_id,
        workspace_id=workspace.workspace_id,
        grant=WorkspaceGrant(
            principal_id=principal_id,
            workspace_id=workspace.workspace_id,
        ),
        agent_options=AgentRequestOptions(execution_location=ExecutionLocation.BRIDGE),
    )
    worker = Worker(
        store,
        _ProgressiveUnusedAdapter(),
        _profile("worker", ModelRole.WORKER),
        execution_scope=scope,
    )
    waiting = await worker.load_resume_state(attempt.attempt_id)
    assert waiting.action is ResumeAction.WAIT_REMOTE
    assert waiting.approval_request is None
    assert waiting.pending_call is not None
    assert waiting.pending_call.status is ToolCallStatus.RUNNING
    facts = RemoteWaitFacts(
        call_id=public.call_id,
        tool_call_id=internal_id,
        workspace_id=workspace.workspace_id,
    )
    payload = payload_from_remote_wait(facts)
    assert payload["reason"] == "remote_assignment_pending"
    assert "provider" not in payload

    durable = ToolResult(
        call_id=internal_id,
        status=ToolCallStatus.SUCCEEDED,
        result={"content": "safe"},
        output="safe",
        completed_at=datetime(2026, 8, 15, 12, 0, 1, tzinfo=UTC),
    )
    await store.complete_tool_call(durable)
    replay = await worker.load_resume_state(attempt.attempt_id)
    assert replay.action is ResumeAction.CONTINUE
    assert len(replay.replay_results) == 1
    assert replay.replay_results[0].call_id == public.call_id
    public_result = replay.replay_results[0]
    await store.append_agent_history(
        AgentHistoryRecord(
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            sequence=2,
            idempotency_key=f"{attempt.attempt_id}:2",
            item=public_result,
            created_at=T0,
        )
    )
    again = await worker.load_resume_state(attempt.attempt_id)
    assert again.action is ResumeAction.CONTINUE
    assert again.replay_results == ()


class _ProgressiveUnusedAdapter:
    @property
    def name(self) -> str:
        return "unused"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        del request
        raise AssertionError("remote wait resume must not call the provider")


def _source_chain_proposal() -> PlanProposal:
    return PlanProposal(
        summary="source facts",
        final_node="tail",
        nodes=(
            {
                "key": "root",
                "name": "Root",
                "instruction": "return the module",
                "output": {"kind": "TEXT"},
            },
            {
                "key": "tail",
                "name": "Tail",
                "instruction": "use the returned module",
                "depends_on": ["root"],
                "output": {"kind": "TEXT"},
            },
        ),
    )


@pytest.mark.asyncio
async def test_progressive_global_report_records_returned_ast_facts(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_source_chain_proposal(), None)
    worker = RevisionWorkerAdapter(
        ("def run():\n    return 1\n", "ready")
    )
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert any("function:run" in request.input for request in worker.requests[1:])
    assert all("/home/" not in request.input for request in worker.requests)
    reports = [
        event
        for event in await store.list_events(run.run_id)
        if event.event_type is EventType.CONTROLLER_DECISION
        and "global_report" in event.payload
    ]
    assert len(reports) == 1
    global_report = reports[0].payload["global_report"]
    assert global_report["result"] == "PASS"
    assert global_report["syntax_report_count"] >= 1
    assert any(check["kind"] == "AST" for check in global_report["checks"])


@pytest.mark.asyncio
async def test_progressive_unknown_ast_fact_stays_conservative(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(
        PlanProposal(
            summary="dynamic source",
            final_node="source",
            nodes=(
                {
                    "key": "source",
                    "name": "Source",
                    "instruction": "return the module",
                    "output": {"kind": "TEXT"},
                },
            ),
        ),
        None,
    )
    worker = RevisionWorkerAdapter(("value = eval('1')\n",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_revisions=0))

    finished = await controller.execute(run.run_id)

    assert finished.status is not RunStatus.SUCCEEDED
    reports = [
        event
        for event in await store.list_events(run.run_id)
        if event.event_type is EventType.CONTROLLER_DECISION
        and "global_report" in event.payload
    ]
    assert reports
    global_report = reports[0].payload["global_report"]
    assert global_report["result"] != "PASS"
    assert global_report["syntax_report_count"] >= 1


@pytest.mark.asyncio
async def test_deterministic_verification_does_not_fabricate_usage(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("measured"), None)
    worker = RevisionWorkerAdapter(('{"ok":true}',))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.usage == Usage(
        input_tokens=2,
        output_tokens=2,
        strong_model_tokens=2,
    )
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1


@pytest.mark.asyncio
async def test_role_token_ceiling_stops_before_next_provider_call(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("ceiling"), None)
    worker = RevisionWorkerAdapter(('{"ok":true}', '{"ok":true}'))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_total_tokens=2))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert len(worker.requests) == 0
    assert finished.usage.total_tokens == 2
