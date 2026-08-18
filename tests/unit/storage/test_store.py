"""Targeted tests for the SQLite operation set."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from prp_runtime.control.progressive import ProgressiveRound, RoundStatus
from prp_runtime.control.reservations import ReservationRequest
from prp_runtime.domain.enums import (
    AttemptStatus,
    BridgeClaimStatus,
    ExecutionLocation,
    ExecutionStrategy,
    ModelRole,
    ReservationStatus,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    ToolEffect,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, InternalError, StateError
from prp_runtime.domain.events import EventType, assert_sequence_chain, payload_from_model
from prp_runtime.domain.models import (
    AgentHistoryRecord,
    AgentRequestOptions,
    AgentToolResult,
    AgentTurn,
    Artifact,
    ArtifactKind,
    Attempt,
    Budget,
    ErrorCategory,
    ErrorInfo,
    Evidence,
    EvidenceKind,
    ExecutionScope,
    NativeRunRequest,
    OutputRequirement,
    Run,
    Session,
    SessionStatus,
    Usage,
    VerificationResult,
    WorkUnit,
    new_artifact_id,
    new_evidence_id,
)
from prp_runtime.domain.values import (
    ModelRef,
    ResourceClaim,
    new_attempt_id,
    new_run_id,
    new_session_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
    new_workspace_id,
)
from prp_runtime.planning.planner import new_planning_work_unit
from prp_runtime.policy.models import (
    ApprovalDecision,
    ApprovalIssuer,
    ApprovalOutcome,
    ApprovalRequest,
    CapabilityBudget,
    CapabilityScope,
    Lease,
    LeaseStatus,
)
from prp_runtime.runtime.event_bus import EventBus
from prp_runtime.storage.sqlite import (
    DanglingReferenceError,
    DuplicateEntityError,
    MissingEntityError,
    SqliteStore,
)
from prp_runtime.tools.models import ToolCall, ToolResult
from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
)
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

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
WORKER = ModelRef(provider="openai_compatible", model="weak-model")
PLANNER = ModelRef(provider="anthropic", model="strong-model")


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "store-test.db"


@pytest_asyncio.fixture
async def store(database_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(database_path) as opened:
        yield opened


def make_run(**overrides: object) -> Run:
    data: dict[str, object] = {
        "run_id": new_run_id(),
        "request": NativeRunRequest(input="summarise the report"),
        "created_at": T0,
    }
    data.update(overrides)
    return Run(**data)  # type: ignore[arg-type]


def make_work_unit(run_id: str, **overrides: object) -> WorkUnit:
    data: dict[str, object] = {
        "work_unit_id": new_work_unit_id(),
        "run_id": run_id,
        "name": "unit",
        "instruction": "do the work",
        "created_at": T0,
    }
    data.update(overrides)
    return WorkUnit(**data)  # type: ignore[arg-type]


def make_attempt(run_id: str, work_unit_id: str, **overrides: object) -> Attempt:
    data: dict[str, object] = {
        "attempt_id": new_attempt_id(),
        "run_id": run_id,
        "work_unit_id": work_unit_id,
        "role": ModelRole.WORKER,
        "model": WORKER,
        "created_at": T0,
    }
    data.update(overrides)
    return Attempt(**data)  # type: ignore[arg-type]


def make_reservation_request(
    run_id: str, work_unit_id: str, **overrides: object
) -> ReservationRequest:
    data: dict[str, object] = {
        "run_id": run_id,
        "work_unit_id": work_unit_id,
        "dispatch_key": "dispatch-1",
        "attempt_units": 1,
        "token_upper_bound": 8,
        "strong_token_upper_bound": 4,
        "capacity_key": "worker-capacity",
    }
    data.update(overrides)
    return ReservationRequest(**data)  # type: ignore[arg-type]


def make_workspace(owner_id: str = "owner-1", alias: str = "project-main") -> Workspace:
    return Workspace(
        workspace_id=new_workspace_id(),
        owner_id=owner_id,
        alias=alias,
        source=WorkspaceSource(
            source_type=WorkspaceSourceType.SERVER_ALIAS,
            server_alias="repo-main",
        ),
        created_at=T0,
    )


def make_manifest(*, suffix: str = "") -> SnapshotManifest:
    return SnapshotManifest(
        entries=(
            SnapshotEntry(
                path=f"src/main{suffix}.py",
                sha256="a" * 64,
                size=4,
                entry_type=SnapshotEntryType.FILE,
            ),
            SnapshotEntry(
                path="src",
                sha256="b" * 64,
                size=0,
                entry_type=SnapshotEntryType.DIRECTORY,
            ),
        )
    )


def make_snapshot(workspace_id: str) -> Snapshot:
    return Snapshot(
        snapshot_id=new_snapshot_id(),
        workspace_id=workspace_id,
        status=SnapshotStatus.READY,
        created_at=T0,
        completed_at=T0 + timedelta(seconds=1),
    )


def make_change_set(
    run: Run,
    workspace: Workspace,
    base: Snapshot,
    new: Snapshot,
    call: ToolCall,
) -> ChangeSet:
    return ChangeSet(
        change_set_id="cs_" + "a" * 32,
        run_id=run.run_id,
        tool_call_id=call.call_id,
        workspace_id=workspace.workspace_id,
        base_snapshot_id=base.snapshot_id,
        new_snapshot_id=new.snapshot_id,
        patch=Patch(
            base_snapshot_id=base.snapshot_id,
            unified_diff="--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-old\n+new\n",
        ),
        files=(
            FileChange(
                path="src/main.py",
                action=FileChangeAction.MODIFY,
                before=FileContent(sha256="a" * 64, size=4),
                after=FileContent(sha256="b" * 64, size=4),
            ),
        ),
        created_at=T0,
    )


async def seed(store: SqliteStore) -> tuple[Run, WorkUnit, Attempt]:
    run = make_run()
    await store.create_run(run)
    unit = make_work_unit(run.run_id)
    await store.create_work_unit(unit)
    attempt = make_attempt(run.run_id, unit.work_unit_id)
    await store.create_attempt(attempt)
    return run, unit, attempt


@pytest.mark.asyncio
async def test_agent_history_append_is_idempotent_and_conflict_checked(
    store: SqliteStore,
) -> None:
    run, unit, attempt = await seed(store)
    first = AgentHistoryRecord(
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        sequence=1,
        idempotency_key=f"{attempt.attempt_id}:1",
        item=AgentTurn(text="public turn"),
        created_at=T0,
    )

    assert await store.append_agent_history(first) == first
    assert await store.append_agent_history(first) == first
    assert await store.list_agent_history(attempt.attempt_id) == (first,)
    assert [event.event_type for event in await store.list_events(run.run_id)] == [
        EventType.AGENT_HISTORY_RECORDED
    ]

    with pytest.raises(StateError, match="agent history"):
        await store.append_agent_history(
            first.model_copy(update={"item": AgentTurn(text="different turn")})
        )
    with pytest.raises(StateError, match="agent history"):
        await store.append_agent_history(
            first.model_copy(update={"idempotency_key": "different-key"})
        )

    second = AgentHistoryRecord(
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        sequence=2,
        idempotency_key=f"{attempt.attempt_id}:2",
        item=AgentToolResult(
            call_id="tc_history",
            status=ToolCallStatus.SUCCEEDED,
            result={"content": "safe"},
        ),
        created_at=T0,
    )
    await store.append_agent_history(second)
    assert await store.list_agent_history(attempt.attempt_id) == (first, second)
    assert len(await store.list_events(run.run_id)) == 2


def make_session(
    workspace: Workspace,
    *,
    principal_id: str | None = None,
    **overrides: object,
) -> Session:
    owner = workspace.owner_id if principal_id is None else principal_id
    values: dict[str, object] = {
        "session_id": new_session_id(),
        "principal_id": owner,
        "workspace_id": workspace.workspace_id,
        "grant": {
            "principal_id": owner,
            "workspace_id": workspace.workspace_id,
            "access": (ResourceAccess.READ,),
        },
        "created_at": T0,
    }
    values.update(overrides)
    return Session(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_execution_scope_restores_session_facts_and_non_session_is_none(
    store: SqliteStore,
) -> None:
    workspace = make_workspace(owner_id="prn_owner_1")
    await store.create_workspace(workspace)
    session = make_session(workspace)
    await store.create_session(session)
    run = make_run()
    await store.create_run(run)
    await store.attach_run_to_session(
        session.session_id, run.run_id, principal_id="prn_owner_1"
    )

    scope = await store.get_execution_scope(run.run_id, principal_id="prn_owner_1")

    assert isinstance(scope, ExecutionScope)
    assert scope.run_id == run.run_id
    assert scope.session_id == session.session_id
    assert scope.workspace_id == workspace.workspace_id
    assert scope.agent_options == session.agent_options
    assert await store.get_execution_scope(new_run_id(), principal_id="prn_owner_1") is None
    assert await store.get_execution_scope(run.run_id, principal_id="other-owner") is None
    assert "root" not in scope.model_fields


@pytest.mark.asyncio
async def test_execution_scope_rechecks_session_and_workspace_lifecycle(
    store: SqliteStore,
) -> None:
    expired_workspace = make_workspace(owner_id="prn_owner_expired")
    await store.create_workspace(expired_workspace)
    expired = make_session(expired_workspace, expires_at=T0 + timedelta(seconds=1))
    await store.create_session(expired)
    expired_run = make_run()
    await store.create_run(expired_run)
    await store.attach_run_to_session(
        expired.session_id, expired_run.run_id, principal_id="prn_owner_expired"
    )
    with pytest.raises(StateError, match="expired"):
        await store.get_execution_scope(expired_run.run_id, principal_id="prn_owner_expired")

    revoked_workspace = make_workspace(owner_id="prn_owner_revoked")
    await store.create_workspace(revoked_workspace)
    revoked = make_session(
        revoked_workspace,
        status=SessionStatus.REVOKED,
        revoked_at=T0 + timedelta(seconds=1),
    )
    await store.create_session(revoked)
    revoked_run = make_run()
    await store.create_run(revoked_run)
    await store.attach_run_to_session(
        revoked.session_id, revoked_run.run_id, principal_id="prn_owner_revoked"
    )
    with pytest.raises(StateError, match="not active"):
        await store.get_execution_scope(revoked_run.run_id, principal_id="prn_owner_revoked")


async def seed_tool_context(
    store: SqliteStore,
    *,
    manifest_suffix: str = "",
    workspace_alias: str = "project-main",
) -> tuple[Run, WorkUnit, Workspace, Snapshot, ToolCall]:
    run = make_run()
    await store.create_run(run)
    unit = make_work_unit(run.run_id)
    await store.create_work_unit(unit)
    workspace = make_workspace(alias=workspace_alias)
    await store.create_workspace(workspace)
    snapshot = make_snapshot(workspace.workspace_id)
    await store.create_snapshot(
        snapshot,
        make_manifest(suffix=manifest_suffix),
        owner_id=workspace.owner_id,
    )
    call = ToolCall(
        call_id=new_tool_call_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        tool_name="read_file",
        effect=ToolEffect.READ,
        arguments={"path": "src/main.py"},
        snapshot_id=snapshot.snapshot_id,
        requested_at=T0,
    )
    return run, unit, workspace, snapshot, call


@pytest.mark.asyncio
async def test_progressive_round_round_trip_is_immutable_and_evented(
    store: SqliteStore,
) -> None:
    run, _, _, snapshot, _ = await seed_tool_context(store)
    progressive_round = ProgressiveRound(
        round_id="round_" + "a" * 32,
        run_id=run.run_id,
        round_index=0,
        graph_version=1,
        base_snapshot_id=snapshot.snapshot_id,
        status=RoundStatus.PLANNED,
        created_at=T0,
    )

    persisted = await store.create_round(progressive_round)
    replay = await store.create_progressive_round(progressive_round)
    restored = await store.get_progressive_round(progressive_round.round_id)
    listed = await store.list_rounds(run.run_id)

    assert persisted == replay == restored
    assert listed == (progressive_round,)
    with pytest.raises(DuplicateEntityError, match="different immutable"):
        await store.create_round(progressive_round.model_copy(update={"graph_version": 2}))
    events = await store.list_events(run.run_id)
    assert any(event.event_type is EventType.ROUND_CREATED for event in events)


@pytest.mark.asyncio
async def test_planned_progressive_round_closes_once(store: SqliteStore) -> None:
    run, _, _, snapshot, _ = await seed_tool_context(store)
    planned = ProgressiveRound(
        round_id="round_" + "b" * 32,
        run_id=run.run_id,
        round_index=0,
        graph_version=1,
        base_snapshot_id=snapshot.snapshot_id,
        status=RoundStatus.PLANNED,
        created_at=T0,
    )
    await store.create_round(planned)
    failed = planned.model_copy(
        update={
            "status": RoundStatus.FAILED,
            "failure_reason": "merge roots unavailable",
            "completed_at": T0 + timedelta(seconds=1),
        }
    )

    assert await store.update_round(failed) == failed
    assert await store.update_progressive_round(failed) == failed
    with pytest.raises(StateError, match="immutable"):
        await store.update_round(
            failed.model_copy(update={"failure_reason": "different fact"})
        )


@pytest.mark.asyncio
async def test_planner_attempt_uses_existing_work_unit_foreign_key_and_usage_columns(
    store: SqliteStore,
) -> None:
    run = make_run()
    await store.create_run(run)
    planning_unit = new_planning_work_unit(run.run_id)
    await store.create_work_unit(planning_unit)
    usage = Usage(
        input_tokens=3,
        output_tokens=2,
        strong_model_tokens=5,
        elapsed_ms=7,
    )
    attempt = make_attempt(
        run.run_id,
        planning_unit.work_unit_id,
        role=ModelRole.PLANNER,
        model=PLANNER,
        status=AttemptStatus.SUCCEEDED,
        usage=usage,
        started_at=T0,
        completed_at=T0,
    )

    await store.create_attempt(attempt)

    assert await store.get_work_unit(planning_unit.work_unit_id) == planning_unit
    assert await store.list_run_attempts(run.run_id) == (attempt,)
    assert (await store.list_run_attempts(run.run_id))[0].usage == usage


# --- runs -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_round_trips_with_every_field(store: SqliteStore) -> None:
    run = make_run(
        request=NativeRunRequest(
            input="  analyse   ",
            instructions="be concise",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.PROGRESSIVE,
            budget=Budget(max_total_tokens=500, max_attempts=4, deadline=T0),
            output=OutputRequirement(kind=ArtifactKind.JSON, json_schema='{"type":"object"}'),
        ),
        status=RunStatus.RUNNING,
        strategy=ExecutionStrategy.PROGRESSIVE,
        graph_version=3,
        usage=Usage(input_tokens=7, output_tokens=9, strong_model_tokens=5, elapsed_ms=120),
        started_at=T0 + timedelta(seconds=1),
    )
    await store.create_run(run)
    assert await store.get_run(run.run_id) == run


@pytest.mark.asyncio
async def test_run_final_work_unit_id_round_trips_only_for_same_run_and_graph(
    store: SqliteStore,
) -> None:
    run = make_run(graph_version=3)
    await store.create_run(run)
    unit = make_work_unit(run.run_id, graph_version=3)
    await store.create_work_unit(unit)
    planned = run.model_copy(update={"final_work_unit_id": unit.work_unit_id})
    await store.update_run(planned)
    assert await store.get_run(run.run_id) == planned

    other_run = make_run(graph_version=3)
    await store.create_run(other_run)
    with pytest.raises(DanglingReferenceError):
        await store.update_run(
            other_run.model_copy(update={"final_work_unit_id": unit.work_unit_id})
        )

    other_graph_unit = make_work_unit(run.run_id, graph_version=4)
    await store.create_work_unit(other_graph_unit)
    with pytest.raises(DanglingReferenceError):
        await store.update_run(
            run.model_copy(update={"final_work_unit_id": other_graph_unit.work_unit_id})
        )


@pytest.mark.asyncio
async def test_duplicate_run_id_is_rejected(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(DuplicateEntityError):
        await store.create_run(run)


@pytest.mark.asyncio
async def test_missing_run_reads_and_updates_fail_with_run_not_found(
    store: SqliteStore,
) -> None:
    with pytest.raises(MissingEntityError) as excinfo:
        await store.get_run("run_missing")
    assert excinfo.value.code is ErrorCode.RUN_NOT_FOUND
    with pytest.raises(MissingEntityError):
        await store.update_run(make_run())


@pytest.mark.asyncio
async def test_event_hints_publish_after_commit_and_not_after_rollback(
    database_path: Path,
) -> None:
    bus = EventBus()
    run = make_run()
    async with SqliteStore(database_path, event_bus=bus) as store:
        subscription = await bus.subscribe(run.run_id)
        await store.create_run(run)
        with pytest.raises(RuntimeError, match="rollback"):
            async with store.transaction():
                await store.append_event(
                    run.run_id, EventType.RUN_CREATED, {"request": {}}
                )
                raise RuntimeError("rollback")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(subscription.get(), timeout=0.01)
        assert await store.list_events(run.run_id) == ()

        await store.append_event(run.run_id, EventType.RUN_CREATED, {"request": {}})
        assert await asyncio.wait_for(subscription.get(), timeout=1.0) == 1
        await subscription.close()
    await bus.close()


# --- workspace and snapshots ---------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_round_trip_and_owner_scoped_queries(store: SqliteStore) -> None:
    first = make_workspace(owner_id="owner-1", alias="first")
    second = make_workspace(owner_id="owner-2", alias="second")
    await store.create_workspace(first)
    await store.create_workspace(second)

    assert await store.get_workspace(first.workspace_id, owner_id="owner-1") == first
    assert await store.list_workspaces(owner_id="owner-1") == (first,)
    with pytest.raises(MissingEntityError):
        await store.get_workspace(first.workspace_id, owner_id="owner-2")
    with pytest.raises(DuplicateEntityError):
        await store.create_workspace(first)


@pytest.mark.asyncio
async def test_snapshot_manifest_round_trips_sorted_and_is_idempotent(
    store: SqliteStore,
) -> None:
    workspace = make_workspace()
    await store.create_workspace(workspace)
    manifest = make_manifest()
    snapshot = make_snapshot(workspace.workspace_id)

    persisted = await store.create_snapshot(snapshot, manifest, owner_id="owner-1")
    assert persisted.file_count == 2
    assert persisted.total_size == 4
    assert await store.get_snapshot(snapshot.snapshot_id, owner_id="owner-1") == persisted
    round_trip = await store.get_snapshot_manifest(snapshot.snapshot_id, owner_id="owner-1")
    assert round_trip.manifest_hash == manifest.manifest_hash
    assert round_trip.total_size == manifest.total_size
    assert tuple(entry.path for entry in round_trip.entries) == tuple(
        sorted(entry.path for entry in manifest.entries)
    )
    assert await store.list_snapshots(workspace.workspace_id, owner_id="owner-1") == (persisted,)

    replay = await store.create_snapshot(
        make_snapshot(workspace.workspace_id), manifest, owner_id="owner-1"
    )
    assert replay.snapshot_id == snapshot.snapshot_id


@pytest.mark.asyncio
async def test_snapshot_owner_scope_and_manifest_conflicts_are_rejected(
    store: SqliteStore,
) -> None:
    first = make_workspace(owner_id="owner-1", alias="first")
    second = make_workspace(owner_id="owner-2", alias="second")
    await store.create_workspace(first)
    await store.create_workspace(second)
    manifest = make_manifest()
    snapshot = make_snapshot(first.workspace_id)
    await store.create_snapshot(snapshot, manifest, owner_id="owner-1")

    with pytest.raises(MissingEntityError):
        await store.get_snapshot(snapshot.snapshot_id, owner_id="owner-2")
    with pytest.raises(MissingEntityError):
        await store.get_snapshot_manifest(snapshot.snapshot_id, owner_id="owner-2")
    with pytest.raises(DuplicateEntityError):
        await store.create_snapshot(
            make_snapshot(second.workspace_id), manifest, owner_id="owner-2"
        )
    with pytest.raises(DuplicateEntityError):
        await store.create_snapshot(
            snapshot,
            make_manifest(suffix="-other"),
            owner_id="owner-1",
        )


@pytest.mark.asyncio
async def test_change_sets_round_trip_are_tool_idempotent_and_scoped(
    store: SqliteStore,
) -> None:
    run, _, workspace, base, call = await seed_tool_context(store)
    write_call = call.model_copy(update={"tool_name": "apply_patch", "effect": ToolEffect.WRITE})
    await store.create_tool_call(
        write_call,
        workspace_id=workspace.workspace_id,
        idempotency_key="patch-request-1",
    )
    new_snapshot = make_snapshot(workspace.workspace_id)
    await store.create_snapshot(
        new_snapshot,
        make_manifest(suffix="-updated"),
        owner_id=workspace.owner_id,
    )
    change_set = make_change_set(run, workspace, base, new_snapshot, write_call)

    assert await store.create_change_set(change_set) == change_set
    assert await store.get_change_set(change_set.change_set_id) == change_set
    assert await store.list_change_sets(run_id=run.run_id) == (change_set,)
    assert await store.list_change_sets(tool_call_id=write_call.call_id) == (change_set,)
    assert await store.list_change_sets(workspace_id=workspace.workspace_id) == (change_set,)
    assert (
        await store.create_change_set(
            change_set.model_copy(update={"change_set_id": "cs_" + "b" * 32})
        )
        == change_set
    )
    with pytest.raises(DuplicateEntityError, match="different ChangeSet"):
        await store.create_change_set(
            change_set.model_copy(
                update={
                    "patch": Patch(
                        base_snapshot_id=base.snapshot_id,
                        unified_diff="different patch",
                    )
                }
            )
        )
    with pytest.raises(ValueError, match="requires a run"):
        await store.list_change_sets()


# --- tool calls and results ----------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_lifecycle_round_trips_and_appends_auditable_events(
    store: SqliteStore,
) -> None:
    run, unit, workspace, snapshot, call = await seed_tool_context(store)
    created = await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="tool-request-1",
    )
    assert created == call
    assert await store.get_tool_call(call.call_id) == call

    awaiting = await store.await_tool_call(call.call_id, reason="write requires approval")
    running = await store.start_tool_call(call.call_id, approved=True, started_at=T0)
    assert awaiting.status is ToolCallStatus.AWAITING_APPROVAL
    assert running.status is ToolCallStatus.RUNNING

    result = ToolResult.from_call(
        running,
        status=ToolCallStatus.SUCCEEDED,
        result={"content": "done"},
        output="done",
        truncated=True,
        changed_paths=("src/main.py",),
        exit_code=0,
        completed_at=T0 + timedelta(seconds=1),
    )
    assert await store.complete_tool_call(result) == result
    assert (await store.get_tool_call(call.call_id)).status is ToolCallStatus.SUCCEEDED
    assert await store.get_tool_result(call.call_id) == result
    assert await store.list_tool_calls(run.run_id, work_unit_id=unit.work_unit_id) == (
        call.model_copy(update={"status": ToolCallStatus.SUCCEEDED}),
    )
    assert [event.event_type for event in await store.list_events(run.run_id)] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_AWAITING_APPROVAL,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_SUCCEEDED,
    ]
    assert snapshot.snapshot_id == call.snapshot_id


async def seed_bridge_tool_call(store: SqliteStore) -> tuple[Run, Session, ToolCall]:
    workspace = make_workspace(owner_id="prn_bridge")
    await store.create_workspace(workspace)
    session = make_session(
        workspace,
        agent_options=AgentRequestOptions(execution_location=ExecutionLocation.BRIDGE),
    )
    await store.create_session(session)
    run, unit, _, _, _ = await seed_tool_context(
        store,
        workspace_alias="unused",
    )
    await store.attach_run_to_session(
        session.session_id, run.run_id, principal_id="prn_bridge"
    )
    # seed_tool_context creates a second workspace; build the call in the scoped one.
    scoped_snapshot = make_snapshot(workspace.workspace_id)
    await store.create_snapshot(
        scoped_snapshot,
        make_manifest(suffix="-bridge"),
        owner_id=workspace.owner_id,
    )
    scoped_call = ToolCall(
        call_id=new_tool_call_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        tool_name="read_file",
        effect=ToolEffect.READ,
        arguments={"path": "src/main.py"},
        snapshot_id=scoped_snapshot.snapshot_id,
        requested_at=T0,
    )
    await store.create_tool_call(
        scoped_call,
        workspace_id=workspace.workspace_id,
        idempotency_key=scoped_call.call_id,
    )
    await store.start_tool_call(scoped_call.call_id, started_at=T0)
    return run, session, scoped_call


@pytest.mark.asyncio
async def test_bridge_claim_is_owner_scoped_idempotent_and_single_active(
    store: SqliteStore,
) -> None:
    run, session, call = await seed_bridge_tool_call(store)
    first = await store.claim_tool_call(
        session.session_id,
        run.run_id,
        call.call_id,
        principal_id="prn_bridge",
        claimant_id="bridge-a",
        idempotency_key="claim-key-1",
        claimed_at=T0,
        expires_at=T0 + timedelta(minutes=1),
    )
    replay = await store.claim_tool_call(
        session.session_id,
        run.run_id,
        call.call_id,
        principal_id="prn_bridge",
        claimant_id="bridge-a",
        idempotency_key="claim-key-1",
        claimed_at=T0,
        expires_at=T0 + timedelta(minutes=1),
    )
    assert replay == first
    assert await store.get_bridge_claim(first.claim_id, principal_id="prn_bridge") == first
    with pytest.raises(DuplicateEntityError, match="active Bridge claim"):
        await store.claim_tool_call(
            session.session_id,
            run.run_id,
            call.call_id,
            principal_id="prn_bridge",
            claimant_id="bridge-b",
            idempotency_key="claim-key-2",
            claimed_at=T0,
        )
    with pytest.raises(MissingEntityError):
        await store.get_bridge_claim(first.claim_id, principal_id="prn_other")
    assert [event.event_type for event in await store.list_events(run.run_id)] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_STARTED,
        EventType.BRIDGE_CLAIM_CREATED,
    ]


@pytest.mark.asyncio
async def test_bridge_claim_expiry_allows_a_new_key_without_replaying_old_claim(
    store: SqliteStore,
) -> None:
    run, session, call = await seed_bridge_tool_call(store)
    first = await store.claim_tool_call(
        session.session_id,
        run.run_id,
        call.call_id,
        principal_id="prn_bridge",
        claimant_id="bridge-a",
        idempotency_key="claim-key-1",
        claimed_at=T0,
        expires_at=T0 + timedelta(seconds=5),
    )
    second = await store.claim_tool_call(
        session.session_id,
        run.run_id,
        call.call_id,
        principal_id="prn_bridge",
        claimant_id="bridge-b",
        idempotency_key="claim-key-2",
        claimed_at=T0 + timedelta(seconds=6),
    )
    expired = await store.get_bridge_claim(first.claim_id, principal_id="prn_bridge")
    assert expired.status is BridgeClaimStatus.EXPIRED
    assert second.status is BridgeClaimStatus.ACTIVE


@pytest.mark.asyncio
async def test_bridge_result_completes_and_settles_atomically_then_replays(
    store: SqliteStore,
) -> None:
    run, session, call = await seed_bridge_tool_call(store)
    claim = await store.claim_tool_call(
        session.session_id,
        run.run_id,
        call.call_id,
        principal_id="prn_bridge",
        claimant_id="prn_bridge",
        idempotency_key="claim-result-key",
        claimed_at=T0,
        expires_at=T0 + timedelta(minutes=1),
    )
    result = ToolResult(
        call_id=call.call_id,
        status=ToolCallStatus.SUCCEEDED,
        result={"content": "bridge result"},
        output="bridge result",
        completed_at=T0 + timedelta(seconds=1),
    )
    completed, replayed = await store.submit_bridge_tool_result(
        session.session_id,
        run.run_id,
        call.call_id,
        result,
        principal_id="prn_bridge",
        claimant_id="prn_bridge",
        settled_at=T0 + timedelta(seconds=1),
    )
    assert completed == result
    assert replayed is False
    assert (await store.get_tool_call(call.call_id)).status is ToolCallStatus.SUCCEEDED
    settled = await store.get_bridge_claim(claim.claim_id, principal_id="prn_bridge")
    assert settled.status is BridgeClaimStatus.SETTLED

    replay, replayed = await store.submit_bridge_tool_result(
        session.session_id,
        run.run_id,
        call.call_id,
        result.model_copy(update={"completed_at": T0 + timedelta(seconds=2)}),
        principal_id="prn_bridge",
        claimant_id="prn_bridge",
        settled_at=T0 + timedelta(seconds=2),
    )
    assert replay == result
    assert replayed is True
    assert [event.event_type for event in await store.list_events(run.run_id)] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_STARTED,
        EventType.BRIDGE_CLAIM_CREATED,
        EventType.TOOL_CALL_SUCCEEDED,
        EventType.BRIDGE_CLAIM_SETTLED,
    ]
    with pytest.raises(StateError, match="conflicting result"):
        await store.submit_bridge_tool_result(
            session.session_id,
            run.run_id,
            call.call_id,
            result.model_copy(update={"output": "different"}),
            principal_id="prn_bridge",
            claimant_id="prn_bridge",
        )


@pytest.mark.asyncio
async def test_expired_bridge_claim_cannot_submit_a_result(
    store: SqliteStore,
) -> None:
    run, session, call = await seed_bridge_tool_call(store)
    await store.claim_tool_call(
        session.session_id,
        run.run_id,
        call.call_id,
        principal_id="prn_bridge",
        claimant_id="prn_bridge",
        idempotency_key="expired-result-key",
        claimed_at=T0,
        expires_at=T0 + timedelta(seconds=1),
    )
    result = ToolResult(
        call_id=call.call_id,
        status=ToolCallStatus.SUCCEEDED,
        completed_at=T0 + timedelta(seconds=2),
    )
    with pytest.raises(StateError, match="expired"):
        await store.submit_bridge_tool_result(
            session.session_id,
            run.run_id,
            call.call_id,
            result,
            principal_id="prn_bridge",
            claimant_id="prn_bridge",
            settled_at=T0 + timedelta(seconds=2),
        )
    with pytest.raises(MissingEntityError):
        await store.get_tool_result(call.call_id)
    assert (await store.get_tool_call(call.call_id)).status is ToolCallStatus.RUNNING


@pytest.mark.asyncio
async def test_requested_tool_rejection_is_atomic_and_idempotent(
    store: SqliteStore,
) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="tool-request-1",
    )

    rejected = await store.reject_tool_call(
        call.call_id,
        reason="grant_denied",
        completed_at=T0,
    )
    assert rejected.status is ToolCallStatus.REJECTED
    assert (await store.get_tool_call(call.call_id)).status is ToolCallStatus.REJECTED
    assert await store.get_tool_result(call.call_id) == rejected
    replay = await store.reject_tool_call(
        call.call_id,
        reason="grant_denied",
        completed_at=T0 + timedelta(seconds=1),
    )
    assert replay == rejected
    with pytest.raises(StateError, match="conflicting result"):
        await store.reject_tool_call(call.call_id, reason="scope_mismatch")
    with pytest.raises(ValueError, match="REJECTED"):
        await store.start_tool_call(call.call_id)

    events = await store.list_events(run.run_id)
    assert [event.event_type for event in events] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_REJECTED,
    ]
    assert "arguments" not in events[-1].payload


@pytest.mark.asyncio
async def test_tool_call_create_and_result_completion_are_idempotent(
    store: SqliteStore,
) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="tool-request-1",
    )
    replay = await store.create_tool_call(
        call.model_copy(update={"call_id": new_tool_call_id()}),
        workspace_id=workspace.workspace_id,
        idempotency_key="tool-request-1",
    )
    assert replay.call_id == call.call_id
    with pytest.raises(DuplicateEntityError, match="idempotency key"):
        await store.create_tool_call(
            call.model_copy(
                update={"call_id": new_tool_call_id(), "arguments": {"path": "other.py"}}
            ),
            workspace_id=workspace.workspace_id,
            idempotency_key="tool-request-1",
        )

    running = await store.start_tool_call(call.call_id, started_at=T0)
    result = ToolResult.from_call(
        running,
        status=ToolCallStatus.SUCCEEDED,
        output="done",
        completed_at=T0 + timedelta(seconds=1),
    )
    assert await store.complete_tool_call(call.call_id, result) == result
    assert await store.complete_tool_call(result) == result
    with pytest.raises(StateError, match="conflicting result"):
        await store.complete_tool_call(
            ToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.SUCCEEDED,
                output="different",
                completed_at=T0 + timedelta(seconds=2),
            )
        )
    assert len(await store.list_events(run.run_id)) == 3


@pytest.mark.asyncio
async def test_tool_completion_rolls_back_call_result_and_event_together(
    store: SqliteStore,
) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="tool-request-1",
    )
    running = await store.start_tool_call(call.call_id, started_at=T0)
    result = ToolResult.from_call(
        running,
        status=ToolCallStatus.SUCCEEDED,
        output="done",
        completed_at=T0 + timedelta(seconds=1),
    )
    with pytest.raises(RuntimeError, match="rollback"):
        async with store.transaction():
            await store.complete_tool_call(result)
            raise RuntimeError("rollback")
    assert (await store.get_tool_call(call.call_id)).status is ToolCallStatus.RUNNING
    with pytest.raises(MissingEntityError):
        await store.get_tool_result(call.call_id)
    assert [event.event_type for event in await store.list_events(run.run_id)] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.TOOL_CALL_STARTED,
    ]


@pytest.mark.asyncio
async def test_concurrent_same_result_has_one_persisted_result_and_event(
    store: SqliteStore,
    database_path: Path,
) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="tool-request-1",
    )
    running = await store.start_tool_call(call.call_id, started_at=T0)
    result = ToolResult.from_call(
        running,
        status=ToolCallStatus.SUCCEEDED,
        output="done",
        completed_at=T0 + timedelta(seconds=1),
    )
    async with SqliteStore(database_path) as other:
        first, second = await asyncio.gather(
            store.complete_tool_call(result),
            other.complete_tool_call(result),
        )
        assert first == result
        assert second == result
    assert await store.get_tool_result(call.call_id) == result
    events = await store.list_events(run.run_id)
    assert [event.event_type for event in events].count(EventType.TOOL_CALL_SUCCEEDED) == 1


@pytest.mark.asyncio
async def test_tool_facts_survive_store_reopen_and_unknown_outcome_is_preserved(
    store: SqliteStore,
    database_path: Path,
) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="tool-request-1",
    )
    await store.start_tool_call(call.call_id, started_at=T0)
    await store.close()
    async with SqliteStore(database_path) as reopened:
        unknown = await reopened.mark_tool_call_unknown(
            call.call_id,
            completed_at=T0 + timedelta(seconds=1),
        )
        assert unknown.status is ToolCallStatus.UNKNOWN
        assert (await reopened.get_tool_call(call.call_id)).status is ToolCallStatus.UNKNOWN
        assert (await reopened.get_tool_result(call.call_id)).status is ToolCallStatus.UNKNOWN
    assert run.run_id == call.run_id


@pytest.mark.asyncio
async def test_tool_listing_is_scoped_to_the_requested_run_and_work_unit(
    store: SqliteStore,
) -> None:
    first_run, first_unit, first_workspace, _, first_call = await seed_tool_context(store)
    second_run, second_unit, second_workspace, _, second_call = await seed_tool_context(
        store, manifest_suffix="-other", workspace_alias="project-other"
    )
    await store.create_tool_call(
        first_call,
        workspace_id=first_workspace.workspace_id,
        idempotency_key="same-request-key",
    )
    await store.create_tool_call(
        second_call,
        workspace_id=second_workspace.workspace_id,
        idempotency_key="same-request-key",
    )
    assert await store.list_tool_calls(first_run.run_id) == (first_call,)
    assert await store.list_tool_calls(second_run.run_id) == (second_call,)
    assert (
        await store.list_tool_calls(first_run.run_id, work_unit_id=second_unit.work_unit_id)
        == ()
    )
    assert await store.list_tool_calls(
        second_run.run_id, work_unit_id=second_unit.work_unit_id
    ) == (
        second_call,
    )
    assert first_unit.work_unit_id != second_unit.work_unit_id


def make_capability_scope(workspace_id: str) -> CapabilityScope:
    return CapabilityScope(
        tools=("read_file",),
        effects=(ToolEffect.READ,),
        workspace_id=workspace_id,
        paths=("src/**",),
        budget=CapabilityBudget(
            max_calls=5,
            max_output_bytes=4096,
            max_wall_clock_ms=30_000,
        ),
        expires_at=T0 + timedelta(hours=1),
    )


def make_approval_request(
    run: Run, workspace: Workspace, call: ToolCall
) -> ApprovalRequest:
    return ApprovalRequest(
        call_id=call.call_id,
        run_id=run.run_id,
        workspace_id=workspace.workspace_id,
        tool_name=call.tool_name,
        effect=call.effect,
        scope=make_capability_scope(workspace.workspace_id),
        reason="read the requested source file",
        requested_at=T0,
    )


@pytest.mark.asyncio
async def test_approval_decision_and_lease_are_owner_scoped_and_idempotent(
    store: SqliteStore,
) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="approval-tool-request",
    )
    request = make_approval_request(run, workspace, call)
    assert await store.create_approval(request, owner_id=workspace.owner_id) == request
    assert await store.create_approval(request, owner_id=workspace.owner_id) == request
    assert await store.get_approval(request.request_id, owner_id=workspace.owner_id) == request
    with pytest.raises(MissingEntityError):
        await store.get_approval(request.request_id, owner_id="other-owner")
    assert await store.list_approvals(owner_id=workspace.owner_id, run_id=run.run_id) == (
        request,
    )

    decision = ApprovalDecision(
        approval_request_id=request.request_id,
        outcome=ApprovalOutcome.ALLOW,
        issuer=ApprovalIssuer.USER,
        decided_at=T0,
    )
    assert (
        await store.decide_approval(
            request.request_id, decision, owner_id=workspace.owner_id
        )
        == decision
    )
    assert (
        await store.get_approval_decision(
            request.request_id, owner_id=workspace.owner_id
        )
        == decision
    )
    assert (
        await store.decide_approval(
            request.request_id, decision, owner_id=workspace.owner_id
        )
        == decision
    )
    with pytest.raises(StateError, match="immutable"):
        await store.decide_approval(
            request.request_id,
            decision.model_copy(
                update={"outcome": ApprovalOutcome.DENY, "reason": "not needed"}
            ),
            owner_id=workspace.owner_id,
        )

    lease = Lease(
        approval_request_id=request.request_id,
        call_id=call.call_id,
        scope=request.scope,
        issuer=ApprovalIssuer.USER,
        issued_at=T0,
        expires_at=T0 + timedelta(minutes=10),
    )
    broader_scope = CapabilityScope(
        tools=("read_file",),
        effects=(ToolEffect.READ,),
        workspace_id=workspace.workspace_id,
        paths=("src/**", "README.md"),
        budget=request.scope.budget,
        expires_at=request.scope.expires_at,
    )
    with pytest.raises(StateError, match="exceeds"):
        await store.create_lease(
            lease.model_copy(update={"scope": broader_scope}),
            owner_id=workspace.owner_id,
        )
    assert await store.create_lease(lease, owner_id=workspace.owner_id) == lease
    assert await store.create_lease(lease, owner_id=workspace.owner_id) == lease
    assert await store.get_lease(lease.lease_id, owner_id=workspace.owner_id) == lease
    with pytest.raises(MissingEntityError):
        await store.get_lease(lease.lease_id, owner_id="other-owner")
    assert await store.list_leases(owner_id=workspace.owner_id, run_id=run.run_id) == (
        lease,
    )

    revoked = await store.revoke_lease(
        lease.lease_id,
        owner_id=workspace.owner_id,
        at=T0 + timedelta(minutes=1),
        reason="user revoked approval",
    )
    assert revoked.status is LeaseStatus.REVOKED
    with pytest.raises(StateError, match="expired or revoked"):
        await store.get_active_lease(
            lease.lease_id,
            owner_id=workspace.owner_id,
            at=T0 + timedelta(minutes=1),
        )
    assert (
        await store.revoke_lease(
            lease.lease_id,
            owner_id=workspace.owner_id,
            at=T0 + timedelta(minutes=1),
            reason="user revoked approval",
        )
        == revoked
    )
    with pytest.raises(StateError, match="immutable"):
        await store.revoke_lease(
            lease.lease_id,
            owner_id=workspace.owner_id,
            at=T0 + timedelta(minutes=2),
            reason="different reason",
        )
    expiring_lease = Lease(
        approval_request_id=request.request_id,
        call_id=call.call_id,
        scope=request.scope,
        issuer=ApprovalIssuer.SERVER,
        issued_at=T0,
        expires_at=T0 + timedelta(minutes=10),
    )
    await store.create_lease(expiring_lease, owner_id=workspace.owner_id)
    expired = await store.expire_lease(
        expiring_lease.lease_id,
        owner_id=workspace.owner_id,
        at=T0 + timedelta(minutes=10),
    )
    assert expired.status is LeaseStatus.EXPIRED
    with pytest.raises(StateError, match="expired or revoked"):
        await store.get_active_lease(
            expiring_lease.lease_id,
            owner_id=workspace.owner_id,
            at=T0 + timedelta(minutes=10),
        )
    assert [event.event_type for event in await store.list_events(run.run_id)] == [
        EventType.TOOL_CALL_REQUESTED,
        EventType.APPROVAL_REQUESTED,
        EventType.APPROVAL_DECIDED,
        EventType.LEASE_CREATED,
        EventType.LEASE_REVOKED,
        EventType.LEASE_CREATED,
        EventType.LEASE_EXPIRED,
    ]


@pytest.mark.asyncio
async def test_deny_is_persisted_but_cannot_create_a_lease(store: SqliteStore) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="approval-tool-request",
    )
    request = make_approval_request(run, workspace, call)
    await store.create_approval(request, owner_id=workspace.owner_id)
    decision = ApprovalDecision(
        approval_request_id=request.request_id,
        outcome=ApprovalOutcome.DENY,
        issuer=ApprovalIssuer.SERVER,
        reason="policy denied the requested effect",
        decided_at=T0,
    )
    await store.decide_approval(request.request_id, decision, owner_id=workspace.owner_id)
    with pytest.raises(StateError, match="ALLOW"):
        await store.create_lease(
            Lease(
                approval_request_id=request.request_id,
                call_id=call.call_id,
                scope=request.scope,
                issuer=ApprovalIssuer.SERVER,
                issued_at=T0,
                expires_at=T0 + timedelta(minutes=10),
            ),
            owner_id=workspace.owner_id,
        )


@pytest.mark.asyncio
async def test_approval_and_lease_reject_secret_text_at_the_store_boundary(
    store: SqliteStore,
) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="approval-tool-request",
    )
    request = make_approval_request(run, workspace, call).model_copy(
        update={"reason": "token=should-not-persist"}
    )
    with pytest.raises(ValueError, match="secret"):
        await store.create_approval(request, owner_id=workspace.owner_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [("tool_name", "write_file"), ("effect", ToolEffect.WRITE)],
)
async def test_approval_rejects_tool_call_metadata_mismatch(
    store: SqliteStore, field: str, value: object
) -> None:
    run, _, workspace, _, call = await seed_tool_context(store)
    await store.create_tool_call(
        call,
        workspace_id=workspace.workspace_id,
        idempotency_key="approval-tool-request",
    )
    request = make_approval_request(run, workspace, call).model_copy(
        update={field: value}
    )

    with pytest.raises(ValueError, match="does not match the persisted tool call"):
        await store.create_approval(request, owner_id=workspace.owner_id)


@pytest.mark.asyncio
async def test_update_run_persists_terminal_state_and_error(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    failed = run.model_copy(
        update={
            "status": RunStatus.FAILED,
            "strategy": ExecutionStrategy.DIRECT,
            "started_at": T0,
            "completed_at": T0 + timedelta(seconds=2),
            "error": ErrorInfo(
                category=ErrorCategory.PROVIDER_ERROR, message="upstream rejected request"
            ),
        }
    )
    await store.update_run(failed)
    stored = await store.get_run(run.run_id)
    assert stored.status is RunStatus.FAILED
    assert stored.error is not None
    assert stored.error.category is ErrorCategory.PROVIDER_ERROR
    assert stored.completed_at == T0 + timedelta(seconds=2)


@pytest.mark.asyncio
async def test_run_usage_accumulates_atomically(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    assert await store.get_run_usage(run.run_id) == Usage()
    await store.add_run_usage(run.run_id, Usage(input_tokens=3, output_tokens=4, elapsed_ms=10))
    total = await store.add_run_usage(
        run.run_id, Usage(input_tokens=1, output_tokens=1, strong_model_tokens=2, elapsed_ms=5)
    )
    assert total == Usage(
        input_tokens=4, output_tokens=5, strong_model_tokens=2, elapsed_ms=15
    )
    assert (await store.get_run(run.run_id)).usage == total
    with pytest.raises(MissingEntityError):
        await store.add_run_usage("run_missing", Usage())


@pytest.mark.asyncio
async def test_update_run_does_not_clobber_accumulated_usage(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    await store.add_run_usage(run.run_id, Usage(input_tokens=10))
    await store.update_run(run.model_copy(update={"status": RunStatus.RUNNING, "started_at": T0}))
    assert (await store.get_run(run.run_id)).usage == Usage(input_tokens=10)


@pytest.mark.asyncio
async def test_concurrent_usage_updates_are_not_lost(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    await asyncio.gather(
        *(store.add_run_usage(run.run_id, Usage(input_tokens=1)) for _ in range(20))
    )
    assert (await store.get_run_usage(run.run_id)).input_tokens == 20


# --- work units -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_work_unit_round_trips_with_edges_and_claims(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    first = make_work_unit(run.run_id, name="collect")
    second = make_work_unit(
        run.run_id,
        name="summarise",
        graph_version=2,
        acceptance_criteria="mentions every section",
        depends_on=(first.work_unit_id,),
        resource_claims=(
            ResourceClaim(resource="report.md", access=ResourceAccess.READ),
            ResourceClaim(resource="summary.md", access=ResourceAccess.WRITE),
        ),
        output=OutputRequirement(kind=ArtifactKind.JSON, json_schema='{"type":"object"}'),
        status=WorkUnitStatus.READY,
    )
    await store.create_work_units([second, first])
    assert await store.get_work_unit(second.work_unit_id) == second
    assert await store.get_work_unit(first.work_unit_id) == first


@pytest.mark.asyncio
async def test_work_unit_lineage_and_fingerprints_round_trip(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    unit = make_work_unit(
        run.run_id,
        lineage_key="stable-node",
        dependency_fingerprint="1" * 64,
        content_fingerprint="a" * 64,
    )
    await store.create_work_unit(unit)
    assert await store.get_work_unit(unit.work_unit_id) == unit


@pytest.mark.asyncio
async def test_work_graph_rejects_duplicate_current_lineage(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    first = make_work_unit(
        run.run_id,
        graph_version=2,
        lineage_key="same-lineage",
        dependency_fingerprint="1" * 64,
        content_fingerprint="a" * 64,
    )
    second = make_work_unit(
        run.run_id,
        graph_version=2,
        lineage_key="same-lineage",
        dependency_fingerprint="2" * 64,
        content_fingerprint="b" * 64,
    )
    with pytest.raises(DuplicateEntityError, match="duplicate lineage"):
        await store.create_graph((first, second))


@pytest.mark.asyncio
async def test_list_work_units_filters_by_graph_version(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    first = make_work_unit(run.run_id, name="v1")
    revised = make_work_unit(run.run_id, name="v2", graph_version=2)
    await store.create_work_units([first, revised])
    assert await store.list_work_units(run.run_id) == (first, revised)
    assert await store.list_work_units(run.run_id, graph_version=2) == (revised,)
    assert await store.list_work_units(run.run_id, graph_version=9) == ()


@pytest.mark.asyncio
async def test_create_graph_requires_one_run_version_and_internal_edges(
    store: SqliteStore,
) -> None:
    run = make_run()
    other_run = make_run()
    await store.create_run(run)
    await store.create_run(other_run)
    first = make_work_unit(run.run_id, graph_version=2, name="first")
    second = make_work_unit(
        run.run_id,
        graph_version=2,
        name="second",
        depends_on=(first.work_unit_id,),
    )
    await store.create_graph((second, first))
    assert len(await store.list_work_units(run.run_id, graph_version=2)) == 2

    mixed = (
        make_work_unit(run.run_id, graph_version=3),
        make_work_unit(other_run.run_id, graph_version=3),
    )
    with pytest.raises(StateError, match="one run_id"):
        await store.create_graph(mixed)
    with pytest.raises(DanglingReferenceError, match="outside the graph"):
        await store.create_graph(
            (
                make_work_unit(
                    run.run_id,
                    graph_version=3,
                    depends_on=(new_work_unit_id(),),
                ),
            )
        )
    assert await store.list_work_units(run.run_id, graph_version=3) == ()


@pytest.mark.asyncio
async def test_work_unit_with_unknown_run_is_rejected(store: SqliteStore) -> None:
    with pytest.raises(DanglingReferenceError):
        await store.create_work_unit(make_work_unit(new_run_id()))


@pytest.mark.asyncio
async def test_work_unit_edge_to_unknown_unit_is_rejected(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(DanglingReferenceError):
        await store.create_work_unit(
            make_work_unit(run.run_id, depends_on=(new_work_unit_id(),))
        )


@pytest.mark.asyncio
async def test_update_work_unit_persists_status_only(store: SqliteStore) -> None:
    run, unit, _ = await seed(store)
    await store.update_work_unit(unit.model_copy(update={"status": WorkUnitStatus.SUCCEEDED}))
    stored = await store.get_work_unit(unit.work_unit_id)
    assert stored.status is WorkUnitStatus.SUCCEEDED
    assert stored.instruction == unit.instruction
    with pytest.raises(MissingEntityError) as excinfo:
        await store.update_work_unit(make_work_unit(run.run_id))
    assert excinfo.value.code is ErrorCode.WORK_UNIT_NOT_FOUND


# --- attempts -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attempt_round_trips_including_usage_and_error(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    assert await store.get_attempt(attempt.attempt_id) == attempt
    assert (await store.get_attempt(attempt.attempt_id)).usage is None

    completed = attempt.model_copy(
        update={
            "status": AttemptStatus.FAILED,
            "provider_request_id": "req_42",
            "usage": Usage(input_tokens=5, output_tokens=6, elapsed_ms=70),
            "error": ErrorInfo(category=ErrorCategory.TIMEOUT, message="upstream timed out"),
            "started_at": T0,
            "completed_at": T0 + timedelta(seconds=1),
        }
    )
    await store.update_attempt(completed)
    assert await store.get_attempt(attempt.attempt_id) == completed

    escalated = make_attempt(
        run.run_id,
        unit.work_unit_id,
        attempt_index=2,
        role=ModelRole.PLANNER,
        model=PLANNER,
        status=AttemptStatus.SUCCEEDED,
        usage=Usage(input_tokens=1, output_tokens=2, strong_model_tokens=3),
        started_at=T0,
        completed_at=T0 + timedelta(seconds=1),
    )
    await store.create_attempt(escalated)
    assert await store.list_attempts(unit.work_unit_id) == (completed, escalated)


@pytest.mark.asyncio
async def test_duplicate_attempt_index_is_rejected(store: SqliteStore) -> None:
    run, unit, _ = await seed(store)
    with pytest.raises(DuplicateEntityError):
        await store.create_attempt(make_attempt(run.run_id, unit.work_unit_id))


@pytest.mark.asyncio
async def test_attempt_with_unknown_work_unit_is_rejected(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(DanglingReferenceError):
        await store.create_attempt(make_attempt(run.run_id, new_work_unit_id()))


@pytest.mark.asyncio
async def test_list_run_attempts_filters_by_status(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    running = attempt.model_copy(update={"status": AttemptStatus.RUNNING, "started_at": T0})
    await store.update_attempt(running)
    assert await store.list_run_attempts(run.run_id) == (running,)
    assert await store.list_run_attempts(run.run_id, statuses=[AttemptStatus.RUNNING]) == (
        running,
    )
    assert await store.list_run_attempts(run.run_id, statuses=[AttemptStatus.SUCCEEDED]) == ()
    assert await store.list_run_attempts(run.run_id, statuses=[]) == ()


@pytest.mark.asyncio
async def test_missing_attempt_read_and_update_fail(store: SqliteStore) -> None:
    run, unit, _ = await seed(store)
    with pytest.raises(MissingEntityError):
        await store.get_attempt(new_attempt_id())
    with pytest.raises(MissingEntityError):
        await store.update_attempt(make_attempt(run.run_id, unit.work_unit_id, attempt_index=9))


# --- reservations --------------------------------------------------------------


@pytest.mark.asyncio
async def test_reservation_round_trip_and_events(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    unit = make_work_unit(run.run_id)
    await store.create_work_unit(unit)
    request = make_reservation_request(run.run_id, unit.work_unit_id)

    reservation = await store.reserve_reservation(
        request,
        reservation_id="res_round_trip",
        created_at=T0,
        held_at=T0,
    )

    assert reservation.status is ReservationStatus.HELD
    assert await store.get_reservation(reservation.reservation_id) == reservation
    assert await store.list_reservations(run.run_id) == (reservation,)
    events = await store.list_events(run.run_id)
    assert [event.event_type for event in events] == [
        EventType.RESERVATION_CREATED,
        EventType.RESERVATION_HELD,
    ]


@pytest.mark.asyncio
async def test_reservation_same_dispatch_is_idempotent_but_conflicting_request_rejected(
    store: SqliteStore,
) -> None:
    run, unit, _ = await seed(store)
    request = make_reservation_request(run.run_id, unit.work_unit_id)
    first = await store.reserve_reservation(
        request, reservation_id="res_idempotent", created_at=T0, held_at=T0
    )
    second = await store.reserve_reservation(
        request,
        reservation_id="res_different",
        created_at=T0 + timedelta(seconds=1),
        held_at=T0 + timedelta(seconds=1),
    )
    assert second == first
    assert len(await store.list_events(run.run_id)) == 2

    with pytest.raises(StateError, match="different reservation request"):
        await store.reserve_reservation(
            request.model_copy(update={"token_upper_bound": 9}),
            reservation_id="res_conflicting",
            created_at=T0,
            held_at=T0,
        )


@pytest.mark.asyncio
async def test_reservation_settle_release_are_terminal_and_idempotent(store: SqliteStore) -> None:
    run, unit, _ = await seed(store)
    request = make_reservation_request(run.run_id, unit.work_unit_id)
    reservation = await store.reserve_reservation(
        request, reservation_id="res_settle", created_at=T0, held_at=T0
    )
    usage = Usage(input_tokens=3, output_tokens=2, strong_model_tokens=2, elapsed_ms=5)
    settled = await store.settle_reservation(
        reservation.reservation_id,
        measured_usage=usage,
        completed_at=T0 + timedelta(seconds=1),
    )
    assert settled.status is ReservationStatus.SETTLED
    assert settled.measured_usage == usage
    assert await store.settle_reservation(
        reservation.reservation_id,
        measured_usage=usage,
        completed_at=T0 + timedelta(seconds=1),
    ) == settled
    with pytest.raises(StateError, match="terminal fact"):
        await store.settle_reservation(
            reservation.reservation_id,
            measured_usage=None,
            completed_at=T0 + timedelta(seconds=2),
        )

    released = await store.reserve_reservation(
        make_reservation_request(run.run_id, unit.work_unit_id, dispatch_key="dispatch-2"),
        reservation_id="res_release",
        created_at=T0,
        held_at=T0,
    )
    released_fact = await store.release_reservation(
        released.reservation_id,
        completed_at=T0 + timedelta(seconds=1),
    )
    assert released_fact.status is ReservationStatus.RELEASED
    assert await store.release_reservation(
        released.reservation_id,
        completed_at=T0 + timedelta(seconds=1),
    ) == released_fact
    with pytest.raises(StateError, match="RELEASED"):
        await store.release_reservation(
            released.reservation_id,
            expired=True,
            completed_at=T0 + timedelta(seconds=2),
        )


@pytest.mark.asyncio
async def test_reservation_unknown_usage_round_trips_without_zero_filling(
    store: SqliteStore,
) -> None:
    run, unit, _ = await seed(store)
    reservation = await store.reserve_reservation(
        make_reservation_request(run.run_id, unit.work_unit_id),
        reservation_id="res_unknown_usage",
        created_at=T0,
        held_at=T0,
    )
    settled = await store.settle_reservation(
        reservation.reservation_id,
        measured_usage=None,
        completed_at=T0 + timedelta(seconds=1),
    )
    assert settled.measured_usage is None


@pytest.mark.asyncio
async def test_reservation_list_filters_status_and_rejects_missing_entities(
    store: SqliteStore,
) -> None:
    run, unit, _ = await seed(store)
    request = make_reservation_request(run.run_id, unit.work_unit_id)
    pending_like = await store.reserve_reservation(
        request, reservation_id="res_filter", created_at=T0, held_at=T0
    )
    await store.release_reservation(
        pending_like.reservation_id, completed_at=T0 + timedelta(seconds=1)
    )
    assert await store.list_reservations(run.run_id, statuses=[ReservationStatus.HELD]) == ()
    assert len(
        await store.list_reservations(run.run_id, statuses=[ReservationStatus.RELEASED])
    ) == 1
    with pytest.raises(MissingEntityError):
        await store.get_reservation("res_missing")


# --- artifacts and evidence -----------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_and_evidence_round_trip(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        name="answer",
        kind=ArtifactKind.JSON,
        content='{"ok": true}',
        created_at=T0,
    )
    await store.add_artifact(artifact)
    assert await store.get_artifact(artifact.artifact_id) == artifact
    assert await store.list_artifacts(unit.work_unit_id) == (artifact,)

    evidence = Evidence(
        evidence_id=new_evidence_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        artifact_id=artifact.artifact_id,
        kind=EvidenceKind.DETERMINISTIC_CHECK,
        rule="MATCHES_JSON_SCHEMA",
        result=VerificationResult.FAIL,
        detail="schema mismatch",
        created_at=T0,
    )
    await store.add_evidence(evidence)
    stored = await store.list_evidence(unit.work_unit_id)
    assert stored == (evidence,)
    assert stored[0].result is VerificationResult.FAIL
    assert stored[0].passed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("result", list(VerificationResult))
async def test_every_verdict_survives_a_round_trip(
    store: SqliteStore, result: VerificationResult
) -> None:
    """An undecided verdict must not come back as a proven failure."""
    run, unit, attempt = await seed(store)
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        name="answer",
        content="text",
        created_at=T0,
    )
    await store.add_artifact(artifact)
    evidence = Evidence(
        evidence_id=new_evidence_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        artifact_id=artifact.artifact_id,
        kind=EvidenceKind.DETERMINISTIC_CHECK,
        rule="VALID_JSON",
        result=result,
        detail=f"recorded {result.value}",
        created_at=T0,
    )
    await store.add_evidence(evidence)
    stored = await store.list_evidence(unit.work_unit_id)
    assert stored == (evidence,)
    assert stored[0].result is result
    assert stored[0].passed is result.is_pass


@pytest.mark.asyncio
@pytest.mark.parametrize("result", list(VerificationResult))
async def test_the_read_path_derives_passed_from_the_stored_verdict(
    store: SqliteStore, result: VerificationResult
) -> None:
    """Proved against a raw row, not one this store wrote.

    A round trip through ``add_evidence`` only shows the writer and the reader
    agree. Inserting the row directly proves the reader derives ``passed`` from
    the stored verdict rather than trusting whatever the writer believed.
    """
    run, unit, attempt = await seed(store)
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        name="answer",
        content="text",
        created_at=T0,
    )
    await store.add_artifact(artifact)
    async with store.transaction() as connection:
        await connection.execute(
            """
            INSERT INTO evidence (evidence_id, run_id, work_unit_id, artifact_id, kind,
                                  rule, result, detail, created_at)
            VALUES (?, ?, ?, ?, 'DETERMINISTIC_CHECK', 'VALID_JSON', ?, 'a detail', ?)
            """,
            (
                new_evidence_id(),
                run.run_id,
                unit.work_unit_id,
                artifact.artifact_id,
                result.value,
                T0.isoformat(),
            ),
        )

    stored = await store.list_evidence(unit.work_unit_id)
    assert len(stored) == 1
    assert stored[0].result is result
    assert stored[0].passed is result.is_pass


@pytest.mark.asyncio
async def test_a_model_review_round_trips_without_a_rule(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        name="answer",
        content="text",
        created_at=T0,
    )
    await store.add_artifact(artifact)
    evidence = Evidence(
        evidence_id=new_evidence_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        artifact_id=artifact.artifact_id,
        kind=EvidenceKind.MODEL_REVIEW,
        result=VerificationResult.INCONCLUSIVE,
        detail="a reviewer could not decide",
        created_at=T0,
    )
    await store.add_evidence(evidence)
    stored = await store.list_evidence(unit.work_unit_id)
    assert stored == (evidence,)
    assert stored[0].rule is None


@pytest.mark.asyncio
async def test_artifact_and_evidence_reject_dangling_references(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    with pytest.raises(DanglingReferenceError):
        await store.add_artifact(
            Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=unit.work_unit_id,
                attempt_id=new_attempt_id(),
                name="answer",
                content="text",
                created_at=T0,
            )
        )
    with pytest.raises(DanglingReferenceError):
        await store.add_evidence(
            Evidence(
                evidence_id=new_evidence_id(),
                run_id=run.run_id,
                work_unit_id=unit.work_unit_id,
                artifact_id=new_artifact_id(),
                kind=EvidenceKind.MODEL_REVIEW,
                result=VerificationResult.PASS,
                detail="looks right",
                created_at=T0,
            )
        )


@pytest.mark.asyncio
async def test_two_attempts_may_produce_the_same_artifact_name(store: SqliteStore) -> None:
    run, unit, first_attempt = await seed(store)
    second_attempt = make_attempt(run.run_id, unit.work_unit_id, attempt_index=2)
    await store.create_attempt(second_attempt)
    for attempt in (first_attempt, second_attempt):
        await store.add_artifact(
            Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                name="answer",
                content=f"from {attempt.attempt_index}",
                created_at=T0,
            )
        )
    assert len(await store.list_artifacts(unit.work_unit_id)) == 2


# --- events ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_are_appended_with_monotonic_sequences(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    assert await store.last_sequence(run.run_id) is None
    created = await store.append_event(
        run.run_id,
        EventType.RUN_CREATED,
        payload_from_model("request", run.request),
        timestamp=T0,
    )
    started = await store.append_event(run.run_id, EventType.RUN_STARTED, timestamp=T0)
    assert (created.sequence, started.sequence) == (1, 2)
    assert await store.last_sequence(run.run_id) == 2
    ledger = await store.list_events(run.run_id)
    assert ledger == (created, started)
    assert assert_sequence_chain(ledger) is None
    assert ledger[0].payload["request"] == run.request.model_dump(mode="json")


@pytest.mark.asyncio
async def test_event_payload_is_validated_before_it_is_written(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(ValueError, match="payload is missing"):
        await store.append_event(run.run_id, EventType.WORK_UNIT_STARTED)
    assert await store.list_events(run.run_id) == ()


@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
@pytest.mark.asyncio
async def test_non_finite_event_payload_is_rejected_and_not_persisted(
    store: SqliteStore, bad_value: float
) -> None:
    """A non-finite number in the payload is refused at the write, not stored.

    ``json.dumps(..., allow_nan=False)`` runs before any row is inserted, so the
    rejection must be provable from the ledger, not only from the raised error.
    """
    run = make_run()
    await store.create_run(run)
    with pytest.raises(ValueError):
        await store.append_event(run.run_id, EventType.RUN_RESUMED, {"value": bad_value})
    assert await store.last_sequence(run.run_id) is None
    assert await store.list_events(run.run_id) == ()
    # The store is still usable: a finite payload after the rejection is
    # accepted and round-trips normally.
    appended = await store.append_event(run.run_id, EventType.RUN_RESUMED, {"value": 1.5})
    assert appended.sequence == 1
    ledger = await store.list_events(run.run_id)
    assert ledger[0].payload["value"] == 1.5


@pytest.mark.asyncio
async def test_event_for_unknown_run_is_rejected(store: SqliteStore) -> None:
    with pytest.raises(DanglingReferenceError) as excinfo:
        await store.append_event(new_run_id(), EventType.RUN_STARTED)
    assert excinfo.value.code is ErrorCode.RUN_NOT_FOUND


@pytest.mark.asyncio
async def test_after_sequence_cursor_is_stable(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    for _ in range(5):
        await store.append_event(run.run_id, EventType.RUN_RESUMED)
    assert [event.sequence for event in await store.list_events(run.run_id)] == [1, 2, 3, 4, 5]
    assert [
        event.sequence for event in await store.list_events(run.run_id, after_sequence=3)
    ] == [4, 5]
    assert [
        event.sequence for event in await store.list_events(run.run_id, after_sequence=0, limit=2)
    ] == [1, 2]
    assert await store.list_events(run.run_id, after_sequence=5) == ()
    with pytest.raises(ValueError):
        await store.list_events(run.run_id, limit=0)


@pytest.mark.asyncio
async def test_ledgers_of_different_runs_are_independent(store: SqliteStore) -> None:
    first = make_run()
    second = make_run()
    await store.create_run(first)
    await store.create_run(second)
    await store.append_event(first.run_id, EventType.RUN_STARTED)
    await store.append_event(second.run_id, EventType.RUN_STARTED)
    assert (await store.last_sequence(first.run_id)) == 1
    assert (await store.last_sequence(second.run_id)) == 1
    assert len(await store.list_events(first.run_id)) == 1


@pytest.mark.asyncio
async def test_concurrent_appends_produce_unique_sequences(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    await asyncio.gather(
        *(store.append_event(run.run_id, EventType.RUN_RESUMED) for _ in range(25))
    )
    ledger = await store.list_events(run.run_id)
    assert [event.sequence for event in ledger] == list(range(1, 26))
    assert assert_sequence_chain(ledger) is None


@pytest.mark.asyncio
async def test_concurrent_appends_from_two_connections_do_not_duplicate(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as writer:
        run = make_run()
        await writer.create_run(run)
        async with SqliteStore(database_path) as other:
            await asyncio.gather(
                *(
                    connection.append_event(run.run_id, EventType.RUN_RESUMED)
                    for connection in (writer, other, writer, other, writer, other)
                )
            )
            ledger = await writer.list_events(run.run_id)
    assert [event.sequence for event in ledger] == [1, 2, 3, 4, 5, 6]


# --- transaction atomicity ------------------------------------------------------


@pytest.mark.asyncio
async def test_state_and_events_commit_together(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    async with store.transaction():
        await store.update_run(
            run.model_copy(update={"status": RunStatus.RUNNING, "started_at": T0})
        )
        await store.append_event(run.run_id, EventType.RUN_STARTED)
        assert store.in_transaction is True
    assert store.in_transaction is False
    assert (await store.get_run(run.run_id)).status is RunStatus.RUNNING
    assert len(await store.list_events(run.run_id)) == 1


@pytest.mark.asyncio
async def test_state_and_events_roll_back_together(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(RuntimeError):
        async with store.transaction():
            await store.update_run(
                run.model_copy(update={"status": RunStatus.RUNNING, "started_at": T0})
            )
            await store.append_event(run.run_id, EventType.RUN_STARTED)
            raise RuntimeError("controller failed")
    assert (await store.get_run(run.run_id)).status is RunStatus.PENDING
    assert await store.list_events(run.run_id) == ()
    assert store.in_transaction is False


@pytest.mark.asyncio
async def test_a_failed_graph_commit_leaves_no_partial_units(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    good = make_work_unit(run.run_id, name="good")
    broken = make_work_unit(run.run_id, name="broken", depends_on=(new_work_unit_id(),))
    with pytest.raises(DanglingReferenceError):
        await store.create_work_units([good, broken])
    assert await store.list_work_units(run.run_id) == ()


@pytest.mark.asyncio
async def test_create_graph_rolls_back_rows_inserted_before_a_duplicate(
    store: SqliteStore,
) -> None:
    run = make_run()
    await store.create_run(run)
    existing = make_work_unit(run.run_id, graph_version=1, name="existing")
    await store.create_work_unit(existing)
    first = make_work_unit(run.run_id, graph_version=2, name="first")
    duplicate = make_work_unit(
        run.run_id,
        graph_version=2,
        name="duplicate",
        work_unit_id=existing.work_unit_id,
    )
    with pytest.raises(DuplicateEntityError):
        await store.create_graph((first, duplicate))
    assert await store.list_work_units(run.run_id, graph_version=2) == ()


@pytest.mark.asyncio
async def test_writes_outside_a_transaction_commit_immediately(
    database_path: Path,
) -> None:
    run = make_run()
    async with SqliteStore(database_path) as writer:
        await writer.create_run(run)
    async with SqliteStore(database_path) as reader:
        assert (await reader.get_run(run.run_id)).run_id == run.run_id


@pytest.mark.asyncio
async def test_store_operations_require_an_open_store(database_path: Path) -> None:
    closed = SqliteStore(database_path)
    with pytest.raises(InternalError, match="not open"):
        await closed.get_run("run_1")
    with pytest.raises(InternalError, match="not open"):
        await closed.create_run(make_run())
