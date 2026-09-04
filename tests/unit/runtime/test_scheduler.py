"""Single-snapshot scheduler wave tests with Controller-style callbacks."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prp_runtime.domain.enums import ResourceAccess, WorkUnitStatus
from prp_runtime.domain.models import ErrorCategory, ErrorInfo, WorkUnit
from prp_runtime.domain.values import ResourceClaim
from prp_runtime.runtime.conflicts import ConflictFacts
from prp_runtime.runtime.scheduler import (
    BridgeClientCandidate,
    Scheduler,
    SlotDispatcher,
    WaveOutcome,
    WaveStatus,
    select_bridge_client,
    select_non_conflicting_batch,
    select_non_conflicting_batch_with_reasons,
)
from prp_runtime.workspace.isolation import LocalIsolationBackend

T0 = datetime(2026, 8, 11, tzinfo=UTC)


def unit(
    key: str,
    *,
    status: WorkUnitStatus = WorkUnitStatus.PENDING,
    depends_on: tuple[str, ...] = (),
    resource_claims: tuple[ResourceClaim, ...] = (),
) -> WorkUnit:
    return WorkUnit(
        work_unit_id=f"wu_{key}",
        run_id="run_scheduler",
        name=key,
        instruction=f"do {key}",
        status=status,
        depends_on=depends_on,
        resource_claims=resource_claims,
        created_at=T0,
    )


async def run_wave(
    units: tuple[WorkUnit, ...],
    outcomes: dict[str, WaveOutcome] | None = None,
    *,
    max_concurrency: int = 1,
):
    started: list[str] = []
    finished: list[str] = []
    failed: list[str] = []
    outcome_by_id = outcomes or {}

    async def on_started(value: WorkUnit) -> WorkUnit:
        started.append(value.work_unit_id)
        return value.model_copy(update={"status": WorkUnitStatus.RUNNING})

    async def execute(value: WorkUnit) -> WaveOutcome:
        return outcome_by_id.get(value.work_unit_id, WaveOutcome.success())

    async def on_succeeded(value: WorkUnit) -> None:
        finished.append(value.work_unit_id)

    async def on_failed(value: WorkUnit, error: ErrorInfo) -> None:
        failed.append(value.work_unit_id)

    result = await Scheduler().run_wave(
        units,
        graph_version=1,
        max_concurrency=max_concurrency,
        execute=execute,
        on_started=on_started,
        on_succeeded=on_succeeded,
        on_failed=on_failed,
    )
    return result, started, finished, failed


@pytest.mark.asyncio
async def test_root_wave_dispatches_only_the_snapshot_ready_unit() -> None:
    root = unit("root")
    child = unit("child", depends_on=(root.work_unit_id,))
    result, started, finished, failed = await run_wave((child, root))
    assert result.status is WaveStatus.DISPATCHED
    assert result.ready == (root.work_unit_id,)
    assert result.started == result.succeeded == (root.work_unit_id,)
    assert result.failed == ()
    assert started == finished == [root.work_unit_id]
    assert failed == []


@pytest.mark.asyncio
async def test_diamond_wave_order_is_stable_and_next_wave_is_external() -> None:
    root = unit("root")
    left = unit("left", depends_on=(root.work_unit_id,))
    right = unit("right", depends_on=(root.work_unit_id,))
    join = unit("join", depends_on=(left.work_unit_id, right.work_unit_id))
    first, started, _, _ = await run_wave((join, right, root, left))
    assert first.ready == (root.work_unit_id,)
    assert started == [root.work_unit_id]

    second_units = (
        join,
        right,
        root.model_copy(update={"status": WorkUnitStatus.SUCCEEDED}),
        left,
    )
    second, started, _, _ = await run_wave(second_units, max_concurrency=2)
    assert second.ready == (left.work_unit_id, right.work_unit_id)
    assert started == [left.work_unit_id, right.work_unit_id]


@pytest.mark.asyncio
async def test_wave_failure_calls_failed_callback_once() -> None:
    error = ErrorInfo(category=ErrorCategory.PROVIDER_ERROR, message="fake failure")
    result, started, finished, failed = await run_wave(
        (unit("one"),),
        outcomes={"wu_one": WaveOutcome.failure(error)},
    )
    assert result.status is WaveStatus.DISPATCHED
    assert result.started == result.failed == ("wu_one",)
    assert result.succeeded == ()
    assert started == ["wu_one"]
    assert finished == []
    assert failed == ["wu_one"]


@pytest.mark.asyncio
async def test_wave_honors_the_coordinator_eligible_batch() -> None:
    result, started, finished, failed = await run_wave(
        (unit("first"), unit("second")),
        max_concurrency=2,
    )

    assert result.started == ("wu_first", "wu_second")
    assert started == list(result.started)
    assert finished == list(result.succeeded)
    assert failed == []

    eligible = ("wu_second",)

    async def execute(value: WorkUnit) -> WaveOutcome:
        return WaveOutcome.success()

    async def on_started(value: WorkUnit) -> WorkUnit:
        return value.model_copy(update={"status": WorkUnitStatus.RUNNING})

    async def on_succeeded(value: WorkUnit) -> None:
        return None

    async def on_failed(value: WorkUnit, error: ErrorInfo) -> None:
        raise AssertionError(error.message)

    selected = await Scheduler().run_wave(
        (unit("first"), unit("second")),
        graph_version=1,
        max_concurrency=2,
        eligible_work_unit_ids=eligible,
        execute=execute,
        on_started=on_started,
        on_succeeded=on_succeeded,
        on_failed=on_failed,
    )
    assert selected.ready == eligible
    assert selected.started == selected.succeeded == eligible


@pytest.mark.asyncio
async def test_wave_converts_executor_exception_to_one_failed_unit() -> None:
    failed: list[str] = []

    async def on_started(value: WorkUnit) -> WorkUnit:
        return value.model_copy(update={"status": WorkUnitStatus.RUNNING})

    async def execute(value: WorkUnit) -> WaveOutcome:
        raise RuntimeError("fixture failure")

    async def on_succeeded(value: WorkUnit) -> None:
        raise AssertionError("the fixture must fail")

    async def on_failed(value: WorkUnit, error: ErrorInfo) -> None:
        failed.append(value.work_unit_id)
        assert error.category is ErrorCategory.UNKNOWN

    result = await Scheduler().run_wave(
        (unit("exception"),),
        graph_version=1,
        execute=execute,
        on_started=on_started,
        on_succeeded=on_succeeded,
        on_failed=on_failed,
    )

    assert result.failed == ("wu_exception",)
    assert failed == ["wu_exception"]


@pytest.mark.asyncio
async def test_cancelled_start_stops_the_wave_before_execution() -> None:
    executed: list[str] = []

    async def on_started(value: WorkUnit) -> WorkUnit | None:
        return None

    async def execute(value: WorkUnit) -> WaveOutcome:
        executed.append(value.work_unit_id)
        return WaveOutcome.success()

    async def on_succeeded(value: WorkUnit) -> None:
        raise AssertionError("a cancelled wave must not succeed a unit")

    async def on_failed(value: WorkUnit, error: ErrorInfo) -> None:
        raise AssertionError("a cancelled wave must not fail a unit")

    result = await Scheduler().run_wave(
        (unit("a"), unit("b")),
        graph_version=1,
        max_concurrency=2,
        execute=execute,
        on_started=on_started,
        on_succeeded=on_succeeded,
        on_failed=on_failed,
    )

    assert result.status is WaveStatus.CANCELLED
    assert result.started == ()
    assert result.deferred == ("wu_a", "wu_b")
    assert executed == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ((), WaveStatus.EMPTY),
        ((unit("done", status=WorkUnitStatus.SUCCEEDED),), WaveStatus.COMPLETE),
        ((unit("blocked", status=WorkUnitStatus.BLOCKED),), WaveStatus.BLOCKED),
        ((unit("running", status=WorkUnitStatus.RUNNING),), WaveStatus.WAITING),
    ],
)
async def test_empty_complete_blocked_and_waiting_waves_are_explicit(
    units: tuple[WorkUnit, ...], expected: WaveStatus
) -> None:
    result, started, _, _ = await run_wave(units)
    assert result.status is expected
    assert started == []


def test_resource_batch_selection_allows_read_read_and_defers_conflicts() -> None:
    read_a = unit(
        "a",
        resource_claims=(ResourceClaim(resource="doc", access=ResourceAccess.READ),),
    )
    read_b = unit(
        "b",
        resource_claims=(ResourceClaim(resource="doc", access=ResourceAccess.READ),),
    )
    write = unit(
        "c",
        resource_claims=(ResourceClaim(resource="doc", access=ResourceAccess.WRITE),),
    )
    selected, deferred = select_non_conflicting_batch(
        (write, read_b, read_a), max_concurrency=3
    )
    assert tuple(unit.work_unit_id for unit in selected) == ("wu_a", "wu_b")
    assert tuple(unit.work_unit_id for unit in deferred) == ("wu_c",)


@pytest.mark.asyncio
async def test_non_conflicting_tasks_run_concurrently_with_a_barrier() -> None:
    gate = asyncio.Event()
    both_started = asyncio.Event()
    active = 0
    max_active = 0

    async def execute(value: WorkUnit) -> WaveOutcome:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            both_started.set()
        await both_started.wait()
        active -= 1
        gate.set()
        return WaveOutcome.success()

    async def on_started(value: WorkUnit) -> WorkUnit:
        return value.model_copy(update={"status": WorkUnitStatus.RUNNING})

    async def on_succeeded(value: WorkUnit) -> None:
        return None

    async def on_failed(value: WorkUnit, error: ErrorInfo) -> None:
        raise AssertionError("the barrier fixture must succeed")

    result = await Scheduler().run_wave(
        (unit("a"), unit("b"), unit("c")),
        graph_version=1,
        max_concurrency=2,
        execute=execute,
        on_started=on_started,
        on_succeeded=on_succeeded,
        on_failed=on_failed,
    )
    assert result.started == ("wu_a", "wu_b")
    assert result.deferred == ("wu_c",)
    assert max_active == 2
    assert gate.is_set()


@pytest.mark.asyncio
async def test_max_concurrency_one_never_starts_two_tasks() -> None:
    active = 0
    max_active = 0

    async def execute(value: WorkUnit) -> WaveOutcome:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        active -= 1
        return WaveOutcome.success()

    async def on_started(value: WorkUnit) -> WorkUnit:
        return value.model_copy(update={"status": WorkUnitStatus.RUNNING})

    async def on_succeeded(value: WorkUnit) -> None:
        return None

    async def on_failed(value: WorkUnit, error: ErrorInfo) -> None:
        raise AssertionError("the fixture must succeed")

    result = await Scheduler().run_wave(
        (unit("a"), unit("b")),
        graph_version=1,
        max_concurrency=1,
        execute=execute,
        on_started=on_started,
        on_succeeded=on_succeeded,
        on_failed=on_failed,
    )
    assert len(result.started) == 1
    assert result.deferred == ("wu_b",)
    assert max_active == 1


@pytest.mark.asyncio
async def test_slot_dispatcher_gives_parallel_units_private_roots_and_cleans_them(
    tmp_path: Path,
) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation")
    source = tmp_path / "source"
    source.mkdir()
    (source / "base.txt").write_text("base\n", encoding="utf-8")
    snapshot = backend.create_base_snapshot(source, "ws_scheduler")
    dispatcher = SlotDispatcher(
        backend,
        snapshot_id=snapshot.snapshot_id,
        owner_id="owner_scheduler",
    )
    roots: dict[str, Path] = {}

    async def execute_in_slot(value: WorkUnit, context) -> WaveOutcome:
        roots[value.work_unit_id] = context.path
        (context.path / f"{value.work_unit_id}.txt").write_text(
            value.work_unit_id,
            encoding="utf-8",
        )
        return WaveOutcome.success()

    async def on_started(value: WorkUnit) -> WorkUnit:
        return value.model_copy(update={"status": WorkUnitStatus.RUNNING})

    async def execute(value: WorkUnit) -> WaveOutcome:
        raise AssertionError("slot-aware callback must receive dispatched units")

    async def on_succeeded(value: WorkUnit) -> None:
        return None

    async def on_failed(value: WorkUnit, error: ErrorInfo) -> None:
        raise AssertionError(error.message)

    result = await Scheduler().run_wave(
        (unit("a"), unit("b")),
        graph_version=1,
        max_concurrency=2,
        execute=execute,
        execute_in_slot=execute_in_slot,
        slot_dispatcher=dispatcher,
        on_started=on_started,
        on_succeeded=on_succeeded,
        on_failed=on_failed,
    )

    assert result.succeeded == ("wu_a", "wu_b")
    assert roots["wu_a"] != roots["wu_b"]
    assert backend.active_slot_count == 0


def test_invalid_max_concurrency_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        select_non_conflicting_batch((unit("a"),), max_concurrency=0)


def test_actual_changesets_defer_overlapping_writes_with_reason() -> None:
    base = "snap_" + "a" * 32
    facts = {
        "wu_a": ConflictFacts(changed_paths=("src/main.py",), base_snapshot_id=base),
        "wu_b": ConflictFacts(changed_paths=("src//main.py",), base_snapshot_id=base),
    }

    result = select_non_conflicting_batch_with_reasons(
        (unit("b"), unit("a")), max_concurrency=2, actual_changesets=facts
    )

    assert tuple(value.work_unit_id for value in result.selected) == ("wu_a",)
    assert tuple(value.work_unit_id for value in result.deferred) == ("wu_b",)
    assert result.deferred_reasons == (
        ("wu_b", "deferred by wu_a: PATH - write paths overlap"),
    )


def test_actual_disjoint_writes_run_together_and_unknown_is_deferred() -> None:
    base = "snap_" + "a" * 32
    disjoint = {
        "wu_a": ConflictFacts(changed_paths=("a.txt",), base_snapshot_id=base),
        "wu_b": ConflictFacts(changed_paths=("b.txt",), base_snapshot_id=base),
    }
    parallel = select_non_conflicting_batch_with_reasons(
        (unit("b"), unit("a")), max_concurrency=2, actual_changesets=disjoint
    )
    assert tuple(value.work_unit_id for value in parallel.selected) == ("wu_a", "wu_b")
    assert parallel.deferred_reasons == ()

    unknown = select_non_conflicting_batch_with_reasons(
        (unit("a"), unit("b")), max_concurrency=2, actual_changesets={"wu_a": disjoint["wu_a"]}
    )
    assert tuple(value.work_unit_id for value in unknown.selected) == ("wu_a",)
    assert unknown.deferred_reasons[0][0] == "wu_b"
    assert "UNKNOWN" in unknown.deferred_reasons[0][1]


@pytest.mark.asyncio
async def test_wave_result_retains_deferred_reasons() -> None:
    result, _, _, _ = await run_wave(
        (unit("a"), unit("b")),
        max_concurrency=1,
    )

    assert result.deferred_reasons == (("wu_b", "deferred: max concurrency reached"),)


def test_select_bridge_client_is_live_scoped_and_capacity_aware() -> None:
    live = BridgeClientCandidate(
        client_id="cli_b",
        workspace_id="ws_project",
        tools=("read_file",),
        liveness="LIVE",
    )
    other_ws = BridgeClientCandidate(
        client_id="cli_a",
        workspace_id="ws_other",
        tools=("read_file",),
        liveness="LIVE",
    )
    stale = BridgeClientCandidate(
        client_id="cli_c",
        workspace_id="ws_project",
        tools=("read_file",),
        liveness="STALE",
    )
    full = BridgeClientCandidate(
        client_id="cli_d",
        workspace_id="ws_project",
        tools=("read_file",),
        liveness="LIVE",
        active_claims=1,
        max_active_claims=1,
    )
    chosen = select_bridge_client(
        workspace_id="ws_project",
        tool_name="read_file",
        candidates=(other_ws, live, stale, full),
    )
    assert chosen.selected is live
    reasons = dict(chosen.skipped)
    assert reasons["cli_a"] == "workspace scope mismatch"
    assert reasons["cli_c"] == "client is not live"
    assert reasons["cli_d"] == "lease capacity exhausted"


def test_select_bridge_client_rejects_duplicate_active_claim() -> None:
    client = BridgeClientCandidate(
        client_id="cli_a",
        workspace_id="ws_project",
        tools=("read_file",),
        liveness="LIVE",
    )
    chosen = select_bridge_client(
        workspace_id="ws_project",
        tool_name="read_file",
        candidates=(client,),
        claimed_call_ids=("tc_one",),
        call_id="tc_one",
    )
    assert chosen.selected is None
    assert chosen.skipped == (("cli_a", "duplicate active claim"),)


def test_select_bridge_client_falls_back_without_starving_or_duplicating() -> None:
    first = BridgeClientCandidate(
        client_id="cli_a",
        workspace_id="ws_project",
        tools=("read_file", "apply_patch"),
        liveness="STALE",
    )
    second = BridgeClientCandidate(
        client_id="cli_b",
        workspace_id="ws_project",
        tools=("read_file",),
        liveness="LIVE",
    )
    third = BridgeClientCandidate(
        client_id="cli_c",
        workspace_id="ws_project",
        tools=("read_file",),
        liveness="LIVE",
    )
    chosen = select_bridge_client(
        workspace_id="ws_project",
        tool_name="read_file",
        candidates=(third, first, second),
    )
    assert chosen.selected is second
    replay = select_bridge_client(
        workspace_id="ws_project",
        tool_name="read_file",
        candidates=(second, third),
        claimed_call_ids=("tc_shared",),
        call_id="tc_shared",
    )
    assert replay.selected is None
    assert "duplicate active claim" in {reason for _client, reason in replay.skipped}




def test_select_bridge_client_does_not_union_capabilities() -> None:
    lister = BridgeClientCandidate(
        client_id="cli_list",
        workspace_id="ws_project",
        tools=("list_files",),
        effects=("READ",),
        liveness="LIVE",
        snapshot_id="snap_ready01",
    )
    reader = BridgeClientCandidate(
        client_id="cli_read",
        workspace_id="ws_project",
        tools=("read_file",),
        effects=("READ",),
        liveness="LIVE",
        snapshot_id="snap_ready01",
    )
    chosen = select_bridge_client(
        workspace_id="ws_project",
        tool_name="read_file",
        effect="READ",
        snapshot_id="snap_ready01",
        candidates=(lister, reader),
        call_id="tc_read01",
    )
    assert chosen.selected is reader
    assert dict(chosen.skipped)["cli_list"] == "tool capability mismatch"


def test_select_bridge_client_skips_disabled_offline_and_stale_snapshot() -> None:
    disabled = BridgeClientCandidate(
        client_id="cli_disabled",
        workspace_id="ws_project",
        tools=("read_file",),
        effects=("READ",),
        liveness="LIVE",
        status="DISABLED",
        snapshot_id="snap_ready01",
    )
    offline = BridgeClientCandidate(
        client_id="cli_offline",
        workspace_id="ws_project",
        tools=("read_file",),
        effects=("READ",),
        liveness="OFFLINE",
        snapshot_id="snap_ready01",
    )
    stale = BridgeClientCandidate(
        client_id="cli_stale",
        workspace_id="ws_project",
        tools=("read_file",),
        effects=("READ",),
        liveness="LIVE",
        snapshot_id="snap_old01",
    )
    writer = BridgeClientCandidate(
        client_id="cli_write",
        workspace_id="ws_project",
        tools=("read_file",),
        effects=("WRITE",),
        liveness="LIVE",
        snapshot_id="snap_ready01",
    )
    chosen = select_bridge_client(
        workspace_id="ws_project",
        tool_name="read_file",
        effect="READ",
        snapshot_id="snap_ready01",
        candidates=(disabled, offline, stale, writer),
        call_id="tc_read02",
    )
    assert chosen.selected is None
    reasons = dict(chosen.skipped)
    assert reasons["cli_disabled"] == "client is disabled"
    assert reasons["cli_offline"] == "client is not live"
    assert reasons["cli_stale"] == "stale snapshot"
    assert reasons["cli_write"] == "effect capability mismatch"
