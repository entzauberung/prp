"""Single-snapshot scheduler wave tests with Controller-style callbacks."""

import asyncio
from datetime import UTC, datetime

import pytest

from prp_runtime.domain.enums import ResourceAccess, WorkUnitStatus
from prp_runtime.domain.models import ErrorCategory, ErrorInfo, WorkUnit
from prp_runtime.domain.values import ResourceClaim
from prp_runtime.runtime.scheduler import (
    Scheduler,
    WaveOutcome,
    WaveStatus,
    select_non_conflicting_batch,
)

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


def test_invalid_max_concurrency_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        select_non_conflicting_batch((unit("a"),), max_concurrency=0)
