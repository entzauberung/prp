"""Targeted tests for restart recovery and event cursor behaviour."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from prp_runtime.control.reservations import ReservationRequest
from prp_runtime.domain.enums import (
    AttemptStatus,
    ModelRole,
    ReservationStatus,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import (
    Artifact,
    Attempt,
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
    Run,
    Usage,
    WorkUnit,
    new_artifact_id,
)
from prp_runtime.domain.values import (
    ModelRef,
    new_attempt_id,
    new_run_id,
    new_work_unit_id,
)
from prp_runtime.storage.recovery import (
    RECOVERY_REASON,
    RecoveryReport,
    recover_after_restart,
)
from prp_runtime.storage.sqlite import SqliteStore

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
WORKER = ModelRef(provider="openai_compatible", model="weak-model")


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "recovery-test.db"


async def build_running_run(
    store: SqliteStore, *, attempt_count: int = 1
) -> tuple[Run, WorkUnit, tuple[Attempt, ...]]:
    """Persist a run that looks like it was executing when the process stopped."""
    run = Run(
        run_id=new_run_id(),
        request=NativeRunRequest(input="analyse the report"),
        status=RunStatus.RUNNING,
        created_at=T0,
        started_at=T0,
    )
    await store.create_run(run)
    unit = WorkUnit(
        work_unit_id=new_work_unit_id(),
        run_id=run.run_id,
        name="unit",
        instruction="do the work",
        status=WorkUnitStatus.RUNNING,
        created_at=T0,
    )
    await store.create_work_unit(unit)
    await store.append_event(run.run_id, EventType.RUN_CREATED, {"request": {"input": "x"}})
    await store.append_event(run.run_id, EventType.RUN_STARTED)
    attempts: list[Attempt] = []
    for index in range(1, attempt_count + 1):
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_index=index,
            role=ModelRole.WORKER,
            model=WORKER,
            status=AttemptStatus.RUNNING,
            created_at=T0,
            started_at=T0,
        )
        await store.create_attempt(attempt)
        await store.append_event(
            run.run_id,
            EventType.ATTEMPT_STARTED,
            {"work_unit_id": unit.work_unit_id, "attempt_id": attempt.attempt_id},
        )
        attempts.append(attempt)
    return run, unit, tuple(attempts)


async def create_held_reservation(
    store: SqliteStore,
    run: Run,
    unit: WorkUnit,
    *,
    dispatch_key: str = "recovery-dispatch",
) -> str:
    reservation = await store.reserve_reservation(
        ReservationRequest(
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            dispatch_key=dispatch_key,
        ),
        created_at=T0,
        held_at=T0,
    )
    return reservation.reservation_id


# --- restart semantics ----------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_marks_in_flight_attempts_interrupted(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        run, unit, (attempt,) = await build_running_run(store)
        sequence_before = await store.last_sequence(run.run_id)

    # A new store on the same file stands in for the restarted process.
    async with SqliteStore(database_path) as store:
        report = await recover_after_restart(store)
        assert report.changed is True
        assert report.interrupted_attempt_ids == (attempt.attempt_id,)
        assert report.failed_work_unit_ids == (unit.work_unit_id,)
        assert report.affected_run_ids == (run.run_id,)
        assert report.recoverable_run_ids == ()
        assert report.blocked_run_ids == (run.run_id,)

        recovered = await store.get_attempt(attempt.attempt_id)
        assert recovered.status is AttemptStatus.INTERRUPTED
        assert recovered.completed_at is not None
        assert recovered.started_at == attempt.started_at
        assert recovered.error is None
        assert recovered.usage is None

        events = await store.list_events(run.run_id, after_sequence=sequence_before)
        assert len(events) == 2
        assert events[0].event_type is EventType.ATTEMPT_INTERRUPTED
        assert events[0].payload == {
            "work_unit_id": unit.work_unit_id,
            "attempt_id": attempt.attempt_id,
            "reason": RECOVERY_REASON,
        }
        assert events[1].event_type is EventType.WORK_UNIT_FAILED
        assert events[1].payload["work_unit_id"] == unit.work_unit_id
        assert events[1].payload["error"] == {
            "category": ErrorCategory.UNKNOWN.value,
            "message": "process restart interrupted an in-flight attempt",
        }


@pytest.mark.asyncio
async def test_restart_fails_running_unit_without_settling_run(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        run, unit, _ = await build_running_run(store)
        await recover_after_restart(store)
        assert (await store.get_run(run.run_id)).status is RunStatus.RUNNING
        assert (await store.get_work_unit(unit.work_unit_id)).status is WorkUnitStatus.FAILED
        assert (await store.get_run(run.run_id)).error is None


@pytest.mark.asyncio
async def test_recovery_is_idempotent(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        run, _, _ = await build_running_run(store)
        first = await recover_after_restart(store)
        assert first.changed is True
        sequence_after_first = await store.last_sequence(run.run_id)

        second = await recover_after_restart(store)
        assert second == RecoveryReport(scanned_at=second.scanned_at)
        assert second.changed is False
        assert await store.last_sequence(run.run_id) == sequence_after_first
        interrupted_events = [
            event
            for event in await store.list_events(run.run_id)
            if event.event_type is EventType.ATTEMPT_INTERRUPTED
        ]
        assert len(interrupted_events) == 1
        failed_events = [
            event
            for event in await store.list_events(run.run_id)
            if event.event_type is EventType.WORK_UNIT_FAILED
        ]
        assert len(failed_events) == 1


@pytest.mark.asyncio
async def test_recovery_on_a_clean_database_changes_nothing(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        report = await recover_after_restart(store)
        assert report.changed is False
        assert report.interrupted_attempt_ids == ()
        assert report.affected_run_ids == ()
        assert report.released_reservation_ids == ()


@pytest.mark.asyncio
async def test_recovery_reports_pending_runs_that_were_never_dispatched(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        run = Run(
            run_id=new_run_id(),
            request=NativeRunRequest(input="resume pending work"),
            created_at=T0,
        )
        await store.create_run(run)

        first = await recover_after_restart(store)
        second = await recover_after_restart(store)

        assert first.recoverable_run_ids == (run.run_id,)
        assert first.blocked_run_ids == ()
        assert second.recoverable_run_ids == (run.run_id,)
        assert second.blocked_run_ids == ()
        assert second.changed is False


@pytest.mark.asyncio
async def test_recovery_blocks_pending_run_with_dispatched_attempt(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        run = Run(
            run_id=new_run_id(),
            request=NativeRunRequest(input="diagnose inconsistent state"),
            created_at=T0,
        )
        await store.create_run(run)
        unit = WorkUnit(
            work_unit_id=new_work_unit_id(),
            run_id=run.run_id,
            name="unit",
            instruction="do the work",
            status=WorkUnitStatus.PENDING,
            created_at=T0,
        )
        await store.create_work_unit(unit)
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_index=1,
            role=ModelRole.WORKER,
            model=WORKER,
            status=AttemptStatus.RUNNING,
            created_at=T0,
            started_at=T0,
        )
        await store.create_attempt(attempt)

        report = await recover_after_restart(store)

        assert report.recoverable_run_ids == ()
        assert report.blocked_run_ids == (run.run_id,)
        assert (await store.get_run(run.run_id)).status is RunStatus.PENDING


@pytest.mark.asyncio
async def test_restart_releases_held_reservation_without_inflight_attempt(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        run = Run(
            run_id=new_run_id(),
            request=NativeRunRequest(input="reserve work"),
            status=RunStatus.RUNNING,
            created_at=T0,
            started_at=T0,
        )
        await store.create_run(run)
        unit = WorkUnit(
            work_unit_id=new_work_unit_id(),
            run_id=run.run_id,
            name="unit",
            instruction="do the work",
            status=WorkUnitStatus.READY,
            created_at=T0,
        )
        await store.create_work_unit(unit)
        await store.append_event(run.run_id, EventType.RUN_CREATED, {"request": {"input": "x"}})
        await store.append_event(run.run_id, EventType.RUN_STARTED)
        reservation_id = await create_held_reservation(store, run, unit)
        before = await store.last_sequence(run.run_id)

    async with SqliteStore(database_path) as store:
        first = await recover_after_restart(store)
        assert first.released_reservation_ids == (reservation_id,)
        assert first.changed is True
        released = await store.get_reservation(reservation_id)
        assert released.status is ReservationStatus.RELEASED
        events = await store.list_events(run.run_id, after_sequence=before)
        assert [event.event_type for event in events] == [
            EventType.RESERVATION_RELEASED
        ]

        second = await recover_after_restart(store)
        assert second.released_reservation_ids == ()
        assert second.changed is False
        assert await store.last_sequence(run.run_id) == before + 1


@pytest.mark.asyncio
async def test_restart_preserves_held_reservation_with_running_attempt(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        run, unit, (attempt,) = await build_running_run(store)
        reservation_id = await create_held_reservation(store, run, unit)

        report = await recover_after_restart(store)

        assert report.released_reservation_ids == ()
        assert report.interrupted_attempt_ids == (attempt.attempt_id,)
        assert (await store.get_reservation(reservation_id)).status is ReservationStatus.HELD
        assert not any(
            event.event_type is EventType.RESERVATION_RELEASED
            for event in await store.list_events(run.run_id)
        )

        second = await recover_after_restart(store)
        assert second.changed is False
        assert second.released_reservation_ids == ()
        assert (await store.get_reservation(reservation_id)).status is ReservationStatus.HELD


@pytest.mark.asyncio
async def test_recovery_does_not_touch_completed_entities(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        run, unit, (running,) = await build_running_run(store)
        succeeded = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_index=2,
            role=ModelRole.WORKER,
            model=WORKER,
            status=AttemptStatus.SUCCEEDED,
            usage=Usage(input_tokens=3, output_tokens=4),
            created_at=T0,
            started_at=T0,
            completed_at=T0 + timedelta(seconds=1),
        )
        failed = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_index=3,
            role=ModelRole.WORKER,
            model=WORKER,
            status=AttemptStatus.FAILED,
            error=ErrorInfo(category=ErrorCategory.TIMEOUT, message="upstream timed out"),
            created_at=T0,
            started_at=T0,
            completed_at=T0 + timedelta(seconds=2),
        )
        pending = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_index=4,
            role=ModelRole.WORKER,
            model=WORKER,
            created_at=T0,
        )
        for attempt in (succeeded, failed, pending):
            await store.create_attempt(attempt)
        artifact = Artifact(
            artifact_id=new_artifact_id(),
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_id=succeeded.attempt_id,
            name="answer",
            content="done",
            created_at=T0,
        )
        await store.add_artifact(artifact)

        report = await recover_after_restart(store)

        assert report.interrupted_attempt_ids == (running.attempt_id,)
        assert report.failed_work_unit_ids == (unit.work_unit_id,)
        assert (await store.get_work_unit(unit.work_unit_id)).status is WorkUnitStatus.FAILED
        assert await store.get_attempt(succeeded.attempt_id) == succeeded
        assert await store.get_attempt(failed.attempt_id) == failed
        assert (await store.get_attempt(pending.attempt_id)).status is AttemptStatus.PENDING
        assert await store.get_artifact(artifact.artifact_id) == artifact


@pytest.mark.asyncio
async def test_recovery_spans_multiple_runs_without_duplicate_sequences(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        first_run, _, first_attempts = await build_running_run(store, attempt_count=2)
        second_run, _, second_attempts = await build_running_run(store)

        report = await recover_after_restart(store)

        assert set(report.affected_run_ids) == {first_run.run_id, second_run.run_id}
        assert set(report.interrupted_attempt_ids) == {
            attempt.attempt_id for attempt in first_attempts + second_attempts
        }
        for run_id, expected_interrupted in (
            (first_run.run_id, 2),
            (second_run.run_id, 1),
        ):
            ledger = await store.list_events(run_id)
            assert assert_sequence_chain(ledger) is None
            assert (
                sum(
                    1
                    for event in ledger
                    if event.event_type is EventType.ATTEMPT_INTERRUPTED
                )
                == expected_interrupted
            )
            assert (
                sum(
                    1
                    for event in ledger
                    if event.event_type is EventType.WORK_UNIT_FAILED
                )
                == 1
            )


@pytest.mark.asyncio
async def test_recovery_is_atomic_when_an_event_append_fails(
    database_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with SqliteStore(database_path) as store:
        run, _, attempts = await build_running_run(store, attempt_count=2)
        sequence_before = await store.last_sequence(run.run_id)
        original = SqliteStore.append_event
        calls = {"count": 0}

        async def failing_append(self: SqliteStore, *args: object, **kwargs: object) -> object:
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("ledger write failed")
            return await original(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(SqliteStore, "append_event", failing_append)
        with pytest.raises(RuntimeError, match="ledger write failed"):
            await recover_after_restart(store)
        monkeypatch.undo()

        for attempt in attempts:
            assert (await store.get_attempt(attempt.attempt_id)).status is AttemptStatus.RUNNING
        unit_id = attempts[0].work_unit_id
        assert (await store.get_work_unit(unit_id)).status is WorkUnitStatus.RUNNING
        assert await store.last_sequence(run.run_id) == sequence_before


# --- event cursor ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_cursor_beyond_the_last_sequence_returns_no_events(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        run, _, _ = await build_running_run(store)
        last = await store.last_sequence(run.run_id)
        assert last is not None
        assert await store.list_events(run.run_id, after_sequence=last) == ()
        assert await store.list_events(run.run_id, after_sequence=last + 1000) == ()
        assert await store.list_events(new_run_id(), after_sequence=5) == ()


@pytest.mark.asyncio
async def test_cursor_rejects_a_negative_value(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        run, _, _ = await build_running_run(store)
        with pytest.raises(ValueError, match="must not be negative"):
            await store.list_events(run.run_id, after_sequence=-1)


@pytest.mark.asyncio
async def test_a_reconnecting_subscriber_resumes_from_its_cursor(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        run, _, _ = await build_running_run(store)
        seen = await store.list_events(run.run_id)
        cursor = seen[-1].sequence

    async with SqliteStore(database_path) as store:
        assert await store.list_events(run.run_id, after_sequence=cursor) == ()
        await recover_after_restart(store)
        fresh = await store.list_events(run.run_id, after_sequence=cursor)
        assert [event.event_type for event in fresh] == [
            EventType.ATTEMPT_INTERRUPTED,
            EventType.WORK_UNIT_FAILED,
        ]
        assert fresh[0].sequence == cursor + 1
        assert await store.list_events(run.run_id, after_sequence=fresh[-1].sequence) == ()


@pytest.mark.asyncio
async def test_cursor_paging_covers_the_whole_ledger(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        run, _, _ = await build_running_run(store, attempt_count=3)
        collected: list[int] = []
        cursor = 0
        while True:
            page = await store.list_events(run.run_id, after_sequence=cursor, limit=2)
            if not page:
                break
            collected.extend(event.sequence for event in page)
            cursor = page[-1].sequence
        assert collected == list(range(1, len(collected) + 1))
        assert collected == [event.sequence for event in await store.list_events(run.run_id)]
