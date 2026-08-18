"""Startup recovery.

A process restart leaves attempts that were in flight. Their upstream outcome
cannot be proven, so they become ``INTERRUPTED`` instead of succeeded or failed.
The corresponding running work unit becomes ``FAILED`` so a committed graph can
propagate the interruption instead of waiting forever. Completed entities are
never rewritten.

Recovery is idempotent: a second scan finds nothing in flight and appends no
further events.
"""

from typing import cast

from pydantic import Field

from prp_runtime.domain.enums import (
    AttemptStatus,
    MergeLedgerStatus,
    ReservationStatus,
    ToolCallStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import Attempt, DomainModel, ErrorCategory, ErrorInfo, WorkUnit
from prp_runtime.domain.transitions import (
    RecoveryAction,
    RecoveryDecision,
    decide_attempt_recovery,
    decide_reservation_recovery,
    decide_run_recovery,
    decide_tool_call_recovery,
    decide_work_unit_recovery,
    recover_attempt_on_restart,
    transition_work_unit,
)
from prp_runtime.domain.values import UtcTimestamp, utc_now
from prp_runtime.storage.sqlite import SqliteStore

__all__ = [
    "RECOVERY_REASON",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryReport",
    "decide_attempt_recovery",
    "decide_reservation_recovery",
    "decide_run_recovery",
    "decide_tool_call_recovery",
    "decide_work_unit_recovery",
    "recover_after_restart",
]

RECOVERY_REASON = "process_restart"
RECOVERY_ERROR = ErrorInfo(
    category=ErrorCategory.UNKNOWN,
    message="process restart interrupted an in-flight attempt",
)


class RecoveryReport(DomainModel):
    """What one startup scan changed."""

    interrupted_attempt_ids: tuple[str, ...] = ()
    interrupted_tool_call_ids: tuple[str, ...] = ()
    blocked_tool_call_ids: tuple[str, ...] = ()
    failed_work_unit_ids: tuple[str, ...] = ()
    affected_run_ids: tuple[str, ...] = ()
    released_reservation_ids: tuple[str, ...] = ()
    unknown_merge_ids: tuple[str, ...] = ()
    recoverable_run_ids: tuple[str, ...] = ()
    blocked_run_ids: tuple[str, ...] = ()
    scanned_at: UtcTimestamp = Field(default_factory=utc_now)

    @property
    def changed(self) -> bool:
        return bool(
            self.interrupted_attempt_ids
            or self.interrupted_tool_call_ids
            or self.released_reservation_ids
            or self.unknown_merge_ids
        )


async def recover_after_restart(store: SqliteStore) -> RecoveryReport:
    """Mark in-flight attempts as interrupted and record one event for each.

    Attempts that never started stay ``PENDING``: they were not dispatched, so
    there is nothing to interrupt.
    """
    in_flight = await store.list_attempts_with_status([AttemptStatus.RUNNING])
    in_flight_tools = await store.list_tool_calls_with_status([ToolCallStatus.RUNNING])
    held_reservations = await store.list_reservations(
        statuses=[ReservationStatus.HELD]
    )
    running_merges = await store.list_merge_ledgers(
        statuses=[MergeLedgerStatus.RUNNING]
    )
    pending_runs = await store.list_recoverable_runs()
    recoverable_runs: list[str] = []
    blocked_runs: list[str] = []
    for run in pending_runs:
        attempts = await store.list_run_attempts(run.run_id)
        decision = decide_run_recovery(
            run.status,
            has_dispatched_attempt=any(
                attempt.status is not AttemptStatus.PENDING for attempt in attempts
            ),
        )
        if decision.action is RecoveryAction.CONTINUE:
            recoverable_runs.append(run.run_id)
        elif decision.action is RecoveryAction.BLOCK:
            blocked_runs.append(run.run_id)
    if not in_flight and not in_flight_tools and not held_reservations and not running_merges:
        return RecoveryReport(
            recoverable_run_ids=tuple(recoverable_runs),
            blocked_run_ids=tuple(blocked_runs),
        )

    recovered_at = utc_now()
    interrupted: list[str] = []
    failed_work_units: list[str] = []
    affected_runs: list[str] = []
    released_reservations: list[str] = []
    interrupted_tool_calls: list[str] = []
    blocked_tool_calls: list[str] = []
    unknown_merge_ids: list[str] = []

    def block_run(run_id: str) -> None:
        if run_id not in blocked_runs:
            blocked_runs.append(run_id)

    in_flight_work_unit_ids = {
        attempt.work_unit_id for attempt in in_flight
    }
    unconfirmed_attempts = await store.list_attempts_with_status(
        [AttemptStatus.INTERRUPTED, AttemptStatus.UNKNOWN]
    )
    unconfirmed_work_unit_ids = {
        attempt.work_unit_id for attempt in unconfirmed_attempts
    }
    unconfirmed_work_unit_ids.update(call.work_unit_id for call in in_flight_tools)
    running_units: dict[str, WorkUnit] = {}
    async with store.transaction():
        for call in in_flight_tools:
            decision = decide_tool_call_recovery(call.status)
            if decision.action is RecoveryAction.INTERRUPT:
                await store.mark_tool_call_unknown(
                    call.call_id,
                    completed_at=recovered_at,
                    message=decision.reason,
                )
                interrupted_tool_calls.append(call.call_id)
                if call.run_id not in affected_runs:
                    affected_runs.append(call.run_id)
                block_run(call.run_id)
            elif decision.action is RecoveryAction.BLOCK:
                blocked_tool_calls.append(call.call_id)
                if call.run_id not in blocked_runs:
                    blocked_runs.append(call.run_id)

        for attempt in in_flight:
            status = recover_attempt_on_restart(attempt.status)
            # A recorded start can sit ahead of this clock after a clock step, so
            # the close-out time is never allowed to precede the start.
            completed_at = recovered_at
            if attempt.started_at is not None and completed_at < attempt.started_at:
                completed_at = attempt.started_at
            updated = Attempt.model_validate(
                attempt.model_dump() | {"status": status, "completed_at": completed_at}
            )
            await store.update_attempt(updated)
            await store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_INTERRUPTED,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "reason": RECOVERY_REASON,
                },
                timestamp=recovered_at,
            )
            interrupted.append(attempt.attempt_id)
            if attempt.run_id not in affected_runs:
                affected_runs.append(attempt.run_id)
            block_run(attempt.run_id)
            if attempt.work_unit_id not in running_units:
                unit = await store.get_work_unit(attempt.work_unit_id)
                if unit.status is WorkUnitStatus.RUNNING:
                    running_units[unit.work_unit_id] = unit

        for unit in running_units.values():
            decision = decide_work_unit_recovery(
                unit.status,
                has_running_attempt=True,
            )
            target_status = cast(WorkUnitStatus, decision.target_status)
            failed = WorkUnit.model_validate(
                unit.model_dump()
                | {
                    "status": transition_work_unit(unit.status, target_status)
                }
            )
            await store.update_work_unit(failed)
            await store.append_event(
                unit.run_id,
                EventType.WORK_UNIT_FAILED,
                {
                    "work_unit_id": unit.work_unit_id,
                    "error": RECOVERY_ERROR.model_dump(mode="json"),
                },
                timestamp=recovered_at,
            )
            failed_work_units.append(unit.work_unit_id)

        for reservation in held_reservations:
            decision = decide_reservation_recovery(
                reservation.status,
                has_inflight_attempt=(
                    reservation.request.work_unit_id in in_flight_work_unit_ids
                ),
                has_unconfirmed_attempt=(
                    reservation.request.work_unit_id in unconfirmed_work_unit_ids
                ),
            )
            if decision.action is not RecoveryAction.RELEASE:
                continue
            released = await store.release_reservation(
                reservation.reservation_id,
                completed_at=recovered_at,
            )
            if released.status is ReservationStatus.RELEASED:
                released_reservations.append(released.reservation_id)

        for merge in running_merges:
            unknown = await store.mark_merge_unknown(
                merge.merge_id,
                completed_at=recovered_at,
            )
            if unknown.status is MergeLedgerStatus.UNKNOWN:
                unknown_merge_ids.append(unknown.merge_id)
                if unknown.run_id not in affected_runs:
                    affected_runs.append(unknown.run_id)
                block_run(unknown.run_id)

    return RecoveryReport(
        interrupted_attempt_ids=tuple(interrupted),
        interrupted_tool_call_ids=tuple(interrupted_tool_calls),
        blocked_tool_call_ids=tuple(blocked_tool_calls),
        failed_work_unit_ids=tuple(failed_work_units),
        affected_run_ids=tuple(affected_runs),
        released_reservation_ids=tuple(released_reservations),
        unknown_merge_ids=tuple(unknown_merge_ids),
        recoverable_run_ids=tuple(
            run_id for run_id in recoverable_runs if run_id not in blocked_runs
        ),
        blocked_run_ids=tuple(blocked_runs),
        scanned_at=recovered_at,
    )
