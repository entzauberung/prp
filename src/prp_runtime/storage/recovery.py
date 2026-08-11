"""Startup recovery.

A process restart leaves attempts that were in flight. Their upstream outcome
cannot be proven, so they become ``INTERRUPTED`` instead of succeeded or failed.
The corresponding running work unit becomes ``FAILED`` so a committed graph can
propagate the interruption instead of waiting forever. Completed entities are
never rewritten.

Recovery is idempotent: a second scan finds nothing in flight and appends no
further events.
"""

from pydantic import Field

from prp_runtime.domain.enums import AttemptStatus, WorkUnitStatus
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import Attempt, DomainModel, ErrorCategory, ErrorInfo, WorkUnit
from prp_runtime.domain.transitions import recover_attempt_on_restart, transition_work_unit
from prp_runtime.domain.values import UtcTimestamp, utc_now
from prp_runtime.storage.sqlite import SqliteStore

__all__ = ["RECOVERY_REASON", "RecoveryReport", "recover_after_restart"]

RECOVERY_REASON = "process_restart"
RECOVERY_ERROR = ErrorInfo(
    category=ErrorCategory.UNKNOWN,
    message="process restart interrupted an in-flight attempt",
)


class RecoveryReport(DomainModel):
    """What one startup scan changed."""

    interrupted_attempt_ids: tuple[str, ...] = ()
    failed_work_unit_ids: tuple[str, ...] = ()
    affected_run_ids: tuple[str, ...] = ()
    scanned_at: UtcTimestamp = Field(default_factory=utc_now)

    @property
    def changed(self) -> bool:
        return bool(self.interrupted_attempt_ids)


async def recover_after_restart(store: SqliteStore) -> RecoveryReport:
    """Mark in-flight attempts as interrupted and record one event for each.

    Attempts that never started stay ``PENDING``: they were not dispatched, so
    there is nothing to interrupt.
    """
    in_flight = await store.list_attempts_with_status([AttemptStatus.RUNNING])
    if not in_flight:
        return RecoveryReport()

    recovered_at = utc_now()
    interrupted: list[str] = []
    failed_work_units: list[str] = []
    affected_runs: list[str] = []
    running_units: dict[str, WorkUnit] = {}
    async with store.transaction():
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
            if attempt.work_unit_id not in running_units:
                unit = await store.get_work_unit(attempt.work_unit_id)
                if unit.status is WorkUnitStatus.RUNNING:
                    running_units[unit.work_unit_id] = unit

        for unit in running_units.values():
            failed = WorkUnit.model_validate(
                unit.model_dump()
                | {
                    "status": transition_work_unit(
                        unit.status, WorkUnitStatus.FAILED
                    )
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

    return RecoveryReport(
        interrupted_attempt_ids=tuple(interrupted),
        failed_work_unit_ids=tuple(failed_work_units),
        affected_run_ids=tuple(affected_runs),
        scanned_at=recovered_at,
    )
