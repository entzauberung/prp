"""Pure ready-frontier calculation over committed WorkUnit facts."""

from enum import StrEnum, unique

from pydantic import Field

from prp_runtime.domain.enums import WorkUnitStatus
from prp_runtime.domain.models import DomainModel, WorkUnit
from prp_runtime.domain.values import WorkUnitId

__all__ = [
    "BlockedReason",
    "BlockedUnit",
    "FrontierResult",
    "compute_frontier",
]


@unique
class BlockedReason(StrEnum):
    """Why a unit cannot become ready in the current graph version."""

    OWN_BLOCKED = "OWN_BLOCKED"
    DANGLING_DEPENDENCY = "DANGLING_DEPENDENCY"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    DEPENDENCY_CANCELLED = "DEPENDENCY_CANCELLED"
    DEPENDENCY_INVALIDATED = "DEPENDENCY_INVALIDATED"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"


class BlockedUnit(DomainModel):
    work_unit_id: WorkUnitId
    reason: BlockedReason
    dependency_ids: tuple[WorkUnitId, ...] = ()


class FrontierResult(DomainModel):
    """Stable classification of one committed graph version."""

    graph_version: int = Field(ge=1)
    ready: tuple[WorkUnitId, ...] = ()
    waiting: tuple[WorkUnitId, ...] = ()
    blocked: tuple[WorkUnitId, ...] = ()
    blocked_details: tuple[BlockedUnit, ...] = ()
    complete: tuple[WorkUnitId, ...] = ()


def _blocked_reason(
    unit: WorkUnit,
    by_id: dict[str, WorkUnit],
) -> BlockedUnit | None:
    if unit.status is WorkUnitStatus.BLOCKED:
        return BlockedUnit(
            work_unit_id=unit.work_unit_id,
            reason=BlockedReason.OWN_BLOCKED,
        )
    dependencies = tuple(sorted(unit.depends_on))
    dangling = tuple(dependency for dependency in dependencies if dependency not in by_id)
    if dangling:
        return BlockedUnit(
            work_unit_id=unit.work_unit_id,
            reason=BlockedReason.DANGLING_DEPENDENCY,
            dependency_ids=dangling,
        )
    statuses = {by_id[dependency].status for dependency in dependencies}
    reason_by_status = (
        (WorkUnitStatus.FAILED, BlockedReason.DEPENDENCY_FAILED),
        (WorkUnitStatus.CANCELLED, BlockedReason.DEPENDENCY_CANCELLED),
        (WorkUnitStatus.INVALIDATED, BlockedReason.DEPENDENCY_INVALIDATED),
        (WorkUnitStatus.BLOCKED, BlockedReason.DEPENDENCY_BLOCKED),
    )
    for status, reason in reason_by_status:
        if status in statuses:
            return BlockedUnit(
                work_unit_id=unit.work_unit_id,
                reason=reason,
                dependency_ids=tuple(
                    dependency
                    for dependency in dependencies
                    if by_id[dependency].status is status
                ),
            )
    return None


def compute_frontier(
    work_units: tuple[WorkUnit, ...],
    *,
    graph_version: int,
) -> FrontierResult:
    """Classify one graph version without executing or changing any unit."""
    if graph_version < 1:
        raise ValueError("graph_version must be at least 1")
    current = tuple(unit for unit in work_units if unit.graph_version == graph_version)
    by_id = {unit.work_unit_id: unit for unit in current}
    if len(by_id) != len(current):
        raise ValueError("frontier input contains duplicate work unit ids")

    ready: list[str] = []
    waiting: list[str] = []
    blocked: list[str] = []
    blocked_details: list[BlockedUnit] = []
    complete: list[str] = []
    for unit in sorted(current, key=lambda value: value.work_unit_id):
        if unit.status in {
            WorkUnitStatus.SUCCEEDED,
            WorkUnitStatus.FAILED,
            WorkUnitStatus.CANCELLED,
            WorkUnitStatus.INVALIDATED,
        }:
            complete.append(unit.work_unit_id)
            continue
        if unit.status is WorkUnitStatus.RUNNING:
            waiting.append(unit.work_unit_id)
            continue
        reason = _blocked_reason(unit, by_id)
        if reason is not None:
            blocked.append(unit.work_unit_id)
            blocked_details.append(reason)
            continue
        dependencies = tuple(by_id[dependency] for dependency in unit.depends_on)
        if all(dependency.status is WorkUnitStatus.SUCCEEDED for dependency in dependencies):
            ready.append(unit.work_unit_id)
        else:
            waiting.append(unit.work_unit_id)

    return FrontierResult(
        graph_version=graph_version,
        ready=tuple(ready),
        waiting=tuple(waiting),
        blocked=tuple(blocked),
        blocked_details=tuple(blocked_details),
        complete=tuple(complete),
    )
