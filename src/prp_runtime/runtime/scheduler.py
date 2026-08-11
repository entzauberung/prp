"""Bounded, side-effect-free PLANNED scheduler waves."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum, unique

from pydantic import Field

from prp_runtime.domain.models import DomainModel, ErrorInfo, WorkUnit
from prp_runtime.domain.values import WorkUnitId
from prp_runtime.planning.frontier import compute_frontier

__all__ = [
    "PlannedExecutor",
    "Scheduler",
    "select_non_conflicting_batch",
    "WaveOutcome",
    "WaveResult",
    "WaveStatus",
]

PlannedExecutor = Callable[[WorkUnit], Awaitable["WaveOutcome"]]
StartedCallback = Callable[[WorkUnit], Awaitable[WorkUnit | None]]
FinishedCallback = Callable[[WorkUnit], Awaitable[None]]
FailedCallback = Callable[[WorkUnit, ErrorInfo], Awaitable[None]]


@unique
class WaveStatus(StrEnum):
    DISPATCHED = "DISPATCHED"
    COMPLETE = "COMPLETE"
    WAITING = "WAITING"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    EMPTY = "EMPTY"


class WaveOutcome(DomainModel):
    """A fake/provider-independent result supplied by the executor."""

    succeeded: bool
    error: ErrorInfo | None = None

    @classmethod
    def success(cls) -> "WaveOutcome":
        return cls(succeeded=True)

    @classmethod
    def failure(cls, error: ErrorInfo) -> "WaveOutcome":
        return cls(succeeded=False, error=error)

    def model_post_init(self, __context: object) -> None:
        if self.succeeded and self.error is not None:
            raise ValueError("a successful wave outcome must not carry an error")
        if not self.succeeded and self.error is None:
            raise ValueError("a failed wave outcome must carry an error")


class WaveResult(DomainModel):
    """Facts about one finite wave, in stable WorkUnit ID order."""

    graph_version: int = Field(ge=1)
    status: WaveStatus
    ready: tuple[WorkUnitId, ...] = ()
    started: tuple[WorkUnitId, ...] = ()
    succeeded: tuple[WorkUnitId, ...] = ()
    failed: tuple[WorkUnitId, ...] = ()
    deferred: tuple[WorkUnitId, ...] = ()


def select_non_conflicting_batch(
    ready_units: Sequence[WorkUnit],
    *,
    max_concurrency: int,
) -> tuple[tuple[WorkUnit, ...], tuple[WorkUnit, ...]]:
    """Select a stable bounded batch; conflicting units remain deferred."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    selected: list[WorkUnit] = []
    deferred: list[WorkUnit] = []
    for unit in sorted(ready_units, key=lambda value: value.work_unit_id):
        if len(selected) >= max_concurrency or any(
            claim.conflicts_with(other)
            for claim in unit.resource_claims
            for selected_unit in selected
            for other in selected_unit.resource_claims
        ):
            deferred.append(unit)
            continue
        selected.append(unit)
    return tuple(selected), tuple(deferred)


class Scheduler:
    """Run one snapshot wave sequentially through Controller callbacks."""

    async def run_wave(
        self,
        work_units: Sequence[WorkUnit],
        *,
        graph_version: int,
        max_concurrency: int = 1,
        execute: PlannedExecutor,
        on_started: StartedCallback,
        on_succeeded: FinishedCallback,
        on_failed: FailedCallback,
    ) -> WaveResult:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        frontier = compute_frontier(tuple(work_units), graph_version=graph_version)
        ready = frontier.ready
        if not work_units:
            return WaveResult(graph_version=graph_version, status=WaveStatus.EMPTY)
        if not ready:
            if frontier.complete and not frontier.waiting and not frontier.blocked:
                status = WaveStatus.COMPLETE
            elif frontier.blocked and not frontier.waiting:
                status = WaveStatus.BLOCKED
            else:
                status = WaveStatus.WAITING
            return WaveResult(
                graph_version=graph_version,
                status=status,
                ready=(),
            )

        by_id = {unit.work_unit_id: unit for unit in work_units}
        selected, deferred = select_non_conflicting_batch(
            tuple(by_id[work_unit_id] for work_unit_id in ready),
            max_concurrency=max_concurrency,
        )
        started: list[str] = []
        succeeded: list[str] = []
        failed: list[str] = []
        running_units: list[WorkUnit] = []
        deferred_ids = [unit.work_unit_id for unit in deferred]
        cancelled = False
        for index, unit in enumerate(selected):
            started_unit = await on_started(unit)
            if started_unit is None:
                cancelled = True
                deferred_ids.extend(
                    pending.work_unit_id for pending in selected[index:]
                )
                break
            running_units.append(started_unit)
            started.append(started_unit.work_unit_id)

        outcomes = await asyncio.gather(*(execute(unit) for unit in running_units))
        for running, outcome in zip(running_units, outcomes, strict=True):
            work_unit_id = running.work_unit_id
            if outcome.succeeded:
                await on_succeeded(running)
                succeeded.append(work_unit_id)
            else:
                assert outcome.error is not None
                await on_failed(running, outcome.error)
                failed.append(work_unit_id)
        return WaveResult(
            graph_version=graph_version,
            status=WaveStatus.CANCELLED if cancelled else WaveStatus.DISPATCHED,
            ready=ready,
            started=tuple(started),
            succeeded=tuple(succeeded),
            failed=tuple(failed),
            deferred=tuple(deferred_ids),
        )
