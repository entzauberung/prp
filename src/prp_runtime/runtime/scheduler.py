"""Bounded, side-effect-free PLANNED scheduler waves."""

import asyncio
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from enum import StrEnum, unique

from pydantic import Field, model_validator

from prp_runtime.domain.models import DomainModel, ErrorCategory, ErrorInfo, WorkUnit
from prp_runtime.domain.values import WorkUnitId
from prp_runtime.planning.frontier import compute_frontier
from prp_runtime.runtime.conflicts import (
    ConflictFacts,
    ConflictReport,
    classify_conflict,
    facts_from_claims,
)
from prp_runtime.workspace.changes import ChangeSet
from prp_runtime.workspace.isolation import IsolationBackend, SlotContext

__all__ = [
    "BridgeClientCandidate",
    "BridgeClientSelection",
    "BridgeWaveClaim",
    "assign_bridge_wave_claims",
    "PlannedExecutor",
    "SlotAwarePlannedExecutor",
    "SlotDispatcher",
    "BatchSelection",
    "Scheduler",
    "StartDecision",
    "select_bridge_client",
    "select_non_conflicting_batch",
    "select_non_conflicting_batch_with_reasons",
    "WaveOutcome",
    "WaveResult",
    "WaveStatus",
]

PlannedExecutor = Callable[[WorkUnit], Awaitable["WaveOutcome"]]
SlotAwarePlannedExecutor = Callable[[WorkUnit, SlotContext], Awaitable["WaveOutcome"]]
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
    change_set: ChangeSet | None = None

    @classmethod
    def success(cls, change_set: ChangeSet | None = None) -> "WaveOutcome":
        return cls(succeeded=True, change_set=change_set)

    @classmethod
    def failure(cls, error: ErrorInfo) -> "WaveOutcome":
        return cls(succeeded=False, error=error)

    def model_post_init(self, __context: object) -> None:
        if self.succeeded and self.error is not None:
            raise ValueError("a successful wave outcome must not carry an error")
        if not self.succeeded and self.error is None:
            raise ValueError("a failed wave outcome must carry an error")
        if not self.succeeded and self.change_set is not None:
            raise ValueError("a failed wave outcome must not carry a ChangeSet")


class StartDecision(DomainModel):
    """Controller admission result for one selected unit."""

    deferred: bool = False
    deferred_reason: str | None = None
    error: ErrorInfo | None = None

    @classmethod
    def defer(cls, reason: str = "admission deferred") -> "StartDecision":
        return cls(deferred=True, deferred_reason=reason)

    @classmethod
    def reject(cls, error: ErrorInfo) -> "StartDecision":
        return cls(error=error)

    @classmethod
    def cancel(cls) -> "StartDecision":
        return cls(deferred=False)


StartedCallback = Callable[
    [WorkUnit], Awaitable[WorkUnit | StartDecision | None]
]


class WaveResult(DomainModel):
    """Facts about one finite wave, in stable WorkUnit ID order."""

    graph_version: int = Field(ge=1)
    status: WaveStatus
    ready: tuple[WorkUnitId, ...] = ()
    started: tuple[WorkUnitId, ...] = ()
    succeeded: tuple[WorkUnitId, ...] = ()
    failed: tuple[WorkUnitId, ...] = ()
    deferred: tuple[WorkUnitId, ...] = ()
    deferred_reasons: tuple[tuple[WorkUnitId, str], ...] = ()
    change_sets: tuple[ChangeSet, ...] = ()

    @model_validator(mode="after")
    def _change_sets_are_unique(self) -> "WaveResult":
        ids = [change_set.change_set_id for change_set in self.change_sets]
        if len(ids) != len(set(ids)):
            raise ValueError("wave contains duplicate ChangeSet ids")
        return self


ActualChangeFacts = ChangeSet | ConflictFacts


class BatchSelection(DomainModel):
    """Stable selected/deferred units and reasons for one admission pass."""

    selected: tuple[WorkUnit, ...]
    deferred: tuple[WorkUnit, ...]
    deferred_reasons: tuple[tuple[WorkUnitId, str], ...] = ()


class SlotDispatcher:
    """Acquire one private slot around each dispatched work-unit execution."""

    def __init__(
        self,
        backend: IsolationBackend,
        *,
        snapshot_id: str,
        owner_id: str,
        lease_seconds: int = 300,
    ) -> None:
        self._backend = backend
        self._snapshot_id = snapshot_id
        self._owner_id = owner_id
        self._lease_seconds = lease_seconds

    def context_for(self, unit: WorkUnit) -> SlotContext:
        """Build a fresh owner-bound context for one unit."""
        return SlotContext(
            self._backend,
            snapshot_id=self._snapshot_id,
            work_unit_id=unit.work_unit_id,
            owner_id=self._owner_id,
            lease_seconds=self._lease_seconds,
        )

    async def dispatch(
        self,
        unit: WorkUnit,
        execute: PlannedExecutor,
        *,
        execute_in_slot: SlotAwarePlannedExecutor | None = None,
    ) -> WaveOutcome:
        """Run one unit with cleanup guaranteed after success or failure."""
        context = self.context_for(unit)
        try:
            context.acquire()
            if execute_in_slot is not None:
                return await execute_in_slot(unit, context)
            return await execute(unit)
        finally:
            context.cleanup()


def _declared_conflict_reason(unit: WorkUnit, selected: WorkUnit) -> str | None:
    if any(
        claim.conflicts_with(other)
        for claim in unit.resource_claims
        for other in selected.resource_claims
    ):
        return f"deferred by {selected.work_unit_id}: declared resource access overlaps"
    return None


def _runtime_facts(
    unit: WorkUnit,
    actual_changesets: Mapping[WorkUnitId, ActualChangeFacts],
) -> ConflictFacts:
    actual = actual_changesets.get(unit.work_unit_id)
    if actual is None:
        return facts_from_claims(unit.resource_claims, unknown=True)
    if isinstance(actual, ConflictFacts):
        return actual
    return ConflictFacts.from_changeset(actual, claims=unit.resource_claims)


def _runtime_conflict_reason(
    unit: WorkUnit,
    selected: WorkUnit,
    actual_changesets: Mapping[WorkUnitId, ActualChangeFacts],
) -> str | None:
    report: ConflictReport = classify_conflict(
        _runtime_facts(unit, actual_changesets),
        _runtime_facts(selected, actual_changesets),
    )
    if not report.conflict:
        return None
    return f"deferred by {selected.work_unit_id}: {report.kind.value} - {report.reason}"


def select_non_conflicting_batch_with_reasons(
    ready_units: Sequence[WorkUnit],
    *,
    max_concurrency: int,
    actual_changesets: Mapping[WorkUnitId, ActualChangeFacts] | None = None,
) -> BatchSelection:
    """Select a stable batch and preserve deterministic admission explanations."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    selected: list[WorkUnit] = []
    deferred: list[WorkUnit] = []
    reasons: list[tuple[WorkUnitId, str]] = []
    for unit in sorted(ready_units, key=lambda value: value.work_unit_id):
        if len(selected) >= max_concurrency:
            deferred.append(unit)
            reasons.append((unit.work_unit_id, "deferred: max concurrency reached"))
            continue
        reason: str | None = None
        for selected_unit in selected:
            reason = (
                _runtime_conflict_reason(unit, selected_unit, actual_changesets)
                if actual_changesets is not None
                else _declared_conflict_reason(unit, selected_unit)
            )
            if reason is not None:
                break
        if reason is not None:
            deferred.append(unit)
            reasons.append((unit.work_unit_id, reason))
            continue
        selected.append(unit)
    return BatchSelection(
        selected=tuple(selected),
        deferred=tuple(deferred),
        deferred_reasons=tuple(reasons),
    )


def select_non_conflicting_batch(
    ready_units: Sequence[WorkUnit],
    *,
    max_concurrency: int,
    actual_changesets: Mapping[WorkUnitId, ActualChangeFacts] | None = None,
) -> tuple[tuple[WorkUnit, ...], tuple[WorkUnit, ...]]:
    """Select a stable bounded batch; conflicting units remain deferred."""
    result = select_non_conflicting_batch_with_reasons(
        ready_units,
        max_concurrency=max_concurrency,
        actual_changesets=actual_changesets,
    )
    return result.selected, result.deferred


class BridgeClientCandidate(DomainModel):
    """Public eligibility facts for one model-free Bridge client."""

    client_id: str
    workspace_id: str
    tools: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    liveness: str
    snapshot_id: str | None = None
    status: str = "ACTIVE"
    fingerprint: str | None = None
    active_claims: int = Field(default=0, ge=0)
    max_active_claims: int = Field(default=1, ge=1)


class BridgeClientSelection(DomainModel):
    """Deterministic client choice plus skipped ineligible reasons."""

    selected: BridgeClientCandidate | None = None
    skipped: tuple[tuple[str, str], ...] = ()


def select_bridge_client(
    *,
    workspace_id: str,
    tool_name: str,
    candidates: Sequence[BridgeClientCandidate],
    claimed_call_ids: Collection[str] = (),
    call_id: str | None = None,
    effect: str | None = None,
    snapshot_id: str | None = None,
) -> BridgeClientSelection:
    """Choose one live, scoped, capacity-ready client without client authority."""
    skipped: list[tuple[str, str]] = []
    if call_id is not None and call_id in claimed_call_ids:
        for candidate in sorted(candidates, key=lambda item: item.client_id):
            skipped.append((candidate.client_id, "duplicate active claim"))
        return BridgeClientSelection(selected=None, skipped=tuple(skipped))
    eligible: list[BridgeClientCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.client_id):
        if str(candidate.status).upper() == "DISABLED":
            skipped.append((candidate.client_id, "client is disabled"))
            continue
        if candidate.liveness != "LIVE":
            skipped.append((candidate.client_id, "client is not live"))
            continue
        if candidate.workspace_id != workspace_id:
            skipped.append((candidate.client_id, "workspace scope mismatch"))
            continue
        if tool_name not in candidate.tools:
            skipped.append((candidate.client_id, "tool capability mismatch"))
            continue
        if effect is not None and candidate.effects and effect not in candidate.effects:
            skipped.append((candidate.client_id, "effect capability mismatch"))
            continue
        if (
            snapshot_id is not None
            and candidate.snapshot_id is not None
            and candidate.snapshot_id != snapshot_id
        ):
            skipped.append((candidate.client_id, "stale snapshot"))
            continue
        if candidate.active_claims >= candidate.max_active_claims:
            skipped.append((candidate.client_id, "lease capacity exhausted"))
            continue
        eligible.append(candidate)
    selected = eligible[0] if eligible else None
    return BridgeClientSelection(selected=selected, skipped=tuple(skipped))



class BridgeWaveClaim(DomainModel):
    """Server-owned claim facts for one ready Progressive work unit."""

    run_id: str
    work_unit_id: str
    graph_version: int = Field(ge=1)
    snapshot_id: str
    workspace_id: str
    tool_name: str
    client_id: str


def assign_bridge_wave_claims(
    ready_work_unit_ids: Sequence[str],
    *,
    run_id: str,
    graph_version: int,
    snapshot_id: str,
    workspace_id: str,
    tool_name: str,
    candidates: Sequence[BridgeClientCandidate],
    claimed_call_ids: Collection[str] = (),
    call_ids: Mapping[str, str] | None = None,
) -> tuple[tuple[BridgeWaveClaim, ...], tuple[tuple[str, str], ...]]:
    """Assign each ready unit to at most one live scoped client."""
    pool = list(candidates)
    claims: list[BridgeWaveClaim] = []
    skipped: list[tuple[str, str]] = []
    mapping = call_ids or {}
    for work_unit_id in ready_work_unit_ids:
        selection = select_bridge_client(
            workspace_id=workspace_id,
            tool_name=tool_name,
            candidates=pool,
            claimed_call_ids=claimed_call_ids,
            call_id=mapping.get(work_unit_id),
            snapshot_id=snapshot_id,
        )
        skipped.extend(selection.skipped)
        if selection.selected is None:
            continue
        claims.append(
            BridgeWaveClaim(
                run_id=run_id,
                work_unit_id=work_unit_id,
                graph_version=graph_version,
                snapshot_id=snapshot_id,
                workspace_id=workspace_id,
                tool_name=tool_name,
                client_id=selection.selected.client_id,
            )
        )
        pool = [
            candidate.model_copy(update={"active_claims": candidate.active_claims + 1})
            if candidate.client_id == selection.selected.client_id
            else candidate
            for candidate in pool
        ]
    return tuple(claims), tuple(skipped)


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
        eligible_work_unit_ids: Collection[WorkUnitId] | None = None,
        actual_changesets: Mapping[WorkUnitId, ActualChangeFacts] | None = None,
        slot_dispatcher: SlotDispatcher | None = None,
        execute_in_slot: SlotAwarePlannedExecutor | None = None,
        propagate_executor_exceptions: bool = False,
    ) -> WaveResult:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if execute_in_slot is not None and slot_dispatcher is None:
            raise ValueError("execute_in_slot requires slot_dispatcher")
        frontier = compute_frontier(tuple(work_units), graph_version=graph_version)
        eligible = (
            None
            if eligible_work_unit_ids is None
            else frozenset(eligible_work_unit_ids)
        )
        known_ids = {unit.work_unit_id for unit in work_units}
        if eligible is not None and not eligible <= known_ids:
            raise ValueError("eligible work units must belong to the graph")
        ready = (
            frontier.ready
            if eligible is None
            else tuple(unit_id for unit_id in frontier.ready if unit_id in eligible)
        )
        if not work_units:
            return WaveResult(graph_version=graph_version, status=WaveStatus.EMPTY)
        if not ready:
            if frontier.ready:
                status = WaveStatus.WAITING
            elif (
                frontier.complete
                and not frontier.waiting
                and not frontier.blocked
            ):
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
        selection = select_non_conflicting_batch_with_reasons(
            tuple(by_id[work_unit_id] for work_unit_id in ready),
            max_concurrency=max_concurrency,
            actual_changesets=actual_changesets,
        )
        selected = selection.selected
        deferred = selection.deferred
        started: list[str] = []
        succeeded: list[str] = []
        failed: list[str] = []
        change_sets: list[ChangeSet] = []
        running_units: list[WorkUnit] = []
        deferred_ids = [unit.work_unit_id for unit in deferred]
        deferred_reasons = dict(selection.deferred_reasons)
        cancelled = False
        for index, unit in enumerate(selected):
            admission = await on_started(unit)
            if isinstance(admission, StartDecision):
                if admission.deferred:
                    deferred_ids.append(unit.work_unit_id)
                    deferred_reasons[unit.work_unit_id] = (
                        admission.deferred_reason or "admission deferred"
                    )
                    continue
                if admission.error is not None:
                    # Admission already persisted the unit's non-running
                    # terminal decision. The failed callback is reserved for
                    # units that actually entered RUNNING.
                    failed.append(unit.work_unit_id)
                    continue
                cancelled = True
                deferred_ids.extend(
                    pending.work_unit_id for pending in selected[index + 1 :]
                )
                for pending in selected[index + 1 :]:
                    deferred_reasons[pending.work_unit_id] = "wave start cancelled"
                break
            started_unit = admission
            if started_unit is None:
                cancelled = True
                deferred_ids.extend(
                    pending.work_unit_id for pending in selected[index:]
                )
                for pending in selected[index:]:
                    deferred_reasons[pending.work_unit_id] = "wave start cancelled"
                break
            running_units.append(started_unit)
            started.append(started_unit.work_unit_id)

        async def dispatch(unit: WorkUnit) -> WaveOutcome:
            try:
                if slot_dispatcher is None:
                    return await execute(unit)
                return await slot_dispatcher.dispatch(
                    unit,
                    execute,
                    execute_in_slot=execute_in_slot,
                )
            except Exception:
                if propagate_executor_exceptions:
                    raise
                return WaveOutcome.failure(
                    ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message="planned unit execution failed unexpectedly",
                    )
                )

        outcomes = await asyncio.gather(*(dispatch(unit) for unit in running_units))
        for running, outcome in zip(running_units, outcomes, strict=True):
            work_unit_id = running.work_unit_id
            if outcome.succeeded:
                await on_succeeded(running)
                succeeded.append(work_unit_id)
                if outcome.change_set is not None:
                    change_sets.append(outcome.change_set)
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
            deferred_reasons=tuple(
                (work_unit_id, deferred_reasons[work_unit_id])
                for work_unit_id in deferred_ids
            ),
            change_sets=tuple(change_sets),
        )
