"""Deterministic, side-effect-free planning for PLANNED execution batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path

from pydantic import Field, model_validator

from prp_runtime.domain.enums import ResourceAccess, ToolEffect
from prp_runtime.domain.models import DomainModel, WorkUnit
from prp_runtime.domain.values import WorkUnitId
from prp_runtime.runtime.conflicts import (
    ConflictFact,
    ConflictFacts,
    ConflictKind,
    ConflictReport,
    conflicts_from_changesets,
)
from prp_runtime.workspace.changes import ChangeSet
from prp_runtime.workspace.merge import (
    GitMergeBackend,
    MergeResult,
    MergeStatus,
    StagedChangeSet,
    merge_change_sets,
    promote_merge,
)

__all__ = [
    "CoordinationBatch",
    "CoordinationMode",
    "CoordinationPlan",
    "Coordinator",
    "conflicts_from_changesets",
    "GitMergeBackend",
    "MergeResult",
    "MergeStatus",
    "StagedChangeSet",
    "merge_change_sets",
    "plan_coordination",
    "promote_merge",
    "resolve_work_unit_effect",
]


@unique
class CoordinationMode(StrEnum):
    """Execution isolation posture for one planned batch."""

    READ_ONLY = "READ_ONLY"
    PARALLEL_WRITE = "PARALLEL_WRITE"
    SERIAL = "SERIAL"


class CoordinationBatch(DomainModel):
    """One stable batch with explicit isolation and reason facts."""

    mode: CoordinationMode
    work_unit_ids: tuple[WorkUnitId, ...]
    shared_immutable_snapshot: bool
    requires_isolated_slots: bool
    reasons: tuple[tuple[WorkUnitId, str], ...] = ()

    @model_validator(mode="after")
    def _batch_is_safe(self) -> CoordinationBatch:
        if not self.work_unit_ids:
            raise ValueError("coordination batch must contain a work unit")
        if len(set(self.work_unit_ids)) != len(self.work_unit_ids):
            raise ValueError("coordination batch contains duplicate work units")
        if self.mode is CoordinationMode.READ_ONLY:
            if not self.shared_immutable_snapshot or self.requires_isolated_slots:
                raise ValueError("read-only batch must share an immutable snapshot")
        elif not self.requires_isolated_slots:
            raise ValueError("write or serial batch requires isolated slots")
        if tuple(unit_id for unit_id, _ in self.reasons) != self.work_unit_ids:
            raise ValueError("coordination reasons must cover batch units in order")
        return self


class CoordinationPlan(DomainModel):
    """A complete stable plan without executing providers or touching a workspace."""

    capacity: int = Field(ge=1)
    batches: tuple[CoordinationBatch, ...]

    @model_validator(mode="after")
    def _capacity_is_respected(self) -> CoordinationPlan:
        all_units = [unit_id for batch in self.batches for unit_id in batch.work_unit_ids]
        if len(all_units) != len(set(all_units)):
            raise ValueError("coordination plan contains duplicate work units")
        for batch in self.batches:
            if (
                batch.mode is not CoordinationMode.SERIAL
                and len(batch.work_unit_ids) > self.capacity
            ):
                raise ValueError("coordination batch exceeds capacity")
        return self


@dataclass
class _BatchDraft:
    mode: CoordinationMode
    work_unit_ids: list[WorkUnitId]
    reasons: list[tuple[WorkUnitId, str]]


def _pair_key(left: WorkUnitId, right: WorkUnitId) -> tuple[WorkUnitId, WorkUnitId]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _claims_conflict(left: WorkUnit, right: WorkUnit) -> bool:
    return any(
        claim.conflicts_with(other)
        for claim in left.resource_claims
        for other in right.resource_claims
    )


def _report_for(
    left: WorkUnit,
    right: WorkUnit,
    reports: Mapping[tuple[WorkUnitId, WorkUnitId], ConflictReport],
) -> ConflictReport | None:
    report = reports.get(_pair_key(left.work_unit_id, right.work_unit_id))
    if report is not None:
        return report
    if _claims_conflict(left, right):
        return ConflictReport(
            kind=ConflictKind.READ_WRITE,
            reason="declared resource claims overlap",
            facts=(
                ConflictFact(
                    kind=ConflictKind.READ_WRITE,
                    reason="declared resource claims overlap",
                ),
            ),
        )
    return None


def _reason_for_report(report: ConflictReport, other: WorkUnit) -> str:
    return f"serialized by {other.work_unit_id}: {report.kind.value} - {report.reason}"


def resolve_work_unit_effect(work_unit: WorkUnit) -> ToolEffect | None:
    """Infer the strongest effect proven by declared resource claims.

    READ is granted only when every declared claim is explicitly read-only. A
    write claim wins over reads, while no claims cannot prove a read effect and
    therefore remains conservative/serial. Command and network effects are not
    representable by a resource claim and must be supplied by a trusted adapter.
    """
    accesses = {claim.access for claim in work_unit.resource_claims}
    if ResourceAccess.WRITE in accesses:
        return ToolEffect.WRITE
    if accesses == {ResourceAccess.READ}:
        return ToolEffect.READ
    return None


def plan_coordination(
    work_units: Sequence[WorkUnit],
    *,
    effects: Mapping[WorkUnitId, ToolEffect | None],
    conflicts: Mapping[tuple[WorkUnitId, WorkUnitId], ConflictReport] | None = None,
    actual_changesets: Mapping[WorkUnitId, ChangeSet | ConflictFacts] | None = None,
    capacity: int = 4,
) -> CoordinationPlan:
    """Plan stable batches; no provider or workspace side effect occurs."""
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    ordered = tuple(sorted(work_units, key=lambda unit: unit.work_unit_id))
    by_id = {unit.work_unit_id: unit for unit in ordered}
    if len(by_id) != len(ordered):
        raise ValueError("coordination input contains duplicate work units")
    reports: dict[tuple[WorkUnitId, WorkUnitId], ConflictReport] = {
        _pair_key(left, right): report
        for (left, right), report in (conflicts or {}).items()
    }
    if actual_changesets is not None:
        for pair, derived_report in conflicts_from_changesets(
            ordered, actual_changesets
        ).items():
            existing = reports.get(pair)
            if existing is None or not existing.conflict:
                reports[pair] = derived_report
    drafts: list[_BatchDraft] = []

    for unit in ordered:
        effect = effects.get(unit.work_unit_id)
        if effect is ToolEffect.READ:
            mode = CoordinationMode.READ_ONLY
            default_reason = "read-only unit shares immutable snapshot"
        elif effect is ToolEffect.WRITE:
            mode = CoordinationMode.PARALLEL_WRITE
            default_reason = "write unit receives an isolated slot"
        else:
            mode = CoordinationMode.SERIAL
            default_reason = "effect is unknown or has external side effects"

        if mode is CoordinationMode.SERIAL:
            reason = default_reason
            for prior in ordered:
                if prior.work_unit_id == unit.work_unit_id:
                    break
                conflict_report = _report_for(unit, prior, reports)
                if conflict_report is not None and conflict_report.conflict:
                    reason = _reason_for_report(conflict_report, prior)
                    break
            drafts.append(_BatchDraft(mode, [unit.work_unit_id], [(unit.work_unit_id, reason)]))
            continue

        conflicting_prior: tuple[ConflictReport, WorkUnit] | None = None
        for draft in drafts:
            if draft.mode is not mode:
                continue
            for prior_id in draft.work_unit_ids:
                prior = by_id[prior_id]
                conflict_report = _report_for(unit, prior, reports)
                if conflict_report is not None and conflict_report.conflict:
                    conflicting_prior = conflict_report, prior
                    break
            if conflicting_prior is not None:
                break
        if conflicting_prior is not None:
            report, prior = conflicting_prior
            reason = _reason_for_report(report, prior)
            drafts.append(
                _BatchDraft(
                    CoordinationMode.SERIAL,
                    [unit.work_unit_id],
                    [(unit.work_unit_id, reason)],
                )
            )
            continue

        target = next(
            (
                draft
                for draft in drafts
                if draft.mode is mode and len(draft.work_unit_ids) < capacity
            ),
            None,
        )
        if target is None:
            drafts.append(
                _BatchDraft(mode, [unit.work_unit_id], [(unit.work_unit_id, default_reason)])
            )
        else:
            target.work_unit_ids.append(unit.work_unit_id)
            target.reasons.append((unit.work_unit_id, default_reason))

    batches = tuple(
        CoordinationBatch(
            mode=draft.mode,
            work_unit_ids=tuple(draft.work_unit_ids),
            shared_immutable_snapshot=draft.mode is CoordinationMode.READ_ONLY,
            requires_isolated_slots=draft.mode is not CoordinationMode.READ_ONLY,
            reasons=tuple(draft.reasons),
        )
        for draft in sorted(drafts, key=lambda item: item.work_unit_ids[0])
    )
    return CoordinationPlan(capacity=capacity, batches=batches)


class Coordinator:
    """Stateless facade for callers that prefer an object boundary."""

    def plan(
        self,
        work_units: Sequence[WorkUnit],
        *,
        effects: Mapping[WorkUnitId, ToolEffect | None],
        conflicts: Mapping[tuple[WorkUnitId, WorkUnitId], ConflictReport] | None = None,
        actual_changesets: Mapping[WorkUnitId, ChangeSet | ConflictFacts] | None = None,
        capacity: int = 4,
    ) -> CoordinationPlan:
        return plan_coordination(
            work_units,
            effects=effects,
            conflicts=conflicts,
            actual_changesets=actual_changesets,
            capacity=capacity,
        )

    def merge_candidate(
        self,
        base_root: Path,
        changes: Sequence[StagedChangeSet],
        *,
        staging_root: Path,
        verify: Callable[[Path], None] | None = None,
    ) -> MergeResult:
        """Create a verified temporary candidate without promoting it."""
        return merge_change_sets(
            base_root,
            changes,
            staging_root=staging_root,
            verify=verify,
        )
