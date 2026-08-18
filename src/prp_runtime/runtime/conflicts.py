"""Deterministic conflict facts for ChangeSets and resource claims."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum, unique
from typing import Self

from pydantic import Field, field_validator

from prp_runtime.analysis.syntax import SymbolChangeAction, SymbolKind, SyntaxReport
from prp_runtime.domain.enums import ResourceAccess
from prp_runtime.domain.models import DomainModel, WorkUnit
from prp_runtime.domain.values import ResourceClaim, WorkUnitId
from prp_runtime.workspace.changes import ChangeSet, FileChange

__all__ = [
    "ConflictFact",
    "ConflictFacts",
    "ConflictKind",
    "ConflictReport",
    "classify_change_sets",
    "classify_conflict",
    "classify_facts",
    "classify_file_changes",
    "classify_syntax_conflict",
    "conflicts_from_changesets",
    "facts_from_claims",
    "normalize_relative_path",
]

@unique
class ConflictKind(StrEnum):
    """Stable categories used by the admission layer."""

    NO_CONFLICT = "NO_CONFLICT"
    PATH = "PATH"
    READ_WRITE = "READ_WRITE"
    BASE = "BASE"
    STRUCTURAL = "STRUCTURAL"
    UNKNOWN = "UNKNOWN"


def normalize_relative_path(path: str) -> str:
    """Return a canonical relative POSIX path or reject an unsafe path."""
    if (
        not isinstance(path, str)
        or not path.strip()
        or path.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", path)
        or "\\" in path
    ):
        raise ValueError("conflict path must be relative POSIX syntax")
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("conflict path must not contain parent segments")
        parts.append(part)
    if not parts:
        raise ValueError("conflict path must not be empty")
    return "/".join(parts)


def _canonical_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({normalize_relative_path(path) for path in paths}))


def _overlap(left: Iterable[str], right: Iterable[str]) -> tuple[str, ...]:
    """Return stable file or directory path overlaps."""
    left_paths = _canonical_paths(left)
    right_paths = _canonical_paths(right)
    overlaps = {
        candidate
        for candidate in left_paths
        for other in right_paths
        if (
            candidate == other
            or candidate.startswith(f"{other}/")
            or other.startswith(f"{candidate}/")
        )
        for candidate in (candidate, other)
    }
    return tuple(sorted(overlaps))


def _resource_overlap(left: Iterable[str], right: Iterable[str]) -> tuple[str, ...]:
    """Return exact logical-resource overlaps without imposing path rules."""
    return tuple(sorted(set(left) & set(right)))


class ConflictFacts(DomainModel):
    """Normalized, path/resource/base facts used by the classifier."""

    changed_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    read_resources: tuple[str, ...] = ()
    write_resources: tuple[str, ...] = ()
    base_snapshot_id: str | None = None
    base_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    unknown: bool = False

    @field_validator("changed_paths", "read_paths", "write_paths")
    @classmethod
    def _canonicalize_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_paths(value)

    @field_validator("read_resources", "write_resources")
    @classmethod
    def _canonicalize_resources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not resource.strip() for resource in value):
            raise ValueError("conflict resources must not be blank")
        return tuple(sorted(set(value)))

    @classmethod
    def from_changeset(
        cls,
        change_set: ChangeSet,
        *,
        claims: Iterable[ResourceClaim] = (),
        base_hash: str | None = None,
        unknown: bool = False,
    ) -> Self:
        """Build facts from one immutable ChangeSet and its declared claims."""
        read_resources = tuple(
            claim.resource for claim in claims if claim.access is ResourceAccess.READ
        )
        write_resources = tuple(
            claim.resource for claim in claims if claim.access is ResourceAccess.WRITE
        )
        changed_paths = tuple(change.path for change in change_set.files)
        return cls(
            changed_paths=changed_paths,
            write_paths=changed_paths,
            read_resources=read_resources,
            write_resources=write_resources,
            base_snapshot_id=change_set.base_snapshot_id,
            base_hash=base_hash,
            unknown=unknown,
        )


def facts_from_claims(
    claims: Iterable[ResourceClaim], *, unknown: bool = False
) -> ConflictFacts:
    """Build planner-only facts; callers can mark missing runtime facts unknown."""
    read_resources = tuple(
        claim.resource for claim in claims if claim.access is ResourceAccess.READ
    )
    write_resources = tuple(
        claim.resource for claim in claims if claim.access is ResourceAccess.WRITE
    )
    return ConflictFacts(
        read_resources=read_resources,
        write_resources=write_resources,
        unknown=unknown,
    )


class ConflictFact(DomainModel):
    """One explainable conflict observation."""

    kind: ConflictKind
    reason: str = Field(min_length=1, max_length=512)
    paths: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()

    @field_validator("paths", "resources", "symbols")
    @classmethod
    def _sort_fact_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))


class ConflictReport(DomainModel):
    """Stable, auditable result of comparing two sets of conflict facts."""

    kind: ConflictKind
    reason: str = Field(min_length=1, max_length=512)
    facts: tuple[ConflictFact, ...]

    @property
    def conflict(self) -> bool:
        """Whether admission must treat the pair as conflicting."""
        return self.kind is not ConflictKind.NO_CONFLICT

    @property
    def is_conflict(self) -> bool:
        """Readable alias for callers that use predicate-style naming."""
        return self.conflict

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return all stable reasons, including the primary reason first."""
        return tuple(fact.reason for fact in self.facts)

    @property
    def classification(self) -> ConflictKind:
        """Readable alias for the primary category."""
        return self.kind


_FACT_ORDER = {
    ConflictKind.BASE: 0,
    ConflictKind.PATH: 1,
    ConflictKind.READ_WRITE: 2,
    ConflictKind.STRUCTURAL: 3,
    ConflictKind.UNKNOWN: 4,
    ConflictKind.NO_CONFLICT: 5,
}


def _base_fact(left: ConflictFacts, right: ConflictFacts) -> ConflictFact | None:
    if left.base_snapshot_id is None or right.base_snapshot_id is None:
        return ConflictFact(
            kind=ConflictKind.UNKNOWN,
            reason="base snapshot facts are incomplete",
        )
    if left.base_snapshot_id != right.base_snapshot_id:
        return ConflictFact(
            kind=ConflictKind.BASE,
            reason="base snapshot ids differ",
        )
    if (left.base_hash is None) != (right.base_hash is None):
        return ConflictFact(
            kind=ConflictKind.UNKNOWN,
            reason="base snapshot hashes are incomplete",
        )
    if left.base_hash is not None and left.base_hash != right.base_hash:
        return ConflictFact(
            kind=ConflictKind.BASE,
            reason="base snapshot hashes differ",
        )
    return None


def _path_facts(left: ConflictFacts, right: ConflictFacts) -> tuple[ConflictFact, ...]:
    left_writes = (*left.changed_paths, *left.write_paths)
    right_writes = (*right.changed_paths, *right.write_paths)
    write_overlap = _overlap(left_writes, right_writes)
    read_write_overlap = tuple(
        sorted(
            set(_overlap(left_writes, right.read_paths))
            | set(_overlap(right_writes, left.read_paths))
        )
    )
    facts: list[ConflictFact] = []
    if write_overlap:
        facts.append(
            ConflictFact(
                kind=ConflictKind.PATH,
                reason="write paths overlap",
                paths=write_overlap,
            )
        )
    if read_write_overlap:
        facts.append(
            ConflictFact(
                kind=ConflictKind.READ_WRITE,
                reason="read and write paths overlap",
                paths=read_write_overlap,
            )
        )
    return tuple(facts)


def _resource_facts(left: ConflictFacts, right: ConflictFacts) -> tuple[ConflictFact, ...]:
    overlap = tuple(
        sorted(
            set(_resource_overlap(left.read_resources, right.write_resources))
            | set(_resource_overlap(left.write_resources, right.read_resources))
            | set(_resource_overlap(left.write_resources, right.write_resources))
        )
    )
    if not overlap:
        return ()
    return (
        ConflictFact(
            kind=ConflictKind.READ_WRITE,
            reason="resource access overlaps",
            resources=overlap,
        ),
    )


def _single_report(fact: ConflictFact) -> ConflictReport:
    return ConflictReport(kind=fact.kind, reason=fact.reason, facts=(fact,))


def _changed_symbols(report: SyntaxReport) -> tuple[str, ...]:
    return tuple(
        sorted(
            change.key
            for change in report.changes
            if change.action
            in (
                SymbolChangeAction.ADD,
                SymbolChangeAction.MODIFY,
                SymbolChangeAction.DELETE,
            )
        )
    )


def _module_or_global_symbols(report: SyntaxReport) -> tuple[str, ...]:
    return tuple(
        sorted(
            change.key
            for change in report.changes
            if "." not in change.key.split(":", maxsplit=1)[1]
            and change.after is not None
            and change.after.kind in (SymbolKind.IMPORT, SymbolKind.ASSIGNMENT)
            or "." not in change.key.split(":", maxsplit=1)[1]
            and change.before is not None
            and change.before.kind in (SymbolKind.IMPORT, SymbolKind.ASSIGNMENT)
        )
    )


def classify_syntax_conflict(
    left_change: FileChange,
    right_change: FileChange,
    *,
    left_report: SyntaxReport | None = None,
    right_report: SyntaxReport | None = None,
) -> ConflictReport:
    """Classify same-file Python symbol overlap with conservative fallback."""
    paths = _overlap((left_change.path,), (right_change.path,))
    if not paths:
        return _single_report(
            ConflictFact(
                kind=ConflictKind.NO_CONFLICT,
                reason="file paths do not overlap",
            )
        )
    if left_report is None or right_report is None:
        return _single_report(
            ConflictFact(
                kind=ConflictKind.PATH,
                reason="syntax facts unavailable; same file is serialized",
                paths=paths,
            )
        )
    if left_report.language != "python" or right_report.language != "python":
        return _single_report(
            ConflictFact(
                kind=ConflictKind.PATH,
                reason="non-Python syntax uses file-level serialization",
                paths=paths,
            )
        )
    if (
        left_report.unknown
        or right_report.unknown
        or not left_report.parse_ok
        or not right_report.parse_ok
    ):
        return _single_report(
            ConflictFact(
                kind=ConflictKind.UNKNOWN,
                reason="Python parse facts are unknown; same file is serialized",
                paths=paths,
            )
        )

    left_symbols = _changed_symbols(left_report)
    right_symbols = _changed_symbols(right_report)
    global_symbols = tuple(
        sorted(
            set(_module_or_global_symbols(left_report))
            | set(_module_or_global_symbols(right_report))
        )
    )
    if global_symbols:
        return _single_report(
            ConflictFact(
                kind=ConflictKind.STRUCTURAL,
                reason="module or global symbols changed; same file is serialized",
                paths=paths,
                symbols=global_symbols,
            )
        )
    overlap = tuple(sorted(set(left_symbols) & set(right_symbols)))
    if overlap:
        return _single_report(
            ConflictFact(
                kind=ConflictKind.STRUCTURAL,
                reason="same Python symbols changed",
                paths=paths,
                symbols=overlap,
            )
        )
    return _single_report(
        ConflictFact(
            kind=ConflictKind.NO_CONFLICT,
            reason=(
                "Python symbol changes do not overlap; "
                "semantic compatibility remains unverified"
            ),
            paths=paths,
        )
    )


classify_file_changes = classify_syntax_conflict


def classify_facts(left: ConflictFacts, right: ConflictFacts) -> ConflictReport:
    """Compare normalized facts with conservative unknown handling."""
    facts: list[ConflictFact] = []
    base_fact = _base_fact(left, right)
    if base_fact is not None:
        facts.append(base_fact)
    facts.extend(_path_facts(left, right))
    facts.extend(_resource_facts(left, right))
    if left.unknown or right.unknown:
        facts.append(
            ConflictFact(
                kind=ConflictKind.UNKNOWN,
                reason="change facts are marked unknown",
            )
        )
    if not facts:
        facts.append(
            ConflictFact(
                kind=ConflictKind.NO_CONFLICT,
                reason="no overlapping path, resource, or base facts",
            )
        )
    ordered = tuple(
        sorted(
            facts,
            key=lambda fact: (
                _FACT_ORDER[fact.kind],
                fact.kind.value,
                fact.paths,
                fact.resources,
                fact.reason,
            ),
        )
    )
    primary = ordered[0]
    return ConflictReport(kind=primary.kind, reason=primary.reason, facts=ordered)


def classify_conflict(
    left: ChangeSet | ConflictFacts,
    right: ChangeSet | ConflictFacts,
    *,
    left_claims: Iterable[ResourceClaim] = (),
    right_claims: Iterable[ResourceClaim] = (),
    left_base_hash: str | None = None,
    right_base_hash: str | None = None,
) -> ConflictReport:
    """Classify two ChangeSets or already-normalized fact collections."""
    left_facts = (
        left
        if isinstance(left, ConflictFacts)
        else ConflictFacts.from_changeset(
            left,
            claims=left_claims,
            base_hash=left_base_hash,
        )
    )
    right_facts = (
        right
        if isinstance(right, ConflictFacts)
        else ConflictFacts.from_changeset(
            right,
            claims=right_claims,
            base_hash=right_base_hash,
        )
    )
    return classify_facts(left_facts, right_facts)


def _facts_for_work_unit(
    work_unit: WorkUnit,
    actual: ChangeSet | ConflictFacts | None,
) -> ConflictFacts:
    """Combine trusted runtime facts with the unit's declared claims."""
    claims = work_unit.resource_claims
    if actual is None:
        return facts_from_claims(claims, unknown=True)
    if isinstance(actual, ChangeSet):
        return ConflictFacts.from_changeset(actual, claims=claims)
    claim_facts = facts_from_claims(claims)
    return ConflictFacts(
        changed_paths=actual.changed_paths,
        read_paths=actual.read_paths,
        write_paths=actual.write_paths,
        read_resources=(*actual.read_resources, *claim_facts.read_resources),
        write_resources=(*actual.write_resources, *claim_facts.write_resources),
        base_snapshot_id=actual.base_snapshot_id,
        base_hash=actual.base_hash,
        unknown=actual.unknown,
    )


def conflicts_from_changesets(
    work_units: Sequence[WorkUnit],
    actual_changesets: Mapping[WorkUnitId, ChangeSet | ConflictFacts],
) -> dict[tuple[WorkUnitId, WorkUnitId], ConflictReport]:
    """Build stable pair reports from runtime ChangeSets and claims.

    A missing runtime fact is represented as unknown, so an admission caller
    cannot accidentally treat an unobserved write as safe. Pair keys and input
    traversal are canonicalized by work-unit id for order-independent output.
    """
    ordered = tuple(sorted(work_units, key=lambda unit: unit.work_unit_id))
    reports: dict[tuple[WorkUnitId, WorkUnitId], ConflictReport] = {}
    for index, left in enumerate(ordered):
        left_id = left.work_unit_id
        left_facts = _facts_for_work_unit(left, actual_changesets.get(left_id))
        for right in ordered[index + 1 :]:
            right_id = right.work_unit_id
            right_facts = _facts_for_work_unit(right, actual_changesets.get(right_id))
            reports[(left_id, right_id)] = classify_facts(left_facts, right_facts)
    return reports


classify_change_sets = classify_conflict
