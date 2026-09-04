"""Deterministic planning tests for isolated PLANNED batches."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from prp_runtime.domain.enums import ResourceAccess, ToolEffect
from prp_runtime.domain.models import WorkUnit
from prp_runtime.domain.values import ResourceClaim
from prp_runtime.runtime.conflicts import ConflictFact, ConflictKind, ConflictReport
from prp_runtime.runtime.coordinator import (
    CoordinationMode,
    Coordinator,
    MergeStatus,
    StagedChangeSet,
    merge_change_sets,
    plan_coordination,
    promote_merge,
    resolve_work_unit_effect,
)
from prp_runtime.workspace.merge import MergeError
from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
)

T0 = datetime(2026, 8, 15, tzinfo=UTC)


def unit(key: str, *, resource: str | None = None, write: bool = False) -> WorkUnit:
    claims = () if resource is None else (
        ResourceClaim(
            resource=resource,
            access=ResourceAccess.WRITE if write else ResourceAccess.READ,
        ),
    )
    return WorkUnit(
        work_unit_id=f"wu_{key}",
        run_id="run_coordinator",
        name=key,
        instruction=f"do {key}",
        resource_claims=claims,
        created_at=T0,
    )


def conflict(kind: ConflictKind, reason: str = "test conflict") -> ConflictReport:
    fact = ConflictFact(kind=kind, reason=reason)
    return ConflictReport(kind=kind, reason=reason, facts=(fact,))


def test_plan_groups_read_only_and_independent_writes_stably() -> None:
    units = (
        unit("write_b", resource="b", write=True),
        unit("read_b", resource="doc"),
        unit("read_a", resource="doc"),
        unit("write_a", resource="a", write=True),
    )
    effects = {
        "wu_read_a": ToolEffect.READ,
        "wu_read_b": ToolEffect.READ,
        "wu_write_a": ToolEffect.WRITE,
        "wu_write_b": ToolEffect.WRITE,
    }

    plan = plan_coordination(units, effects=effects, capacity=2)

    assert [batch.mode for batch in plan.batches] == [
        CoordinationMode.READ_ONLY,
        CoordinationMode.PARALLEL_WRITE,
    ]
    assert plan.batches[0].work_unit_ids == ("wu_read_a", "wu_read_b")
    assert plan.batches[1].work_unit_ids == ("wu_write_a", "wu_write_b")
    assert plan.batches[0].shared_immutable_snapshot is True
    assert plan.batches[1].requires_isolated_slots is True


def test_conflict_and_unknown_effects_are_serial_with_reasons() -> None:
    write_a = unit("a", resource="same", write=True)
    write_b = unit("b", resource="same", write=True)
    unknown = unit("unknown")
    effects = {
        "wu_a": ToolEffect.WRITE,
        "wu_b": ToolEffect.WRITE,
        "wu_unknown": None,
    }
    reports = {("wu_a", "wu_b"): conflict(ConflictKind.PATH, "same file")}

    plan = Coordinator().plan(
        (unknown, write_b, write_a), effects=effects, conflicts=reports, capacity=4
    )

    assert [batch.mode for batch in plan.batches] == [
        CoordinationMode.PARALLEL_WRITE,
        CoordinationMode.SERIAL,
        CoordinationMode.SERIAL,
    ]
    assert plan.batches[0].work_unit_ids == ("wu_a",)
    assert "PATH - same file" in plan.batches[1].reasons[0][1]
    assert "unknown or has external side effects" in plan.batches[2].reasons[0][1]


@pytest.mark.parametrize("effect", (ToolEffect.COMMAND, ToolEffect.NETWORK, None))
def test_external_or_unproven_effects_never_join_a_parallel_batch(
    effect: ToolEffect | None,
) -> None:
    units = (unit("b"), unit("a"))
    effects = {work_unit.work_unit_id: effect for work_unit in units}

    plan = plan_coordination(units, effects=effects, capacity=4)

    assert [batch.mode for batch in plan.batches] == [
        CoordinationMode.SERIAL,
        CoordinationMode.SERIAL,
    ]
    assert [batch.work_unit_ids for batch in plan.batches] == [
        ("wu_a",),
        ("wu_b",),
    ]


def test_work_unit_effect_requires_explicit_resource_proof() -> None:
    read = unit("read", resource="document")
    write = unit("write", resource="document", write=True)
    mixed = WorkUnit(
        work_unit_id="wu_mixed",
        run_id="run_coordinator",
        name="mixed",
        instruction="do mixed",
        resource_claims=(
            ResourceClaim(resource="document", access=ResourceAccess.READ),
            ResourceClaim(resource="output", access=ResourceAccess.WRITE),
        ),
        created_at=T0,
    )
    unknown = unit("unknown")

    assert resolve_work_unit_effect(read) is ToolEffect.READ
    assert resolve_work_unit_effect(write) is ToolEffect.WRITE
    assert resolve_work_unit_effect(mixed) is ToolEffect.WRITE
    assert resolve_work_unit_effect(unknown) is None


def test_capacity_splits_parallel_batches_without_duplicate_units() -> None:
    units = tuple(unit(str(index)) for index in range(5))
    effects = {unit.work_unit_id: ToolEffect.READ for unit in units}

    plan = plan_coordination(units, effects=effects, capacity=2)

    assert [batch.work_unit_ids for batch in plan.batches] == [
        ("wu_0", "wu_1"),
        ("wu_2", "wu_3"),
        ("wu_4",),
    ]
    flattened = [unit_id for batch in plan.batches for unit_id in batch.work_unit_ids]
    assert len(flattened) == len(set(flattened)) == 5


def test_actual_changesets_are_admission_conflicts_for_parallel_writes() -> None:
    left = unit("a", resource="left", write=True)
    right = unit("b", resource="right", write=True)
    actual = {
        left.work_unit_id: _change_set("a", "src/main.py", "old", "left"),
        right.work_unit_id: _change_set("b", "src/main.py", "old", "right"),
    }

    plan = plan_coordination(
        (right, left),
        effects={left.work_unit_id: ToolEffect.WRITE, right.work_unit_id: ToolEffect.WRITE},
        actual_changesets=actual,
        capacity=2,
    )

    assert plan.batches[0].mode is CoordinationMode.PARALLEL_WRITE
    assert plan.batches[0].work_unit_ids == (left.work_unit_id,)
    assert plan.batches[1].mode is CoordinationMode.SERIAL
    assert plan.batches[1].work_unit_ids == (right.work_unit_id,)
    assert "PATH" in plan.batches[1].reasons[0][1]


def _digest(content: str) -> FileContent:
    import hashlib

    return FileContent(sha256=hashlib.sha256(content.encode()).hexdigest(), size=len(content))


def _change_set(key: str, path: str, before: str, after: str) -> ChangeSet:
    return ChangeSet(
        change_set_id=f"cs_{key}",
        run_id="run_coordinator",
        tool_call_id=f"tc_{key}",
        workspace_id="ws_coordinator",
        base_snapshot_id="snap_" + "a" * 32,
        new_snapshot_id="snap_" + key + "b" * (31 - len(key)),
        patch=Patch(
            base_snapshot_id="snap_" + "a" * 32,
            unified_diff=f"--- a/{path}\n+++ b/{path}\n",
        ),
        files=(
            FileChange(
                path=path,
                action=FileChangeAction.MODIFY,
                before=_digest(before),
                after=_digest(after),
            ),
        ),
        created_at=T0,
    )


def _snapshot(base: Path, name: str) -> Path:
    root = base / name
    root.mkdir()
    return root


def test_temp_git_merge_verifies_and_promotes_only_staging(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.txt").write_text("a0")
    (base / "b.txt").write_text("b0")
    left_root = _snapshot(tmp_path, "left")
    right_root = _snapshot(tmp_path, "right")
    # Each isolated root is a complete snapshot, not a patch-only directory.
    for root in (left_root, right_root):
        (root / "a.txt").write_text("a0")
        (root / "b.txt").write_text("b0")
    (left_root / "a.txt").write_text("a1")
    (right_root / "b.txt").write_text("b1")
    result = Coordinator().merge_candidate(
        base,
        (
            StagedChangeSet(_change_set("right", "b.txt", "b0", "b1"), right_root),
            StagedChangeSet(_change_set("left", "a.txt", "a0", "a1"), left_root),
        ),
        staging_root=tmp_path / "staging",
        verify=lambda root: (
            None
            if (root / "a.txt").read_text() == "a1"
            and (root / "b.txt").read_text() == "b1"
            else (_ for _ in ()).throw(AssertionError("merged content mismatch"))
        ),
    )

    assert result.status is MergeStatus.MERGED
    assert result.verified is True
    assert result.applied_change_set_ids == ("cs_left", "cs_right")
    destination = tmp_path / "promoted"
    promoted = promote_merge(result, destination)
    assert promoted.promoted is True
    assert (destination / "a.txt").read_text() == "a1"
    assert (destination / "b.txt").read_text() == "b1"
    assert (base / "a.txt").read_text() == "a0"
    assert (base / "b.txt").read_text() == "b0"
    assert promote_merge(promoted, destination) == promoted


def test_merge_conflict_is_explicit_and_does_not_promote(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "same.txt").write_text("base")
    left_root = _snapshot(tmp_path, "left")
    right_root = _snapshot(tmp_path, "right")
    for root, content in ((left_root, "left"), (right_root, "right")):
        (root / "same.txt").write_text(content)
    result = merge_change_sets(
        base,
        (
            StagedChangeSet(_change_set("left", "same.txt", "base", "left"), left_root),
            StagedChangeSet(_change_set("right", "same.txt", "base", "right"), right_root),
        ),
        staging_root=tmp_path / "staging",
    )

    assert result.status is MergeStatus.CONFLICT
    assert result.conflict_report is not None
    assert result.conflict_report.kind is ConflictKind.PATH
    assert result.verified is False
    assert result.promoted is False
    assert (base / "same.txt").read_text() == "base"
    assert (left_root / "same.txt").read_text() == "left"
    assert (right_root / "same.txt").read_text() == "right"
    with pytest.raises(MergeError, match="verified merged result"):
        promote_merge(result, tmp_path / "promoted")
    assert not (tmp_path / "promoted").exists()


def test_unverified_merged_candidate_cannot_self_promote(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.txt").write_text("a0")
    left_root = _snapshot(tmp_path, "left")
    (left_root / "a.txt").write_text("a1")
    result = Coordinator().merge_candidate(
        base,
        (StagedChangeSet(_change_set("left", "a.txt", "a0", "a1"), left_root),),
        staging_root=tmp_path / "staging",
    )

    assert result.status is MergeStatus.MERGED
    assert result.verified is False
    assert result.promoted is False
    assert (base / "a.txt").read_text() == "a0"
    with pytest.raises(MergeError, match="verified merged result"):
        promote_merge(result, tmp_path / "promoted")
    assert not (tmp_path / "promoted").exists()
    assert (base / "a.txt").read_text() == "a0"
