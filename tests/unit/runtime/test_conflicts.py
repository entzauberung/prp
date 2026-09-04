"""Matrix tests for deterministic ChangeSet conflict facts."""

from datetime import UTC, datetime

from prp_runtime.domain.enums import ResourceAccess
from prp_runtime.domain.models import WorkUnit
from prp_runtime.domain.values import ResourceClaim
from prp_runtime.runtime.conflicts import (
    ConflictFacts,
    ConflictKind,
    ConflictReport,
    classify_conflict,
    classify_facts,
    conflicts_from_changesets,
)
from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
)

T0 = datetime(2026, 8, 14, tzinfo=UTC)
BASE = "snap_" + "a" * 32
NEW = "snap_" + "b" * 32
BASE_HASH = "c" * 64


def change_set(path: str, *, base: str = BASE) -> ChangeSet:
    content = FileContent(sha256="d" * 64, size=1)
    return ChangeSet(
        change_set_id="cs_" + "e" * 32,
        run_id="run_" + "f" * 32,
        tool_call_id="tc_" + "g" * 32,
        workspace_id="ws_" + "h" * 32,
        base_snapshot_id=base,
        new_snapshot_id=NEW,
        patch=Patch(base_snapshot_id=base, unified_diff="--- a/x\n+++ b/x\n"),
        files=(
            FileChange(path=path, action=FileChangeAction.MODIFY, before=content, after=content),
        ),
        created_at=T0,
    )


def test_path_conflicts_are_normalized_and_symmetric() -> None:
    left = ConflictFacts(
        changed_paths=("src//main.py",),
        base_snapshot_id=BASE,
        base_hash=BASE_HASH,
    )
    right = ConflictFacts(
        changed_paths=("src/main.py",),
        base_snapshot_id=BASE,
        base_hash=BASE_HASH,
    )

    result = classify_facts(left, right)
    reverse = classify_facts(right, left)

    assert result.kind is ConflictKind.PATH
    assert result.conflict is True
    assert result.reasons == ("write paths overlap",)
    assert result == reverse


def test_read_read_is_safe_but_read_write_is_conflict() -> None:
    read_a = ConflictFacts(read_paths=("docs/guide.md",), base_snapshot_id=BASE)
    read_b = ConflictFacts(read_paths=("docs/guide.md",), base_snapshot_id=BASE)
    write = ConflictFacts(write_paths=("docs/guide.md",), base_snapshot_id=BASE)

    assert classify_facts(read_a, read_b).kind is ConflictKind.NO_CONFLICT
    assert classify_facts(read_a, write).kind is ConflictKind.READ_WRITE


def test_parent_child_and_add_delete_paths_are_conservative_conflicts() -> None:
    parent = ConflictFacts(write_paths=("src",), base_snapshot_id=BASE)
    child = ConflictFacts(changed_paths=("src/main.py",), base_snapshot_id=BASE)

    parent_child = classify_facts(parent, child)
    assert parent_child.kind is ConflictKind.PATH
    assert parent_child.facts[0].paths == ("src", "src/main.py")

    added = ChangeSet(
        change_set_id="cs_" + "i" * 32,
        run_id="run_" + "f" * 32,
        tool_call_id="tc_" + "j" * 32,
        workspace_id="ws_" + "h" * 32,
        base_snapshot_id=BASE,
        new_snapshot_id="snap_" + "k" * 32,
        patch=Patch(base_snapshot_id=BASE, unified_diff="--- /dev/null\n+++ b/src/main.py\n"),
        files=(
            FileChange(
                path="src/main.py",
                action=FileChangeAction.ADD,
                after=FileContent(sha256="d" * 64, size=1),
            ),
        ),
        created_at=T0,
    )
    deleted = change_set("src/main.py")
    assert classify_conflict(added, deleted).kind is ConflictKind.PATH


def test_base_hash_and_resource_conflicts_have_stable_reasons() -> None:
    left = change_set("src/main.py")
    right = change_set("src/other.py", base="snap_" + "i" * 32)
    claims = (ResourceClaim(resource="database", access=ResourceAccess.WRITE),)

    result = classify_conflict(
        left,
        right,
        left_claims=claims,
        right_claims=claims,
        left_base_hash=BASE_HASH,
        right_base_hash="1" * 64,
    )

    assert result.kind is ConflictKind.BASE
    assert result.reason == "base snapshot ids differ"
    assert [fact.kind for fact in result.facts] == [
        ConflictKind.BASE,
        ConflictKind.READ_WRITE,
    ]


def test_unknown_facts_block_conservatively_and_disjoint_writes_pass() -> None:
    unknown = ConflictFacts(base_snapshot_id=BASE, unknown=True)
    disjoint_left = ConflictFacts(write_paths=("a.txt",), base_snapshot_id=BASE)
    disjoint_right = ConflictFacts(write_paths=("b.txt",), base_snapshot_id=BASE)

    assert classify_facts(unknown, disjoint_right).kind is ConflictKind.UNKNOWN
    assert classify_facts(disjoint_left, disjoint_right).kind is ConflictKind.NO_CONFLICT


def test_changeset_adapter_uses_changed_paths_as_writes() -> None:
    report = classify_conflict(
        change_set("src/main.py"),
        change_set("src/main.py"),
        left_base_hash=BASE_HASH,
        right_base_hash=BASE_HASH,
    )

    assert report.kind is ConflictKind.PATH
    assert report.facts[0].paths == ("src/main.py",)


def test_changesets_and_claims_build_order_independent_admission_reports() -> None:
    left = WorkUnit(
        work_unit_id="wu_left",
        run_id="run_conflicts",
        name="left",
        instruction="left",
        resource_claims=(ResourceClaim(resource="shared", access=ResourceAccess.WRITE),),
        created_at=T0,
    )
    right = WorkUnit(
        work_unit_id="wu_right",
        run_id="run_conflicts",
        name="right",
        instruction="right",
        resource_claims=(ResourceClaim(resource="shared", access=ResourceAccess.WRITE),),
        created_at=T0,
    )

    forward = conflicts_from_changesets(
        (right, left),
        {
            left.work_unit_id: change_set("src/main.py"),
            right.work_unit_id: change_set("src/main.py"),
        },
    )
    reverse = conflicts_from_changesets(
        (left, right),
        {
            left.work_unit_id: change_set("src/main.py"),
            right.work_unit_id: change_set("src/main.py"),
        },
    )

    assert forward == reverse
    assert forward[("wu_left", "wu_right")].kind is ConflictKind.PATH
    assert ConflictKind.READ_WRITE in {
        fact.kind for fact in forward[("wu_left", "wu_right")].facts
    }


def test_missing_changeset_is_unknown_and_blocks_admission() -> None:
    left = WorkUnit(
        work_unit_id="wu_left",
        run_id="run_conflicts",
        name="left",
        instruction="left",
        created_at=T0,
    )
    right = WorkUnit(
        work_unit_id="wu_right",
        run_id="run_conflicts",
        name="right",
        instruction="right",
        created_at=T0,
    )

    reports = conflicts_from_changesets(
        (left, right),
        {left.work_unit_id: change_set("src/main.py")},
    )

    assert reports[("wu_left", "wu_right")].kind is ConflictKind.UNKNOWN


def test_parallel_fake_client_changesets_are_explicit_and_durable() -> None:
    left = change_set("src/main.py")
    overlapping = change_set("src/main.py")
    disjoint = change_set("src/other.py")

    conflict = classify_conflict(left, overlapping, left_base_hash=BASE_HASH, right_base_hash=BASE_HASH)
    compatible = classify_conflict(left, disjoint, left_base_hash=BASE_HASH, right_base_hash=BASE_HASH)

    assert conflict.conflict is True
    assert conflict.kind is ConflictKind.PATH
    assert compatible.kind is ConflictKind.NO_CONFLICT
    restored = ConflictReport.model_validate_json(conflict.model_dump_json())
    assert restored == conflict
    assert restored.conflict is True

