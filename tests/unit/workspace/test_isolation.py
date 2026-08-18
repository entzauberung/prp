"""Isolated snapshot/slot lifecycle tests with temporary directories."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from prp_runtime.domain.values import new_run_id, new_tool_call_id
from prp_runtime.workspace.isolation import (
    IsolationCapacityError,
    IsolationError,
    IsolationOwnershipError,
    LocalIsolationBackend,
    SlotContext,
    SlotStatus,
)


def _source(root: Path, name: str = "source") -> Path:
    source = root / name
    source.mkdir()
    (source / "README.md").write_text("base\n", encoding="utf-8")
    return source


def test_slots_are_private_from_source_and_cleanup_is_idempotent(tmp_path: Path) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation")
    source = _source(tmp_path)
    snapshot = backend.create_base_snapshot(source, "ws_test")
    slot = backend.create_slot(snapshot.snapshot_id, "wu_test", "owner_a")
    slot_path = backend.slot_path(slot.slot_id, owner_id="owner_a")
    (slot_path / "README.md").write_text("slot\n", encoding="utf-8")
    (slot_path / "new.txt").write_text("new\n", encoding="utf-8")

    assert (source / "README.md").read_text(encoding="utf-8") == "base\n"
    assert slot.base_hash == snapshot.content_hash
    assert "isolation" not in slot.model_dump_json()
    assert backend.cleanup(slot.slot_id, owner_id="owner_a") is True
    assert backend.cleanup(slot.slot_id, owner_id="owner_a") is True
    assert backend.get_slot(slot.slot_id, owner_id="owner_a").status is SlotStatus.CLEANED


def test_snapshot_source_cannot_overlap_isolation_storage(tmp_path: Path) -> None:
    storage_root = tmp_path / "isolation-overlap"
    backend = LocalIsolationBackend(storage_root)
    source = storage_root / "source"
    source.mkdir()
    (source / "README.md").write_text("unsafe overlap\n", encoding="utf-8")

    with pytest.raises(IsolationError, match="outside isolation storage"):
        backend.create_base_snapshot(source, "ws_test")


def test_promote_creates_a_new_immutable_snapshot(tmp_path: Path) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation")
    snapshot = backend.create_base_snapshot(_source(tmp_path, "source-2"), "ws_test")
    slot = backend.create_slot(snapshot.snapshot_id, "wu_test", "owner_a")
    slot_path = backend.slot_path(slot.slot_id, owner_id="owner_a")
    (slot_path / "README.md").write_text("changed\n", encoding="utf-8")

    promoted = backend.promote(slot.slot_id, owner_id="owner_a")

    assert promoted.snapshot_id != snapshot.snapshot_id
    assert promoted.content_hash != snapshot.content_hash
    assert backend.get_slot(slot.slot_id, owner_id="owner_a").status is SlotStatus.PROMOTED
    backend.cleanup(slot.slot_id, owner_id="owner_a")


def test_promote_changes_records_relative_file_facts(tmp_path: Path) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation")
    source = _source(tmp_path)
    snapshot = backend.create_base_snapshot(source, "ws_test")
    slot = backend.create_slot(snapshot.snapshot_id, "wu_test", "owner_a")
    slot_path = backend.slot_path(slot.slot_id, owner_id="owner_a")
    (slot_path / "README.md").write_text("changed\n", encoding="utf-8")
    (slot_path / "new.txt").write_text("new\n", encoding="utf-8")

    change_set = backend.promote_changes(
        slot.slot_id,
        owner_id="owner_a",
        run_id=new_run_id(),
        tool_call_id=new_tool_call_id(),
        work_unit_id="wu_test",
    )

    assert change_set is not None
    assert change_set.base_snapshot_id == snapshot.snapshot_id
    assert {change.path for change in change_set.files} == {"README.md", "new.txt"}
    assert str(tmp_path) not in change_set.model_dump_json()


def test_invalid_changeset_lineage_does_not_promote_the_slot(tmp_path: Path) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation")
    source = _source(tmp_path, "source-lineage")
    snapshot = backend.create_base_snapshot(source, "ws_test")
    slot = backend.create_slot(snapshot.snapshot_id, "wu_lineage", "owner_a")
    backend.slot_path(slot.slot_id, owner_id="owner_a").joinpath("README.md").write_text(
        "changed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        backend.promote_changes(
            slot.slot_id,
            owner_id="owner_a",
            run_id="invalid",
            tool_call_id="invalid",
        )

    assert backend.get_slot(slot.slot_id, owner_id="owner_a").status is SlotStatus.ACTIVE
    backend.cleanup(slot.slot_id, owner_id="owner_a")


def test_capacity_and_owner_limits_fail_closed(tmp_path: Path) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation", max_slots=1, max_bytes=8)
    snapshot = backend.create_base_snapshot(_source(tmp_path, "source-2"), "ws_test")
    with pytest.raises(IsolationCapacityError):
        backend.create_slot(snapshot.snapshot_id, "wu_test", "owner_a")

    backend = LocalIsolationBackend(tmp_path / "isolation-2", max_slots=1)
    snapshot = backend.create_base_snapshot(_source(tmp_path), "ws_test")
    slot = backend.create_slot(snapshot.snapshot_id, "wu_test", "owner_a")
    with pytest.raises(IsolationOwnershipError):
        backend.slot_path(slot.slot_id, owner_id="owner_b")


def test_slot_context_binds_identity_and_cleans_in_a_finally_boundary(
    tmp_path: Path,
) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation-context")
    snapshot = backend.create_base_snapshot(_source(tmp_path, "source-context"), "ws_test")
    context = SlotContext(
        backend,
        snapshot_id=snapshot.snapshot_id,
        work_unit_id="wu_context",
        owner_id="owner_context",
    )

    with context:
        assert context.slot.base_snapshot_id == snapshot.snapshot_id
        assert context.slot.work_unit_id == "wu_context"
        assert context.path.is_dir()
        assert backend.active_slot_count == 1

    assert backend.active_slot_count == 0
    assert context.cleanup() is True
