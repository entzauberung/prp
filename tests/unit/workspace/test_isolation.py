"""Isolated snapshot/slot lifecycle tests with temporary directories."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import ExecutionLocation, ExecutionStrategy, IsolationMode
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.values import new_run_id, new_tool_call_id
from prp_runtime.workspace.isolation import (
    DEFAULT_ISOLATION_MAX_BYTES,
    DEFAULT_ISOLATION_MAX_SLOTS,
    ExecutionCopyMode,
    IsolationCapacityError,
    IsolationError,
    IsolationOwnershipError,
    LocalIsolationBackend,
    MAX_ISOLATION_MAX_BYTES,
    MAX_ISOLATION_MAX_SLOTS,
    SlotContext,
    SlotStatus,
    select_execution_copy_mode,
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
    with pytest.raises(IsolationCapacityError) as raised:
        backend.create_slot(snapshot.snapshot_id, "wu_test", "owner_a")
    assert raised.value.detail.code is ErrorCode.RESOURCE_BUDGET_EXCEEDED
    assert raised.value.detail.field in {"max_slots", "max_copied_bytes"}
    assert backend.active_slot_count == 0

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


def test_select_execution_copy_mode_keeps_copy_backed_parallel_paths() -> None:
    assert (
        select_execution_copy_mode(
            execution_location=ExecutionLocation.LOCAL,
            isolation_mode=IsolationMode.HOST,
            strategy=ExecutionStrategy.DIRECT,
            concurrency=1,
        )
        is ExecutionCopyMode.IN_PLACE
    )
    assert (
        select_execution_copy_mode(
            execution_location=ExecutionLocation.LOCAL,
            isolation_mode=IsolationMode.HOST,
            strategy=ExecutionStrategy.PLANNED,
            concurrency=1,
        )
        is ExecutionCopyMode.COPY_BACKED
    )
    assert (
        select_execution_copy_mode(
            execution_location=ExecutionLocation.LOCAL,
            isolation_mode=IsolationMode.HOST,
            strategy=ExecutionStrategy.DIRECT,
            concurrency=2,
        )
        is ExecutionCopyMode.COPY_BACKED
    )


def test_in_place_mode_cannot_create_copied_snapshots_or_slots(tmp_path: Path) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation-inplace")
    source = _source(tmp_path, "source-inplace")
    with pytest.raises(IsolationError, match="in-place"):
        backend.create_base_snapshot(
            source, "ws_test", copy_mode=ExecutionCopyMode.IN_PLACE
        )
    snapshot = backend.create_base_snapshot(source, "ws_test")
    with pytest.raises(IsolationError, match="in-place"):
        backend.create_slot(
            snapshot.snapshot_id,
            "wu_test",
            "owner_a",
            copy_mode=ExecutionCopyMode.IN_PLACE,
        )
    slot = backend.create_slot(snapshot.snapshot_id, "wu_test", "owner_a")
    assert slot.status is SlotStatus.ACTIVE
    backend.cleanup(slot.slot_id, owner_id="owner_a")


def test_default_isolation_capacity_is_bounded(tmp_path: Path) -> None:
    backend = LocalIsolationBackend(tmp_path / "isolation-defaults")
    assert backend.max_slots == DEFAULT_ISOLATION_MAX_SLOTS == 2
    assert backend.max_bytes == DEFAULT_ISOLATION_MAX_BYTES == 256 * 1024 * 1024
    assert DEFAULT_ISOLATION_MAX_SLOTS < 8
    assert DEFAULT_ISOLATION_MAX_BYTES < 10 * 1024 * 1024 * 1024


def test_invalid_isolation_capacity_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        LocalIsolationBackend(tmp_path / "bad-slots", max_slots=0)
    with pytest.raises(ValueError, match="positive"):
        LocalIsolationBackend(tmp_path / "bad-bytes", max_bytes=0)
    with pytest.raises(ValueError, match="bounded"):
        LocalIsolationBackend(
            tmp_path / "too-many-slots",
            max_slots=MAX_ISOLATION_MAX_SLOTS + 1,
        )
    with pytest.raises(ValueError, match="bounded"):
        LocalIsolationBackend(
            tmp_path / "too-many-bytes",
            max_bytes=MAX_ISOLATION_MAX_BYTES + 1,
        )


def test_default_slots_reject_a_third_copy_before_copytree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_calls: list[object] = []
    real_copytree = shutil.copytree

    def counting_copytree(*args: object, **kwargs: object) -> object:
        copy_calls.append((args, kwargs))
        return real_copytree(*args, **kwargs)

    monkeypatch.setattr(
        "prp_runtime.workspace.isolation.shutil.copytree",
        counting_copytree,
    )
    backend = LocalIsolationBackend(tmp_path / "isolation-slots")
    snapshot = backend.create_base_snapshot(_source(tmp_path, "source-slots"), "ws_test")
    first = backend.create_slot(snapshot.snapshot_id, "wu_one", "owner_a")
    second = backend.create_slot(snapshot.snapshot_id, "wu_two", "owner_a")
    copies_before_overflow = len(copy_calls)
    with pytest.raises(IsolationCapacityError, match="slot") as raised:
        backend.create_slot(snapshot.snapshot_id, "wu_three", "owner_a")
    assert raised.value.detail.field == "max_slots"
    assert raised.value.detail.code is ErrorCode.RESOURCE_BUDGET_EXCEEDED
    assert len(copy_calls) == copies_before_overflow
    assert backend.active_slot_count == 2
    first_path = backend.slot_path(first.slot_id, owner_id="owner_a")
    second_path = backend.slot_path(second.slot_id, owner_id="owner_a")
    (first_path / "README.md").write_text("first\n", encoding="utf-8")
    assert (second_path / "README.md").read_text(encoding="utf-8") == "base\n"
    assert (tmp_path / "source-slots" / "README.md").read_text(encoding="utf-8") == "base\n"
    assert backend.cleanup(first.slot_id, owner_id="owner_a") is True
    assert backend.active_slot_count == 1
    third = backend.create_slot(snapshot.snapshot_id, "wu_three", "owner_a")
    assert backend.active_slot_count == 2
    backend.cleanup(second.slot_id, owner_id="owner_a")
    backend.cleanup(third.slot_id, owner_id="owner_a")


def test_byte_overflow_and_cleanup_reclaim_capacity_before_copytree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_calls: list[object] = []
    real_copytree = shutil.copytree

    def counting_copytree(*args: object, **kwargs: object) -> object:
        copy_calls.append((args, kwargs))
        return real_copytree(*args, **kwargs)

    monkeypatch.setattr(
        "prp_runtime.workspace.isolation.shutil.copytree",
        counting_copytree,
    )
    source = _source(tmp_path, "source-bytes")
    measuring = LocalIsolationBackend(tmp_path / "measure")
    measured = measuring.create_base_snapshot(source, "ws_test")
    measuring.close()
    backend = LocalIsolationBackend(
        tmp_path / "isolation-bytes",
        max_slots=2,
        max_bytes=measured.total_bytes,
    )
    snapshot = backend.create_base_snapshot(source, "ws_test")
    used_after_snapshot = backend.used_bytes
    copies_before_slot = len(copy_calls)
    with pytest.raises(IsolationCapacityError, match="byte") as raised:
        backend.create_slot(snapshot.snapshot_id, "wu_bytes", "owner_a")
    assert raised.value.detail.field == "max_copied_bytes"
    assert raised.value.detail.code is ErrorCode.RESOURCE_BUDGET_EXCEEDED
    assert len(copy_calls) == copies_before_slot
    assert backend.active_slot_count == 0
    assert backend.used_bytes == used_after_snapshot

    roomy = LocalIsolationBackend(
        tmp_path / "isolation-reclaim",
        max_slots=1,
        max_bytes=measured.total_bytes * 3,
    )
    snapshot = roomy.create_base_snapshot(source, "ws_test")
    used_after_snapshot = roomy.used_bytes
    slot = roomy.create_slot(snapshot.snapshot_id, "wu_reclaim", "owner_a")
    used_with_slot = roomy.used_bytes
    assert used_with_slot == used_after_snapshot + snapshot.total_bytes
    assert roomy.active_slot_count == 1
    assert roomy.cleanup(slot.slot_id, owner_id="owner_a") is True
    assert roomy.cleanup(slot.slot_id, owner_id="owner_a") is True
    assert roomy.active_slot_count == 0
    assert roomy.used_bytes == used_after_snapshot
    replacement = roomy.create_slot(snapshot.snapshot_id, "wu_later", "owner_a")
    assert replacement.status is SlotStatus.ACTIVE
    assert roomy.active_slot_count == 1
    roomy.cleanup(replacement.slot_id, owner_id="owner_a")
