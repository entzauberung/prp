"""Targeted tests for atomic, base-rooted unified patch application."""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prp_runtime.domain.enums import AgentMode, ToolCallStatus, ToolEffect
from prp_runtime.domain.values import (
    new_run_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
)
from prp_runtime.policy.engine import PolicyOutcome, PolicyReasonCode, decide_tool_call
from prp_runtime.tools.models import ToolCall
from prp_runtime.tools.patch import (
    PatchRequest,
    PatchRunner,
    PatchStaleError,
    PatchValidationError,
    build_patch_definition,
    materialize_patched_contents,
)
from prp_runtime.workspace import WorkspaceBackend, WorkspaceBackendError
from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
)
from prp_runtime.workspace.models import Snapshot, SnapshotManifest, SnapshotStatus

T0 = datetime(2026, 8, 14, tzinfo=UTC)
WORKSPACE_ID = "ws_" + "a" * 32
OWNER_ID = "owner-1"


class RecordingStore:
    """Small transactional Store double for patch promotion tests."""

    def __init__(self, *, fail_change_set: bool = False) -> None:
        self.snapshots: list[Snapshot] = []
        self.change_sets: list[ChangeSet] = []
        self.fail_change_set = fail_change_set

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[object]:
        snapshots = list(self.snapshots)
        change_sets = list(self.change_sets)
        try:
            yield object()
        except BaseException:
            self.snapshots = snapshots
            self.change_sets = change_sets
            raise

    async def create_snapshot(
        self,
        snapshot: Snapshot,
        manifest: SnapshotManifest,
        *,
        owner_id: str,
        file_contents: object | None = None,
    ) -> Snapshot:
        assert owner_id == OWNER_ID
        assert snapshot.file_count == len(manifest.entries)
        del file_contents
        self.snapshots.append(snapshot)
        return snapshot

    async def create_change_set(self, change_set: ChangeSet) -> ChangeSet:
        if self.fail_change_set:
            raise RuntimeError("store failed")
        self.change_sets.append(change_set)
        return change_set

    async def list_change_sets(self, *, tool_call_id: str) -> tuple[ChangeSet, ...]:
        return tuple(
            change_set
            for change_set in self.change_sets
            if change_set.tool_call_id == tool_call_id
        )


def make_call(base_snapshot_id: str) -> ToolCall:
    return ToolCall(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        tool_name="apply_patch",
        effect=ToolEffect.WRITE,
        arguments={},
        status=ToolCallStatus.RUNNING,
        snapshot_id=base_snapshot_id,
        requested_at=T0,
    )


def make_base_snapshot(manifest: SnapshotManifest) -> Snapshot:
    return Snapshot(
        snapshot_id=new_snapshot_id(),
        workspace_id=WORKSPACE_ID,
        status=SnapshotStatus.READY,
        created_at=T0,
        completed_at=T0,
        file_count=len(manifest.entries),
        total_size=manifest.total_size,
    )


def modification(base_snapshot_id: str, path: str = "src/main.py") -> PatchRequest:
    return PatchRequest(
        patch=Patch(
            base_snapshot_id=base_snapshot_id,
            unified_diff=(
                f"--- a/{path}\n"
                f"+++ b/{path}\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            ),
        )
    )


@pytest.mark.asyncio
async def test_patch_applies_in_staging_and_records_snapshot_and_change_set(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = source / "main.py"
    target.write_text("old\n", encoding="utf-8")

    with WorkspaceBackend(tmp_path) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        store = RecordingStore()
        result = await PatchRunner(
            backend,
            store,
            owner_id=OWNER_ID,
            base_snapshot=base,
            base_manifest=manifest,
        ).apply(make_call(base.snapshot_id), modification(base.snapshot_id))

        assert target.read_text(encoding="utf-8") == "new\n"
        assert backend.snapshot_manifest().manifest_hash == result.manifest_hash

    assert len(store.snapshots) == 1
    assert len(store.change_sets) == 1
    assert store.change_sets[0].base_snapshot_id == base.snapshot_id
    assert store.change_sets[0].new_snapshot_id == result.new_snapshot_id
    assert store.change_sets[0].files[0].path == "src/main.py"
    assert store.change_sets[0].files[0].after is not None
    assert store.change_sets[0].files[0].after.sha256 == hashlib.sha256(b"new\n").hexdigest()
    assert store.change_sets[0].files[0].after.size == 4
    assert not tuple(tmp_path.glob(".prp-patch-*"))


@pytest.mark.asyncio
async def test_retry_replays_the_persisted_change_set_without_reapplying_the_patch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "main.py"
    target.write_text("old\n", encoding="utf-8")

    with WorkspaceBackend(tmp_path) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        store = RecordingStore()
        runner = PatchRunner(
            backend,
            store,
            owner_id=OWNER_ID,
            base_snapshot=base,
            base_manifest=manifest,
        )
        call = make_call(base.snapshot_id)
        first = await runner.apply(call, modification(base.snapshot_id, "main.py"))
        replay = await runner.apply(call, modification(base.snapshot_id, "main.py"))

        assert replay == first
        assert target.read_text(encoding="utf-8") == "new\n"
        assert len(store.snapshots) == 1
        assert len(store.change_sets) == 1


@pytest.mark.asyncio
async def test_delete_is_recorded_and_rename_or_duplicate_paths_are_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = source / "main.py"
    target.write_text("old\n", encoding="utf-8")

    with WorkspaceBackend(tmp_path) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        store = RecordingStore()
        delete = PatchRequest(
            patch=Patch(
                base_snapshot_id=base.snapshot_id,
                unified_diff=(
                    "--- a/src/main.py\n"
                    "+++ /dev/null\n"
                    "@@ -1 +1,0 @@\n"
                    "-old\n"
                ),
            )
        )
        result = await PatchRunner(
            backend,
            store,
            owner_id=OWNER_ID,
            base_snapshot=base,
            base_manifest=manifest,
        ).apply(make_call(base.snapshot_id), delete)
        assert not target.exists()
        assert store.change_sets[0].files[0].action.value == "DELETE"
        assert store.change_sets[0].files[0].after is None
        assert result.changed_paths == ("src/main.py",)

    with WorkspaceBackend(tmp_path) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        runner = PatchRunner(
            backend,
            RecordingStore(),
            owner_id=OWNER_ID,
            base_snapshot=base,
            base_manifest=manifest,
        )
        rename = PatchRequest(
            patch=Patch(
                base_snapshot_id=base.snapshot_id,
                unified_diff=(
                    "--- a/src/other.py\n"
                    "+++ b/src/main.py\n"
                    "@@ -0,0 +1 @@\n"
                    "+new\n"
                ),
            )
        )
        duplicate = PatchRequest(
            patch=Patch(
                base_snapshot_id=base.snapshot_id,
                unified_diff=(
                    "--- a/src/other.py\n"
                    "+++ b/src/other.py\n"
                    "@@ -0,0 +1 @@\n"
                    "+new\n"
                    "--- a/src/other.py\n"
                    "+++ b/src/other.py\n"
                    "@@ -0,0 +1 @@\n"
                    "+again\n"
                ),
            )
        )
        with pytest.raises(PatchValidationError, match="rename"):
            await runner.apply(make_call(base.snapshot_id), rename)
        with pytest.raises(PatchValidationError, match="duplicate"):
            await runner.apply(make_call(base.snapshot_id), duplicate)


@pytest.mark.asyncio
async def test_store_failure_rolls_back_workspace_and_snapshot_facts(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = source / "main.py"
    target.write_text("old\n", encoding="utf-8")

    with WorkspaceBackend(tmp_path) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        store = RecordingStore(fail_change_set=True)
        runner = PatchRunner(
            backend,
            store,
            owner_id=OWNER_ID,
            base_snapshot=base,
            base_manifest=manifest,
        )
        with pytest.raises(RuntimeError, match="store failed"):
            await runner.apply(make_call(base.snapshot_id), modification(base.snapshot_id))
        assert target.read_text(encoding="utf-8") == "old\n"

    assert store.snapshots == []
    assert store.change_sets == []
    assert not tuple(tmp_path.glob(".prp-patch-*"))


@pytest.mark.asyncio
async def test_stale_base_escape_and_binary_patch_inputs_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = source / "main.py"
    target.write_text("old\n", encoding="utf-8")

    with WorkspaceBackend(tmp_path) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        runner = PatchRunner(
            backend,
            RecordingStore(),
            owner_id=OWNER_ID,
            base_snapshot=base,
            base_manifest=manifest,
        )
        target.write_text("changed elsewhere\n", encoding="utf-8")
        with pytest.raises(PatchStaleError, match="no longer matches"):
            await runner.apply(make_call(base.snapshot_id), modification(base.snapshot_id))

    unaffected_change_root = tmp_path / "unaffected-change"
    unaffected_change_root.mkdir()
    unaffected_target = unaffected_change_root / "main.py"
    unaffected_target.write_text("old\nkeep this line\n", encoding="utf-8")
    with WorkspaceBackend(unaffected_change_root) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        runner = PatchRunner(
            backend,
            RecordingStore(),
            owner_id=OWNER_ID,
            base_snapshot=base,
            base_manifest=manifest,
        )
        unaffected_target.write_text("old\nchanged elsewhere\n", encoding="utf-8")
        patch = PatchRequest(
            patch=Patch(
                base_snapshot_id=base.snapshot_id,
                unified_diff=(
                    "--- a/main.py\n"
                    "+++ b/main.py\n"
                    "@@ -1 +1 @@\n"
                    "-old\n"
                    "+new\n"
                ),
            )
        )
        with pytest.raises(PatchStaleError, match="matches"):
            await runner.apply(make_call(base.snapshot_id), patch)

    binary_root = tmp_path / "binary"
    binary_root.mkdir()
    (binary_root / "data.bin").write_bytes(b"\x00old\n")
    with WorkspaceBackend(binary_root) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        runner = PatchRunner(
            backend,
            RecordingStore(),
            owner_id=OWNER_ID,
            base_snapshot=base,
            base_manifest=manifest,
        )
        with pytest.raises(PatchValidationError, match="bounded text"):
            await runner.apply(
                make_call(base.snapshot_id), modification(base.snapshot_id, "data.bin")
            )
        with pytest.raises(PatchValidationError, match="workspace-relative"):
            await runner.apply(
                make_call(base.snapshot_id),
                modification(base.snapshot_id, "../escape.py"),
            )


def test_promotion_failure_restores_the_original_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    target = source / "main.py"
    target.write_text("old\n", encoding="utf-8")
    original_replace = os.replace
    failed = False

    def fail_staged_replace(source_name: str, destination_name: str, **kwargs: object) -> None:
        nonlocal failed
        if source_name == "new-0" and not failed:
            failed = True
            raise OSError("promotion failed")
        original_replace(source_name, destination_name, **kwargs)

    monkeypatch.setattr("prp_runtime.workspace.backend.os.replace", fail_staged_replace)
    with WorkspaceBackend(tmp_path) as backend:
        transaction = backend.stage_text_changes({"src/main.py": (True, "new\n")})
        with pytest.raises(WorkspaceBackendError, match="promotion failed"):
            transaction.commit()
        assert target.read_text(encoding="utf-8") == "old\n"

    assert not tuple(tmp_path.glob(".prp-patch-*"))


def test_partial_multi_file_promotion_restores_every_original_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("first-old\n", encoding="utf-8")
    second.write_text("second-old\n", encoding="utf-8")
    original_replace = os.replace
    failed = False

    def fail_second_promotion(source_name: str, destination_name: str, **kwargs: object) -> None:
        nonlocal failed
        if source_name == "new-1" and not failed:
            failed = True
            raise OSError("second promotion failed")
        original_replace(source_name, destination_name, **kwargs)

    monkeypatch.setattr("prp_runtime.workspace.backend.os.replace", fail_second_promotion)
    with WorkspaceBackend(tmp_path) as backend:
        transaction = backend.stage_text_changes(
            {
                "first.py": (True, "first-new\n"),
                "second.py": (True, "second-new\n"),
            }
        )
        with pytest.raises(WorkspaceBackendError, match="patch promotion failed"):
            transaction.commit()
        assert first.read_text(encoding="utf-8") == "first-old\n"
        assert second.read_text(encoding="utf-8") == "second-old\n"

    assert not tuple(tmp_path.glob(".prp-patch-*"))


def test_symlink_is_rejected_before_patch_processing(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.py").write_text("secret\n", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError, match="symlink"):
            backend.snapshot_manifest()


def test_patch_is_a_write_tool_and_plan_mode_denies_it(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("old\n", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        manifest = backend.snapshot_manifest()
        base = make_base_snapshot(manifest)
        definition = build_patch_definition(
            PatchRunner(
                backend,
                RecordingStore(),
                owner_id=OWNER_ID,
                base_snapshot=base,
                base_manifest=manifest,
            )
        )
        decision = decide_tool_call(
            make_call(base.snapshot_id), AgentMode.PLAN, known_tools=(definition.name,)
        )

    assert definition.effect is ToolEffect.WRITE
    assert decision.outcome is PolicyOutcome.DENY
    assert decision.reason_code is PolicyReasonCode.PLAN_SIDE_EFFECT


def test_materialize_patched_contents_applies_diff_without_a_live_root() -> None:
    previous = {"src/main.py": "old\n", "kept.txt": "keep\n"}
    patch = Patch(
        base_snapshot_id="snap_" + "a" * 32,
        unified_diff=(
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        ),
    )
    files = (
        FileChange(
            path="src/main.py",
            action=FileChangeAction.MODIFY,
            before=FileContent(sha256=hashlib.sha256(b"old\n").hexdigest(), size=4),
            after=FileContent(sha256=hashlib.sha256(b"new\n").hexdigest(), size=4),
        ),
    )
    assert materialize_patched_contents(previous, patch, files) == {
        "src/main.py": "new\n",
        "kept.txt": "keep\n",
    }
