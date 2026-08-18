"""Targeted tests for bounded manifest and read-only Git diff backends."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import ToolCallStatus, ToolEffect
from prp_runtime.domain.values import new_run_id, new_tool_call_id, new_work_unit_id
from prp_runtime.tools.diff import (
    DiffParseError,
    DiffRequest,
    DiffStatus,
    DiffToolRunner,
    GitDiffBackend,
    ManifestDiffBackend,
    build_diff_definitions,
    change_set_diff,
    parse_git_name_status,
    parse_git_status,
)
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.models import ToolCall
from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
)
from prp_runtime.workspace.models import SnapshotEntry, SnapshotEntryType, SnapshotManifest

T0 = datetime(2026, 8, 14, tzinfo=UTC)
BASE = "snap_" + "a" * 32
NEW = "snap_" + "b" * 32


def entry(path: str, digest: str, size: int = 1) -> SnapshotEntry:
    return SnapshotEntry(
        path=path,
        sha256=digest * 64 if len(digest) == 1 else digest,
        size=size,
        entry_type=SnapshotEntryType.FILE,
    )


def manifest(*entries: SnapshotEntry) -> SnapshotManifest:
    return SnapshotManifest(entries=entries)


def test_manifest_diff_distinguishes_modify_add_delete_and_content_rename() -> None:
    base = manifest(
        entry("delete.py", "c"),
        entry("keep.py", "d"),
        entry("modify.py", "a"),
        entry("old.py", "b"),
    )
    new = manifest(
        entry("added.py", "e"),
        entry("keep.py", "d"),
        entry("modify.py", "f", size=2),
        entry("renamed.py", "b"),
    )

    result = ManifestDiffBackend().compare(base, new)

    assert [(item.path, item.status) for item in result.entries] == [
        ("added.py", DiffStatus.ADDED),
        ("delete.py", DiffStatus.DELETED),
        ("modify.py", DiffStatus.MODIFIED),
        ("renamed.py", DiffStatus.RENAMED),
    ]
    renamed = result.entries[-1]
    assert renamed.old_path == "old.py"
    assert result.summary.total == 4
    assert result.summary.added == 1
    assert result.summary.deleted == 1
    assert result.summary.modified == 1
    assert result.summary.renamed == 1
    assert result.base_manifest_hash == base.manifest_hash
    assert result.new_manifest_hash == new.manifest_hash


def test_manifest_diff_is_stable_and_bounded() -> None:
    base = manifest(entry("b.py", "a"), entry("a.py", "b"))
    new = manifest(entry("b.py", "c"), entry("a.py", "d"))
    first = ManifestDiffBackend(max_entries=1).compare(base, new)
    second = ManifestDiffBackend(max_entries=1).compare(
        SnapshotManifest(entries=tuple(reversed(base.entries))),
        SnapshotManifest(entries=tuple(reversed(new.entries))),
    )

    assert first == second
    assert first.truncated is True
    assert len(first.entries) == 1


def test_change_set_projection_uses_the_same_status_view() -> None:
    change_set = ChangeSet(
        change_set_id="cs_" + "c" * 32,
        run_id="run_" + "d" * 32,
        tool_call_id="tc_" + "e" * 32,
        workspace_id="ws_" + "f" * 32,
        base_snapshot_id=BASE,
        new_snapshot_id=NEW,
        patch=Patch(base_snapshot_id=BASE, unified_diff="--- a/a.py\n+++ b/a.py\n"),
        files=(
            FileChange(
                path="a.py",
                action=FileChangeAction.MODIFY,
                before=FileContent(sha256="a" * 64, size=1),
                after=FileContent(sha256="b" * 64, size=2),
            ),
            FileChange(
                path="new.py",
                action=FileChangeAction.ADD,
                after=FileContent(sha256="c" * 64, size=1),
            ),
        ),
        created_at=T0,
    )

    result = change_set_diff(change_set)

    assert [item.status for item in result.entries] == [
        DiffStatus.MODIFIED,
        DiffStatus.ADDED,
    ]
    assert result.summary.total == 2


def test_git_argv_is_read_only_and_parser_preserves_rename_and_paths(tmp_path: Path) -> None:
    backend = GitDiffBackend(tmp_path)
    assert backend.diff_argv()[-1] == "--"
    assert backend.status_argv()[-1] == "--"
    assert not set(backend.diff_argv()) & {"add", "commit", "reset", "checkout"}
    assert not set(backend.status_argv()) & {"add", "commit", "reset", "checkout"}

    result = backend.parse_diff(
        b"A\0added.py\0M\0src/changed.py\0D\0removed.py\0R100\0old.py\0new.py\0"
    )
    assert [item.status for item in result.entries] == [
        DiffStatus.ADDED,
        DiffStatus.MODIFIED,
        DiffStatus.DELETED,
        DiffStatus.RENAMED,
    ]
    assert result.entries[-1].old_path == "old.py"


def test_git_status_keeps_untracked_distinct_from_added() -> None:
    result = parse_git_status(
        b"?? untracked.py\0A  added.py\0 M modified.py\0D  deleted.py\0R  old.py\0new.py\0"
    )

    assert [item.status for item in result.entries] == [
        DiffStatus.UNTRACKED,
        DiffStatus.ADDED,
        DiffStatus.MODIFIED,
        DiffStatus.DELETED,
        DiffStatus.RENAMED,
    ]
    assert result.entries[-1].old_path == "old.py"


@pytest.mark.parametrize(
    "output",
    [b"A\0../escape.py\0", b"R100\0old.py\0/tmp/new.py\0", b"X\0bad.py\0"],
)
def test_git_parser_rejects_escape_and_unknown_records(output: bytes) -> None:
    with pytest.raises((DiffParseError, ValidationError, ValueError)):
        parse_git_name_status(output)


def test_git_parser_rejects_oversized_output() -> None:
    with pytest.raises(DiffParseError, match="size limit"):
        parse_git_name_status(b"A\0file.py\0", max_output_bytes=1)


def make_verified_change_set() -> tuple[SnapshotManifest, SnapshotManifest, ChangeSet]:
    base = manifest(entry("main.py", "a"))
    after = FileContent(sha256="b" * 64, size=2)
    change_set = ChangeSet(
        change_set_id="cs_" + "c" * 32,
        run_id="run_" + "d" * 32,
        tool_call_id="tc_" + "e" * 32,
        workspace_id="ws_" + "f" * 32,
        base_snapshot_id=BASE,
        new_snapshot_id=NEW,
        patch=Patch(
            base_snapshot_id=BASE,
            unified_diff="--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-old\n+new\n",
        ),
        files=(
            FileChange(
                path="main.py",
                action=FileChangeAction.MODIFY,
                before=FileContent(sha256="a" * 64, size=1),
                after=after,
            ),
        ),
        created_at=T0,
    )
    return base, manifest(entry("main.py", "b", size=2)), change_set


@pytest.mark.asyncio
async def test_diff_handlers_are_read_only_and_verify_manifest_before_returning(
    tmp_path: Path,
) -> None:
    base, current, change_set = make_verified_change_set()
    runner = DiffToolRunner(
        change_set,
        base_manifest=base,
        manifest_provider=lambda: current,
    )
    definitions = build_diff_definitions(runner)
    assert [definition.name for definition in definitions] == ["get_diff", "get_status"]
    assert all(definition.effect is ToolEffect.READ for definition in definitions)
    call = ToolCall(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        tool_name="get_diff",
        effect=ToolEffect.READ,
        arguments={},
        status=ToolCallStatus.RUNNING,
        requested_at="2026-08-14T12:00:00+00:00",
    )
    diff_result = await definitions[0].handler(
        ExecutionContext(call=call, arguments=DiffRequest(), workspace_id="ws-test")
    )
    status_result = await definitions[1].handler(
        ExecutionContext(
            call=call.model_copy(update={"tool_name": "get_status"}),
            arguments=DiffRequest(),
            workspace_id="ws-test",
        )
    )
    assert diff_result.status is ToolCallStatus.SUCCEEDED
    assert diff_result.result is not None
    assert diff_result.result["patch_sha256"] == change_set.patch_sha256
    assert status_result.status is ToolCallStatus.SUCCEEDED
    assert status_result.result is not None
    assert status_result.result["patch"] == ""
    del tmp_path


@pytest.mark.asyncio
async def test_diff_handler_returns_structured_failure_on_manifest_mismatch() -> None:
    base, _, change_set = make_verified_change_set()
    runner = DiffToolRunner(
        change_set,
        base_manifest=base,
        manifest_provider=lambda: manifest(entry("main.py", "c", size=2)),
    )
    definition = build_diff_definitions(runner)[0]
    call = ToolCall(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        tool_name="get_diff",
        effect=ToolEffect.READ,
        arguments={},
        status=ToolCallStatus.RUNNING,
        requested_at="2026-08-14T12:00:00+00:00",
    )
    result = await definition.handler(
        ExecutionContext(call=call, arguments=DiffRequest(), workspace_id="ws-test")
    )
    assert result.status is ToolCallStatus.FAILED
    assert result.error is not None
    assert "does not match ChangeSet" in result.error.message


def test_change_set_patch_is_bounded_but_hash_is_retained() -> None:
    base, _, change_set = make_verified_change_set()
    large_change_set = change_set.model_copy(
        update={
            "patch": Patch(
                base_snapshot_id=BASE,
                unified_diff="x" * (128 * 1024 + 100),
            )
        }
    )
    result = change_set_diff(large_change_set)
    assert result.patch_truncated is True
    assert len(result.patch.encode("utf-8")) <= 128 * 1024
    assert result.patch_sha256 == large_change_set.patch_sha256
    del base
