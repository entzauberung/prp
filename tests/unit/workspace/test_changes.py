"""Contract tests for bounded snapshot-rooted ChangeSet facts."""

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
    inherit_unchanged_file_contents,
    overlay_changed_file_contents,
)
from prp_runtime.workspace.models import SnapshotEntry, SnapshotEntryType, SnapshotManifest

T0 = datetime(2026, 8, 14, tzinfo=UTC)
BASE = "snap_" + "a" * 32
NEW = "snap_" + "b" * 32


def content(digest: str = "a" * 64, size: int = 1) -> FileContent:
    return FileContent(sha256=digest, size=size)


def patch() -> Patch:
    return Patch(base_snapshot_id=BASE, unified_diff="--- a/src/main.py\n+++ b/src/main.py\n")


def change_set(*files: FileChange, **overrides: object) -> ChangeSet:
    values: dict[str, object] = {
        "change_set_id": "cs_" + "c" * 32,
        "run_id": "run_" + "d" * 32,
        "tool_call_id": "tc_" + "e" * 32,
        "workspace_id": "ws_" + "f" * 32,
        "base_snapshot_id": BASE,
        "new_snapshot_id": NEW,
        "patch": patch(),
        "files": files,
        "created_at": T0,
    }
    values.update(overrides)
    return ChangeSet(**values)  # type: ignore[arg-type]


def test_file_changes_record_only_relative_paths_and_complete_content_facts() -> None:
    added = FileChange(path="src/new.py", action=FileChangeAction.ADD, after=content())
    modified = FileChange(
        path="src/main.py",
        action=FileChangeAction.MODIFY,
        before=content(),
        after=content("b" * 64, 2),
    )
    deleted = FileChange(path="src/old.py", action=FileChangeAction.DELETE, before=content())

    assert added.before is None
    assert modified.after is not None
    assert deleted.after is None
    with pytest.raises(ValidationError):
        FileChange(path="../secret", action=FileChangeAction.ADD, after=content())
    with pytest.raises(ValidationError):
        FileChange(path="src/bad.py", action=FileChangeAction.ADD, before=content())
    with pytest.raises(ValidationError):
        FileContent(sha256="A" * 64, size=1)


def test_change_set_requires_matching_base_and_unique_changed_paths() -> None:
    changed = FileChange(path="src/main.py", action=FileChangeAction.ADD, after=content())
    result = change_set(changed)

    assert result.patch_sha256 == result.patch.sha256
    with pytest.raises(ValidationError, match="base snapshot"):
        change_set(changed, base_snapshot_id="snap_" + "g" * 32)
    with pytest.raises(ValidationError, match="duplicate"):
        change_set(changed, changed)
    with pytest.raises(ValidationError, match="new snapshot"):
        change_set(changed, new_snapshot_id=BASE)


def _file(path: str, payload: str) -> SnapshotEntry:
    encoded = payload.encode("utf-8")
    return SnapshotEntry(
        path=path,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size=len(encoded),
        entry_type=SnapshotEntryType.FILE,
    )


def test_inherit_and_overlay_file_contents_keep_unchanged_and_replace_changed() -> None:
    kept = "kept\n"
    gone = "gone\n"
    updated = "updated\n"
    old = SnapshotManifest(entries=(_file("kept.txt", kept), _file("gone.txt", gone)))
    new = SnapshotManifest(entries=(_file("kept.txt", kept), _file("new.txt", updated)))
    inherited = inherit_unchanged_file_contents(
        {"kept.txt": kept, "gone.txt": gone}, old, new
    )
    assert inherited == {"kept.txt": kept}
    overlay = overlay_changed_file_contents(
        inherited,
        (
            FileChange(path="gone.txt", action=FileChangeAction.DELETE, before=content()),
            FileChange(path="new.txt", action=FileChangeAction.ADD, after=content()),
        ),
        {"new.txt": updated},
    )
    assert overlay == {"kept.txt": kept, "new.txt": updated}
