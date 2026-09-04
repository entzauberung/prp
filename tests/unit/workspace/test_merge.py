"""R0 contract tests for temporary DEV merge candidates."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prp_runtime.domain.values import (
    new_run_id,
    new_snapshot_id,
    new_tool_call_id,
    new_workspace_id,
)
from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
    new_change_set_id,
)
from prp_runtime.workspace.merge import (
    GitMergeBackend,
    MergeError,
    MergeStatus,
    StagedChangeSet,
    merge_candidate_file_contents,
    merge_candidate_manifest,
    promote_merge,
)


def _content(value: str) -> FileContent:
    encoded = value.encode("utf-8")
    return FileContent(sha256=hashlib.sha256(encoded).hexdigest(), size=len(encoded))


def test_temporary_merge_candidate_is_dev_only_and_does_not_expose_or_change_base(
    tmp_path: Path,
) -> None:
    base_snapshot_id = new_snapshot_id()
    before = "before\n"
    after = "after\n"
    base_root = tmp_path / "base"
    staged_root = tmp_path / "staged"
    staging_root = tmp_path / "merge"
    base_root.mkdir()
    staged_root.mkdir()
    (base_root / "README.md").write_text(before, encoding="utf-8")
    (staged_root / "README.md").write_text(after, encoding="utf-8")
    change_set = ChangeSet(
        change_set_id=new_change_set_id(),
        run_id=new_run_id(),
        tool_call_id=new_tool_call_id(),
        workspace_id=new_workspace_id(),
        base_snapshot_id=base_snapshot_id,
        new_snapshot_id=new_snapshot_id(),
        patch=Patch(base_snapshot_id=base_snapshot_id, unified_diff="@@ -1 +1 @@"),
        files=(
            FileChange(
                path="README.md",
                action=FileChangeAction.MODIFY,
                before=_content(before),
                after=_content(after),
            ),
        ),
        created_at=datetime(2026, 8, 17, 12, 0, tzinfo=UTC),
    )

    result = GitMergeBackend().merge(
        base_root,
        (StagedChangeSet(change_set=change_set, root=staged_root),),
        staging_root=staging_root,
    )

    assert result.status is MergeStatus.MERGED
    assert result.dev_only is True
    assert result.verified is False
    assert result.promoted is False
    assert str(staging_root) not in result.model_dump_json()
    assert "staging_root" not in result.model_dump(mode="json")
    assert (base_root / "README.md").read_text(encoding="utf-8") == before
    manifest = merge_candidate_manifest(result.staging_root)
    contents = merge_candidate_file_contents(result.staging_root, manifest)
    assert contents["README.md"] == after
    assert not any(path == ".git" or path.startswith(".git/") for path in contents)
    with pytest.raises(MergeError, match="verified"):
        promote_merge(result, tmp_path / "unverified")


def test_merge_rejects_empty_or_overlapping_roots_without_touching_base(
    tmp_path: Path,
) -> None:
    base_root = tmp_path / "base"
    base_root.mkdir()
    (base_root / "README.md").write_text("base\n", encoding="utf-8")

    empty = GitMergeBackend().merge(
        base_root,
        (),
        staging_root=tmp_path / "empty-merge",
    )
    assert empty.status is MergeStatus.FAILED

    overlapping = GitMergeBackend().merge(
        base_root,
        (),
        staging_root=base_root,
    )
    assert overlapping.status is MergeStatus.FAILED
    assert (base_root / "README.md").read_text(encoding="utf-8") == "base\n"
