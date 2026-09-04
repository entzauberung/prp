"""Targeted tests for descriptor-based workspace access."""

import os
import time
from pathlib import Path

import pytest

from prp_runtime.workspace import (
    SnapshotEntryType,
    WorkspaceBackend,
    WorkspaceBackendError,
)
from prp_runtime.workspace.models import SnapshotEntry, SnapshotEntryType, SnapshotManifest
from prp_runtime.workspace.backend import ExportFile, select_snapshot_export_files


def test_resolve_read_and_list_stay_relative_and_are_deterministic(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("0123456789", encoding="utf-8")
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "nested").mkdir()

    with WorkspaceBackend(tmp_path, max_read_bytes=8) as backend:
        assert backend.resolve("a.txt") == "a.txt"
        entries = backend.list_directory()
        assert [entry.path for entry in entries] == ["a.txt", "nested", "z.txt"]
        assert entries[0].entry_type is SnapshotEntryType.FILE
        assert entries[1].entry_type is SnapshotEntryType.DIRECTORY
        result = backend.read_file("z.txt", offset=2, limit=4)

    assert result.path == "z.txt"
    assert result.content == "2345"
    assert result.bytes_read == 4
    assert result.truncated is True
    assert str(tmp_path) not in result.model_dump_json()


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../secret", "a/../x", "a//b", "a\\b", "C:/x"]
)
def test_absolute_parent_and_non_posix_paths_are_rejected(
    tmp_path: Path, path: str
) -> None:
    (tmp_path / "a").mkdir()
    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError, match="not authorized"):
            backend.resolve(path)


def test_symlink_escape_is_rejected_without_leaking_host_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    (tmp_path / "file-link").symlink_to(outside / "secret.txt")

    with WorkspaceBackend(tmp_path) as backend:
        for path in ("escape/secret.txt", "file-link"):
            with pytest.raises(WorkspaceBackendError) as error:
                backend.resolve(path)
            assert str(tmp_path) not in str(error.value)
            assert str(outside) not in str(error.value)
        with pytest.raises(WorkspaceBackendError, match="symlink"):
            backend.list_directory()


def test_fifo_and_binary_files_are_rejected_or_explicitly_reported(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    (tmp_path / "binary.bin").write_bytes(b"\x00\xff\x01")

    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError, match="special files"):
            backend.resolve("pipe")
        with pytest.raises(WorkspaceBackendError, match="special files"):
            backend.list_directory()
        binary = backend.read_file("binary.bin")

    assert binary.binary is True
    assert binary.content is None
    assert binary.bytes_read == 3


def test_open_read_uses_a_live_descriptor_and_backend_close_is_safe(tmp_path: Path) -> None:
    path = tmp_path / "stable.txt"
    path.write_text("stable", encoding="utf-8")
    backend = WorkspaceBackend(tmp_path)
    stream = backend.open_read("stable.txt")
    backend.close()
    assert stream.read() == b"stable"
    stream.close()
    backend.close()
    with pytest.raises(WorkspaceBackendError, match="closed"):
        backend.resolve("stable.txt")


def test_export_bundle_excludes_secrets_and_does_not_mutate(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / "auth.json").write_text('{"provider":"https://models.internal","api_key":"sk-secret"}', encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git-internal", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ok.txt").write_text("ok", encoding="utf-8")
    before = (tmp_path / "readme.txt").read_bytes()

    with WorkspaceBackend(tmp_path) as backend:
        selection = backend.collect_export_files(
            max_files=8, max_bytes=1024, max_nesting=4
        )

    paths = [item.path for item in selection.files]
    assert paths == ["nested/ok.txt", "readme.txt"]
    assert all(isinstance(item, ExportFile) for item in selection.files)
    assert ".env" in selection.excluded_paths
    assert "auth.json" in selection.excluded_paths
    assert ".git" in selection.excluded_paths
    assert all(".git" not in item.path for item in selection.files)
    assert "SECRET=1" not in "".join(item.content for item in selection.files)
    assert "sk-secret" not in "".join(item.content for item in selection.files)
    assert "https://models.internal" not in "".join(item.content for item in selection.files)
    assert str(tmp_path) not in selection.model_dump_json()
    assert (tmp_path / "readme.txt").read_bytes() == before
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=1"
    assert (git_dir / "config").read_text(encoding="utf-8") == "git-internal"


def test_export_bundle_rejects_oversize_and_deep_nesting(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 32, encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError, match="byte limit"):
            backend.collect_export_files(max_files=8, max_bytes=8, max_nesting=4)

    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "leaf.txt").write_text("leaf", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError, match="nesting"):
            backend.collect_export_files(max_files=8, max_bytes=1024, max_nesting=1)


def test_export_oversize_fails_without_partial_selection(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("aaaa", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bbbb", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError, match="file limit"):
            backend.collect_export_files(max_files=1, max_bytes=1024, max_nesting=4)


def test_export_rejects_symlink_without_leaking_host_path(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-export-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret-value", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "link").symlink_to(outside / "secret.txt")
    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError) as error:
            backend.collect_export_files(max_files=8, max_bytes=1024, max_nesting=4)
    assert str(tmp_path) not in str(error.value)
    assert str(outside) not in str(error.value)
    assert "secret-value" not in str(error.value)
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "ok"



def test_require_base_manifest_rejects_stale_tree_without_host_paths(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        base = backend.snapshot_manifest()
        backend.require_base_manifest(base)
        (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
        with pytest.raises(WorkspaceBackendError, match="stale") as error:
            backend.require_base_manifest(base)
    assert str(tmp_path) not in str(error.value)


def test_export_bundle_keeps_snapshot_manifest_and_drops_later_edits(tmp_path: Path) -> None:
    (tmp_path / "kept.txt").write_text("frozen", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("live-only", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        manifest = backend.snapshot_manifest()
        (tmp_path / "new.txt").write_text("after-run", encoding="utf-8")
        selection = backend.collect_export_files(
            max_files=8,
            max_bytes=1024,
            max_nesting=4,
            snapshot_manifest=manifest,
        )
        paths = [item.path for item in selection.files]
        assert paths == ["extra.txt", "kept.txt"]
        assert "new.txt" not in paths
        assert "after-run" not in "".join(item.content for item in selection.files)
        (tmp_path / "kept.txt").write_text("mutated", encoding="utf-8")
        with pytest.raises(WorkspaceBackendError, match="stale"):
            backend.collect_export_files(
                max_files=8,
                max_bytes=1024,
                max_nesting=4,
                snapshot_manifest=manifest,
            )
        (tmp_path / "kept.txt").write_text("frozen", encoding="utf-8")
        (tmp_path / "extra.txt").unlink()
        with pytest.raises(WorkspaceBackendError, match="stale"):
            backend.collect_export_files(
                max_files=8,
                max_bytes=1024,
                max_nesting=4,
                snapshot_manifest=manifest,
            )


def test_snapshot_export_uses_captured_contents_after_live_edits(tmp_path: Path) -> None:
    (tmp_path / "kept.txt").write_text("frozen", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("live-only", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        manifest, contents = backend.capture_snapshot()
        (tmp_path / "kept.txt").write_text("mutated", encoding="utf-8")
        (tmp_path / "new.txt").write_text("after-run", encoding="utf-8")
        selection = select_snapshot_export_files(
            manifest,
            contents,
            max_files=8,
            max_bytes=1024,
            max_nesting=4,
        )
    paths = [item.path for item in selection.files]
    joined = "".join(item.content for item in selection.files)
    assert paths == ["extra.txt", "kept.txt"]
    assert "frozen" in joined
    assert "live-only" in joined
    assert "mutated" not in joined
    assert "after-run" not in joined
    assert ".env" not in contents
    assert "SECRET=1" not in joined

def test_export_rejects_expired_deadline_and_unknown_binary(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError, match="time limit"):
            backend.collect_export_files(
                max_files=8,
                max_bytes=1024,
                max_nesting=4,
                deadline_monotonic=time.monotonic() - 1,
            )
    (tmp_path / "blob.bin").write_bytes(b"\x00\xff")
    with WorkspaceBackend(tmp_path) as backend:
        with pytest.raises(WorkspaceBackendError, match="unknown binary"):
            backend.collect_export_files(max_files=8, max_bytes=1024, max_nesting=4)

