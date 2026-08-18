"""Targeted tests for descriptor-based workspace access."""

import os
from pathlib import Path

import pytest

from prp_runtime.workspace import (
    SnapshotEntryType,
    WorkspaceBackend,
    WorkspaceBackendError,
)


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
