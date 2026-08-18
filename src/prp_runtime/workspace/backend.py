"""Descriptor-based access to one server-authorized workspace root."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO, Protocol, Self

from pydantic import ConfigDict, Field, StringConstraints

from prp_runtime.domain.models import DomainModel
from prp_runtime.workspace.models import SnapshotEntry, SnapshotEntryType, SnapshotManifest

__all__ = [
    "DirectoryEntry",
    "FileReadResult",
    "WorkspaceBackend",
    "WorkspaceBackendError",
    "WorkspaceBackendProtocol",
    "WorkspacePatchTransaction",
]

RelativeWorkspacePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
]


class WorkspaceBackendError(ValueError):
    """A stable, host-path-free workspace access error."""


class DirectoryEntry(DomainModel):
    """Metadata for one safe child of an authorized directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: RelativeWorkspacePath
    entry_type: SnapshotEntryType
    size: int = Field(ge=0)


class FileReadResult(DomainModel):
    """Bounded text or explicit binary metadata from a workspace file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: RelativeWorkspacePath
    content: str | None = None
    binary: bool = False
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)
    bytes_read: int = Field(ge=0)
    truncated: bool = False


class WorkspaceBackendProtocol(Protocol):
    """Safe primitives required by future file tool handlers."""

    def resolve(self, path: str) -> str:
        """Validate and return a safe relative path."""

    def open_read(self, path: str) -> BinaryIO:
        """Open a regular file through an authorized descriptor."""

    def list_directory(self, path: str = "") -> tuple[DirectoryEntry, ...]:
        """List safe children in deterministic order."""

    def read_file(
        self, path: str, *, offset: int = 0, limit: int | None = None
    ) -> FileReadResult:
        """Read a bounded regular file without exposing its host path."""


_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_MAX_READ_LIMIT = 8 * 1024 * 1024
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_PATH_FLAGS = getattr(os, "O_PATH", _OPEN_FLAGS)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_PATCH_STAGE_PREFIX = ".prp-patch-"


@dataclass
class _StagedPatchFile:
    parent_fd: int
    name: str
    staged_name: str | None
    backup_name: str
    expected_exists: bool
    new_promoted: bool = False
    backup_moved: bool = False


class WorkspacePatchTransaction:
    """Keep staged file replacements reversible until durable facts commit."""

    def __init__(
        self,
        backend: WorkspaceBackend,
        changes: Mapping[str, tuple[bool, str | None]],
        expected_digests: Mapping[str, tuple[str, int]] | None = None,
    ) -> None:
        self._backend = backend
        self._staging_name = f"{_PATCH_STAGE_PREFIX}{secrets.token_hex(12)}"
        self._staging_fd: int | None = None
        self._entries: list[_StagedPatchFile] = []
        self._closed = False
        self._committed = False
        self._prepare(changes, expected_digests or {})

    def _prepare(
        self,
        changes: Mapping[str, tuple[bool, str | None]],
        expected_digests: Mapping[str, tuple[str, int]],
    ) -> None:
        self._backend._ensure_open()
        if not set(expected_digests).issubset(changes):
            raise WorkspaceBackendError("patch staging contains an unknown expected file")
        try:
            os.mkdir(self._staging_name, mode=0o700, dir_fd=self._backend._root_fd)
            self._staging_fd = os.open(
                self._staging_name,
                _OPEN_FLAGS | os.O_DIRECTORY | _NOFOLLOW,
                dir_fd=self._backend._root_fd,
            )
            for index, (path, (expected_exists, content)) in enumerate(changes.items()):
                relative = _normalise_path(path)
                parent_fd, name = self._backend._open_parent(relative)
                try:
                    exists = _is_regular_file(name, parent_fd)
                    if exists != expected_exists:
                        raise WorkspaceBackendError("workspace changed before patch promotion")
                    expected_digest = expected_digests.get(path)
                    if expected_digest is not None:
                        digest, size = _hash_regular_file(name, parent_fd)
                        if (digest, size) != expected_digest:
                            raise WorkspaceBackendError(
                                "workspace content changed before patch promotion"
                            )
                    staged_name = None if content is None else f"new-{index}"
                    entry = _StagedPatchFile(
                        parent_fd=parent_fd,
                        name=name,
                        staged_name=staged_name,
                        backup_name=f"old-{index}",
                        expected_exists=expected_exists,
                    )
                    self._entries.append(entry)
                    if content is not None:
                        assert staged_name is not None
                        self._write_staged(staged_name, content.encode("utf-8"))
                except BaseException:
                    os.close(parent_fd)
                    raise
        except BaseException:
            self._cleanup()
            raise

    def _write_staged(self, name: str, content: bytes) -> None:
        if len(content) > _MAX_READ_LIMIT:
            raise WorkspaceBackendError("patched file exceeds the workspace limit")
        if self._staging_fd is None:
            raise WorkspaceBackendError("patch staging is unavailable")
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=self._staging_fd,
        )
        try:
            offset = 0
            while offset < len(content):
                offset += os.write(fd, content[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

    def commit(self) -> None:
        """Promote every staged file while retaining backups for rollback."""
        if self._closed:
            raise WorkspaceBackendError("patch transaction is closed")
        if self._committed:
            return
        if self._staging_fd is None:
            raise WorkspaceBackendError("patch staging is unavailable")
        try:
            for entry in self._entries:
                if entry.expected_exists:
                    os.replace(
                        entry.name,
                        entry.backup_name,
                        src_dir_fd=entry.parent_fd,
                        dst_dir_fd=self._staging_fd,
                    )
                    entry.backup_moved = True
                if entry.staged_name is not None:
                    os.replace(
                        entry.staged_name,
                        entry.name,
                        src_dir_fd=self._staging_fd,
                        dst_dir_fd=entry.parent_fd,
                    )
                    entry.new_promoted = True
            self._committed = True
        except OSError as error:
            try:
                self.rollback()
            except WorkspaceBackendError as rollback_error:
                raise WorkspaceBackendError("patch promotion rollback failed") from rollback_error
            raise WorkspaceBackendError("workspace patch promotion failed") from error

    def rollback(self) -> None:
        """Restore every promoted entry and remove all staged data."""
        if self._closed:
            return
        rollback_error: OSError | None = None
        if self._staging_fd is not None:
            for entry in reversed(self._entries):
                try:
                    if entry.new_promoted:
                        os.unlink(entry.name, dir_fd=entry.parent_fd)
                        entry.new_promoted = False
                    if entry.backup_moved:
                        os.replace(
                            entry.backup_name,
                            entry.name,
                            src_dir_fd=self._staging_fd,
                            dst_dir_fd=entry.parent_fd,
                        )
                        entry.backup_moved = False
                except OSError as error:
                    rollback_error = error
                    break
        self._cleanup()
        if rollback_error is not None:
            raise WorkspaceBackendError("workspace patch rollback failed") from rollback_error

    def finalize(self) -> None:
        """Discard backups only after the corresponding durable facts commit."""
        if self._closed:
            return
        if not self._committed:
            raise WorkspaceBackendError("cannot finalize an uncommitted patch")
        self._cleanup()

    def _cleanup(self) -> None:
        staging_fd, self._staging_fd = self._staging_fd, None
        if staging_fd is not None:
            for entry in self._entries:
                for name in (entry.staged_name, entry.backup_name):
                    if name is None:
                        continue
                    try:
                        os.unlink(name, dir_fd=staging_fd)
                    except FileNotFoundError:
                        continue
            os.close(staging_fd)
        for entry in self._entries:
            try:
                os.close(entry.parent_fd)
            except OSError:
                continue
        try:
            os.rmdir(self._staging_name, dir_fd=self._backend._root_fd)
        except FileNotFoundError:
            pass
        self._closed = True


class WorkspaceBackend:
    """Own one root descriptor and perform all access relative to that fd."""

    __slots__ = ("_root_fd", "_max_read_bytes", "_closed")

    def __init__(self, root: Path, *, max_read_bytes: int = _MAX_READ_LIMIT) -> None:
        if max_read_bytes <= 0 or max_read_bytes > _MAX_READ_LIMIT:
            raise ValueError(f"max_read_bytes must be between 1 and {_MAX_READ_LIMIT}")
        try:
            root_stat = root.lstat()
        except OSError as error:
            raise WorkspaceBackendError("workspace root is unavailable") from error
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise WorkspaceBackendError("workspace root is not an authorized directory")
        try:
            root_fd = os.open(root, _OPEN_FLAGS | os.O_DIRECTORY | _NOFOLLOW)
        except OSError as error:
            raise WorkspaceBackendError("workspace root cannot be opened safely") from error
        self._root_fd = root_fd
        self._max_read_bytes = max_read_bytes
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the root descriptor; repeated close is harmless."""
        if not self._closed:
            os.close(self._root_fd)
            self._closed = True

    def resolve(self, path: str) -> str:
        """Return a validated relative path after descriptor resolution."""
        relative = _normalise_path(path, allow_root=True)
        fd = self._open_relative(relative)
        try:
            _require_supported_entry(os.fstat(fd), allow_directory=True)
        finally:
            os.close(fd)
        return relative

    def open_read(self, path: str) -> BinaryIO:
        """Open a regular file with symlink following disabled."""
        relative = _normalise_path(path)
        parent_fd, name = self._open_parent(relative)
        try:
            fd = os.open(
                name,
                _OPEN_FLAGS | _NONBLOCK | _NOFOLLOW,
                dir_fd=parent_fd,
            )
            _require_supported_entry(os.fstat(fd), allow_directory=False)
        except BaseException:
            if "fd" in locals():
                os.close(fd)
            raise
        finally:
            os.close(parent_fd)
        return os.fdopen(fd, "rb", closefd=True)

    def list_directory(self, path: str = "") -> tuple[DirectoryEntry, ...]:
        """List regular files and directories with stable relative names."""
        relative = _normalise_path(path, allow_root=True)
        directory_fd = self._open_relative(relative)
        scan_fd: int | None = None
        try:
            directory_stat = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise WorkspaceBackendError("path is not a directory")
            scan_fd = os.open(
                ".",
                _OPEN_FLAGS | os.O_DIRECTORY | _NOFOLLOW,
                dir_fd=directory_fd,
            )
            entries: list[DirectoryEntry] = []
            with os.scandir(scan_fd) as iterator:
                for entry in iterator:
                    entry_stat = entry.stat(follow_symlinks=False)
                    entry_type = _entry_type(entry_stat)
                    child = entry.name if not relative else f"{relative}/{entry.name}"
                    entries.append(
                        DirectoryEntry(
                            path=child,
                            entry_type=entry_type,
                            size=entry_stat.st_size
                            if entry_type is SnapshotEntryType.FILE
                            else 0,
                        )
                    )
            return tuple(sorted(entries, key=lambda item: item.path))
        except WorkspaceBackendError:
            raise
        except OSError as error:
            raise _safe_os_error(error) from error
        finally:
            if scan_fd is not None:
                os.close(scan_fd)
            os.close(directory_fd)

    def read_file(
        self, path: str, *, offset: int = 0, limit: int | None = None
    ) -> FileReadResult:
        """Read UTF-8 text or return explicit binary metadata."""
        if offset < 0:
            raise WorkspaceBackendError("offset must not be negative")
        read_limit = self._max_read_bytes if limit is None else limit
        if read_limit <= 0 or read_limit > self._max_read_bytes:
            raise WorkspaceBackendError("read limit exceeds the workspace limit")
        relative = _normalise_path(path)
        with self.open_read(relative) as stream:
            stream.seek(offset)
            data = stream.read(read_limit + 1)
        truncated = len(data) > read_limit
        bounded = data[:read_limit]
        try:
            content = bounded.decode("utf-8")
            binary = b"\x00" in bounded
        except UnicodeDecodeError:
            content = None
            binary = True
        return FileReadResult(
            path=relative,
            content=None if binary else content,
            binary=binary,
            offset=offset,
            limit=read_limit,
            bytes_read=len(bounded),
            truncated=truncated,
        )

    def read_patch_text(self, path: str) -> str:
        """Read one complete UTF-8 regular file for a bounded patch operation."""
        result = self.read_file(path)
        if result.binary or result.content is None:
            raise WorkspaceBackendError("binary files cannot be patched")
        if result.truncated:
            raise WorkspaceBackendError("patched file exceeds the workspace limit")
        return result.content

    def snapshot_manifest(self) -> SnapshotManifest:
        """Build a deterministic manifest through the root descriptor only."""
        self._ensure_open()
        entries: list[SnapshotEntry] = []
        root_fd = os.dup(self._root_fd)
        try:
            self._append_manifest_entries(root_fd, "", entries)
            return SnapshotManifest(entries=tuple(entries))
        except WorkspaceBackendError:
            raise
        except (OSError, ValueError) as error:
            raise WorkspaceBackendError("workspace manifest cannot be created") from error
        finally:
            os.close(root_fd)

    def stage_text_changes(
        self,
        changes: Mapping[str, tuple[bool, str | None]],
        *,
        expected_digests: Mapping[str, tuple[str, int]] | None = None,
    ) -> WorkspacePatchTransaction:
        """Stage expected file replacements below the authorized root."""
        if not changes:
            raise WorkspaceBackendError("patch must change at least one file")
        return WorkspacePatchTransaction(self, changes, expected_digests)

    def _append_manifest_entries(
        self, directory_fd: int, prefix: str, entries: list[SnapshotEntry]
    ) -> None:
        scan_fd = os.open(
            ".",
            _OPEN_FLAGS | os.O_DIRECTORY | _NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            with os.scandir(scan_fd) as iterator:
                children = sorted(iterator, key=lambda entry: entry.name)
            for child in children:
                child_stat = child.stat(follow_symlinks=False)
                entry_type = _entry_type(child_stat)
                path = child.name if not prefix else f"{prefix}/{child.name}"
                if entry_type is SnapshotEntryType.DIRECTORY:
                    entries.append(
                        SnapshotEntry(
                            path=path,
                            sha256=hashlib.sha256(b"").hexdigest(),
                            size=0,
                            entry_type=entry_type,
                        )
                    )
                    child_fd = os.open(
                        child.name,
                        _OPEN_FLAGS | os.O_DIRECTORY | _NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        self._append_manifest_entries(child_fd, path, entries)
                    finally:
                        os.close(child_fd)
                else:
                    digest, size = _hash_regular_file(child.name, directory_fd)
                    entries.append(
                        SnapshotEntry(
                            path=path,
                            sha256=digest,
                            size=size,
                            entry_type=entry_type,
                        )
                    )
        finally:
            os.close(scan_fd)

    def _open_relative(self, relative: str) -> int:
        self._ensure_open()
        current = os.dup(self._root_fd)
        if not relative:
            return current
        try:
            for component in relative.split("/"):
                next_fd = os.open(
                    component,
                    _PATH_FLAGS | _NOFOLLOW,
                    dir_fd=current,
                )
                os.close(current)
                current = next_fd
            return current
        except OSError as error:
            os.close(current)
            raise _safe_os_error(error) from error

    def _open_parent(self, relative: str) -> tuple[int, str]:
        self._ensure_open()
        components = relative.split("/")
        current = os.dup(self._root_fd)
        try:
            for component in components[:-1]:
                next_fd = os.open(
                    component,
                    _PATH_FLAGS | _NOFOLLOW,
                    dir_fd=current,
                )
                os.close(current)
                current = next_fd
                if not stat.S_ISDIR(os.fstat(current).st_mode):
                    raise WorkspaceBackendError("path is not a directory")
            return current, components[-1]
        except BaseException:
            os.close(current)
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise WorkspaceBackendError("workspace backend is closed")


def _normalise_path(path: str, *, allow_root: bool = False) -> str:
    if not isinstance(path, str):
        raise WorkspaceBackendError("path must be a relative POSIX string")
    if allow_root and path in {"", "."}:
        return ""
    if not path or path.startswith(("/", "\\")) or _DRIVE_RE.match(path) or "\\" in path:
        raise WorkspaceBackendError("path is not authorized")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise WorkspaceBackendError("path is not authorized")
    return "/".join(parts)


def _require_supported_entry(entry: os.stat_result, *, allow_directory: bool) -> None:
    if stat.S_ISLNK(entry.st_mode):
        raise WorkspaceBackendError("symlink paths are not supported")
    if stat.S_ISDIR(entry.st_mode):
        if allow_directory:
            return
        raise WorkspaceBackendError("path is a directory")
    if not stat.S_ISREG(entry.st_mode):
        raise WorkspaceBackendError("special files are not supported")


def _entry_type(entry: os.stat_result) -> SnapshotEntryType:
    if stat.S_ISREG(entry.st_mode):
        return SnapshotEntryType.FILE
    if stat.S_ISDIR(entry.st_mode):
        return SnapshotEntryType.DIRECTORY
    if stat.S_ISLNK(entry.st_mode):
        raise WorkspaceBackendError("symlink entries are not supported")
    raise WorkspaceBackendError("special files are not supported")


def _is_regular_file(name: str, parent_fd: int) -> bool:
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    _require_supported_entry(entry, allow_directory=False)
    return True


def _hash_regular_file(name: str, parent_fd: int) -> tuple[str, int]:
    fd = os.open(name, _OPEN_FLAGS | _NONBLOCK | _NOFOLLOW, dir_fd=parent_fd)
    try:
        _require_supported_entry(os.fstat(fd), allow_directory=False)
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(fd, 65_536):
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size
    finally:
        os.close(fd)


def _safe_os_error(error: OSError) -> WorkspaceBackendError:
    if error.errno in {errno.ELOOP, errno.EMLINK}:
        return WorkspaceBackendError("symlink paths are not supported")
    if error.errno in {errno.ENOENT, errno.ENOTDIR}:
        return WorkspaceBackendError("path does not exist")
    if error.errno in {errno.EACCES, errno.EPERM}:
        return WorkspaceBackendError("path is not accessible")
    return WorkspaceBackendError("workspace path operation failed")
