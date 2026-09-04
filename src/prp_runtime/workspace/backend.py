"""Descriptor-based access to one server-authorized workspace root."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import secrets
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO, Protocol, Self

from pydantic import ConfigDict, Field, StringConstraints

from prp_runtime.domain.models import DomainModel
from prp_runtime.workspace.models import (
    MAX_ENTRY_SIZE_BYTES,
    SnapshotEntry,
    SnapshotEntryType,
    SnapshotManifest,
)

__all__ = [
    "DirectoryEntry",
    "ExportBundleSelection",
    "ExportFile",
    "FileReadResult",
    "WorkspaceBackend",
    "WorkspaceBackendError",
    "WorkspaceBackendProtocol",
    "WorkspacePatchTransaction",
    "select_snapshot_export_files",
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


class ExportFile(DomainModel):
    """One authorized text file selected for a CLOUD export bundle."""

    path: RelativeWorkspacePath
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    content: str


class ExportBundleSelection(DomainModel):
    """Bounded export inventory with no host path or secret payload."""

    files: tuple[ExportFile, ...] = ()
    excluded_paths: tuple[str, ...] = ()
    total_bytes: int = Field(ge=0)

    @property
    def file_count(self) -> int:
        return len(self.files)


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

    def collect_export_files(
        self,
        *,
        max_files: int,
        max_bytes: int,
        max_nesting: int,
        deadline_monotonic: float | None = None,
        snapshot_manifest: SnapshotManifest | None = None,
    ) -> ExportBundleSelection:
        """Select bounded, non-secret regular files without mutating the tree.

        When a snapshot manifest is supplied, declared snapshot files must still
        match the live digest. Mutated or missing snapshot files fail closed;
        live files absent from the snapshot are excluded.
        """
        if max_files <= 0 or max_bytes <= 0 or max_nesting <= 0:
            raise WorkspaceBackendError("export limits must be positive")
        self._ensure_open()
        files: list[ExportFile] = []
        excluded: list[str] = []
        totals = [0]
        allowed = None
        if snapshot_manifest is not None:
            allowed = {
                entry.path: entry.sha256
                for entry in snapshot_manifest.entries
                if entry.entry_type is SnapshotEntryType.FILE
            }
        root_fd = os.dup(self._root_fd)
        try:
            self._collect_export_entries(
                root_fd,
                "",
                0,
                files,
                excluded,
                totals,
                max_files=max_files,
                max_bytes=max_bytes,
                max_nesting=max_nesting,
                deadline_monotonic=deadline_monotonic,
                allowed_digests=allowed,
            )
            if allowed is not None:
                exported = {item.path for item in files}
                required = {
                    path for path in allowed if not _export_path_is_excluded(path)
                }
                if any(path not in exported for path in required):
                    raise WorkspaceBackendError("stale workspace snapshot")
            return ExportBundleSelection(
                files=tuple(files),
                excluded_paths=tuple(excluded),
                total_bytes=totals[0],
            )
        except WorkspaceBackendError:
            raise
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise WorkspaceBackendError("workspace export cannot be created") from error
        finally:
            os.close(root_fd)

    def require_base_manifest(self, manifest: SnapshotManifest) -> None:
        """Reject local mutation when the live tree no longer matches the base."""
        current = self.snapshot_manifest()
        if current.manifest_hash != manifest.manifest_hash:
            raise WorkspaceBackendError("stale workspace manifest")

    def snapshot_manifest(self) -> SnapshotManifest:
        """Build a deterministic manifest through the root descriptor only."""
        return self.capture_snapshot()[0]

    def capture_snapshot(self) -> tuple[SnapshotManifest, dict[str, str]]:
        """Capture the manifest plus UTF-8 contents for exportable files."""
        self._ensure_open()
        entries: list[SnapshotEntry] = []
        contents: dict[str, str] = {}
        root_fd = os.dup(self._root_fd)
        try:
            self._append_manifest_entries(root_fd, "", entries, contents)
            return SnapshotManifest(entries=tuple(entries)), contents
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

    def _collect_export_entries(
        self,
        directory_fd: int,
        prefix: str,
        depth: int,
        files: list[ExportFile],
        excluded: list[str],
        totals: list[int],
        *,
        max_files: int,
        max_bytes: int,
        max_nesting: int,
        deadline_monotonic: float | None,
        allowed_digests: dict[str, str] | None = None,
    ) -> None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise WorkspaceBackendError("export exceeded the time limit")
        if depth > max_nesting:
            raise WorkspaceBackendError("export nesting exceeds the workspace limit")
        scan_fd = os.open(
            ".",
            _OPEN_FLAGS | os.O_DIRECTORY | _NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            with os.scandir(scan_fd) as iterator:
                children = sorted(iterator, key=lambda entry: entry.name)
            for child in children:
                path = child.name if not prefix else f"{prefix}/{child.name}"
                if _export_path_is_excluded(path):
                    excluded.append(path)
                    continue
                child_stat = child.stat(follow_symlinks=False)
                entry_type = _entry_type(child_stat)
                if entry_type is SnapshotEntryType.DIRECTORY:
                    child_fd = os.open(
                        child.name,
                        _OPEN_FLAGS | os.O_DIRECTORY | _NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        self._collect_export_entries(
                            child_fd,
                            path,
                            depth + 1,
                            files,
                            excluded,
                            totals,
                            max_files=max_files,
                            max_bytes=max_bytes,
                            max_nesting=max_nesting,
                            deadline_monotonic=deadline_monotonic,
                            allowed_digests=allowed_digests,
                        )
                    finally:
                        os.close(child_fd)
                    continue
                if allowed_digests is not None and path not in allowed_digests:
                    excluded.append(path)
                    continue
                if len(files) >= max_files:
                    raise WorkspaceBackendError("export exceeds the file limit")
                digest, size, content = _read_export_text(
                    child.name,
                    directory_fd,
                    remaining_bytes=max_bytes - totals[0],
                )
                if allowed_digests is not None and digest != allowed_digests[path]:
                    raise WorkspaceBackendError("stale workspace snapshot")
                files.append(
                    ExportFile(path=path, sha256=digest, size=size, content=content)
                )
                totals[0] += size
        finally:
            os.close(scan_fd)

    def _append_manifest_entries(
        self,
        directory_fd: int,
        prefix: str,
        entries: list[SnapshotEntry],
        contents: dict[str, str],
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
                        self._append_manifest_entries(child_fd, path, entries, contents)
                    finally:
                        os.close(child_fd)
                else:
                    content = None
                    if _export_path_is_excluded(path):
                        digest, size = _hash_regular_file(child.name, directory_fd)
                    else:
                        try:
                            digest, size, content = _read_export_text(
                                child.name,
                                directory_fd,
                                remaining_bytes=MAX_ENTRY_SIZE_BYTES,
                            )
                        except WorkspaceBackendError:
                            digest, size = _hash_regular_file(child.name, directory_fd)
                    entries.append(
                        SnapshotEntry(
                            path=path,
                            sha256=digest,
                            size=size,
                            entry_type=entry_type,
                        )
                    )
                    if content is not None:
                        contents[path] = content
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


_EXCLUDED_EXPORT_NAMES = frozenset(
    {
        ".git",
        ".env",
        "opencode.json",
        "opencode.jsonc",
        "auth.json",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)
_EXCLUDED_EXPORT_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def _export_path_is_excluded(path: str) -> bool:
    parts = path.split("/")
    for part in parts:
        lowered = part.lower()
        if part in _EXCLUDED_EXPORT_NAMES or lowered in _EXCLUDED_EXPORT_NAMES:
            return True
        if part.startswith(".env."):
            return True
        if lowered.endswith(_EXCLUDED_EXPORT_SUFFIXES):
            return True
    return False


def select_snapshot_export_files(
    manifest: SnapshotManifest,
    file_contents: Mapping[str, str],
    *,
    max_files: int,
    max_bytes: int,
    max_nesting: int,
) -> ExportBundleSelection:
    """Assemble a CLOUD export from persisted snapshot bytes, never the live tree."""
    if max_files <= 0 or max_bytes <= 0 or max_nesting <= 0:
        raise WorkspaceBackendError("export limits must be positive")
    files: list[ExportFile] = []
    excluded: list[str] = []
    total_bytes = 0
    for entry in sorted(manifest.entries, key=lambda item: item.path):
        if entry.entry_type is not SnapshotEntryType.FILE:
            continue
        if _export_path_is_excluded(entry.path):
            excluded.append(entry.path)
            continue
        if entry.path.count("/") > max_nesting:
            raise WorkspaceBackendError("export nesting exceeds the workspace limit")
        content = file_contents.get(entry.path)
        if content is None:
            raise WorkspaceBackendError("stale workspace snapshot")
        payload = content.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.sha256 or len(payload) != entry.size:
            raise WorkspaceBackendError("stale workspace snapshot")
        if len(files) >= max_files:
            raise WorkspaceBackendError("export exceeds the file limit")
        if total_bytes + len(payload) > max_bytes:
            raise WorkspaceBackendError("export exceeds the byte limit")
        files.append(
            ExportFile(path=entry.path, sha256=digest, size=len(payload), content=content)
        )
        total_bytes += len(payload)
    return ExportBundleSelection(
        files=tuple(files),
        excluded_paths=tuple(excluded),
        total_bytes=total_bytes,
    )


def _read_export_text(
    name: str, parent_fd: int, *, remaining_bytes: int
) -> tuple[str, int, str]:
    if remaining_bytes <= 0:
        raise WorkspaceBackendError("export exceeds the byte limit")
    fd = os.open(name, _OPEN_FLAGS | _NONBLOCK | _NOFOLLOW, dir_fd=parent_fd)
    try:
        _require_supported_entry(os.fstat(fd), allow_directory=False)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(fd, 65_536):
            size += len(chunk)
            if size > remaining_bytes:
                raise WorkspaceBackendError("export exceeds the byte limit")
            digest.update(chunk)
            chunks.append(chunk)
        payload = b"".join(chunks)
        if b"\x00" in payload:
            raise WorkspaceBackendError("export contains unknown binary content")
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WorkspaceBackendError("export contains unknown binary content") from error
        return digest.hexdigest(), size, content
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
