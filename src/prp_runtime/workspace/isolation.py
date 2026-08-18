"""Immutable snapshot and isolated execution-slot lifecycle.

This backend uses bounded temporary copies. It never shares a mutable source
directory between slots and keeps host paths out of public lifecycle facts.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta
from enum import StrEnum, unique
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import ConfigDict, Field

from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.values import new_snapshot_id, utc_now, validate_snapshot_id
from prp_runtime.workspace.changes import (
    MAX_FILE_CHANGES,
    MAX_PATCH_BYTES,
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
    new_change_set_id,
)

__all__ = [
    "BaseSnapshot",
    "ExecutionSlot",
    "IsolationBackend",
    "IsolationCapacityError",
    "IsolationError",
    "IsolationLeaseError",
    "IsolationOwnershipError",
    "LocalIsolationBackend",
    "SlotContext",
    "SlotStatus",
]

_DEFAULT_MAX_SLOTS = 8
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024
_SlotResult = TypeVar("_SlotResult")


class IsolationError(RuntimeError):
    """Base error for isolated workspace lifecycle failures."""


class IsolationCapacityError(IsolationError):
    """The configured slot or byte capacity would be exceeded."""


class IsolationOwnershipError(IsolationError):
    """A caller does not own the requested execution slot."""


class IsolationLeaseError(IsolationError):
    """A slot lease is expired or otherwise unavailable."""


@unique
class SlotStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PROMOTED = "PROMOTED"
    CLEANED = "CLEANED"


class BaseSnapshot(DomainModel):
    """Public immutable snapshot facts; no filesystem path is retained."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str = Field(pattern=r"^snap_[A-Za-z0-9][A-Za-z0-9_-]{3,127}$")
    workspace_id: str = Field(min_length=1, max_length=128)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_bytes: int = Field(ge=0)
    file_count: int = Field(ge=0)
    created_at: datetime


class ExecutionSlot(DomainModel):
    """One mutable copy bound to a base hash, work unit and lease owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str = Field(pattern=r"^slot_[A-Za-z0-9][A-Za-z0-9_-]{3,127}$")
    workspace_id: str = Field(min_length=1, max_length=128)
    work_unit_id: str = Field(min_length=1, max_length=128)
    base_snapshot_id: str = Field(pattern=r"^snap_[A-Za-z0-9][A-Za-z0-9_-]{3,127}$")
    base_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    owner_id: str = Field(min_length=1, max_length=128)
    lease_expires_at: datetime
    copied_bytes: int = Field(ge=0)
    status: SlotStatus = SlotStatus.ACTIVE


class IsolationBackend(Protocol):
    """Lifecycle operations required by isolated WorkUnit execution."""

    def create_base_snapshot(self, source: Path, workspace_id: str) -> BaseSnapshot:
        """Copy a source directory into an immutable base snapshot."""

    def create_slot(
        self,
        snapshot_id: str,
        work_unit_id: str,
        owner_id: str,
        *,
        lease_seconds: int = 300,
    ) -> ExecutionSlot:
        """Create one private mutable slot from a base snapshot."""

    def cleanup(self, slot_id: str, *, owner_id: str | None = None) -> bool:
        """Release a slot; repeating cleanup is safe."""

    def get_slot(self, slot_id: str, *, owner_id: str | None = None) -> ExecutionSlot:
        """Return slot facts after owner and active-lease validation."""

    def slot_path(self, slot_id: str, *, owner_id: str | None = None) -> Path:
        """Return the private path for internal execution only."""

    def promote_changes(
        self,
        slot_id: str,
        *,
        owner_id: str,
        run_id: str,
        tool_call_id: str,
        work_unit_id: str | None = None,
    ) -> ChangeSet | None:
        """Promote an owned slot and return its tool-bound ChangeSet facts."""


class SlotContext:
    """Private lifecycle handle for one temporary execution slot.

    The host path is intentionally available only through this internal
    context. Public slot facts remain the immutable ``ExecutionSlot`` model,
    while acquire/cleanup enforce the owner and active-lease checks at every
    use.
    """

    __slots__ = (
        "_backend",
        "_snapshot_id",
        "_work_unit_id",
        "_owner_id",
        "_lease_seconds",
        "_slot",
        "_closed",
    )

    def __init__(
        self,
        backend: IsolationBackend,
        *,
        snapshot_id: str,
        work_unit_id: str,
        owner_id: str,
        lease_seconds: int = 300,
    ) -> None:
        if not snapshot_id or not work_unit_id or not owner_id:
            raise ValueError("slot context identities are required")
        if lease_seconds <= 0:
            raise ValueError("slot lease_seconds must be positive")
        self._backend = backend
        self._snapshot_id = snapshot_id
        self._work_unit_id = work_unit_id
        self._owner_id = owner_id
        self._lease_seconds = lease_seconds
        self._slot: ExecutionSlot | None = None
        self._closed = False

    @property
    def slot(self) -> ExecutionSlot:
        """Return the acquired public slot facts after lease validation."""
        if self._slot is None:
            raise IsolationError("execution slot has not been acquired")
        return self._slot

    @property
    def path(self) -> Path:
        """Return the private slot path for internal tool composition only."""
        slot = self.slot
        return self._backend.slot_path(slot.slot_id, owner_id=self._owner_id)

    def promote_changes(
        self,
        *,
        run_id: str,
        tool_call_id: str,
    ) -> ChangeSet | None:
        """Promote this slot into a ChangeSet before the context is cleaned."""
        slot = self.slot
        return self._backend.promote_changes(
            slot.slot_id,
            owner_id=self._owner_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            work_unit_id=slot.work_unit_id,
        )

    def acquire(self) -> ExecutionSlot:
        """Create or revalidate the owned slot and its active lease."""
        if self._closed:
            raise IsolationError("slot context is closed")
        if self._slot is None:
            slot = self._backend.create_slot(
                self._snapshot_id,
                self._work_unit_id,
                self._owner_id,
                lease_seconds=self._lease_seconds,
            )
        else:
            slot = self._backend.get_slot(self._slot.slot_id, owner_id=self._owner_id)
        if (
            slot.owner_id != self._owner_id
            or slot.work_unit_id != self._work_unit_id
            or slot.base_snapshot_id != self._snapshot_id
        ):
            raise IsolationError("execution slot identity mismatch")
        self._slot = slot
        return slot

    def cleanup(self) -> bool:
        """Release the owned slot exactly once, including after failures."""
        if self._closed:
            return True
        if self._slot is None:
            self._closed = True
            return False
        result = self._backend.cleanup(self._slot.slot_id, owner_id=self._owner_id)
        self._closed = True
        return result

    close = cleanup

    async def run(
        self,
        operation: Callable[[SlotContext], Awaitable[_SlotResult]],
    ) -> _SlotResult:
        """Acquire, run one operation, and always release the slot."""
        self.acquire()
        try:
            return await operation(self)
        finally:
            self.cleanup()

    def __enter__(self) -> SlotContext:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()


class _SnapshotRecord:
    def __init__(self, facts: BaseSnapshot, path: Path) -> None:
        self.facts = facts
        self.path = path


class _SlotRecord:
    def __init__(self, facts: ExecutionSlot, path: Path) -> None:
        self.facts = facts
        self.path = path


def _reject_symlinks(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            candidate = current_path / name
            if candidate.is_symlink():
                raise IsolationError("workspace snapshots do not accept symbolic links")


def _tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    entries: list[Path] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        entries.extend(current_path / name for name in directories)
        entries.extend(current_path / name for name in files)
    for path in sorted(entries, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_dir():
            digest.update(b"D\0" + relative + b"\0")
            continue
        if not path.is_file():
            raise IsolationError("workspace snapshot contains an unsupported entry")
        size = path.stat().st_size
        digest.update(b"F\0" + relative + b"\0" + str(size).encode("ascii") + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
        total_bytes += size
        file_count += 1
    return digest.hexdigest(), total_bytes, file_count


def _file_facts(root: Path) -> dict[str, tuple[str, int]]:
    facts: dict[str, tuple[str, int]] = {}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise IsolationError("workspace changes contain an unsupported entry")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(_CHUNK_SIZE):
                    digest.update(chunk)
            facts[path.relative_to(root).as_posix()] = (digest.hexdigest(), path.stat().st_size)
    return facts


def _unified_diff(base: Path, current: Path, paths: Sequence[str]) -> str:
    chunks: list[str] = []
    for relative in paths:
        before = base / relative
        after = current / relative
        try:
            before_text = before.read_text(encoding="utf-8") if before.is_file() else ""
            after_text = after.read_text(encoding="utf-8") if after.is_file() else ""
            chunks.extend(
                difflib.unified_diff(
                    before_text.splitlines(keepends=True),
                    after_text.splitlines(keepends=True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        except UnicodeDecodeError:
            chunks.append(f"Binary files a/{relative} and b/{relative} differ\n")
    return "".join(chunks) or "No textual changes\n"


def _set_read_only(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            (Path(current) / name).chmod(0o444)
        for name in directories:
            (Path(current) / name).chmod(0o555)
    root.chmod(0o555)


def _set_slot_writable(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            (Path(current) / name).chmod(0o644)
        for name in directories:
            (Path(current) / name).chmod(0o755)
    root.chmod(0o755)


def _ensure_directory(path: Path) -> None:
    try:
        stat_result = path.lstat()
    except OSError as error:
        raise IsolationError("isolation directory is unavailable") from error
    if path.is_symlink() or not path.is_dir() or stat_result.st_ino == 0:
        raise IsolationError("isolation directory is not a safe directory")


def _reject_storage_overlap(source: Path, storage_root: Path) -> None:
    source_root = source.resolve()
    isolation_root = storage_root.resolve()
    try:
        source_root.relative_to(isolation_root)
    except ValueError:
        return
    raise IsolationError("snapshot source must be outside isolation storage")


class LocalIsolationBackend:
    """Bounded temporary-copy backend for isolated WorkUnit slots."""

    def __init__(
        self,
        storage_root: Path,
        *,
        max_slots: int = _DEFAULT_MAX_SLOTS,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        if max_slots <= 0 or max_bytes <= 0:
            raise ValueError("isolation capacity must be positive")
        self._storage_root = storage_root
        self._max_slots = max_slots
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._snapshots: dict[str, _SnapshotRecord] = {}
        self._slots: dict[str, _SlotRecord] = {}
        storage_root.mkdir(parents=True, exist_ok=True)
        _ensure_directory(storage_root)
        self._base_root = storage_root / "bases"
        self._slot_root = storage_root / "slots"
        self._base_root.mkdir(exist_ok=True)
        self._slot_root.mkdir(exist_ok=True)

    @property
    def active_slot_count(self) -> int:
        return sum(record.facts.status is SlotStatus.ACTIVE for record in self._slots.values())

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def create_base_snapshot(self, source: Path, workspace_id: str) -> BaseSnapshot:
        _ensure_directory(source)
        _reject_storage_overlap(source, self._storage_root)
        _reject_symlinks(source)
        source_hash, source_bytes, source_files = _tree_digest(source)
        if self._used_bytes + source_bytes > self._max_bytes:
            raise IsolationCapacityError("isolation byte capacity exceeded")
        snapshot_id = new_snapshot_id()
        target = self._base_root / snapshot_id
        try:
            shutil.copytree(source, target, symlinks=False)
            copied_hash, copied_bytes, copied_files = _tree_digest(target)
            if (copied_hash, copied_bytes, copied_files) != (
                source_hash,
                source_bytes,
                source_files,
            ):
                raise IsolationError("source changed while creating immutable snapshot")
            _set_read_only(target)
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise
        facts = BaseSnapshot(
            snapshot_id=snapshot_id,
            workspace_id=workspace_id,
            content_hash=source_hash,
            total_bytes=source_bytes,
            file_count=source_files,
            created_at=utc_now(),
        )
        self._snapshots[snapshot_id] = _SnapshotRecord(facts, target)
        self._used_bytes += source_bytes
        return facts

    def create_slot(
        self,
        snapshot_id: str,
        work_unit_id: str,
        owner_id: str,
        *,
        lease_seconds: int = 300,
    ) -> ExecutionSlot:
        if lease_seconds <= 0:
            raise ValueError("slot lease_seconds must be positive")
        if self.active_slot_count >= self._max_slots:
            raise IsolationCapacityError("isolation slot capacity exceeded")
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            raise IsolationError("base snapshot is unavailable")
        if self._used_bytes + snapshot.facts.total_bytes > self._max_bytes:
            raise IsolationCapacityError("isolation byte capacity exceeded")
        slot_id = f"slot_{secrets.token_hex(12)}"
        target = self._slot_root / slot_id
        try:
            shutil.copytree(snapshot.path, target, symlinks=False)
            _set_slot_writable(target)
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise
        facts = ExecutionSlot(
            slot_id=slot_id,
            workspace_id=snapshot.facts.workspace_id,
            work_unit_id=work_unit_id,
            base_snapshot_id=snapshot_id,
            base_hash=snapshot.facts.content_hash,
            owner_id=owner_id,
            lease_expires_at=utc_now() + timedelta(seconds=lease_seconds),
            copied_bytes=snapshot.facts.total_bytes,
        )
        self._slots[slot_id] = _SlotRecord(facts, target)
        self._used_bytes += snapshot.facts.total_bytes
        return facts

    def get_slot(self, slot_id: str, *, owner_id: str | None = None) -> ExecutionSlot:
        record = self._slots.get(slot_id)
        if record is None:
            raise IsolationError("execution slot is unavailable")
        self._check_owner(record.facts, owner_id)
        if record.facts.status is SlotStatus.ACTIVE:
            self._check_lease(record.facts)
        return record.facts

    def slot_path(self, slot_id: str, *, owner_id: str | None = None) -> Path:
        """Return a private path for an executor; it is not part of slot facts."""
        record = self._slots.get(slot_id)
        if record is None:
            raise IsolationError("execution slot is unavailable")
        self._check_owner(record.facts, owner_id)
        self._check_lease(record.facts)
        return record.path

    def promote(
        self,
        slot_id: str,
        *,
        owner_id: str,
        new_snapshot_id_value: str | None = None,
    ) -> BaseSnapshot:
        record = self._slots.get(slot_id)
        if record is None:
            raise IsolationError("execution slot is unavailable")
        self._check_owner(record.facts, owner_id)
        self._check_lease(record.facts)
        if record.facts.status is not SlotStatus.ACTIVE:
            raise IsolationError("execution slot is not active")
        _reject_symlinks(record.path)
        digest, total_bytes, file_count = _tree_digest(record.path)
        if self._used_bytes + total_bytes > self._max_bytes:
            raise IsolationCapacityError("isolation byte capacity exceeded")
        snapshot_id = (
            new_snapshot_id()
            if new_snapshot_id_value is None
            else validate_snapshot_id(new_snapshot_id_value)
        )
        target = self._base_root / snapshot_id
        if target.exists() or target.is_symlink():
            raise IsolationError("snapshot id already exists")
        temporary: Path | None = None
        try:
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{snapshot_id}.snapshot-", dir=self._base_root)
            )
            shutil.copytree(record.path, temporary, symlinks=False, dirs_exist_ok=True)
            copied_digest, copied_bytes, copied_files = _tree_digest(temporary)
            if (copied_digest, copied_bytes, copied_files) != (
                digest,
                total_bytes,
                file_count,
            ):
                raise IsolationError("slot changed while creating promoted snapshot")
            _set_read_only(temporary)
            os.replace(temporary, target)
            temporary = None
        except OSError as error:
            raise IsolationError("atomic snapshot promotion failed") from error
        except BaseException:
            raise
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
        facts = BaseSnapshot(
            snapshot_id=snapshot_id,
            workspace_id=record.facts.workspace_id,
            content_hash=digest,
            total_bytes=total_bytes,
            file_count=file_count,
            created_at=utc_now(),
        )
        self._snapshots[snapshot_id] = _SnapshotRecord(facts, target)
        self._used_bytes += total_bytes
        record.facts = record.facts.model_copy(update={"status": SlotStatus.PROMOTED})
        return facts

    def promote_changes(
        self,
        slot_id: str,
        *,
        owner_id: str,
        run_id: str,
        tool_call_id: str,
        work_unit_id: str | None = None,
    ) -> ChangeSet | None:
        """Promote slot content and return its bounded ChangeSet fact."""
        record = self._slots.get(slot_id)
        if record is None:
            raise IsolationError("execution slot is unavailable")
        self._check_owner(record.facts, owner_id)
        self._check_lease(record.facts)
        if work_unit_id is not None and work_unit_id != record.facts.work_unit_id:
            raise IsolationOwnershipError("execution slot work unit mismatch")
        base = self._snapshots.get(record.facts.base_snapshot_id)
        if base is None:
            raise IsolationError("base snapshot is unavailable")
        _reject_symlinks(record.path)
        before = _file_facts(base.path)
        after = _file_facts(record.path)
        changed = sorted(
            path for path in set(before) | set(after) if before.get(path) != after.get(path)
        )
        if not changed:
            self.cleanup(slot_id, owner_id=owner_id)
            return None
        files: list[FileChange] = []
        for path in changed:
            before_fact = before.get(path)
            after_fact = after.get(path)
            files.append(
                FileChange(
                    path=path,
                    action=(
                        FileChangeAction.ADD
                        if before_fact is None
                        else FileChangeAction.DELETE
                        if after_fact is None
                        else FileChangeAction.MODIFY
                    ),
                    before=(
                        None
                        if before_fact is None
                        else FileContent(sha256=before_fact[0], size=before_fact[1])
                    ),
                    after=(
                        None
                        if after_fact is None
                        else FileContent(sha256=after_fact[0], size=after_fact[1])
                    ),
                )
            )
        unified_diff = _unified_diff(base.path, record.path, changed)
        if len(files) > MAX_FILE_CHANGES:
            raise IsolationError("slot ChangeSet contains too many file changes")
        if len(unified_diff.encode("utf-8")) > MAX_PATCH_BYTES:
            raise IsolationError("slot ChangeSet diff exceeds the byte limit")
        change_set = ChangeSet(
            change_set_id=new_change_set_id(),
            run_id=run_id,
            tool_call_id=tool_call_id,
            workspace_id=record.facts.workspace_id,
            base_snapshot_id=record.facts.base_snapshot_id,
            new_snapshot_id=new_snapshot_id(),
            patch=Patch(
                base_snapshot_id=record.facts.base_snapshot_id,
                unified_diff=unified_diff,
            ),
            files=tuple(files),
            created_at=utc_now(),
        )
        self.promote(
            slot_id,
            owner_id=owner_id,
            new_snapshot_id_value=change_set.new_snapshot_id,
        )
        return change_set

    def cleanup(self, slot_id: str, *, owner_id: str | None = None) -> bool:
        record = self._slots.get(slot_id)
        if record is None:
            return False
        self._check_owner(record.facts, owner_id)
        if record.facts.status is SlotStatus.CLEANED:
            return True
        shutil.rmtree(record.path, ignore_errors=True)
        self._used_bytes = max(0, self._used_bytes - record.facts.copied_bytes)
        record.facts = record.facts.model_copy(update={"status": SlotStatus.CLEANED})
        return True

    def close(self) -> None:
        for slot_id in tuple(self._slots):
            self.cleanup(slot_id)
        shutil.rmtree(self._storage_root, ignore_errors=True)

    @staticmethod
    def _check_owner(slot: ExecutionSlot, owner_id: str | None) -> None:
        if owner_id is not None and slot.owner_id != owner_id:
            raise IsolationOwnershipError("execution slot owner mismatch")

    @staticmethod
    def _check_lease(slot: ExecutionSlot) -> None:
        if slot.status is not SlotStatus.ACTIVE:
            raise IsolationLeaseError("execution slot lease is not active")
        if slot.lease_expires_at <= utc_now():
            raise IsolationLeaseError("execution slot lease has expired")
