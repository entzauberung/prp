"""Bounded manifest and read-only Git diff views for authorized workspaces."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection
from enum import StrEnum, unique
from pathlib import Path
from typing import Annotated, Protocol, cast

from pydantic import BaseModel, Field, StringConstraints, model_validator

from prp_runtime.domain.enums import ToolCallStatus, ToolEffect
from prp_runtime.domain.models import DomainModel, ErrorCategory, ErrorInfo
from prp_runtime.domain.values import utc_now
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.models import MAX_TOOL_OUTPUT_BYTES, ToolResult
from prp_runtime.tools.registry import ToolDefinition, ToolHandler
from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
)
from prp_runtime.workspace.models import (
    SnapshotEntry,
    SnapshotEntryType,
    SnapshotManifest,
)

__all__ = [
    "DiffBackend",
    "DiffEntry",
    "DiffParseError",
    "DiffResult",
    "DiffStatus",
    "DiffSummary",
    "DiffManifestMismatchError",
    "DiffRequest",
    "DiffToolRunner",
    "GitDiffBackend",
    "MAX_DIFF_PATCH_BYTES",
    "ManifestDiffBackend",
    "MAX_DIFF_ENTRIES",
    "MAX_DIFF_OUTPUT_BYTES",
    "change_set_diff",
    "DeferredDiffRunner",
    "build_diff_definitions",
    "parse_git_name_status",
    "parse_git_status",
]

MAX_DIFF_ENTRIES = 10_000
MAX_DIFF_OUTPUT_BYTES = 512 * 1024
MAX_DIFF_PATCH_BYTES = 128 * 1024
_PATH_RE = re.compile(r"^[^/\\][^\\]*$")
RelativeDiffPath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
]


@unique
class DiffStatus(StrEnum):
    """Stable status vocabulary for manifest and Git views."""

    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"
    RENAMED = "RENAMED"
    UNTRACKED = "UNTRACKED"


class DiffEntry(DomainModel):
    """One bounded path change without a host path or raw Git record."""

    path: RelativeDiffPath
    status: DiffStatus
    old_path: RelativeDiffPath | None = None
    before: FileContent | None = None
    after: FileContent | None = None

    @model_validator(mode="after")
    def _paths_and_rename_are_consistent(self) -> DiffEntry:
        _validate_relative_path(self.path)
        if self.old_path is not None:
            _validate_relative_path(self.old_path)
        if self.status is DiffStatus.RENAMED:
            if self.old_path is None or self.old_path == self.path:
                raise ValueError("rename diff requires two distinct relative paths")
        elif self.old_path is not None:
            raise ValueError("only a rename diff may carry old_path")
        if self.status is DiffStatus.UNTRACKED and self.before is not None:
            raise ValueError("untracked diff must not carry base content")
        return self


class DiffSummary(DomainModel):
    """Deterministic counts for a bounded diff result."""

    added: int = Field(ge=0)
    modified: int = Field(ge=0)
    deleted: int = Field(ge=0)
    renamed: int = Field(ge=0)
    untracked: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.added + self.modified + self.deleted + self.renamed + self.untracked


class DiffResult(DomainModel):
    """Bounded, stable diff entries and optional snapshot identity facts."""

    entries: tuple[DiffEntry, ...] = ()
    truncated: bool = False
    base_manifest_hash: str | None = None
    new_manifest_hash: str | None = None
    patch: str = Field(default="", max_length=MAX_DIFF_PATCH_BYTES)
    patch_sha256: str | None = None
    patch_truncated: bool = False

    @property
    def summary(self) -> DiffSummary:
        counts = {status: 0 for status in DiffStatus}
        for entry in self.entries:
            counts[entry.status] += 1
        return DiffSummary(
            added=counts[DiffStatus.ADDED],
            modified=counts[DiffStatus.MODIFIED],
            deleted=counts[DiffStatus.DELETED],
            renamed=counts[DiffStatus.RENAMED],
            untracked=counts[DiffStatus.UNTRACKED],
        )


class DiffBackend(Protocol):
    """Common comparison contract for future read-only diff tool wiring."""

    def compare(self, base: SnapshotManifest, new: SnapshotManifest) -> DiffResult:
        """Compare two server-owned manifests without executing a command."""


class ManifestDiffBackend:
    """Compare immutable manifests and infer deterministic content renames."""

    def __init__(self, *, max_entries: int = MAX_DIFF_ENTRIES) -> None:
        if max_entries <= 0 or max_entries > MAX_DIFF_ENTRIES:
            raise ValueError("max_entries exceeds the diff limit")
        self._max_entries = max_entries

    def compare(self, base: SnapshotManifest, new: SnapshotManifest) -> DiffResult:
        base_files = _file_entries(base.entries)
        new_files = _file_entries(new.entries)
        entries: list[DiffEntry] = []
        deleted = set(base_files) - set(new_files)
        added = set(new_files) - set(base_files)

        deleted_by_content = _group_by_content(base_files, deleted)
        added_by_content = _group_by_content(new_files, added)
        for content_key in sorted(set(deleted_by_content) & set(added_by_content)):
            for old_path, new_path in zip(
                sorted(deleted_by_content[content_key]), sorted(added_by_content[content_key])
            ):
                entries.append(
                    DiffEntry(
                        path=new_path,
                        old_path=old_path,
                        status=DiffStatus.RENAMED,
                        before=_file_content(base_files[old_path]),
                        after=_file_content(new_files[new_path]),
                    )
                )
                deleted.remove(old_path)
                added.remove(new_path)

        for path in sorted(set(base_files) & set(new_files)):
            before = _file_content(base_files[path])
            after = _file_content(new_files[path])
            if before != after:
                entries.append(
                    DiffEntry(path=path, status=DiffStatus.MODIFIED, before=before, after=after)
                )
        entries.extend(
            DiffEntry(path=path, status=DiffStatus.DELETED, before=_file_content(base_files[path]))
            for path in sorted(deleted)
        )
        entries.extend(
            DiffEntry(path=path, status=DiffStatus.ADDED, after=_file_content(new_files[path]))
            for path in sorted(added)
        )
        entries.sort(key=lambda entry: (entry.path, entry.old_path or "", entry.status.value))
        return DiffResult(
            entries=tuple(entries[: self._max_entries]),
            truncated=len(entries) > self._max_entries,
            base_manifest_hash=base.manifest_hash,
            new_manifest_hash=new.manifest_hash,
        )


def change_set_diff(change_set: ChangeSet, *, max_entries: int = MAX_DIFF_ENTRIES) -> DiffResult:
    """Project one persisted ChangeSet into the same bounded diff vocabulary."""
    if max_entries <= 0 or max_entries > MAX_DIFF_ENTRIES:
        raise ValueError("max_entries exceeds the diff limit")
    entries = tuple(
        _diff_entry_from_file_change(file_change) for file_change in change_set.files
    )
    entries = tuple(sorted(entries, key=lambda entry: (entry.path, entry.status.value)))
    patch, patch_truncated = _bound_patch(change_set.patch.unified_diff)
    return DiffResult(
        entries=entries[:max_entries],
        truncated=len(entries) > max_entries,
        patch=patch,
        patch_sha256=change_set.patch_sha256,
        patch_truncated=patch_truncated,
    )


class DiffRequest(DomainModel):
    """Empty, closed request: manifests and ChangeSets are server-owned facts."""


class DiffManifestMismatchError(RuntimeError):
    """The current workspace does not match the expected ChangeSet result."""


class DeferredDiffRunner:
    """Keep get_diff/get_status in the catalog until a local patch binds."""

    def __init__(
        self,
        base_manifest: SnapshotManifest,
        manifest_provider: Callable[[], SnapshotManifest],
    ) -> None:
        self._base_manifest = base_manifest
        self._manifest_provider = manifest_provider
        self._runner: DiffToolRunner | None = None

    def bind(self, change_set: ChangeSet) -> None:
        if self._runner is not None:
            return
        self._runner = DiffToolRunner(
            change_set,
            base_manifest=self._base_manifest,
            manifest_provider=self._manifest_provider,
        )

    def get_diff(self) -> DiffResult:
        if self._runner is None:
            raise DiffManifestMismatchError("diff is unavailable before a successful patch")
        return self._runner.get_diff()

    def get_status(self) -> DiffResult:
        if self._runner is None:
            raise DiffManifestMismatchError("status is unavailable before a successful patch")
        return self._runner.get_status()


class DiffToolRunner:
    """Verify one ChangeSet against its current manifest before exposing a view."""

    def __init__(
        self,
        change_set: ChangeSet,
        *,
        base_manifest: SnapshotManifest,
        manifest_provider: Callable[[], SnapshotManifest],
        max_entries: int = 256,
    ) -> None:
        if max_entries <= 0 or max_entries > MAX_DIFF_ENTRIES:
            raise ValueError("max_entries exceeds the diff limit")
        self._change_set = change_set
        self._base_manifest = base_manifest
        self._manifest_provider = manifest_provider
        self._max_entries = max_entries

    def get_diff(self) -> DiffResult:
        current = self._verified_manifest()
        result = change_set_diff(self._change_set, max_entries=self._max_entries)
        return result.model_copy(
            update={
                "base_manifest_hash": self._base_manifest.manifest_hash,
                "new_manifest_hash": current.manifest_hash,
            }
        )

    def get_status(self) -> DiffResult:
        result = self.get_diff()
        return result.model_copy(
            update={"patch": "", "patch_sha256": None, "patch_truncated": False}
        )

    def _verified_manifest(self) -> SnapshotManifest:
        expected = _manifest_after_changes(self._base_manifest, self._change_set.files)
        current = self._manifest_provider()
        if current.manifest_hash != expected.manifest_hash:
            raise DiffManifestMismatchError("workspace manifest does not match ChangeSet")
        return current


def build_diff_definitions(runner: DiffToolRunner) -> tuple[ToolDefinition, ToolDefinition]:
    """Build read-only get_diff and get_status handlers for one server-owned run."""

    async def diff_handler(context: BaseModel) -> ToolResult:
        return await _run_diff_handler(context, runner.get_diff)

    async def status_handler(context: BaseModel) -> ToolResult:
        return await _run_diff_handler(context, runner.get_status)

    return (
        ToolDefinition(
            name="get_diff",
            description="Inspect the bounded diff for the authorized ChangeSet.",
            effect=ToolEffect.READ,
            argument_model=DiffRequest,
            handler=cast(ToolHandler, diff_handler),
        ),
        ToolDefinition(
            name="get_status",
            description="Inspect the current status of the authorized ChangeSet.",
            effect=ToolEffect.READ,
            argument_model=DiffRequest,
            handler=cast(ToolHandler, status_handler),
        ),
    )


async def _run_diff_handler(
    context: BaseModel, operation: Callable[[], DiffResult]
) -> ToolResult:
    if not isinstance(context, ExecutionContext):
        raise TypeError("diff tools require an execution context")
    if not isinstance(context.arguments, DiffRequest):
        raise TypeError("diff tool received invalid arguments")
    try:
        result = operation()
    except DiffManifestMismatchError:
        return ToolResult.from_call(
            context.call,
            status=ToolCallStatus.FAILED,
            error=ErrorInfo(
                category=ErrorCategory.INTERNAL,
                message="workspace manifest does not match ChangeSet",
            ),
            completed_at=utc_now(),
        )
    payload = result.model_dump(mode="json")
    output = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    if len(output.encode("utf-8")) > MAX_TOOL_OUTPUT_BYTES:
        summary_payload = {
            "summary": result.summary.model_dump(mode="json"),
            "truncated": True,
            "base_manifest_hash": result.base_manifest_hash,
            "new_manifest_hash": result.new_manifest_hash,
            "patch_sha256": result.patch_sha256,
            "patch_truncated": result.patch_truncated,
        }
        payload = summary_payload
        output = json.dumps(
            summary_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    return ToolResult.from_call(
        context.call,
        status=ToolCallStatus.SUCCEEDED,
        result=payload,
        output=output,
        truncated=False,
        completed_at=utc_now(),
    )


class GitDiffBackend:
    """Build read-only Git argv and parse bounded name/status output."""

    def __init__(
        self, repository_root: Path, *, max_output_bytes: int = MAX_DIFF_OUTPUT_BYTES
    ) -> None:
        try:
            root_stat = repository_root.lstat()
        except OSError as error:
            raise DiffParseError("Git repository root is unavailable") from error
        if repository_root.is_symlink() or not repository_root.is_dir():
            raise DiffParseError("Git repository root is not a directory")
        del root_stat
        if max_output_bytes <= 0 or max_output_bytes > MAX_DIFF_OUTPUT_BYTES:
            raise ValueError("max_output_bytes exceeds the diff limit")
        self._repository_root = repository_root
        self._max_output_bytes = max_output_bytes

    def diff_argv(self) -> tuple[str, ...]:
        """Return a Git diff command containing no write operation."""
        return (
            "git",
            "-C",
            str(self._repository_root),
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--find-renames",
            "--name-status",
            "-z",
            "--",
        )

    def status_argv(self) -> tuple[str, ...]:
        """Return a Git status command containing no write operation."""
        return (
            "git",
            "-C",
            str(self._repository_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
            "--",
        )

    def parse_diff(self, output: bytes) -> DiffResult:
        return parse_git_name_status(output, max_output_bytes=self._max_output_bytes)

    def parse_status(self, output: bytes) -> DiffResult:
        return parse_git_status(output, max_output_bytes=self._max_output_bytes)


class DiffParseError(ValueError):
    """Git or manifest diff data is malformed or exceeds its public bound."""


def parse_git_name_status(
    output: bytes, *, max_output_bytes: int = MAX_DIFF_OUTPUT_BYTES
) -> DiffResult:
    """Parse ``git diff --name-status -z`` without retaining raw output."""
    _ensure_output_bound(output, max_output_bytes)
    records = output.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    entries: list[DiffEntry] = []
    index = 0
    while index < len(records):
        status = _decode(records[index], "Git status")
        index += 1
        if status.startswith("R") and status[1:].isdigit():
            old_path, new_path, index = _read_rename_records(records, index)
            entries.append(DiffEntry(path=new_path, old_path=old_path, status=DiffStatus.RENAMED))
            continue
        if status not in {"A", "M", "D"}:
            raise DiffParseError("unsupported Git diff status")
        if index >= len(records):
            raise DiffParseError("Git diff record has no path")
        path = _decode(records[index], "Git path")
        index += 1
        entries.append(
            DiffEntry(
                path=path,
                status={
                    "A": DiffStatus.ADDED,
                    "M": DiffStatus.MODIFIED,
                    "D": DiffStatus.DELETED,
                }[status],
            )
        )
    return DiffResult(entries=tuple(entries))


def parse_git_status(
    output: bytes, *, max_output_bytes: int = MAX_DIFF_OUTPUT_BYTES
) -> DiffResult:
    """Parse ``git status --porcelain=v1 -z`` into safe relative statuses."""
    _ensure_output_bound(output, max_output_bytes)
    records = output.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    entries: list[DiffEntry] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 3 or record[2:3] != b" ":
            raise DiffParseError("Git status record is malformed")
        code = record[:2].decode("ascii", errors="strict")
        path = _decode(record[3:], "Git path")
        if code == "??":
            entries.append(DiffEntry(path=path, status=DiffStatus.UNTRACKED))
        elif "R" in code:
            if index >= len(records):
                raise DiffParseError("Git rename status has no new path")
            new_path = _decode(records[index], "Git path")
            index += 1
            entries.append(DiffEntry(path=new_path, old_path=path, status=DiffStatus.RENAMED))
        elif "A" in code:
            entries.append(DiffEntry(path=path, status=DiffStatus.ADDED))
        elif "M" in code:
            entries.append(DiffEntry(path=path, status=DiffStatus.MODIFIED))
        elif "D" in code:
            entries.append(DiffEntry(path=path, status=DiffStatus.DELETED))
        else:
            raise DiffParseError("unsupported Git status code")
    return DiffResult(entries=tuple(entries))


def _diff_entry_from_file_change(change: FileChange) -> DiffEntry:
    status = {
        FileChangeAction.ADD: DiffStatus.ADDED,
        FileChangeAction.MODIFY: DiffStatus.MODIFIED,
        FileChangeAction.DELETE: DiffStatus.DELETED,
    }[change.action]
    return DiffEntry(path=change.path, status=status, before=change.before, after=change.after)


def _manifest_after_changes(
    base_manifest: SnapshotManifest, file_changes: tuple[FileChange, ...]
) -> SnapshotManifest:
    entries = {entry.path: entry for entry in base_manifest.entries}
    for change in file_changes:
        current = entries.get(change.path)
        if change.action is FileChangeAction.ADD:
            if current is not None or change.after is None:
                raise DiffManifestMismatchError("ChangeSet add does not match base manifest")
            after = change.after
        elif change.action is FileChangeAction.MODIFY:
            if current is None or current.entry_type is not SnapshotEntryType.FILE:
                raise DiffManifestMismatchError("ChangeSet modify does not match base manifest")
            if change.before != _file_content(current) or change.after is None:
                raise DiffManifestMismatchError("ChangeSet modify content does not match base")
            after = change.after
        else:
            if current is None or current.entry_type is not SnapshotEntryType.FILE:
                raise DiffManifestMismatchError("ChangeSet delete does not match base manifest")
            if change.before != _file_content(current) or change.after is not None:
                raise DiffManifestMismatchError("ChangeSet delete content does not match base")
            entries.pop(change.path)
            continue
        entries[change.path] = SnapshotEntry(
            path=change.path,
            sha256=after.sha256,
            size=after.size,
            entry_type=SnapshotEntryType.FILE,
        )
    return SnapshotManifest(entries=tuple(entries.values()))


def _bound_patch(value: str, max_bytes: int = MAX_DIFF_PATCH_BYTES) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _file_entries(entries: Collection[SnapshotEntry]) -> dict[str, SnapshotEntry]:
    return {entry.path: entry for entry in entries if entry.entry_type is SnapshotEntryType.FILE}


def _group_by_content(
    entries: dict[str, SnapshotEntry], paths: set[str]
) -> dict[tuple[str, int], list[str]]:
    grouped: dict[tuple[str, int], list[str]] = {}
    for path in paths:
        entry = entries[path]
        grouped.setdefault((entry.sha256, entry.size), []).append(path)
    return grouped


def _file_content(entry: SnapshotEntry) -> FileContent:
    return FileContent(sha256=entry.sha256, size=entry.size)


def _read_rename_records(records: list[bytes], index: int) -> tuple[str, str, int]:
    if index + 1 >= len(records):
        raise DiffParseError("Git rename record is incomplete")
    old_path = _decode(records[index], "Git old path")
    new_path = _decode(records[index + 1], "Git new path")
    return old_path, new_path, index + 2


def _decode(value: bytes, label: str) -> str:
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DiffParseError(f"{label} is not valid UTF-8") from error
    if not decoded:
        raise DiffParseError(f"{label} is empty")
    return decoded


def _validate_relative_path(path: str) -> None:
    if (
        _PATH_RE.fullmatch(path) is None
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("diff path must be workspace-relative")


def _ensure_output_bound(output: bytes, max_output_bytes: int) -> None:
    if max_output_bytes <= 0 or max_output_bytes > MAX_DIFF_OUTPUT_BYTES:
        raise ValueError("max_output_bytes exceeds the diff limit")
    if len(output) > max_output_bytes:
        raise DiffParseError("Git diff output exceeds the size limit")
