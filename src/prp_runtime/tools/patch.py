"""Bounded unified-patch application with reversible workspace promotion."""

from __future__ import annotations

import hashlib
import re
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel

from prp_runtime.domain.enums import ToolCallStatus, ToolEffect
from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.values import SnapshotId, UtcTimestamp, utc_now
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.models import ToolCall, ToolResult
from prp_runtime.tools.registry import ToolDefinition, ToolHandler
from prp_runtime.workspace.backend import WorkspaceBackend, WorkspaceBackendError
from prp_runtime.workspace.changes import (
    MAX_CHANGED_FILE_BYTES,
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
    new_change_set_id,
)
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotEntry,
    SnapshotEntryType,
    SnapshotManifest,
    SnapshotStatus,
)

__all__ = [
    "PatchApplyError",
    "PatchApplyResult",
    "PatchRequest",
    "PatchRunner",
    "PatchStaleError",
    "PatchValidationError",
    "build_patch_definition",
]

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*?)?(?:\n)?$"
)


class PatchApplyError(RuntimeError):
    """A bounded patch cannot be applied without risking an audit gap."""


class PatchStaleError(PatchApplyError):
    """The current workspace content no longer matches the declared base."""


class PatchValidationError(PatchApplyError):
    """The submitted unified diff is outside the supported safe subset."""


class PatchRequest(DomainModel):
    """Structured tool input containing one base-rooted unified diff."""

    patch: Patch


class PatchApplyResult(DomainModel):
    """Small, audit-safe projection of a committed patch transition."""

    change_set_id: str
    base_snapshot_id: SnapshotId
    new_snapshot_id: SnapshotId
    changed_paths: tuple[str, ...]
    manifest_hash: str
    completed_at: UtcTimestamp


class PatchStore(Protocol):
    """The atomic snapshot and ChangeSet persistence boundary for this tool."""

    def transaction(self) -> AbstractAsyncContextManager[object]:
        """Open one transaction that nests Store operations."""

    async def create_snapshot(
        self, snapshot: Snapshot, manifest: SnapshotManifest, *, owner_id: str
    ) -> Snapshot:
        """Create or replay an immutable snapshot."""

    async def create_change_set(self, change_set: ChangeSet) -> ChangeSet:
        """Create or replay one tool-bound ChangeSet."""

    async def list_change_sets(self, *, tool_call_id: str) -> tuple[ChangeSet, ...]:
        """Find durable ChangeSets for idempotent tool-call replay."""

    async def get_change_set(self, change_set_id: str) -> ChangeSet:
        """Read one persisted ChangeSet after an atomic patch commit."""


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_count: int
    lines: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _FilePatch:
    path: str
    action: FileChangeAction
    hunks: tuple[_Hunk, ...]


class PatchRunner:
    """Apply an approved patch only when its server-owned base still matches."""

    def __init__(
        self,
        backend: WorkspaceBackend,
        store: PatchStore,
        *,
        owner_id: str,
        base_snapshot: Snapshot,
        base_manifest: SnapshotManifest,
    ) -> None:
        if base_snapshot.status is not SnapshotStatus.READY:
            raise ValueError("patch base snapshot must be READY")
        if base_snapshot.file_count != len(base_manifest.entries):
            raise ValueError("patch base manifest does not match snapshot metadata")
        self._backend = backend
        self._store = store
        self._owner_id = owner_id
        self._base_snapshot = base_snapshot
        self._base_manifest = base_manifest

    async def apply(self, call: ToolCall, request: PatchRequest) -> PatchApplyResult:
        """Validate, stage, persist, and promote one complete patch transition."""
        patch = request.patch
        if call.snapshot_id != patch.base_snapshot_id:
            raise PatchStaleError("patch base does not match the tool call snapshot")
        if patch.base_snapshot_id != self._base_snapshot.snapshot_id:
            raise PatchStaleError("patch base is not available to this workspace runner")
        if call.status is not ToolCallStatus.RUNNING:
            raise PatchApplyError("patch tool call must be running")
        existing = await self._existing_change_set(call, patch)
        if existing is not None:
            return self._replay_result(existing)
        if self._backend.snapshot_manifest().manifest_hash != self._base_manifest.manifest_hash:
            raise PatchStaleError("workspace no longer matches the patch base snapshot")

        parsed = _parse_unified_diff(patch.unified_diff)
        changes, expected_digests, file_changes, manifest = self._prepare_changes(parsed)
        if manifest.manifest_hash == self._base_manifest.manifest_hash:
            raise PatchValidationError("patch does not change the base snapshot")
        staged = self._backend.stage_text_changes(changes, expected_digests=expected_digests)
        try:
            staged.commit()
            completed_at = utc_now()
            requested_snapshot = Snapshot(
                snapshot_id=_new_snapshot_id(),
                workspace_id=self._base_snapshot.workspace_id,
                status=SnapshotStatus.READY,
                created_at=completed_at,
                completed_at=completed_at,
                file_count=len(manifest.entries),
                total_size=manifest.total_size,
            )
            async with self._store.transaction():
                persisted_snapshot = await self._store.create_snapshot(
                    requested_snapshot, manifest, owner_id=self._owner_id
                )
                change_set = ChangeSet(
                    change_set_id=new_change_set_id(),
                    run_id=call.run_id,
                    tool_call_id=call.call_id,
                    workspace_id=self._base_snapshot.workspace_id,
                    base_snapshot_id=patch.base_snapshot_id,
                    new_snapshot_id=persisted_snapshot.snapshot_id,
                    patch=patch,
                    files=file_changes,
                    created_at=completed_at,
                )
                persisted_change_set = await self._store.create_change_set(change_set)
        except BaseException:
            staged.rollback()
            raise
        else:
            staged.finalize()
        return PatchApplyResult(
            change_set_id=persisted_change_set.change_set_id,
            base_snapshot_id=patch.base_snapshot_id,
            new_snapshot_id=persisted_snapshot.snapshot_id,
            changed_paths=tuple(change.path for change in file_changes),
            manifest_hash=manifest.manifest_hash,
            completed_at=completed_at,
        )

    async def _existing_change_set(
        self, call: ToolCall, patch: Patch
    ) -> ChangeSet | None:
        existing = await self._store.list_change_sets(tool_call_id=call.call_id)
        if len(existing) > 1:
            raise PatchApplyError("tool call has multiple ChangeSets")
        if not existing:
            return None
        change_set = existing[0]
        if (
            change_set.run_id != call.run_id
            or change_set.tool_call_id != call.call_id
            or change_set.workspace_id != self._base_snapshot.workspace_id
            or change_set.base_snapshot_id != patch.base_snapshot_id
            or change_set.patch != patch
        ):
            raise PatchApplyError("tool call already has a different ChangeSet")
        return change_set

    def _replay_result(self, change_set: ChangeSet) -> PatchApplyResult:
        expected_manifest = _manifest_after_changes(self._base_manifest, change_set.files)
        actual_manifest = self._backend.snapshot_manifest()
        if actual_manifest.manifest_hash != expected_manifest.manifest_hash:
            raise PatchStaleError("workspace no longer matches the idempotent ChangeSet")
        return PatchApplyResult(
            change_set_id=change_set.change_set_id,
            base_snapshot_id=change_set.base_snapshot_id,
            new_snapshot_id=change_set.new_snapshot_id,
            changed_paths=tuple(change.path for change in change_set.files),
            manifest_hash=actual_manifest.manifest_hash,
            completed_at=change_set.created_at,
        )

    def _prepare_changes(
        self, parsed: tuple[_FilePatch, ...]
    ) -> tuple[
        dict[str, tuple[bool, str | None]],
        dict[str, tuple[str, int]],
        tuple[FileChange, ...],
        SnapshotManifest,
    ]:
        entries = {entry.path: entry for entry in self._base_manifest.entries}
        changes: dict[str, tuple[bool, str | None]] = {}
        expected_digests: dict[str, tuple[str, int]] = {}
        file_changes: list[FileChange] = []
        for file_patch in parsed:
            current = entries.get(file_patch.path)
            if file_patch.action is FileChangeAction.ADD:
                if current is not None:
                    raise PatchValidationError("added file already exists in the base snapshot")
                source = ""
            else:
                if current is None or current.entry_type is not SnapshotEntryType.FILE:
                    raise PatchValidationError("patch target is not a base snapshot file")
                try:
                    source = self._backend.read_patch_text(file_patch.path)
                except WorkspaceBackendError as error:
                    raise PatchValidationError("patch target is not a bounded text file") from error
                current_content = _file_content(source)
                if (current_content.sha256, current_content.size) != (
                    current.sha256,
                    current.size,
                ):
                    raise PatchStaleError("patch target content no longer matches the base")
                expected_digests[file_patch.path] = (current.sha256, current.size)
            updated = _apply_hunks(source, file_patch)
            before = (
                None
                if current is None
                else FileContent(sha256=current.sha256, size=current.size)
            )
            if file_patch.action is FileChangeAction.DELETE:
                if updated:
                    raise PatchValidationError("deleted file patch must produce empty content")
                entries.pop(file_patch.path, None)
                after = None
                changes[file_patch.path] = (True, None)
            else:
                after = _file_content(updated)
                entries[file_patch.path] = SnapshotEntry(
                    path=file_patch.path,
                    sha256=after.sha256,
                    size=after.size,
                    entry_type=SnapshotEntryType.FILE,
                )
                changes[file_patch.path] = (current is not None, updated)
            file_changes.append(
                FileChange(
                    path=file_patch.path,
                    action=file_patch.action,
                    before=before,
                    after=after,
                )
            )
        return (
            changes,
            expected_digests,
            tuple(file_changes),
            SnapshotManifest(entries=tuple(entries.values())),
        )


def build_patch_definition(runner: PatchRunner) -> ToolDefinition:
    """Build the policy-classified write tool around one workspace runner."""

    async def handler(context: BaseModel) -> ToolResult:
        if not isinstance(context, ExecutionContext):
            raise TypeError("apply_patch requires an execution context")
        if not isinstance(context.arguments, PatchRequest):
            raise TypeError("apply_patch received an invalid argument model")
        result = await runner.apply(context.call, context.arguments)
        return ToolResult.from_call(
            context.call,
            status=ToolCallStatus.SUCCEEDED,
            result=result.model_dump(mode="json"),
            changed_paths=result.changed_paths,
            completed_at=result.completed_at,
        )

    return ToolDefinition(
        name="apply_patch",
        description="Apply one validated patch to the authorized workspace.",
        effect=ToolEffect.WRITE,
        argument_model=PatchRequest,
        handler=cast(ToolHandler, handler),
    )


def _parse_unified_diff(value: str) -> tuple[_FilePatch, ...]:
    lines = value.splitlines(keepends=True)
    parsed: list[_FilePatch] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(("diff --git ", "index ")):
            index += 1
            continue
        if not line.startswith("--- "):
            raise PatchValidationError("patch contains an unsupported diff record")
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise PatchValidationError("patch file header is incomplete")
        old_path = _header_path(lines[index][4:])
        new_path = _header_path(lines[index + 1][4:])
        path, action = _patch_action(old_path, new_path)
        index += 2
        hunks: list[_Hunk] = []
        while index < len(lines) and lines[index].startswith("@@ "):
            hunk, index = _parse_hunk(lines, index)
            hunks.append(hunk)
        if not hunks:
            raise PatchValidationError("patch file has no content hunks")
        parsed.append(_FilePatch(path=path, action=action, hunks=tuple(hunks)))
    if not parsed:
        raise PatchValidationError("patch contains no file changes")
    paths = [file_patch.path for file_patch in parsed]
    if len(paths) != len(set(paths)):
        raise PatchValidationError("patch contains duplicate file paths")
    return tuple(parsed)


def _header_path(value: str) -> str | None:
    path = value.split("\t", 1)[0].rstrip("\r\n")
    if path == "/dev/null":
        return None
    if not path.startswith(("a/", "b/")):
        raise PatchValidationError("patch path must use a relative diff prefix")
    candidate = path[2:]
    try:
        FileChange(
            path=candidate,
            action=FileChangeAction.ADD,
            after=FileContent(sha256="0" * 64, size=0),
        )
    except ValueError as error:
        raise PatchValidationError("patch path is not workspace-relative") from error
    return candidate


def _patch_action(old_path: str | None, new_path: str | None) -> tuple[str, FileChangeAction]:
    if old_path is None and new_path is not None:
        return new_path, FileChangeAction.ADD
    if old_path is not None and new_path is None:
        return old_path, FileChangeAction.DELETE
    if old_path is None or new_path is None or old_path != new_path:
        raise PatchValidationError("patch rename and malformed null paths are not supported")
    return old_path, FileChangeAction.MODIFY


def _parse_hunk(lines: list[str], index: int) -> tuple[_Hunk, int]:
    match = _HUNK_HEADER_RE.fullmatch(lines[index])
    if match is None:
        raise PatchValidationError("patch hunk header is invalid")
    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    new_count = int(match.group(4) or "1")
    index += 1
    body: list[tuple[str, str]] = []
    while index < len(lines) and not lines[index].startswith(("@@ ", "--- ", "diff --git ")):
        line = lines[index]
        if not line or line[0] not in {" ", "+", "-"}:
            raise PatchValidationError("patch hunk contains an unsupported record")
        body.append((line[0], line[1:]))
        index += 1
    if sum(kind in {" ", "-"} for kind, _ in body) != old_count:
        raise PatchValidationError("patch hunk old line count is invalid")
    if sum(kind in {" ", "+"} for kind, _ in body) != new_count:
        raise PatchValidationError("patch hunk new line count is invalid")
    return _Hunk(old_start=old_start, old_count=old_count, lines=tuple(body)), index


def _apply_hunks(source: str, file_patch: _FilePatch) -> str:
    source_lines = source.splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    for hunk in file_patch.hunks:
        old_index = 0 if hunk.old_start == 0 else hunk.old_start - 1
        if old_index < cursor or old_index > len(source_lines):
            raise PatchValidationError("patch hunk position is invalid")
        output.extend(source_lines[cursor:old_index])
        cursor = old_index
        for kind, line in hunk.lines:
            if kind in {" ", "-"}:
                if cursor >= len(source_lines) or source_lines[cursor] != line:
                    raise PatchStaleError("patch hunk does not match the workspace base")
                cursor += 1
            if kind in {" ", "+"}:
                output.append(line)
    output.extend(source_lines[cursor:])
    return "".join(output)


def _manifest_after_changes(
    base_manifest: SnapshotManifest, file_changes: tuple[FileChange, ...]
) -> SnapshotManifest:
    entries = {entry.path: entry for entry in base_manifest.entries}
    for change in file_changes:
        current = entries.get(change.path)
        if change.action is FileChangeAction.ADD:
            if current is not None or change.after is None:
                raise PatchApplyError("stored ChangeSet has an invalid added file")
            after = change.after
        elif change.action is FileChangeAction.MODIFY:
            if current is None or current.entry_type is not SnapshotEntryType.FILE:
                raise PatchApplyError("stored ChangeSet has an invalid modified file")
            if change.before != FileContent(sha256=current.sha256, size=current.size):
                raise PatchApplyError("stored ChangeSet has an invalid base file")
            if change.after is None:
                raise PatchApplyError("stored ChangeSet has no modified file content")
            after = change.after
        else:
            if current is None or current.entry_type is not SnapshotEntryType.FILE:
                raise PatchApplyError("stored ChangeSet has an invalid deleted file")
            if change.before != FileContent(sha256=current.sha256, size=current.size):
                raise PatchApplyError("stored ChangeSet has an invalid deleted file base")
            if change.after is not None:
                raise PatchApplyError("stored ChangeSet has deleted file content")
            entries.pop(change.path)
            continue
        entries[change.path] = SnapshotEntry(
            path=change.path,
            sha256=after.sha256,
            size=after.size,
            entry_type=SnapshotEntryType.FILE,
        )
    return SnapshotManifest(entries=tuple(entries.values()))


def _file_content(value: str) -> FileContent:
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_CHANGED_FILE_BYTES:
        raise PatchValidationError("patched file exceeds the content limit")
    return FileContent(sha256=hashlib.sha256(encoded).hexdigest(), size=len(encoded))


def _new_snapshot_id() -> str:
    from prp_runtime.domain.values import new_snapshot_id

    return new_snapshot_id()
