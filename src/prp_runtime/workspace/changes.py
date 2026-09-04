"""Immutable patch and ChangeSet facts rooted in an authorized snapshot."""

import hashlib
import re
from enum import StrEnum, unique
from collections.abc import Mapping
from typing import Annotated
from uuid import uuid4

from pydantic import Field, StringConstraints, ValidationError, model_validator

from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.values import RunId, SnapshotId, ToolCallId, UtcTimestamp, WorkspaceId
from prp_runtime.workspace.models import SnapshotEntry, SnapshotEntryType, SnapshotManifest

__all__ = [
    "BRIDGE_RESULT_SELF_ASSERTION_KEYS",
    "ChangeSet",
    "ChangeSetId",
    "FileChange",
    "FileChangeAction",
    "FileContent",
    "Patch",
    "apply_patch_facts_to_manifest",
    "assert_bridge_result_is_not_self_asserted",
    "inherit_unchanged_file_contents",
    "overlay_changed_file_contents",
    "new_change_set_id",
    "parse_bridge_patch_facts",
    "validate_patch_facts_against_manifest",
]

MAX_PATCH_BYTES = 262_144
MAX_FILE_CHANGES = 10_000
MAX_CHANGED_FILE_BYTES = 1_073_741_824
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ChangeSetId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=4,
        max_length=67,
        pattern=r"^cs_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
    ),
]
RelativeChangePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


@unique
class FileChangeAction(StrEnum):
    """The only file transformations recorded in a ChangeSet."""

    ADD = "ADD"
    MODIFY = "MODIFY"
    DELETE = "DELETE"


class FileContent(DomainModel):
    """Content metadata, without retaining a host path or raw file bytes."""

    sha256: Sha256
    size: int = Field(ge=0, le=MAX_CHANGED_FILE_BYTES)

    @model_validator(mode="after")
    def _digest_is_canonical(self) -> "FileContent":
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("file content digest must be lowercase SHA-256")
        return self


class FileChange(DomainModel):
    """One relative file transformation with before and after content facts."""

    path: RelativeChangePath
    action: FileChangeAction
    before: FileContent | None = None
    after: FileContent | None = None

    @model_validator(mode="after")
    def _change_is_complete_and_relative(self) -> "FileChange":
        if (
            self.path.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z]:", self.path)
            or "\\" in self.path
        ):
            raise ValueError("changed path must be relative POSIX syntax")
        if any(part in {"", ".", ".."} for part in self.path.split("/")):
            raise ValueError("changed path must not contain empty, dot, or parent segments")
        expected = {
            FileChangeAction.ADD: (None, "after"),
            FileChangeAction.MODIFY: ("before", "after"),
            FileChangeAction.DELETE: ("before", None),
        }[self.action]
        before, after = expected
        if (before is None) != (self.before is None) or (after is None) != (self.after is None):
            raise ValueError(f"{self.action.value} change has invalid before/after content")
        return self


class Patch(DomainModel):
    """One bounded unified diff explicitly rooted in a base snapshot."""

    base_snapshot_id: SnapshotId
    unified_diff: Annotated[str, StringConstraints(min_length=1, max_length=MAX_PATCH_BYTES)]

    @model_validator(mode="after")
    def _diff_is_byte_bounded(self) -> "Patch":
        if len(self.unified_diff.encode("utf-8")) > MAX_PATCH_BYTES:
            raise ValueError("unified diff exceeds the byte limit")
        return self

    @property
    def sha256(self) -> str:
        """Return the canonical digest used for durable patch identity."""
        return hashlib.sha256(self.unified_diff.encode("utf-8")).hexdigest()


class ChangeSet(DomainModel):
    """An auditable, immutable transition from one snapshot to another."""

    change_set_id: ChangeSetId
    run_id: RunId
    tool_call_id: ToolCallId
    workspace_id: WorkspaceId
    base_snapshot_id: SnapshotId
    new_snapshot_id: SnapshotId
    patch: Patch
    files: tuple[FileChange, ...]
    created_at: UtcTimestamp

    @model_validator(mode="after")
    def _facts_share_one_base_and_unique_paths(self) -> "ChangeSet":
        if self.patch.base_snapshot_id != self.base_snapshot_id:
            raise ValueError("patch base snapshot must match the ChangeSet base snapshot")
        if self.new_snapshot_id == self.base_snapshot_id:
            raise ValueError("ChangeSet must produce a new snapshot")
        if not self.files:
            raise ValueError("ChangeSet must record at least one changed file")
        if len(self.files) > MAX_FILE_CHANGES:
            raise ValueError("ChangeSet has too many changed files")
        paths = [change.path for change in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("ChangeSet contains duplicate changed paths")
        return self

    @property
    def patch_sha256(self) -> str:
        """Expose the stable digest of the bounded unified diff."""
        return self.patch.sha256


def new_change_set_id() -> str:
    """Generate one opaque ChangeSet identity without encoding host details."""
    return f"cs_{uuid4().hex}"


BRIDGE_RESULT_SELF_ASSERTION_KEYS = frozenset(
    {
        "accepted",
        "promoted",
        "candidate_accepted",
        "merge_status",
    }
)


def assert_bridge_result_is_not_self_asserted(
    payload: Mapping[str, object] | None,
) -> None:
    """Reject client or model claims that a candidate or ChangeSet is accepted."""
    if not payload:
        return
    claimed = sorted(key for key in BRIDGE_RESULT_SELF_ASSERTION_KEYS if key in payload)
    if claimed:
        raise ValueError(
            "client self-assertion cannot create an accepted candidate or ChangeSet"
        )


def parse_bridge_patch_facts(
    payload: Mapping[str, object] | None,
) -> tuple[Patch, tuple[FileChange, ...]] | None:
    """Return base-bound patch facts, or None when the result has no patch payload."""
    assert_bridge_result_is_not_self_asserted(payload)
    if payload is None:
        return None
    patch_payload = payload.get("patch")
    files_payload = payload.get("files")
    patch_is_object = isinstance(patch_payload, Mapping)
    files_is_list = isinstance(files_payload, list)
    if not patch_is_object and not files_is_list:
        return None
    if not patch_is_object or not files_is_list:
        raise ValueError("write result patch facts must include patch and files")
    try:
        patch = Patch.model_validate(patch_payload)
        files = tuple(FileChange.model_validate(item) for item in files_payload)
    except ValidationError as error:
        raise ValueError("write result patch facts are invalid") from error
    if not files:
        raise ValueError("write result patch facts must include patch and files")
    return patch, files


def validate_patch_facts_against_manifest(
    manifest: SnapshotManifest,
    files: tuple[FileChange, ...],
) -> None:
    """Require before hashes to match the authorized base snapshot inventory."""
    by_path = {
        entry.path: entry
        for entry in manifest.entries
        if entry.entry_type is SnapshotEntryType.FILE
    }
    for change in files:
        current = by_path.get(change.path)
        if change.action is FileChangeAction.ADD:
            if current is not None:
                raise ValueError("ADD path already exists in the base snapshot")
            continue
        if current is None:
            raise ValueError("base snapshot does not contain the changed path")
        if (
            change.before is None
            or current.sha256 != change.before.sha256
            or current.size != change.before.size
        ):
            raise ValueError("ChangeSet before facts do not match the base snapshot")


def apply_patch_facts_to_manifest(
    manifest: SnapshotManifest,
    files: tuple[FileChange, ...],
) -> SnapshotManifest:
    """Apply validated file facts to a base manifest without reading a workspace root."""
    entries = {entry.path: entry for entry in manifest.entries}
    for change in files:
        if change.action is FileChangeAction.DELETE:
            entries.pop(change.path, None)
            continue
        if change.after is None:
            raise ValueError("write result patch facts are invalid")
        entries[change.path] = SnapshotEntry(
            path=change.path,
            sha256=change.after.sha256,
            size=change.after.size,
            entry_type=SnapshotEntryType.FILE,
        )
    return SnapshotManifest(
        entries=tuple(sorted(entries.values(), key=lambda entry: entry.path))
    )


def inherit_unchanged_file_contents(
    previous: Mapping[str, str],
    old_manifest: SnapshotManifest,
    new_manifest: SnapshotManifest,
) -> dict[str, str]:
    """Keep stored bytes only for files whose digest did not change."""
    old_digests = {
        entry.path: entry.sha256
        for entry in old_manifest.entries
        if entry.entry_type is SnapshotEntryType.FILE
    }
    inherited: dict[str, str] = {}
    for entry in new_manifest.entries:
        if entry.entry_type is not SnapshotEntryType.FILE:
            continue
        if old_digests.get(entry.path) != entry.sha256:
            continue
        text = previous.get(entry.path)
        if text is not None:
            inherited[entry.path] = text
    return inherited


def overlay_changed_file_contents(
    current: Mapping[str, str],
    files: tuple[FileChange, ...],
    updated: Mapping[str, str],
) -> dict[str, str]:
    """Replace or drop stored bytes for files listed in a ChangeSet."""
    contents = dict(current)
    for change in files:
        if change.action is FileChangeAction.DELETE:
            contents.pop(change.path, None)
            continue
        text = updated.get(change.path)
        if text is not None:
            contents[change.path] = text
    return contents
