"""Immutable patch and ChangeSet facts rooted in an authorized snapshot."""

import hashlib
import re
from enum import StrEnum, unique
from typing import Annotated
from uuid import uuid4

from pydantic import Field, StringConstraints, model_validator

from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.values import RunId, SnapshotId, ToolCallId, UtcTimestamp, WorkspaceId

__all__ = [
    "ChangeSet",
    "ChangeSetId",
    "FileChange",
    "FileChangeAction",
    "FileContent",
    "Patch",
    "new_change_set_id",
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
