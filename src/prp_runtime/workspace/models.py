"""Closed domain contracts for server-owned workspaces and snapshots."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from enum import StrEnum, unique
from typing import Annotated

from pydantic import AfterValidator, AliasChoices, Field, StringConstraints, model_validator

from prp_runtime.domain.models import ClientId, DomainModel
from prp_runtime.domain.values import SnapshotId, UtcTimestamp, WorkspaceId

__all__ = [
    "Snapshot",
    "SnapshotEntry",
    "SnapshotEntryType",
    "SnapshotManifest",
    "SnapshotStatus",
    "Workspace",
    "WorkspaceRoot",
    "WorkspaceRootMapping",
    "WorkspaceSource",
    "WorkspaceSourceType",
    "WorkspaceStatus",
    "canonical_manifest_hash",
    "BridgeManifestAcceptance",
    "BridgeManifestPublication",
]

MAX_MANIFEST_ENTRIES = 100_000
MAX_ENTRY_SIZE_BYTES = 1_073_741_824
MAX_MANIFEST_SIZE_BYTES = 10_737_418_240
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

Alias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]
RootPath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4096),
]
GrantId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    ),
]


@unique
class WorkspaceSourceType(StrEnum):
    """The only source descriptors accepted by the public contract."""

    SERVER_ALIAS = "SERVER_ALIAS"
    BRIDGE_GRANT = "BRIDGE_GRANT"


def _absolute_root(value: str) -> str:
    """Validate root syntax without checking the filesystem."""
    if "\x00" in value or not value.startswith("/"):
        raise ValueError("workspace root must be an absolute POSIX path")
    return value


class WorkspaceRoot(DomainModel):
    """One server-only alias to an absolute workspace root."""

    alias: Alias
    root: Annotated[RootPath, AfterValidator(_absolute_root)]


class WorkspaceRootMapping(DomainModel):
    """Immutable server-owned roots with a redacted public representation."""

    entries: tuple[WorkspaceRoot, ...] = Field(
        default=(),
        exclude=True,
        repr=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_json_mapping(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                "entries": tuple(
                    {"alias": alias, "root": root}
                    for alias, root in sorted(value.items(), key=lambda item: str(item[0]))
                )
            }
        return value

    @model_validator(mode="after")
    def _aliases_are_unique(self) -> "WorkspaceRootMapping":
        aliases = [entry.alias for entry in self.entries]
        if len(aliases) != len(set(aliases)):
            raise ValueError("workspace root aliases must be unique")
        return self

    @property
    def aliases(self) -> tuple[str, ...]:
        """Configured aliases in deterministic order."""
        return tuple(entry.alias for entry in self.entries)

    def root_for(self, alias: str) -> str:
        """Return a configured root without exposing the complete mapping."""
        for entry in self.entries:
            if entry.alias == alias:
                return entry.root
        raise KeyError(f"workspace alias is not configured: {alias}")


@unique
class WorkspaceStatus(StrEnum):
    """Workspace lifecycle controlled by the owning service."""

    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    REVOKED = "REVOKED"

    @property
    def is_terminal(self) -> bool:
        return self is WorkspaceStatus.REVOKED


@unique
class SnapshotStatus(StrEnum):
    """Immutable snapshot lifecycle."""

    CREATING = "CREATING"
    READY = "READY"
    INVALIDATED = "INVALIDATED"

    @property
    def is_terminal(self) -> bool:
        return self in {SnapshotStatus.READY, SnapshotStatus.INVALIDATED}


@unique
class SnapshotEntryType(StrEnum):
    """Entry types that do not expose symlink targets."""

    FILE = "FILE"
    DIRECTORY = "DIRECTORY"


class SnapshotEntry(DomainModel):
    """One bounded, relative entry in a snapshot manifest."""

    path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
    ]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] = Field(
        validation_alias=AliasChoices("sha256", "digest")
    )
    size: int = Field(ge=0, le=MAX_ENTRY_SIZE_BYTES)
    entry_type: SnapshotEntryType

    @model_validator(mode="after")
    def _path_is_relative_and_digest_is_canonical(self) -> "SnapshotEntry":
        if (
            self.path.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z]:", self.path)
            or "\\" in self.path
        ):
            raise ValueError("snapshot path must be relative POSIX syntax")
        parts = self.path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("snapshot path must not contain empty, dot, or parent segments")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("snapshot digest must be lowercase SHA-256")
        return self

    @property
    def digest(self) -> str:
        """Compatibility name for the canonical SHA-256 field."""
        return self.sha256


def canonical_manifest_hash(entries: Iterable[SnapshotEntry]) -> str:
    """Return a content-only SHA-256 independent of input entry order."""
    canonical = [
        entry.model_dump(mode="json")
        for entry in sorted(entries, key=lambda candidate: candidate.path)
    ]
    payload = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SnapshotManifest(DomainModel):
    """Bounded immutable content inventory for one snapshot."""

    entries: tuple[SnapshotEntry, ...] = ()

    @model_validator(mode="after")
    def _entries_are_unique_and_bounded(self) -> "SnapshotManifest":
        if len(self.entries) > MAX_MANIFEST_ENTRIES:
            raise ValueError("snapshot manifest has too many entries")
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("snapshot manifest contains duplicate paths")
        if sum(entry.size for entry in self.entries) > MAX_MANIFEST_SIZE_BYTES:
            raise ValueError("snapshot manifest is too large")
        return self

    @property
    def total_size(self) -> int:
        return sum(entry.size for entry in self.entries)

    @property
    def manifest_hash(self) -> str:
        return canonical_manifest_hash(self.entries)


class BridgeManifestPublication(DomainModel):
    """Client-owned bounded inventory. No root, secret or file content."""

    snapshot_id: SnapshotId
    client_id: ClientId
    workspace_id: WorkspaceId
    entries: tuple[SnapshotEntry, ...] = ()
    manifest_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None

    @model_validator(mode="after")
    def _identity_is_content_free_and_canonical(self) -> "BridgeManifestPublication":
        manifest = SnapshotManifest(entries=self.entries)
        if self.manifest_hash is not None and self.manifest_hash != manifest.manifest_hash:
            raise ValueError("manifest hash does not match published entries")
        return self

    @property
    def manifest(self) -> SnapshotManifest:
        return SnapshotManifest(entries=self.entries)


class BridgeManifestAcceptance(DomainModel):
    """Server-authorized snapshot identity for one registered client workspace."""

    snapshot_id: SnapshotId
    client_id: ClientId
    workspace_id: WorkspaceId
    manifest_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    file_count: int = Field(ge=0)
    total_size: int = Field(ge=0)
    status: SnapshotStatus = SnapshotStatus.READY


class WorkspaceSource(DomainModel):
    """A server-owned alias or an opaque bridge grant, never a host path."""

    source_type: WorkspaceSourceType
    server_alias: Alias | None = None
    bridge_grant: GrantId | None = None

    @model_validator(mode="after")
    def _source_matches_type(self) -> "WorkspaceSource":
        if self.source_type is WorkspaceSourceType.SERVER_ALIAS:
            if self.server_alias is None or self.bridge_grant is not None:
                raise ValueError("SERVER_ALIAS requires only server_alias")
        elif self.bridge_grant is None or self.server_alias is not None:
            raise ValueError("BRIDGE_GRANT requires only bridge_grant")
        return self


class Workspace(DomainModel):
    """A service-owned workspace identity with no filesystem location."""

    workspace_id: WorkspaceId
    owner_id: Alias
    alias: Alias
    source: WorkspaceSource
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    created_at: UtcTimestamp
    closed_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _lifecycle_is_consistent(self) -> "Workspace":
        if self.status.is_terminal and self.closed_at is None:
            raise ValueError("a revoked workspace must have closed_at")
        if not self.status.is_terminal and self.closed_at is not None:
            raise ValueError("an active workspace must not have closed_at")
        return self


class Snapshot(DomainModel):
    """An immutable identity for a workspace state."""

    snapshot_id: SnapshotId
    workspace_id: WorkspaceId
    status: SnapshotStatus = SnapshotStatus.CREATING
    created_at: UtcTimestamp
    completed_at: UtcTimestamp | None = None
    file_count: int | None = Field(default=None, ge=0)
    total_size: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _lifecycle_is_consistent(self) -> "Snapshot":
        if self.status.is_terminal and self.completed_at is None:
            raise ValueError("a terminal snapshot must have completed_at")
        if not self.status.is_terminal and self.completed_at is not None:
            raise ValueError("a creating snapshot must not have completed_at")
        return self
