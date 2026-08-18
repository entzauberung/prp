"""Resolve logical server-owned Workspaces to safe backend handles."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Self

from prp_runtime.domain.values import WorkspaceId
from prp_runtime.workspace.backend import WorkspaceBackend, WorkspaceBackendError
from prp_runtime.workspace.models import (
    SnapshotManifest,
    Workspace,
    WorkspaceRootMapping,
    WorkspaceSourceType,
    WorkspaceStatus,
)

__all__ = ["ResolvedWorkspace", "WorkspaceResolveError", "WorkspaceResolver"]


class WorkspaceResolveError(ValueError):
    """A stable, host-path-free workspace resolution failure."""


class ResolvedWorkspace:
    """A logical workspace identity paired with one owned backend handle."""

    __slots__ = (
        "workspace_id",
        "owner_id",
        "server_alias",
        "_backend",
        "_workspace_root",
        "_closed",
    )

    def __init__(
        self,
        *,
        workspace_id: WorkspaceId,
        owner_id: str,
        server_alias: str,
        backend: WorkspaceBackend,
        workspace_root: Path | None = None,
    ) -> None:
        self.workspace_id = workspace_id
        self.owner_id = owner_id
        self.server_alias = server_alias
        self._backend = backend
        self._workspace_root = workspace_root
        self._closed = False

    @property
    def backend(self) -> WorkspaceBackend:
        """Return the backend while the resolved handle is open."""
        if self._closed:
            raise WorkspaceResolveError("workspace handle is closed")
        return self._backend

    def close(self) -> None:
        """Close the backend; repeated close is harmless."""
        if not self._closed:
            self._backend.close()
            self._closed = True

    def snapshot_manifest(self) -> SnapshotManifest:
        """Build a relative manifest while preserving the safe error boundary."""
        try:
            return self.backend.snapshot_manifest()
        except WorkspaceBackendError as error:
            raise WorkspaceResolveError("workspace manifest is unavailable") from error

    def _tool_workspace_cwd(self) -> Path:
        """Return the server-owned cwd only to the internal tool composition."""
        if self._workspace_root is None:
            raise WorkspaceResolveError("workspace tool cwd is unavailable")
        return self._workspace_root

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            "ResolvedWorkspace("
            f"workspace_id={self.workspace_id!r}, "
            f"owner_id={self.owner_id!r}, "
            f"server_alias={self.server_alias!r})"
        )


class WorkspaceResolver:
    """Resolve only active SERVER_ALIAS workspaces from server-owned settings."""

    __slots__ = ("_roots",)

    def __init__(self, roots: WorkspaceRootMapping) -> None:
        self._roots = roots

    def resolve(self, workspace: Workspace, *, owner_id: str) -> ResolvedWorkspace:
        """Validate identity and open one non-cached backend handle."""
        if workspace.owner_id != owner_id:
            raise WorkspaceResolveError("workspace owner does not match the request")
        if workspace.status is not WorkspaceStatus.ACTIVE:
            raise WorkspaceResolveError("workspace is not active")
        if workspace.source.source_type is not WorkspaceSourceType.SERVER_ALIAS:
            raise WorkspaceResolveError("workspace source is not available to the cloud")
        server_alias = workspace.source.server_alias
        if server_alias is None:
            raise WorkspaceResolveError("workspace server alias is missing")
        try:
            root = self._roots.root_for(server_alias)
        except KeyError as error:
            raise WorkspaceResolveError("workspace server alias is not configured") from error
        root_path = Path(root)
        self._validate_root_path(root_path)
        try:
            backend = WorkspaceBackend(root_path)
        except WorkspaceBackendError as error:
            raise WorkspaceResolveError("configured workspace root is unavailable") from error
        return ResolvedWorkspace(
            workspace_id=workspace.workspace_id,
            owner_id=workspace.owner_id,
            server_alias=server_alias,
            backend=backend,
            workspace_root=root_path,
        )

    @staticmethod
    def _validate_root_path(root: Path) -> None:
        """Reject symlinked or non-canonical components before backend open."""
        if not root.is_absolute() or root.anchor != "/":
            raise WorkspaceResolveError("configured workspace root is not an absolute path")
        current = Path(root.anchor)
        for component in root.parts[1:]:
            if component in {"", ".", ".."}:
                raise WorkspaceResolveError("configured workspace root is not canonical")
            current /= component
            try:
                mode = current.lstat().st_mode
            except OSError as error:
                raise WorkspaceResolveError(
                    "configured workspace root is unavailable"
                ) from error
            if stat.S_ISLNK(mode):
                raise WorkspaceResolveError(
                    "configured workspace root contains a symlink"
                )
