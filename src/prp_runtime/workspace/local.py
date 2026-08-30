"""Resolve a caller-provided local directory into a bounded workspace handle."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Self

from prp_runtime.domain.values import WorkspaceId
from prp_runtime.workspace.backend import WorkspaceBackend, WorkspaceBackendError
from prp_runtime.workspace.resolver import WorkspaceResolveError, WorkspaceResolver

__all__ = [
    "LocalWorkspaceHandle",
    "canonicalize_local_root",
    "local_workspace_id",
    "resolve_local_workspace",
]


class LocalWorkspaceHandle:
    """An owned local directory handle with a redacted public identity."""

    __slots__ = ("workspace_id", "owner_id", "_backend", "_workspace_root", "_closed")

    def __init__(
        self,
        *,
        workspace_id: WorkspaceId,
        owner_id: str,
        backend: WorkspaceBackend,
        workspace_root: Path,
    ) -> None:
        self.workspace_id = workspace_id
        self.owner_id = owner_id
        self._backend = backend
        self._workspace_root = workspace_root
        self._closed = False

    @property
    def backend(self) -> WorkspaceBackend:
        """Return the backend while the local handle is open."""
        if self._closed:
            raise WorkspaceResolveError("workspace handle is closed")
        return self._backend

    def close(self) -> None:
        """Close the backend; repeated close is harmless."""
        if not self._closed:
            self._backend.close()
            self._closed = True

    def public_identity(self) -> dict[str, str]:
        """Return logical identity only; the host root stays private."""
        return {"workspace_id": self.workspace_id, "owner_id": self.owner_id}

    def _tool_workspace_cwd(self) -> Path:
        """Return the process-local cwd only to internal tool composition."""
        if self._closed:
            raise WorkspaceResolveError("workspace handle is closed")
        return self._workspace_root

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            "LocalWorkspaceHandle("
            f"workspace_id={self.workspace_id!r}, "
            f"owner_id={self.owner_id!r})"
        )


def canonicalize_local_root(root: Path | str) -> Path:
    """Expand relative/tilde input to one absolute root, then reject unsafe paths."""
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    canonical = Path(os.path.normpath(path))
    WorkspaceResolver._validate_root_path(canonical, kind="local")
    return canonical


def local_workspace_id(*, owner_id: str, root: Path) -> str:
    """Derive a stable local workspace id from owner and canonical root."""
    digest = hashlib.sha256(f"{owner_id}:{root}".encode("utf-8")).hexdigest()
    return f"ws_{digest[:32]}"


def resolve_local_workspace(
    root: Path | str,
    *,
    owner_id: str = "local-owner",
    workspace_id: WorkspaceId | None = None,
) -> LocalWorkspaceHandle:
    """Validate a caller-provided directory and open one backend handle."""
    root_path = canonicalize_local_root(root)
    try:
        backend = WorkspaceBackend(root_path)
    except WorkspaceBackendError as error:
        raise WorkspaceResolveError("local workspace root is unavailable") from error
    return LocalWorkspaceHandle(
        workspace_id=workspace_id or local_workspace_id(owner_id=owner_id, root=root_path),
        owner_id=owner_id,
        backend=backend,
        workspace_root=root_path,
    )
