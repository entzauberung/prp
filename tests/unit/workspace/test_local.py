"""Targeted tests for local workspace path bounds and public redaction."""

import json
from pathlib import Path

import pytest

from prp_runtime.workspace.backend import WorkspaceBackendError
from prp_runtime.workspace.local import (
    canonicalize_local_root,
    local_workspace_id,
    resolve_local_workspace,
)
from prp_runtime.workspace.resolver import WorkspaceResolveError


def test_local_handle_reads_only_inside_the_authorized_root(tmp_path: Path) -> None:
    root = tmp_path / "local-root"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")

    handle = resolve_local_workspace(root)
    try:
        result = handle.backend.read_file("src/main.py")
        assert result.path == "src/main.py"
        assert result.content == "print('ok')\n"
        dumped = result.model_dump_json()
        assert json.loads(dumped)["path"] == "src/main.py"
        assert str(root) not in dumped
        assert str(root) not in json.dumps(handle.public_identity())
        assert str(root) not in repr(handle)
    finally:
        handle.close()
        handle.close()
    with pytest.raises(WorkspaceResolveError, match="closed"):
        _ = handle.backend


def test_equivalent_local_roots_share_one_workspace_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(tmp_path)
    owner_id = "prn_default"
    handles = [
        resolve_local_workspace(root, owner_id=owner_id),
        resolve_local_workspace("ws", owner_id=owner_id),
        resolve_local_workspace(nested / ".." / "ws", owner_id=owner_id),
    ]
    try:
        identities = {handle.workspace_id for handle in handles}
        assert len(identities) == 1
        workspace_id = next(iter(identities))
        canonical = canonicalize_local_root(root)
        assert workspace_id == local_workspace_id(owner_id=owner_id, root=canonical)
        assert str(root) not in workspace_id
        assert str(canonical) not in workspace_id
        for handle in handles:
            dumped = str(handle.public_identity())
            assert str(root) not in dumped
            assert str(canonical) not in dumped
    finally:
        for handle in handles:
            handle.close()


def test_tilde_local_root_canonicalizes_to_the_same_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = home / "proj"
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    handle = resolve_local_workspace("~/proj", owner_id="prn_default")
    try:
        canonical = canonicalize_local_root(root)
        assert handle.workspace_id == local_workspace_id(
            owner_id="prn_default", root=canonical
        )
        assert str(home) not in handle.workspace_id
        assert str(root) not in str(handle.public_identity())
    finally:
        handle.close()


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../secret", "a/../x", "a//b", r"a\b", "C:/x", "C:\\x"],
)
def test_local_handle_rejects_unsafe_paths_before_access(tmp_path: Path, path: str) -> None:
    root = tmp_path / "local-root"
    root.mkdir()
    (root / "a").mkdir()
    handle = resolve_local_workspace(root)
    try:
        with pytest.raises(WorkspaceBackendError, match="not authorized") as excinfo:
            handle.backend.resolve(path)
        assert str(root) not in str(excinfo.value)
        assert path not in str(excinfo.value) or "not authorized" in str(excinfo.value)
    finally:
        handle.close()


def test_local_handle_rejects_symlink_escape_without_host_paths(tmp_path: Path) -> None:
    root = tmp_path / "local-root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (root / "escape").symlink_to(outside, target_is_directory=True)

    handle = resolve_local_workspace(root)
    try:
        with pytest.raises(WorkspaceBackendError) as excinfo:
            handle.backend.resolve("escape/secret.txt")
        assert str(root) not in str(excinfo.value)
        assert str(outside) not in str(excinfo.value)
    finally:
        handle.close()
