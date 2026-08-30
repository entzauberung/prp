"""Targeted tests for the server-owned workspace resolver."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from prp_runtime.domain.values import new_workspace_id
from prp_runtime.workspace.models import (
    Workspace,
    WorkspaceRootMapping,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from prp_runtime.workspace.local import resolve_local_workspace
from prp_runtime.workspace.resolver import WorkspaceResolveError, WorkspaceResolver

T0 = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def make_workspace(**overrides: object) -> Workspace:
    values: dict[str, object] = {
        "workspace_id": new_workspace_id(),
        "owner_id": "tenant-owner",
        "alias": "project-main",
        "source": {
            "source_type": WorkspaceSourceType.SERVER_ALIAS,
            "server_alias": "repo-main",
        },
        "created_at": T0,
    }
    values.update(overrides)
    return Workspace(**values)  # type: ignore[arg-type]


def test_server_alias_resolves_to_a_redacted_lifecycle_handle(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("safe", encoding="utf-8")
    resolver = WorkspaceResolver(
        WorkspaceRootMapping.model_validate({"repo-main": str(root)})
    )

    handle = resolver.resolve(make_workspace(), owner_id="tenant-owner")
    try:
        assert handle.server_alias == "repo-main"
        assert handle.owner_id == "tenant-owner"
        assert [entry.path for entry in handle.snapshot_manifest().entries] == ["README.md"]
        assert str(root) not in repr(handle)
    finally:
        handle.close()
        handle.close()
    with pytest.raises(WorkspaceResolveError, match="closed"):
        _ = handle.backend


@pytest.mark.parametrize(
    "workspace_factory, expected",
    [
        (lambda: make_workspace(), "not configured"),
        (
            lambda: make_workspace(
                source={
                    "source_type": WorkspaceSourceType.BRIDGE_GRANT,
                    "bridge_grant": "grant_01",
                }
            ),
            "not available",
        ),
        (lambda: make_workspace(status=WorkspaceStatus.SUSPENDED), "not active"),
    ],
)
def test_logical_scope_failures_do_not_expose_root(
    tmp_path, workspace_factory, expected: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    resolver = WorkspaceResolver(
        WorkspaceRootMapping.model_validate({"repo-main": str(root)})
    )
    if expected == "not configured":
        resolver = WorkspaceResolver(
            WorkspaceRootMapping.model_validate({"other": str(root)})
        )

    with pytest.raises(WorkspaceResolveError, match=expected) as excinfo:
        resolver.resolve(workspace_factory(), owner_id="tenant-owner")
    assert str(root) not in str(excinfo.value)


def test_owner_mismatch_missing_root_symlink_and_file_fail_closed(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    resolver = WorkspaceResolver(
        WorkspaceRootMapping.model_validate({"repo-main": str(root)})
    )
    with pytest.raises(WorkspaceResolveError, match="owner"):
        resolver.resolve(make_workspace(), owner_id="other-owner")

    for candidate in (tmp_path / "missing", tmp_path / "file"):
        if candidate.name == "file":
            candidate.write_text("not a directory", encoding="utf-8")
        failing = WorkspaceResolver(
            WorkspaceRootMapping.model_validate({"repo-main": str(candidate)})
        )
        with pytest.raises(WorkspaceResolveError) as excinfo:
            failing.resolve(make_workspace(), owner_id="tenant-owner")
        assert str(candidate) not in str(excinfo.value)

    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    failing = WorkspaceResolver(
        WorkspaceRootMapping.model_validate({"repo-main": str(link)})
    )
    with pytest.raises(WorkspaceResolveError) as excinfo:
        failing.resolve(make_workspace(), owner_id="tenant-owner")
    assert str(link) not in str(excinfo.value)


def test_resolver_rejects_an_intermediate_root_symlink(tmp_path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    root = real_parent / "repo"
    root.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    resolver = WorkspaceResolver(
        WorkspaceRootMapping.model_validate({"repo-main": str(alias_parent / "repo")})
    )

    with pytest.raises(WorkspaceResolveError, match="symlink"):
        resolver.resolve(make_workspace(), owner_id="tenant-owner")


def test_resolver_rejects_invalid_root_configuration_without_io() -> None:
    with pytest.raises(ValidationError):
        WorkspaceRootMapping.model_validate({"repo-main": "relative"})


def test_local_root_resolves_to_a_redacted_backend_handle(tmp_path) -> None:
    root = tmp_path / "local-repo"
    root.mkdir()
    (root / "README.md").write_text("safe", encoding="utf-8")

    handle = resolve_local_workspace(root, owner_id="local-owner")
    try:
        assert handle.owner_id == "local-owner"
        assert handle.public_identity() == {
            "workspace_id": handle.workspace_id,
            "owner_id": "local-owner",
        }
        assert handle.backend.read_file("README.md").content == "safe"
        assert str(root) not in repr(handle)
        assert str(root) not in str(handle.public_identity())
        assert not any(path.name == ".snapshot" for path in root.iterdir())
    finally:
        handle.close()
        handle.close()
    with pytest.raises(WorkspaceResolveError, match="closed"):
        _ = handle.backend


def test_local_root_validation_fails_closed_without_exposing_path(tmp_path) -> None:
    root = tmp_path / "local-repo"
    root.mkdir()
    missing = tmp_path / "missing"
    file_root = tmp_path / "file"
    file_root.write_text("not a directory", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)

    cases = [
        ("missing", str(missing)),
        ("file", str(file_root)),
        ("symlink", str(link)),
    ]
    for label, candidate in cases:
        del label
        with pytest.raises(WorkspaceResolveError) as excinfo:
            resolve_local_workspace(candidate)
        assert str(root) not in str(excinfo.value)
        assert str(candidate) not in str(excinfo.value)
        assert ".." not in str(excinfo.value)


def test_local_relative_and_noncanonical_roots_share_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "local-repo"
    root.mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(tmp_path)
    absolute = resolve_local_workspace(root, owner_id="local-owner")
    relative = resolve_local_workspace("local-repo", owner_id="local-owner")
    dotted = resolve_local_workspace(nested / ".." / "local-repo", owner_id="local-owner")
    try:
        assert absolute.workspace_id == relative.workspace_id == dotted.workspace_id
        assert str(root) not in absolute.workspace_id
        assert str(root) not in str(absolute.public_identity())
    finally:
        absolute.close()
        relative.close()
        dotted.close()

