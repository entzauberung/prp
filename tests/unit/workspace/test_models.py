"""Targeted tests for the workspace identity and lifecycle contracts."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from prp_runtime.domain.values import (
    new_snapshot_id,
    new_workspace_id,
    validate_snapshot_id,
    validate_workspace_id,
)
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotEntry,
    SnapshotEntryType,
    SnapshotManifest,
    SnapshotStatus,
    Workspace,
    WorkspaceRoot,
    WorkspaceRootMapping,
    WorkspaceSource,
    WorkspaceSourceType,
    WorkspaceStatus,
    canonical_manifest_hash,
)

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def make_source(
    source_type: WorkspaceSourceType = WorkspaceSourceType.SERVER_ALIAS,
) -> WorkspaceSource:
    if source_type is WorkspaceSourceType.SERVER_ALIAS:
        return WorkspaceSource(source_type=source_type, server_alias="repo-main")
    return WorkspaceSource(source_type=source_type, bridge_grant="grant_01")


def make_workspace(**overrides: object) -> Workspace:
    values: dict[str, object] = {
        "workspace_id": new_workspace_id(),
        "owner_id": "tenant-owner",
        "alias": "project-main",
        "source": make_source(),
        "created_at": T0,
    }
    values.update(overrides)
    return Workspace(**values)  # type: ignore[arg-type]


def make_snapshot(**overrides: object) -> Snapshot:
    values: dict[str, object] = {
        "snapshot_id": new_snapshot_id(),
        "workspace_id": new_workspace_id(),
        "created_at": T0,
    }
    values.update(overrides)
    return Snapshot(**values)  # type: ignore[arg-type]


def make_entry(path: str, *, size: int = 1, sha256: str = "a" * 64) -> SnapshotEntry:
    return SnapshotEntry(
        path=path,
        sha256=sha256,
        size=size,
        entry_type=SnapshotEntryType.FILE,
    )


@pytest.mark.parametrize("source_type", list(WorkspaceSourceType))
def test_source_types_are_closed_and_json_stable(source_type: WorkspaceSourceType) -> None:
    source = make_source(source_type)
    assert WorkspaceSource.model_validate_json(source.model_dump_json()) == source
    with pytest.raises(ValueError):
        WorkspaceSource(source_type="HOST_PATH", server_alias="/home/user/project")


def test_source_accepts_only_one_server_owned_reference() -> None:
    with pytest.raises(ValidationError):
        WorkspaceSource(source_type=WorkspaceSourceType.SERVER_ALIAS)
    with pytest.raises(ValidationError):
        WorkspaceSource(
            source_type=WorkspaceSourceType.SERVER_ALIAS,
            server_alias="repo-main",
            bridge_grant="grant_01",
        )
    with pytest.raises(ValidationError):
        WorkspaceSource(source_type=WorkspaceSourceType.BRIDGE_GRANT, server_alias="repo-main")
    with pytest.raises(ValidationError):
        WorkspaceSource(
            source_type=WorkspaceSourceType.BRIDGE_GRANT,
            bridge_grant="/tmp/project",
        )


def test_workspace_root_mapping_is_strict_and_deterministic() -> None:
    mapping = WorkspaceRootMapping.model_validate(
        {"scratch": "/var/lib/prp/scratch", "repo-main": "/srv/repos/main"}
    )

    assert mapping.aliases == ("repo-main", "scratch")
    assert mapping.root_for("scratch") == "/var/lib/prp/scratch"
    assert mapping.model_dump() == {}
    with pytest.raises(KeyError):
        mapping.root_for("missing")
    with pytest.raises(ValidationError):
        WorkspaceRootMapping(
            entries=(
                WorkspaceRoot(alias="repo-main", root="/srv/one"),
                WorkspaceRoot(alias="repo-main", root="/srv/two"),
            )
        )
    with pytest.raises(ValidationError):
        WorkspaceRootMapping.model_validate({"repo-main": "relative/root"})
    with pytest.raises(ValidationError):
        WorkspaceRootMapping.model_validate({"Repo-main": "/srv/repos/main"})


def test_workspace_rejects_absolute_path_and_unknown_public_fields() -> None:
    with pytest.raises(ValidationError):
        make_workspace(alias="/home/user/project")
    with pytest.raises(ValidationError):
        make_workspace(host_path="/home/user/project")
    assert "host_path" not in make_workspace().model_dump()


def test_workspace_lifecycle_has_one_terminal_state() -> None:
    active = make_workspace()
    assert active.status is WorkspaceStatus.ACTIVE
    with pytest.raises(ValidationError):
        make_workspace(status=WorkspaceStatus.REVOKED)
    revoked = make_workspace(
        status=WorkspaceStatus.REVOKED,
        closed_at=T0 + timedelta(seconds=1),
    )
    assert revoked.status.is_terminal is True
    assert Workspace.model_validate_json(revoked.model_dump_json()) == revoked


def test_snapshot_lifecycle_and_bounded_counts_round_trip() -> None:
    creating = make_snapshot()
    assert creating.status is SnapshotStatus.CREATING
    ready = make_snapshot(
        status=SnapshotStatus.READY,
        completed_at=T0 + timedelta(seconds=1),
        file_count=3,
        total_size=128,
    )
    assert ready.status.is_terminal is True
    assert Snapshot.model_validate_json(ready.model_dump_json()) == ready
    with pytest.raises(ValidationError):
        make_snapshot(file_count=-1)
    with pytest.raises(ValidationError):
        make_snapshot(status=SnapshotStatus.READY)


def test_manifest_hash_is_canonical_and_order_independent() -> None:
    first = make_entry("src/main.py", sha256="a" * 64)
    second = make_entry("README.md", size=2, sha256="b" * 64)
    left = SnapshotManifest(entries=(first, second))
    right = SnapshotManifest(entries=(second, first))
    assert left.manifest_hash == right.manifest_hash
    assert left.total_size == 3
    assert canonical_manifest_hash(left.entries) == left.manifest_hash


def test_manifest_rejects_duplicate_or_unsafe_entries() -> None:
    entry = make_entry("src/main.py")
    with pytest.raises(ValidationError):
        SnapshotManifest(entries=(entry, entry))
    for path in ("/etc/passwd", "../secret", "src/../secret", "C:/secret", "src\\main.py"):
        with pytest.raises(ValidationError):
            make_entry(path)
    with pytest.raises(ValidationError):
        make_entry("bad-hash", sha256="A" * 64)
    with pytest.raises(ValidationError):
        make_entry("bad-hash", sha256="not-a-hash")


def test_workspace_and_snapshot_ids_are_typed_and_bounded() -> None:
    workspace_id = new_workspace_id()
    snapshot_id = new_snapshot_id()
    assert validate_workspace_id(workspace_id) == workspace_id
    assert validate_snapshot_id(snapshot_id) == snapshot_id
    assert TypeAdapter(str).validate_python(workspace_id).startswith("ws_")
    with pytest.raises(ValueError):
        validate_workspace_id(snapshot_id)
    with pytest.raises(ValueError):
        validate_snapshot_id(workspace_id)
