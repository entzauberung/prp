"""R0 contract tests for workspace-scoped tool composition."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from prp_runtime.domain.enums import ExecutionLocation, IsolationMode, ResourceAccess
from prp_runtime.domain.models import AgentRequestOptions, ExecutionScope, WorkspaceGrant
from prp_runtime.domain.values import (
    new_principal_id,
    new_run_id,
    new_session_id,
    new_snapshot_id,
    new_workspace_id,
)
from prp_runtime.runtime.tooling import (
    ToolRuntimeError,
    WorkspaceToolRuntimeFactory,
)
from prp_runtime.tools.registry import ToolRegistry
from prp_runtime.workspace.backend import WorkspaceBackend
from prp_runtime.workspace.isolation import LocalIsolationBackend, SlotContext
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotManifest,
    SnapshotStatus,
    WorkspaceSourceType,
)
from prp_runtime.workspace.resolver import ResolvedWorkspace, WorkspaceResolveError

T0 = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


class FakeBackend:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class FakeExecutor:
    pass


def make_facts(
    *,
    workspace_root: Path | None = None,
    access: tuple[ResourceAccess, ...] = (ResourceAccess.READ,),
) -> tuple[ExecutionScope, ResolvedWorkspace, Snapshot, FakeBackend]:
    principal_id = new_principal_id()
    workspace_id = new_workspace_id()
    backend = FakeBackend()
    resolved = ResolvedWorkspace(
        workspace_id=workspace_id,
        owner_id=principal_id,
        server_alias="project-main",
        backend=backend,  # type: ignore[arg-type]
        workspace_root=workspace_root,
    )
    scope = ExecutionScope(
        run_id=new_run_id(),
        session_id=new_session_id(),
        principal_id=principal_id,
        workspace_id=workspace_id,
        grant=WorkspaceGrant(
            principal_id=principal_id,
            workspace_id=workspace_id,
            access=access,
        ),
    )
    snapshot = Snapshot(
        snapshot_id=new_snapshot_id(),
        workspace_id=workspace_id,
        status=SnapshotStatus.READY,
        created_at=T0,
        completed_at=T0,
        file_count=0,
        total_size=0,
    )
    return scope, resolved, snapshot, backend


def test_runtime_exposes_logical_catalog_and_closes_idempotently() -> None:
    scope, resolved, snapshot, backend = make_facts()
    runtime = WorkspaceToolRuntimeFactory().build(
        scope=scope,
        resolved_workspace=resolved,
        snapshot=snapshot,
        registry=ToolRegistry(),
        executor=FakeExecutor(),  # type: ignore[arg-type]
    )

    assert runtime.workspace_id == scope.workspace_id
    assert runtime.server_alias == "project-main"
    assert runtime.snapshot_id == snapshot.snapshot_id
    assert runtime.catalog == ()
    assert "/" not in repr(runtime)

    runtime.close()
    runtime.close()

    assert backend.close_count == 1
    with pytest.raises(ToolRuntimeError, match="closed"):
        _ = runtime.registry


def test_bridge_source_and_mismatched_facts_fail_closed() -> None:
    scope, resolved, snapshot, _ = make_facts()
    factory = WorkspaceToolRuntimeFactory()

    with pytest.raises(ToolRuntimeError):
        factory.build(
            scope=scope,
            resolved_workspace=resolved,
            snapshot=snapshot.model_copy(update={"workspace_id": new_workspace_id()}),
            registry=ToolRegistry(),
            executor=FakeExecutor(),  # type: ignore[arg-type]
        )

    with pytest.raises(WorkspaceResolveError, match="bridge"):
        factory.build(
            scope=scope,
            resolved_workspace=resolved,
            snapshot=snapshot,
            registry=ToolRegistry(),
            executor=FakeExecutor(),  # type: ignore[arg-type]
            source_type=WorkspaceSourceType.BRIDGE_GRANT,
        )


def test_factory_binds_the_complete_cloud_registry_for_read_write_grant(
    tmp_path: Path,
) -> None:
    scope, resolved, snapshot, backend = make_facts(
        workspace_root=tmp_path,
        access=(ResourceAccess.READ, ResourceAccess.WRITE),
    )
    scope = scope.model_copy(
        update={
            "agent_options": AgentRequestOptions(
                isolation_mode=IsolationMode.HOST,
                execution_location=ExecutionLocation.CLOUD,
            )
        }
    )

    store = object()
    runtime = WorkspaceToolRuntimeFactory().build(
        scope=scope,
        resolved_workspace=resolved,
        snapshot=snapshot,
        store=store,  # type: ignore[arg-type]
        snapshot_manifest=SnapshotManifest(),
        rg_path=Path("/bin/true"),
    )

    assert runtime.registry.names == (
        "list_files",
        "read_file",
        "search_text",
        "get_diff",
        "get_status",
        "apply_patch",
        "run_targeted_test",
    )
    assert {descriptor.name for descriptor in runtime.catalog} == set(runtime.registry.names)
    assert all("root" not in descriptor.model_dump() for descriptor in runtime.catalog)
    assert runtime.executor.approval_store is store

    runtime.close()
    assert backend.close_count == 1


def test_factory_closes_workspace_when_a_runner_fails_during_construction(
    tmp_path: Path,
) -> None:
    scope, resolved, snapshot, backend = make_facts(
        workspace_root=tmp_path,
        access=(ResourceAccess.READ, ResourceAccess.WRITE),
    )

    class FailingSandbox:
        def probe(self) -> object:
            raise RuntimeError("sandbox probe failed")

    with pytest.raises(RuntimeError, match="sandbox probe failed"):
        WorkspaceToolRuntimeFactory().build(
            scope=scope,
            resolved_workspace=resolved,
            snapshot=snapshot,
            store=object(),  # type: ignore[arg-type]
            snapshot_manifest=SnapshotManifest(),
            rg_path=Path("/bin/true"),
            sandbox_backend=FailingSandbox(),  # type: ignore[arg-type]
        )

    assert backend.close_count == 1


def test_factory_binds_server_tools_to_an_owned_slot_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_text("source\n", encoding="utf-8")
    scope, _, snapshot, _ = make_facts(
        workspace_root=root,
        access=(ResourceAccess.READ,),
    )
    resolved = ResolvedWorkspace(
        workspace_id=scope.workspace_id,
        owner_id=scope.principal_id,
        server_alias="project-main",
        backend=WorkspaceBackend(root),
        workspace_root=root,
    )
    isolation = LocalIsolationBackend(tmp_path / "isolation")
    base = isolation.create_base_snapshot(root, scope.workspace_id)
    slot = SlotContext(
        isolation,
        snapshot_id=base.snapshot_id,
        work_unit_id="wu_slot_tools",
        owner_id=scope.principal_id,
    )
    runtime = WorkspaceToolRuntimeFactory().build(
        scope=scope,
        resolved_workspace=resolved,
        snapshot=snapshot,
        store=object(),  # type: ignore[arg-type]
        snapshot_manifest=SnapshotManifest(),
        rg_path=Path("/bin/true"),
        slot_context=slot,
    )

    assert runtime.registry.names == ("list_files", "read_file", "search_text")
    assert slot.path != root
    runtime.close()
    assert isolation.active_slot_count == 0
