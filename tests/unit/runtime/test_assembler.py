"""CLOUD bundle provenance is bound to the requested verified run."""

from datetime import UTC, datetime, timedelta

import pytest

from prp_runtime.domain.enums import ExecutionLocation, ExecutionStrategy, RunStatus
from prp_runtime.domain.models import (
    AgentRequestOptions,
    NativeRunRequest,
    Run,
    Session,
    SessionStatus,
    WorkspaceGrant,
)
from prp_runtime.domain.values import (
    new_principal_id,
    new_run_id,
    new_session_id,
    new_snapshot_id,
    new_workspace_id,
)
from prp_runtime.runtime.assembler import (
    CloudBundleError,
    assemble_cloud_bundle,
    select_verified_cloud_snapshot,
)
from prp_runtime.workspace.backend import ExportBundleSelection, ExportFile
from prp_runtime.workspace.models import Snapshot, SnapshotStatus

T0 = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


def _session(workspace_id: str | None = None) -> Session:
    principal = new_principal_id()
    workspace = workspace_id or new_workspace_id()
    return Session(
        session_id=new_session_id(),
        principal_id=principal,
        workspace_id=workspace,
        grant=WorkspaceGrant(principal_id=principal, workspace_id=workspace),
        agent_options=AgentRequestOptions(execution_location=ExecutionLocation.CLOUD),
        status=SessionStatus.ACTIVE,
        created_at=T0,
    )


def _run(session: Session, *, completed_at: datetime = T0 + timedelta(minutes=1)) -> Run:
    return Run(
        run_id=new_run_id(),
        request=NativeRunRequest(
            input="export the verified workspace",
            agent_options=AgentRequestOptions(execution_location=ExecutionLocation.CLOUD),
        ),
        status=RunStatus.SUCCEEDED,
        strategy=ExecutionStrategy.DIRECT,
        created_at=T0,
        started_at=T0,
        completed_at=completed_at,
    )


def _snapshot(
    workspace_id: str,
    *,
    completed_at: datetime,
    status: SnapshotStatus = SnapshotStatus.READY,
) -> Snapshot:
    return Snapshot(
        snapshot_id=new_snapshot_id(),
        workspace_id=workspace_id,
        status=status,
        created_at=completed_at,
        completed_at=completed_at,
        file_count=1,
        total_size=1,
    )


def test_select_verified_snapshot_rejects_later_ready_and_ambiguous_sets() -> None:
    session = _session()
    run = _run(session)
    frozen = _snapshot(session.workspace_id, completed_at=run.completed_at)
    later = _snapshot(
        session.workspace_id, completed_at=run.completed_at + timedelta(minutes=5)
    )
    selected = select_verified_cloud_snapshot(
        run=run, session=session, snapshots=(frozen, later)
    )
    assert selected.snapshot_id == frozen.snapshot_id

    bound = _snapshot(session.workspace_id, completed_at=run.completed_at)
    selected_bound = select_verified_cloud_snapshot(
        run=run,
        session=session,
        snapshots=(frozen, bound),
        verified_snapshot_id=bound.snapshot_id,
    )
    assert selected_bound.snapshot_id == bound.snapshot_id
    with pytest.raises(CloudBundleError, match="ambiguous"):
        select_verified_cloud_snapshot(
            run=run, session=session, snapshots=(frozen, bound)
        )
    with pytest.raises(CloudBundleError, match="later workspace snapshot|missing or not READY"):
        select_verified_cloud_snapshot(
            run=run,
            session=session,
            snapshots=(frozen, later),
            verified_snapshot_id=later.snapshot_id,
        )


def test_assemble_cloud_bundle_names_requested_run_and_rejects_later_snapshot() -> None:
    session = _session()
    run = _run(session)
    snapshot = _snapshot(session.workspace_id, completed_at=run.completed_at)
    selection = ExportBundleSelection(
        files=(
            ExportFile(
                path="result.txt",
                sha256="a" * 64,
                size=8,
                content="cloud-ok",
            ),
        ),
        excluded_paths=(".env",),
        total_bytes=8,
    )
    bundle = assemble_cloud_bundle(
        run=run,
        session=session,
        snapshot=snapshot,
        selection=selection,
        change_set_ids=("cs_" + "a" * 32,),
        evidence_ids=("ev_" + "b" * 32,),
    )
    assert bundle.run_id == run.run_id
    assert bundle.snapshot_id == snapshot.snapshot_id
    assert bundle.workspace_id == session.workspace_id
    later = _snapshot(
        session.workspace_id, completed_at=run.completed_at + timedelta(hours=1)
    )
    with pytest.raises(CloudBundleError, match="later workspace snapshot"):
        assemble_cloud_bundle(
            run=run, session=session, snapshot=later, selection=selection
        )

def test_assemble_cloud_bundle_rejects_local_bridge_and_unverified_runs() -> None:
    selection = ExportBundleSelection(files=(), excluded_paths=(), total_bytes=0)
    for location in (ExecutionLocation.LOCAL, ExecutionLocation.BRIDGE):
        session = _session()
        session = session.model_copy(
            update={"agent_options": AgentRequestOptions(execution_location=location)}
        )
        run = _run(session).model_copy(
            update={
                "request": NativeRunRequest(
                    input="export the verified workspace",
                    agent_options=AgentRequestOptions(execution_location=location),
                )
            }
        )
        snapshot = _snapshot(session.workspace_id, completed_at=run.completed_at)
        with pytest.raises(CloudBundleError, match="CLOUD export is only available"):
            assemble_cloud_bundle(
                run=run, session=session, snapshot=snapshot, selection=selection
            )
    session = _session()
    run = _run(session).model_copy(update={"status": RunStatus.FAILED})
    snapshot = _snapshot(session.workspace_id, completed_at=run.completed_at)
    with pytest.raises(CloudBundleError, match="succeeded run"):
        assemble_cloud_bundle(
            run=run, session=session, snapshot=snapshot, selection=selection
        )

