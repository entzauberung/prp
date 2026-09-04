"""Result assembly.

The client-facing result is derived from persisted facts only: the artifact a
work unit actually produced, the usage that was actually measured and the run's
recorded status. A model's own claim of completion is never a result.
"""

from collections.abc import Sequence

from prp_runtime.domain.enums import ExecutionLocation, ExecutionStrategy, RunStatus
from prp_runtime.domain.models import (
    Artifact,
    ArtifactKind,
    DomainModel,
    ErrorInfo,
    Run,
    Session,
    Usage,
)
from prp_runtime.domain.values import RunId, SessionId, WorkspaceId
from prp_runtime.runtime.context import ANSWER_ARTIFACT_NAME
from prp_runtime.storage.sqlite import MissingEntityError, SqliteStore
from prp_runtime.workspace.backend import ExportBundleSelection
from prp_runtime.workspace.models import Snapshot, SnapshotStatus

__all__ = [
    "CloudBundleError",
    "CloudBundleFile",
    "CloudResultBundle",
    "RunResult",
    "assemble_cloud_bundle",
    "assemble_run_result",
    "find_answer_artifact",
    "select_verified_cloud_snapshot",
]


class CloudBundleError(ValueError):
    """A structural CLOUD export rejection with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class CloudBundleFile(DomainModel):
    """One authorized file inside a CLOUD result bundle."""

    path: str
    sha256: str
    size: int
    content: str


class CloudResultBundle(DomainModel):
    """Bounded verified CLOUD export; never includes host roots or secrets."""

    run_id: RunId
    session_id: SessionId
    workspace_id: WorkspaceId
    snapshot_id: str
    status: RunStatus
    strategy: ExecutionStrategy | None = None
    files: tuple[CloudBundleFile, ...] = ()
    change_set_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    excluded_paths: tuple[str, ...] = ()
    total_bytes: int = 0
    file_count: int = 0


def select_verified_cloud_snapshot(
    *,
    run: Run,
    session: Session,
    snapshots: Sequence[Snapshot],
    verified_snapshot_id: str | None = None,
) -> Snapshot:
    """Choose the requested run's verified snapshot, never the live last READY."""
    if session.workspace_id is None:
        raise CloudBundleError("bundle_unknown", "session workspace is required")
    if run.completed_at is None:
        raise CloudBundleError(
            "bundle_unverified",
            "only a succeeded run can export a verified bundle",
        )
    scoped = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.workspace_id == session.workspace_id
        and snapshot.status is SnapshotStatus.READY
        and snapshot.completed_at is not None
        and snapshot.completed_at <= run.completed_at
    )
    if verified_snapshot_id is not None:
        matches = tuple(
            snapshot for snapshot in scoped if snapshot.snapshot_id == verified_snapshot_id
        )
        if len(matches) != 1:
            raise CloudBundleError(
                "bundle_unverified",
                "run-bound snapshot is missing or not READY for this run",
            )
        return matches[0]
    if not scoped:
        raise CloudBundleError(
            "bundle_unverified",
            "only a READY snapshot can be exported",
        )
    if len(scoped) != 1:
        raise CloudBundleError(
            "bundle_unknown",
            "export snapshot is ambiguous without run-bound provenance",
        )
    return scoped[0]


def assemble_cloud_bundle(
    *,
    run: Run,
    session: Session,
    snapshot: Snapshot,
    selection: ExportBundleSelection,
    change_set_ids: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
) -> CloudResultBundle:
    """Assemble a CLOUD bundle from authorized verified facts only."""
    session_location = session.agent_options.execution_location
    run_location = run.request.agent_options.execution_location
    if (
        session_location is not ExecutionLocation.CLOUD
        or run_location is not ExecutionLocation.CLOUD
    ):
        raise CloudBundleError(
            "bundle_not_cloud",
            "CLOUD export is only available for CLOUD runs",
        )
    if session.workspace_id != snapshot.workspace_id:
        raise CloudBundleError(
            "bundle_unknown",
            "snapshot workspace does not match the session",
        )
    if run.status is not RunStatus.SUCCEEDED:
        raise CloudBundleError(
            "bundle_unverified",
            "only a succeeded run can export a verified bundle",
        )
    if snapshot.status is not SnapshotStatus.READY:
        raise CloudBundleError(
            "bundle_unverified",
            "only a READY snapshot can be exported",
        )
    if run.completed_at is None or snapshot.completed_at is None:
        raise CloudBundleError(
            "bundle_unverified",
            "export snapshot must be completed with the requested run",
        )
    if snapshot.completed_at > run.completed_at:
        raise CloudBundleError(
            "bundle_unverified",
            "a later workspace snapshot cannot be exported for this run",
        )
    if run.strategy is ExecutionStrategy.PROGRESSIVE and not evidence_ids:
        raise CloudBundleError(
            "bundle_unverified",
            "Progressive export requires verified evidence",
        )
    files = tuple(
        CloudBundleFile(
            path=item.path,
            sha256=item.sha256,
            size=item.size,
            content=item.content,
        )
        for item in selection.files
    )
    return CloudResultBundle(
        run_id=run.run_id,
        session_id=session.session_id,
        workspace_id=session.workspace_id,
        snapshot_id=snapshot.snapshot_id,
        status=run.status,
        strategy=run.strategy,
        files=files,
        change_set_ids=change_set_ids,
        evidence_ids=evidence_ids,
        excluded_paths=selection.excluded_paths,
        total_bytes=selection.total_bytes,
        file_count=selection.file_count,
    )


class RunResult(DomainModel):
    """The assembled outcome of a run."""

    run_id: RunId
    status: RunStatus
    strategy: ExecutionStrategy | None = None
    graph_version: int = 1
    output_text: str | None = None
    output_kind: ArtifactKind | None = None
    artifact_id: str | None = None
    usage: Usage = Usage()
    error: ErrorInfo | None = None


async def find_answer_artifact(store: SqliteStore, run: Run) -> Artifact | None:
    """Read the answer artifact from the run's explicit final work unit."""
    if run.final_work_unit_id is not None:
        try:
            work_unit = await store.get_work_unit(run.final_work_unit_id)
        except MissingEntityError:
            return None
        if work_unit.run_id != run.run_id or work_unit.graph_version != run.graph_version:
            return None
        work_unit_id = work_unit.work_unit_id
    else:
        work_units = await store.list_work_units(
            run.run_id, graph_version=run.graph_version
        )
        if len(work_units) != 1:
            return None
        work_unit_id = work_units[0].work_unit_id
    artifacts = await store.list_artifacts(work_unit_id)
    return next(
        (
            artifact
            for artifact in reversed(artifacts)
            if artifact.name == ANSWER_ARTIFACT_NAME
            and artifact.run_id == run.run_id
            and artifact.work_unit_id == work_unit_id
        ),
        None,
    )


async def assemble_run_result(store: SqliteStore, run_id: str) -> RunResult:
    """Read a run and build its client-facing result."""
    run = await store.get_run(run_id)
    artifact = await find_answer_artifact(store, run)
    return RunResult(
        run_id=run.run_id,
        status=run.status,
        strategy=run.strategy,
        graph_version=run.graph_version,
        output_text=None if artifact is None else artifact.content,
        output_kind=None if artifact is None else artifact.kind,
        artifact_id=None if artifact is None else artifact.artifact_id,
        usage=run.usage,
        error=run.error,
    )
