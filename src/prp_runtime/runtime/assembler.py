"""Result assembly.

The client-facing result is derived from persisted facts only: the artifact a
work unit actually produced, the usage that was actually measured and the run's
recorded status. A model's own claim of completion is never a result.
"""

from prp_runtime.domain.enums import ExecutionStrategy, RunStatus
from prp_runtime.domain.models import (
    Artifact,
    ArtifactKind,
    DomainModel,
    ErrorInfo,
    Run,
    Usage,
)
from prp_runtime.domain.values import RunId
from prp_runtime.runtime.context import ANSWER_ARTIFACT_NAME
from prp_runtime.storage.sqlite import SqliteStore

__all__ = ["RunResult", "assemble_run_result", "find_answer_artifact"]


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
    """The most recent answer artifact of the run's current graph version."""
    latest: Artifact | None = None
    for work_unit in await store.list_work_units(run.run_id, graph_version=run.graph_version):
        for artifact in await store.list_artifacts(work_unit.work_unit_id):
            if artifact.name != ANSWER_ARTIFACT_NAME:
                continue
            if latest is None or artifact.created_at >= latest.created_at:
                latest = artifact
    return latest


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
