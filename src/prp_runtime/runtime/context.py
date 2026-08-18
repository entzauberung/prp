"""Worker context.

A worker sees the instruction of its own work unit and the artifacts of the
dependencies that unit declared. It never receives the whole graph, the event
ledger or other units' history.
"""

from collections.abc import Mapping
from pathlib import Path

from pydantic import PrivateAttr

from prp_runtime.domain.models import (
    DomainModel,
    Evidence,
    Label,
    OutputRequirement,
    WorkUnit,
)
from prp_runtime.domain.values import PrincipalId, RunId, WorkspaceId, WorkUnitId
from prp_runtime.policy.models import (
    DevEvidenceMetadata,
    DevScope,
    serialize_dev_evidence,
)

__all__ = [
    "ANSWER_ARTIFACT_NAME",
    "DependencyArtifact",
    "DevExecutionContext",
    "WorkerContext",
    "build_worker_context",
]

ANSWER_ARTIFACT_NAME = "answer"


class DependencyArtifact(DomainModel):
    """One artifact a work unit is allowed to read."""

    work_unit_name: Label
    artifact_name: Label
    content: str


class DevExecutionContext(DomainModel):
    """DEV scope plus an internal-only temporary host root.

    The root is a private runtime fact and therefore cannot enter ``model_dump``
    or public evidence. Callers can resolve relative paths only through this
    context, which keeps the absolute path out of the DEV contract.
    """

    scope: DevScope
    _temporary_root: Path | None = PrivateAttr(default=None)

    @classmethod
    def from_temporary_root(
        cls,
        scope: DevScope,
        temporary_root: str | Path,
    ) -> "DevExecutionContext":
        """Bind a DEV scope to an internal temporary directory."""
        root = Path(temporary_root)
        if not root.is_absolute():
            raise ValueError("DEV temporary root must be absolute")
        context = cls(scope=scope)
        object.__setattr__(context, "_temporary_root", root)
        return context

    @property
    def internal_temporary_root(self) -> Path:
        """Return the root for internal tool execution only."""
        if self._temporary_root is None:
            raise RuntimeError("DEV context has no temporary root")
        return self._temporary_root

    def resolve_internal_path(self, relative_path: str) -> Path:
        """Resolve one safe relative path without exposing it in evidence."""
        if (
            not relative_path
            or relative_path.startswith(("/", "\\"))
            or (len(relative_path) >= 2 and relative_path[1] == ":")
            or "\\" in relative_path
        ):
            raise ValueError("DEV internal path must be relative POSIX syntax")
        parts = relative_path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("DEV internal path contains an unsafe component")
        return self.internal_temporary_root.joinpath(*parts)

    def evidence_metadata(
        self,
        *,
        principal_id: PrincipalId | None = None,
        workspace_id: WorkspaceId | None = None,
    ) -> DevEvidenceMetadata:
        """Return metadata derived from the authenticated DEV scope."""
        return self.scope.evidence_metadata(
            principal_id=principal_id,
            workspace_id=workspace_id,
        )

    def serialize_evidence(
        self,
        evidence: Evidence | Mapping[str, object],
    ) -> dict[str, object]:
        """Serialize public evidence without private host context."""
        return serialize_dev_evidence(evidence, scope=self.scope)


class WorkerContext(DomainModel):
    """Everything one worker call is allowed to know."""

    run_id: RunId
    work_unit_id: WorkUnitId
    instruction: str
    instructions: str | None = None
    acceptance_criteria: str | None = None
    output: OutputRequirement = OutputRequirement()
    dependencies: tuple[DependencyArtifact, ...] = ()

    def render_instructions(self) -> str | None:
        """The system text for the model, or ``None`` when there is nothing to say."""
        parts: list[str] = []
        if self.instructions is not None:
            parts.append(self.instructions)
        if self.output.json_schema is not None:
            parts.append("Reply with one JSON document that satisfies the provided schema.")
        if self.acceptance_criteria is not None:
            parts.append(f"The result must satisfy: {self.acceptance_criteria}")
        return "\n\n".join(parts) if parts else None

    def render_input(self) -> str:
        """The user text for the model: this unit's task plus its declared inputs."""
        parts = [self.instruction]
        for dependency in self.dependencies:
            parts.append(
                f"### Input from {dependency.work_unit_name} "
                f"({dependency.artifact_name})\n{dependency.content}"
            )
        return "\n\n".join(parts)


def build_worker_context(
    work_unit: WorkUnit,
    *,
    instructions: str | None = None,
    dependencies: tuple[DependencyArtifact, ...] = (),
) -> WorkerContext:
    """Build the context for one work unit."""
    return WorkerContext(
        run_id=work_unit.run_id,
        work_unit_id=work_unit.work_unit_id,
        instruction=work_unit.instruction,
        instructions=instructions,
        acceptance_criteria=work_unit.acceptance_criteria,
        output=work_unit.output,
        dependencies=dependencies,
    )
