"""Worker context.

A worker sees the instruction of its own work unit and the artifacts of the
dependencies that unit declared. It never receives the whole graph, the event
ledger or other units' history.
"""

from prp_runtime.domain.models import DomainModel, Label, OutputRequirement, WorkUnit
from prp_runtime.domain.values import RunId, WorkUnitId

__all__ = [
    "ANSWER_ARTIFACT_NAME",
    "DependencyArtifact",
    "WorkerContext",
    "build_worker_context",
]

ANSWER_ARTIFACT_NAME = "answer"


class DependencyArtifact(DomainModel):
    """One artifact a work unit is allowed to read."""

    work_unit_name: Label
    artifact_name: Label
    content: str


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
