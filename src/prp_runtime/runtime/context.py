"""Worker context.

A worker sees the instruction of its own work unit and the artifacts of the
dependencies that unit declared. It never receives the whole graph, the event
ledger or other units' history.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import Field, PrivateAttr

from prp_runtime.analysis.syntax import BoundSyntaxReport, SyntaxReport
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
    "MAX_STATIC_FACTS",
    "MAX_STATIC_FACT_BYTES",
    "MAX_STATIC_FACTS_TOTAL_BYTES",
    "StaticFact",
    "WorkerContext",
    "build_worker_context",
    "render_static_facts",
    "select_relevant_facts",
    "static_facts_from_syntax_reports",
]

MAX_STATIC_FACTS = 16
MAX_STATIC_FACT_BYTES = 2 * 1024
MAX_STATIC_FACTS_TOTAL_BYTES = 16 * 1024
_NON_SOURCE_SYNTAX_ERRORS = frozenset(
    {
        "source is absent",
        "unsupported language is unknown",
    }
)
_PRIVATE_FACT_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
)

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


class StaticFact(DomainModel):
    """One bounded, public fact a worker may receive for the current unit."""

    key: str = Field(min_length=1, max_length=512)
    kind: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_]{0,31}$")
    summary: str = Field(min_length=1, max_length=1024)
    work_unit_id: str | None = None
    round_id: str | None = None
    artifact_id: str | None = None


def _is_raw_root_text(value: str) -> bool:
    stripped = value.strip()
    if stripped.startswith(("/", "\\")):
        return True
    return (
        len(stripped) >= 3
        and stripped[0].isalpha()
        and stripped[1] == ":"
        and stripped[2] in {"/", "\\"}
    )


def _fact_leaks_private_context(fact: StaticFact) -> bool:
    blob = f"{fact.key}\n{fact.summary}".lower()
    if any(marker in blob for marker in _PRIVATE_FACT_MARKERS):
        return True
    if "token=" in blob or "token:" in blob:
        return True
    return _is_raw_root_text(fact.key) or _is_raw_root_text(fact.summary)


def _coerce_static_fact(value: StaticFact | Mapping[str, object]) -> StaticFact | None:
    if isinstance(value, StaticFact):
        fact = value
    else:
        try:
            fact = StaticFact.model_validate(value)
        except (TypeError, ValueError):
            return None
    if _fact_leaks_private_context(fact):
        return None
    encoded = len(fact.key.encode("utf-8")) + len(fact.summary.encode("utf-8"))
    if encoded > MAX_STATIC_FACT_BYTES:
        return None
    return fact


def _syntax_report_is_actionable(report: SyntaxReport) -> bool:
    if report.symbols or report.changes:
        return True
    if not report.unknown:
        return False
    return (report.parse_error or "") not in _NON_SOURCE_SYNTAX_ERRORS


def static_facts_from_syntax_reports(
    reports: Sequence[BoundSyntaxReport | SyntaxReport],
    *,
    work_unit_id: str | None = None,
    round_id: str | None = None,
    artifact_id: str | None = None,
) -> tuple[StaticFact, ...]:
    """Project bounded AST reports into public worker facts. No repository dump."""
    facts: list[StaticFact] = []
    seen: set[str] = set()
    for item in reports:
        if isinstance(item, BoundSyntaxReport):
            report = item.report
            unit_id = item.work_unit_id
            item_round = item.round_id
            item_artifact = item.artifact_id
        else:
            report = item
            unit_id = work_unit_id
            item_round = round_id
            item_artifact = artifact_id
        if not _syntax_report_is_actionable(report):
            continue
        for symbol in report.symbols:
            key = symbol.key
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                StaticFact(
                    key=key,
                    kind="ast",
                    summary=f"{symbol.kind.value} {symbol.qualified_name} L{symbol.span.start_line}",
                    work_unit_id=unit_id,
                    round_id=item_round,
                    artifact_id=item_artifact,
                )
            )
        for change in report.changes:
            key = f"change:{change.action.value}:{change.key}"
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                StaticFact(
                    key=key,
                    kind="ast",
                    summary=f"{change.action.value} {change.key}",
                    work_unit_id=unit_id,
                    round_id=item_round,
                    artifact_id=item_artifact,
                )
            )
        if report.unknown and not report.symbols:
            key = f"ast-unknown:{item_artifact or unit_id or 'unscoped'}"
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                StaticFact(
                    key=key,
                    kind="ast",
                    summary=report.parse_error or "unknown syntax",
                    work_unit_id=unit_id,
                    round_id=item_round,
                    artifact_id=item_artifact,
                )
            )
    return tuple(facts)


def select_relevant_facts(
    facts: Sequence[StaticFact | Mapping[str, object]],
    *,
    work_unit_id: str | None = None,
    related_work_unit_ids: Sequence[str] = (),
    round_id: str | None = None,
    max_items: int = MAX_STATIC_FACTS,
    max_bytes: int = MAX_STATIC_FACTS_TOTAL_BYTES,
) -> tuple[StaticFact, ...]:
    """Keep facts for the current unit/round only, within item and byte bounds."""
    allowed_units = {unit_id for unit_id in related_work_unit_ids if unit_id}
    if work_unit_id:
        allowed_units.add(work_unit_id)
    selected: list[StaticFact] = []
    seen: set[str] = set()
    total_bytes = 0
    limit = max(0, max_items)
    byte_limit = max(0, max_bytes)
    for raw in facts:
        fact = _coerce_static_fact(raw)
        if fact is None or fact.key in seen:
            continue
        if allowed_units and fact.work_unit_id and fact.work_unit_id not in allowed_units:
            continue
        if round_id and fact.round_id and fact.round_id != round_id:
            continue
        encoded = len(fact.key.encode("utf-8")) + len(fact.summary.encode("utf-8"))
        if total_bytes + encoded > byte_limit:
            break
        selected.append(fact)
        seen.add(fact.key)
        total_bytes += encoded
        if len(selected) >= limit:
            break
    return tuple(selected)


def render_static_facts(facts: Sequence[StaticFact]) -> str:
    """Render selected facts as a bounded public context block."""
    if not facts:
        return ""
    lines = ["### Static facts"]
    for fact in facts:
        lines.append(f"- [{fact.kind}] {fact.key}: {fact.summary}")
    return "\n".join(lines)


class WorkerContext(DomainModel):
    """Everything one worker call is allowed to know."""

    run_id: RunId
    work_unit_id: WorkUnitId
    instruction: str
    instructions: str | None = None
    acceptance_criteria: str | None = None
    output: OutputRequirement = OutputRequirement()
    dependencies: tuple[DependencyArtifact, ...] = ()
    static_facts: tuple[StaticFact, ...] = ()

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
        rendered = render_static_facts(self.static_facts)
        if rendered:
            parts.append(rendered)
        return "\n\n".join(parts)


def build_worker_context(
    work_unit: WorkUnit,
    *,
    instructions: str | None = None,
    dependencies: tuple[DependencyArtifact, ...] = (),
    static_facts: tuple[StaticFact, ...] = (),
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
        static_facts=static_facts,
    )
