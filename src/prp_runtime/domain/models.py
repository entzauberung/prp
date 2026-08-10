"""Native domain contracts.

These models are the only internal truth. Inbound bindings normalise into
``NativeRunRequest``; provider payloads are never embedded. Entities reference
each other by identifier, are frozen, and reject unknown fields.

No model carries a private chain of thought. What leaves the runtime is limited
to plan summaries, work units, artifacts, evidence, decisions and usage.
"""

from enum import StrEnum, unique
from typing import Annotated
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.values import (
    AttemptId,
    ModelRef,
    ResourceClaim,
    RunId,
    UtcTimestamp,
    WorkUnitId,
    utc_now,
)
from prp_runtime.json_support import StrictJsonError, strict_json_loads

__all__ = [
    "ARTIFACT_ID_PREFIX",
    "Artifact",
    "ArtifactId",
    "ArtifactKind",
    "Attempt",
    "Budget",
    "ControllerAction",
    "ControllerDecision",
    "EVIDENCE_ID_PREFIX",
    "ErrorCategory",
    "ErrorInfo",
    "Evidence",
    "EvidenceId",
    "EvidenceKind",
    "NativeRunRequest",
    "OutputRequirement",
    "Run",
    "Usage",
    "VerificationResult",
    "WorkUnit",
    "new_artifact_id",
    "new_evidence_id",
]

# Mirrors the identifier tail used by prp_runtime.domain.values.
_ID_TAIL = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"

ARTIFACT_ID_PREFIX = "art_"
EVIDENCE_ID_PREFIX = "ev_"

ArtifactId = Annotated[str, StringConstraints(pattern=rf"^{ARTIFACT_ID_PREFIX}{_ID_TAIL}$")]
EvidenceId = Annotated[str, StringConstraints(pattern=rf"^{EVIDENCE_ID_PREFIX}{_ID_TAIL}$")]


def new_artifact_id() -> str:
    """Generate a fresh artifact id."""
    return f"{ARTIFACT_ID_PREFIX}{uuid4().hex}"


def new_evidence_id() -> str:
    """Generate a fresh evidence id."""
    return f"{EVIDENCE_ID_PREFIX}{uuid4().hex}"


def _reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


def _require_standard_json(text: str, *, field_message: str) -> None:
    """Confirm ``text`` is standard JSON, via the runtime's only parser.

    Rejection of ``NaN``, ``Infinity`` and ``-Infinity`` lives in
    ``strict_json_loads`` alone; this only translates its refusal into the field
    error pydantic expects, keeping the reason but not the traceback.
    """
    try:
        strict_json_loads(text)
    except StrictJsonError as error:
        raise ValueError(f"{field_message}: {error.reason}") from error


Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
PromptText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonBlankText = Annotated[str, AfterValidator(_reject_blank)]


@unique
class ArtifactKind(StrEnum):
    """Supported artifact payload shapes. No binary or multimodal payload."""

    TEXT = "TEXT"
    JSON = "JSON"


@unique
class EvidenceKind(StrEnum):
    """How a verdict about an artifact was produced."""

    DETERMINISTIC_CHECK = "DETERMINISTIC_CHECK"
    MODEL_REVIEW = "MODEL_REVIEW"


@unique
class VerificationResult(StrEnum):
    """The verdict of one check.

    ``INCONCLUSIVE`` is a first class outcome: a check that cannot decide says so
    instead of reporting a pass it cannot prove.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def is_pass(self) -> bool:
        return self is VerificationResult.PASS


@unique
class ErrorCategory(StrEnum):
    """Classification of a failed attempt or run."""

    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    AUTH = "AUTH"
    INVALID_REQUEST = "INVALID_REQUEST"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    NETWORK = "NETWORK"
    CANCELLED = "CANCELLED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    INTERNAL = "INTERNAL"
    UNKNOWN = "UNKNOWN"


@unique
class ControllerAction(StrEnum):
    """Deterministic control actions. A model can propose, never decide."""

    SELECT_STRATEGY = "SELECT_STRATEGY"
    ESCALATE_STRATEGY = "ESCALATE_STRATEGY"
    ESCALATE_MODEL = "ESCALATE_MODEL"
    ACCEPT_ARTIFACT = "ACCEPT_ARTIFACT"
    REJECT_ARTIFACT = "REJECT_ARTIFACT"
    COMMIT_PLAN = "COMMIT_PLAN"
    REJECT_PLAN = "REJECT_PLAN"
    REVISE_PLAN = "REVISE_PLAN"
    INVALIDATE_WORK_UNIT = "INVALIDATE_WORK_UNIT"
    STOP_ON_BUDGET = "STOP_ON_BUDGET"
    CANCEL = "CANCEL"


class DomainModel(BaseModel):
    """Base contract: immutable and closed to unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorInfo(DomainModel):
    """A classified, redacted failure description.

    The message must never contain an API key, a full upstream request or an
    internal path.
    """

    category: ErrorCategory
    message: NonBlankText


class Usage(DomainModel):
    """Measured consumption. Nothing here is estimated.

    ``strong_model_tokens`` is the part of the token total attributable to
    planner-role calls. ``elapsed_ms`` is provider call time, not wall clock of
    a run.
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    strong_model_tokens: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @model_validator(mode="after")
    def _strong_tokens_are_a_subset(self) -> "Usage":
        if self.strong_model_tokens > self.total_tokens:
            raise ValueError("strong_model_tokens cannot exceed total tokens")
        return self

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            strong_model_tokens=self.strong_model_tokens + other.strong_model_tokens,
            elapsed_ms=self.elapsed_ms + other.elapsed_ms,
        )


class Budget(DomainModel):
    """Optional hard ceilings. ``None`` means no ceiling for that dimension."""

    max_total_tokens: int | None = Field(default=None, ge=0)
    max_strong_model_tokens: int | None = Field(default=None, ge=0)
    max_attempts: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)
    max_plan_revisions: int | None = Field(default=None, ge=0)
    deadline: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _strong_ceiling_fits_total(self) -> "Budget":
        if (
            self.max_total_tokens is not None
            and self.max_strong_model_tokens is not None
            and self.max_strong_model_tokens > self.max_total_tokens
        ):
            raise ValueError("max_strong_model_tokens cannot exceed max_total_tokens")
        return self


class OutputRequirement(DomainModel):
    """What an acceptable result must look like."""

    kind: ArtifactKind = ArtifactKind.TEXT
    json_schema: str | None = None

    @model_validator(mode="after")
    def _schema_requires_json_kind(self) -> "OutputRequirement":
        if self.json_schema is None:
            return self
        if self.kind is not ArtifactKind.JSON:
            raise ValueError("json_schema is only allowed when kind is JSON")
        _require_standard_json(
            self.json_schema, field_message="json_schema must be valid JSON"
        )
        return self


class NativeRunRequest(DomainModel):
    """The single normalised request shape.

    Every inbound binding produces this. ``MANUAL`` routing pins one strategy and
    forbids automatic escalation; ``AUTO`` leaves the choice to the controller.
    """

    input: PromptText
    instructions: PromptText | None = None
    routing_policy: RoutingPolicy = RoutingPolicy.AUTO
    strategy: ExecutionStrategy | None = None
    budget: Budget = Field(default_factory=Budget)
    output: OutputRequirement = Field(default_factory=OutputRequirement)

    @model_validator(mode="after")
    def _routing_matches_strategy(self) -> "NativeRunRequest":
        if self.routing_policy is RoutingPolicy.MANUAL and self.strategy is None:
            raise ValueError("manual routing requires an explicit strategy")
        if self.routing_policy is RoutingPolicy.AUTO and self.strategy is not None:
            raise ValueError("auto routing must not pin a strategy")
        return self


class Run(DomainModel):
    """A persisted run. ``strategy`` is empty until the controller decides."""

    run_id: RunId
    request: NativeRunRequest
    status: RunStatus = RunStatus.PENDING
    strategy: ExecutionStrategy | None = None
    graph_version: int = Field(default=1, ge=1)
    usage: Usage = Field(default_factory=Usage)
    error: ErrorInfo | None = None
    created_at: UtcTimestamp = Field(default_factory=utc_now)
    started_at: UtcTimestamp | None = None
    completed_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _lifecycle_is_consistent(self) -> "Run":
        if self.status is not RunStatus.PENDING and self.started_at is None:
            raise ValueError("a run that left PENDING must have started_at")
        if self.status is RunStatus.PENDING and self.started_at is not None:
            raise ValueError("a PENDING run must not have started_at")
        if self.status.is_terminal and self.completed_at is None:
            raise ValueError("a terminal run must have completed_at")
        if not self.status.is_terminal and self.completed_at is not None:
            raise ValueError("a non-terminal run must not have completed_at")
        if self.status is RunStatus.FAILED and self.error is None:
            raise ValueError("a FAILED run must carry an error")
        if self.status is not RunStatus.FAILED and self.error is not None:
            raise ValueError("only a FAILED run may carry an error")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at cannot precede created_at")
        if (
            self.completed_at is not None
            and self.started_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("completed_at cannot precede started_at")
        if self.request.strategy is not None and self.strategy not in (
            None,
            self.request.strategy,
        ):
            raise ValueError("a manually pinned strategy cannot be replaced")
        return self


class WorkUnit(DomainModel):
    """One unit of work inside a run's execution graph."""

    work_unit_id: WorkUnitId
    run_id: RunId
    graph_version: int = Field(default=1, ge=1)
    name: Label
    instruction: PromptText
    acceptance_criteria: PromptText | None = None
    output: OutputRequirement = Field(default_factory=OutputRequirement)
    status: WorkUnitStatus = WorkUnitStatus.PENDING
    depends_on: tuple[WorkUnitId, ...] = ()
    resource_claims: tuple[ResourceClaim, ...] = ()
    created_at: UtcTimestamp = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _dependencies_and_claims_are_sane(self) -> "WorkUnit":
        if self.work_unit_id in self.depends_on:
            raise ValueError("a work unit cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("duplicate dependency")
        claim_keys = [(claim.resource, claim.access) for claim in self.resource_claims]
        if len(set(claim_keys)) != len(claim_keys):
            raise ValueError("duplicate resource claim")
        return self

    def writes(self) -> tuple[str, ...]:
        """Resources this unit claims for writing."""
        return tuple(
            claim.resource
            for claim in self.resource_claims
            if claim.access is ResourceAccess.WRITE
        )


class Attempt(DomainModel):
    """One provider call for one work unit.

    Raw provider requests and responses are not stored. Produced content lives in
    an ``Artifact``.
    """

    attempt_id: AttemptId
    run_id: RunId
    work_unit_id: WorkUnitId
    attempt_index: int = Field(default=1, ge=1)
    role: ModelRole
    model: ModelRef
    status: AttemptStatus = AttemptStatus.PENDING
    provider_request_id: str | None = None
    usage: Usage | None = None
    error: ErrorInfo | None = None
    created_at: UtcTimestamp = Field(default_factory=utc_now)
    started_at: UtcTimestamp | None = None
    completed_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _lifecycle_is_consistent(self) -> "Attempt":
        if self.status is AttemptStatus.PENDING and self.started_at is not None:
            raise ValueError("a PENDING attempt must not have started_at")
        if self.status is not AttemptStatus.PENDING and self.started_at is None:
            raise ValueError("an attempt that left PENDING must have started_at")
        if self.status.is_terminal and self.completed_at is None:
            raise ValueError("a terminal attempt must have completed_at")
        if not self.status.is_terminal and self.completed_at is not None:
            raise ValueError("a non-terminal attempt must not have completed_at")
        if self.status is AttemptStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a SUCCEEDED attempt must not carry an error")
        if self.status is AttemptStatus.FAILED and self.error is None:
            raise ValueError("a FAILED attempt must carry an error")
        if self.completed_at is not None and self.started_at is not None:
            if self.completed_at < self.started_at:
                raise ValueError("completed_at cannot precede started_at")
        return self


class Artifact(DomainModel):
    """A produced result. Completion is driven by artifacts, not self-reports."""

    artifact_id: ArtifactId
    run_id: RunId
    work_unit_id: WorkUnitId
    attempt_id: AttemptId
    name: Label
    kind: ArtifactKind = ArtifactKind.TEXT
    content: NonBlankText
    created_at: UtcTimestamp = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _json_content_parses(self) -> "Artifact":
        if self.kind is ArtifactKind.JSON:
            _require_standard_json(
                self.content, field_message="JSON artifact content must be valid JSON"
            )
        return self


class Evidence(DomainModel):
    """A verdict about one artifact of one work unit.

    ``result`` is the whole truth and it is required. ``passed`` is a read-only
    projection of it, not a field: there is no way to store a boolean that
    disagrees with the verdict, and no way to record a verdict that was never
    reached.

    The boolean alone cannot be the record. ``FAIL`` and ``INCONCLUSIVE`` both
    project to ``passed is False``, so a check that could not decide would later
    be indistinguishable from one that proved the output wrong. Keeping ``result``
    as the stored form is what stops "not proven" from being read as "proven bad".
    """

    evidence_id: EvidenceId
    run_id: RunId
    work_unit_id: WorkUnitId
    artifact_id: ArtifactId
    kind: EvidenceKind
    rule: Label | None = None
    result: VerificationResult
    detail: NonBlankText
    created_at: UtcTimestamp = Field(default_factory=utc_now)

    @property
    def passed(self) -> bool:
        """Whether this verdict permits acceptance. Derived, never stored."""
        return self.result.is_pass

    @model_validator(mode="after")
    def _deterministic_checks_name_their_rule(self) -> "Evidence":
        if self.kind is EvidenceKind.DETERMINISTIC_CHECK and self.rule is None:
            raise ValueError("DETERMINISTIC_CHECK requires rule")
        return self


class ControllerDecision(DomainModel):
    """A recorded deterministic control action with its reason."""

    run_id: RunId
    action: ControllerAction
    rationale: NonBlankText
    from_strategy: ExecutionStrategy | None = None
    to_strategy: ExecutionStrategy | None = None
    work_unit_id: WorkUnitId | None = None
    evidence_ids: tuple[EvidenceId, ...] = ()
    decided_at: UtcTimestamp = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _strategy_fields_match_action(self) -> "ControllerDecision":
        if self.action is ControllerAction.SELECT_STRATEGY and self.to_strategy is None:
            raise ValueError("SELECT_STRATEGY requires to_strategy")
        if self.action is ControllerAction.ESCALATE_STRATEGY:
            if self.from_strategy is None or self.to_strategy is None:
                raise ValueError("ESCALATE_STRATEGY requires from_strategy and to_strategy")
            if self.from_strategy is self.to_strategy:
                raise ValueError("ESCALATE_STRATEGY requires a different strategy")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("duplicate evidence reference")
        return self
