"""Native domain contracts.

These models are the only internal truth. Inbound bindings normalise into
``NativeRunRequest``; provider payloads are never embedded. Entities reference
each other by identifier, are frozen, and reject unknown fields.

No model carries a private chain of thought. What leaves the runtime is limited
to plan summaries, work units, artifacts, evidence, decisions and usage.
"""

import json
from decimal import Decimal, InvalidOperation
from enum import StrEnum, unique
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StringConstraints,
    model_validator,
)

from prp_runtime.domain.enums import (
    AgentMode,
    AttemptStatus,
    ExecutionLocation,
    ExecutionStrategy,
    IsolationMode,
    MergeLedgerStatus,
    ModelRole,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.values import (
    AttemptId,
    MergeId,
    ModelRef,
    PrincipalId,
    ResourceClaim,
    RunId,
    SessionId,
    SnapshotId,
    UtcTimestamp,
    WorkspaceId,
    WorkUnitId,
    utc_now,
)
from prp_runtime.json_support import StrictJsonError, strict_json_loads

__all__ = [
    "ARTIFACT_ID_PREFIX",
    "MAX_PROVIDER_TOOL_COUNT",
    "MAX_PROVIDER_TOOL_DESCRIPTOR_BYTES",
    "MAX_PROVIDER_TOOLS_BYTES",
    "MAX_AGENT_HISTORY_ITEM_BYTES",
    "AgentRequestOptions",
    "AgentHistoryItem",
    "AgentHistoryRecord",
    "AgentToolCall",
    "AgentToolResult",
    "AgentTurn",
    "ProviderToolDescriptor",
    "Artifact",
    "ArtifactId",
    "ArtifactKind",
    "Attempt",
    "AttemptCost",
    "Budget",
    "ControllerAction",
    "ControllerDecision",
    "EVIDENCE_ID_PREFIX",
    "ErrorCategory",
    "ErrorInfo",
    "ExecutionScope",
    "Evidence",
    "EvidenceId",
    "EvidenceKind",
    "GlobalCheck",
    "GlobalVerificationReport",
    "MAX_ARTIFACT_CONTENT_BYTES",
    "MergeLedger",
    "MAX_PUBLIC_JSON_BYTES",
    "MAX_PUBLIC_TEXT_CHARS",
    "NativeRunRequest",
    "Principal",
    "Session",
    "SessionCreateRequest",
    "SessionStatus",
    "Money",
    "OutputRequirement",
    "RoutingIntent",
    "Run",
    "RunMetrics",
    "Usage",
    "VerificationResult",
    "WorkspaceGrant",
    "WorkUnit",
    "new_artifact_id",
    "new_evidence_id",
]

# Mirrors the identifier tail used by prp_runtime.domain.values.
_ID_TAIL = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"

ARTIFACT_ID_PREFIX = "art_"
EVIDENCE_ID_PREFIX = "ev_"

# Persisted public facts are intentionally bounded before they reach storage.
# The model-level JSON cap is a last line of defence; fields with a larger
# semantic payload (such as an artifact) still have an explicit field check.
MAX_PUBLIC_TEXT_CHARS = 100_000
MAX_PUBLIC_JSON_BYTES = 256 * 1024
MAX_ARTIFACT_CONTENT_BYTES = 256 * 1024

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


def _strict_json_bytes(
    value: object,
    *,
    field_message: str,
    max_bytes: int | None = None,
) -> bytes:
    """Encode one public value using the standard JSON grammar only."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_message} must contain standard JSON") from error
    if max_bytes is not None and len(encoded) > max_bytes:
        raise ValueError(f"{field_message} exceeds the size limit")
    return encoded


def _assert_text_bytes(value: str, *, field_message: str, max_bytes: int) -> None:
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_message} exceeds the size limit")


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
PromptText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_PUBLIC_TEXT_CHARS,
    ),
]
NonBlankText = Annotated[
    str,
    StringConstraints(max_length=MAX_PUBLIC_TEXT_CHARS),
    AfterValidator(_reject_blank),
]


def _reject_persisted_lineage(value: str) -> str:
    if value.startswith(("run_", "wu_", "att_", "art_", "ev_")):
        raise ValueError("lineage_key must be proposal-local, not a persisted id")
    return value


LineageKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
    AfterValidator(_reject_persisted_lineage),
]
Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RoundId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^round_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
    ),
]
FactId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
ProviderRequestId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]


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

    @model_validator(mode="after")
    def _public_shape_is_strict_json(self) -> "DomainModel":
        """Keep persisted model output finite, JSON-safe and bounded."""
        _strict_json_bytes(
            self.model_dump(mode="json"),
            field_message="domain model",
            max_bytes=MAX_PUBLIC_JSON_BYTES,
        )
        return self


MergeChangeSetId = Annotated[
    str,
    StringConstraints(pattern=r"^cs_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"),
]


class MergeLedger(DomainModel):
    """Durable merge facts without filesystem locations or failure bodies."""

    merge_id: MergeId
    run_id: RunId
    workspace_id: WorkspaceId
    base_snapshot_id: SnapshotId
    change_set_ids: tuple[MergeChangeSetId, ...] = ()
    input_digest: Fingerprint
    status: MergeLedgerStatus = MergeLedgerStatus.PLANNED
    merged_snapshot_id: SnapshotId | None = None
    merged_content_hash: Fingerprint | None = None
    promoted_content_hash: Fingerprint | None = None
    created_at: UtcTimestamp = Field(default_factory=utc_now)
    completed_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _lifecycle_facts_match_status(self) -> "MergeLedger":
        if len(self.change_set_ids) != len(set(self.change_set_ids)):
            raise ValueError("merge ledger contains duplicate ChangeSet ids")
        candidate_facts = (self.merged_snapshot_id, self.merged_content_hash)
        if self.status in {MergeLedgerStatus.PLANNED, MergeLedgerStatus.RUNNING}:
            if any(fact is not None for fact in candidate_facts) or self.promoted_content_hash:
                raise ValueError("planned or running merge cannot contain output facts")
            if self.completed_at is not None:
                raise ValueError("planned or running merge cannot be completed")
        elif self.status is MergeLedgerStatus.MERGED:
            if any(fact is None for fact in candidate_facts):
                raise ValueError("merged ledger requires candidate snapshot facts")
            if self.promoted_content_hash is not None or self.completed_at is None:
                raise ValueError("merged ledger has invalid promotion facts")
        elif self.status is MergeLedgerStatus.PROMOTED:
            if any(fact is None for fact in candidate_facts):
                raise ValueError("promoted ledger requires candidate snapshot facts")
            if self.promoted_content_hash is None or self.completed_at is None:
                raise ValueError("promoted ledger requires promotion facts")
        else:
            if any(fact is not None for fact in candidate_facts):
                raise ValueError("unresolved merge cannot contain output facts")
            if self.promoted_content_hash is not None or self.completed_at is None:
                raise ValueError("unresolved merge requires a terminal timestamp")
        return self


MAX_AGENT_ARGUMENT_BYTES = 64 * 1024
MAX_AGENT_RESULT_BYTES = 64 * 1024
MAX_AGENT_TURN_TEXT_CHARS = 100_000
MAX_AGENT_HISTORY_ITEMS = 128
MAX_AGENT_HISTORY_ITEM_BYTES = 128 * 1024
MAX_PROVIDER_TOOL_COUNT = 64
MAX_PROVIDER_TOOL_DESCRIPTOR_BYTES = 16 * 1024
MAX_PROVIDER_TOOLS_BYTES = 64 * 1024


class AgentToolCall(DomainModel):
    """A provider-neutral, bounded request for one registered tool."""

    kind: Literal["tool_call"] = "tool_call"
    call_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    tool_name: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=64,
            pattern=r"^[a-z][a-z0-9_.-]{0,63}$",
        ),
    ]
    arguments: dict[str, JsonValue] = {}

    @model_validator(mode="after")
    def _arguments_are_bounded(self) -> "AgentToolCall":
        _strict_json_bytes(
            self.arguments,
            field_message="agent tool arguments",
            max_bytes=MAX_AGENT_ARGUMENT_BYTES,
        )
        return self


class AgentToolResult(DomainModel):
    """A bounded structured tool observation safe to place in turn history."""

    kind: Literal["tool_result"] = "tool_result"
    call_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    status: ToolCallStatus
    result: dict[str, JsonValue] | None = None
    output: str = Field(default="", max_length=MAX_AGENT_RESULT_BYTES)
    truncated: StrictBool = False

    @model_validator(mode="after")
    def _result_is_terminal_and_bounded(self) -> "AgentToolResult":
        if not self.status.is_terminal:
            raise ValueError("agent tool result must be terminal")
        if self.result is not None:
            _strict_json_bytes(
                self.result,
                field_message="agent tool result",
                max_bytes=MAX_AGENT_RESULT_BYTES,
            )
        _assert_text_bytes(
            self.output,
            field_message="agent tool output",
            max_bytes=MAX_AGENT_RESULT_BYTES,
        )
        return self


class AgentTurn(DomainModel):
    """Exactly one public model turn: final text or tool calls, never hidden CoT."""

    kind: Literal["turn"] = "turn"
    text: str | None = Field(default=None, max_length=MAX_AGENT_TURN_TEXT_CHARS)
    tool_calls: tuple[AgentToolCall, ...] = ()

    @model_validator(mode="after")
    def _turn_has_one_shape(self) -> "AgentTurn":
        has_text = self.text is not None
        has_tools = bool(self.tool_calls)
        if has_text == has_tools:
            raise ValueError("agent turn must contain text or tool calls, exclusively")
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("agent turn tool call ids must be unique")
        return self


AgentHistoryItem = Annotated[
    AgentTurn | AgentToolResult,
    Field(discriminator="kind"),
]

_FORBIDDEN_HISTORY_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "chain_of_thought",
        "cot",
        "credential",
        "env",
        "password",
        "provider_body",
        "raw_provider_body",
        "secret",
        "token",
    }
)


def _assert_public_history(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_HISTORY_KEYS:
                raise ValueError(f"agent history must not contain {key}")
            _assert_public_history(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_public_history(nested)


class AgentHistoryRecord(DomainModel):
    """One bounded public history item for one persisted agent attempt."""

    run_id: RunId
    work_unit_id: WorkUnitId
    attempt_id: AttemptId
    sequence: int = Field(ge=1, le=MAX_AGENT_HISTORY_ITEMS)
    idempotency_key: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
    ]
    item: AgentHistoryItem
    created_at: UtcTimestamp = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _item_is_public_strict_json_and_bounded(self) -> "AgentHistoryRecord":
        data = self.item.model_dump(mode="json")
        _assert_public_history(data)
        _strict_json_bytes(
            data,
            field_message="agent history item",
            max_bytes=MAX_AGENT_HISTORY_ITEM_BYTES,
        )
        return self


ProviderToolName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_.-]{0,63}$",
    ),
]


class ProviderToolDescriptor(DomainModel):
    """The non-executable public view of one server-registered tool.

    Only information needed to form a provider tool schema crosses this
    boundary. Handler, effect, executable and workspace metadata stay in the
    server-side registry.
    """

    name: ProviderToolName
    description: Annotated[str, StringConstraints(max_length=2_048)] = ""
    input_schema: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _descriptor_is_standard_json_and_bounded(self) -> "ProviderToolDescriptor":
        _strict_json_bytes(
            self.model_dump(mode="json"),
            field_message="provider tool descriptor",
            max_bytes=MAX_PROVIDER_TOOL_DESCRIPTOR_BYTES,
        )
        return self


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


def _as_decimal(value: object) -> Decimal:
    """Parse money without first passing through binary floating point."""
    if isinstance(value, bool):
        raise ValueError("money must be a finite decimal")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("money must be a finite decimal") from error
    if not amount.is_finite():
        raise ValueError("money must be a finite decimal")
    return amount


Money = Annotated[Decimal, BeforeValidator(_as_decimal), Field(ge=Decimal("0"))]
TOKENS_PER_MILLION = Decimal(1_000_000)


class AttemptCost(DomainModel):
    """The exact cost of one measured Provider attempt.

    Prices are captured with the attempt and all arithmetic uses ``Decimal``.
    ``None`` usage is represented by the absence of an ``AttemptCost`` rather
    than by a fabricated zero-cost record.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    input_price_per_million_tokens: Money
    output_price_per_million_tokens: Money
    input_cost: Money
    output_cost: Money
    total_cost: Money

    @classmethod
    def from_usage(
        cls,
        usage: Usage | None,
        *,
        input_price_per_million_tokens: Decimal | float | int | str,
        output_price_per_million_tokens: Decimal | float | int | str,
    ) -> "AttemptCost | None":
        """Calculate cost only when the Provider reported token usage."""
        if usage is None:
            return None
        input_price = _as_decimal(input_price_per_million_tokens)
        output_price = _as_decimal(output_price_per_million_tokens)
        input_cost = input_price * Decimal(usage.input_tokens) / TOKENS_PER_MILLION
        output_cost = output_price * Decimal(usage.output_tokens) / TOKENS_PER_MILLION
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            input_price_per_million_tokens=input_price,
            output_price_per_million_tokens=output_price,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )

    @property
    def cost(self) -> Decimal:
        """Compatibility alias for the total measured cost."""
        return self.total_cost

    @model_validator(mode="after")
    def _cost_components_are_consistent(self) -> "AttemptCost":
        expected_input = (
            self.input_price_per_million_tokens * Decimal(self.input_tokens)
            / TOKENS_PER_MILLION
        )
        expected_output = (
            self.output_price_per_million_tokens * Decimal(self.output_tokens)
            / TOKENS_PER_MILLION
        )
        if self.input_cost != expected_input:
            raise ValueError("input_cost does not match tokens and profile price")
        if self.output_cost != expected_output:
            raise ValueError("output_cost does not match tokens and profile price")
        if self.total_cost != self.input_cost + self.output_cost:
            raise ValueError("total_cost does not match cost components")
        return self


class RunMetrics(DomainModel):
    """Additive run facts with explicit unknown values.

    Provider elapsed time is the sum of measured Provider call durations. Wall
    clock is the separately measured run duration; it is never inferred from
    Provider elapsed time. ``None`` means the fact was not measured.
    """

    usage: Usage | None = None
    provider_elapsed_ms: int | None = Field(default=None, ge=0)
    wall_clock_ms: int | None = Field(default=None, ge=0)
    cost: Money | None = None

    @classmethod
    def from_attempt(
        cls,
        *,
        usage: Usage | None,
        cost: AttemptCost | None = None,
        wall_clock_ms: int | None = None,
    ) -> "RunMetrics":
        """Build metrics for one attempt without conflating its clocks."""
        if usage is None and cost is not None:
            raise ValueError("unknown usage cannot carry a known attempt cost")
        return cls(
            usage=usage,
            provider_elapsed_ms=None if usage is None else usage.elapsed_ms,
            wall_clock_ms=wall_clock_ms,
            cost=None if cost is None else cost.total_cost,
        )

    @property
    def total_cost(self) -> Decimal | None:
        """The known aggregate cost, or ``None`` when any cost is unknown."""
        return self.cost

    def __add__(self, other: "RunMetrics") -> "RunMetrics":
        usage = None if self.usage is None or other.usage is None else self.usage + other.usage
        provider_elapsed = _add_known(self.provider_elapsed_ms, other.provider_elapsed_ms)
        wall_clock = _add_known(self.wall_clock_ms, other.wall_clock_ms)
        cost = _add_decimal(self.cost, other.cost)
        return RunMetrics(
            usage=usage,
            provider_elapsed_ms=provider_elapsed,
            wall_clock_ms=wall_clock,
            cost=cost,
        )

    @model_validator(mode="after")
    def _facts_are_consistent(self) -> "RunMetrics":
        if self.usage is None and self.cost is not None:
            raise ValueError("unknown usage cannot carry a known cost")
        if (
            self.usage is not None
            and self.provider_elapsed_ms is not None
            and self.provider_elapsed_ms != self.usage.elapsed_ms
        ):
            raise ValueError("provider_elapsed_ms must match usage.elapsed_ms")
        return self


def _add_known(first: int | None, second: int | None) -> int | None:
    if first is None or second is None:
        return None
    return first + second


def _add_decimal(first: Decimal | None, second: Decimal | None) -> Decimal | None:
    if first is None or second is None:
        return None
    return first + second


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
    json_schema: Annotated[str, StringConstraints(max_length=MAX_PUBLIC_TEXT_CHARS)] | None = None

    @model_validator(mode="after")
    def _schema_requires_json_kind(self) -> "OutputRequirement":
        if self.json_schema is None:
            return self
        if self.kind is not ArtifactKind.JSON:
            raise ValueError("json_schema is only allowed when kind is JSON")
        _assert_text_bytes(
            self.json_schema,
            field_message="json_schema",
            max_bytes=MAX_PUBLIC_JSON_BYTES,
        )
        _require_standard_json(
            self.json_schema, field_message="json_schema must be valid JSON"
        )
        return self


class RoutingIntent(DomainModel):
    """Explicit client facts used for deterministic AUTO strategy selection."""

    requires_cascade: bool = False
    requires_plan: bool = False
    requires_revision: bool = False
    desired_parallelism: int | None = Field(default=None, ge=1)


class AgentRequestOptions(DomainModel):
    """Orthogonal agent execution facts carried by a native request.

    ``LOCAL`` may combine with ``HOST``. That combination does not create a
    Bridge claim and does not weaken HOST YOLO, which still requires
    ``user_explicit=true``.
    """

    agent_mode: AgentMode = AgentMode.NORMAL
    isolation_mode: IsolationMode = IsolationMode.SANDBOXED
    execution_location: ExecutionLocation = ExecutionLocation.CLOUD
    user_explicit: StrictBool = False

    @model_validator(mode="after")
    def _host_yolo_requires_explicit_user_fact(self) -> "AgentRequestOptions":
        if (
            self.agent_mode is AgentMode.YOLO
            and self.isolation_mode is IsolationMode.HOST
            and not self.user_explicit
        ):
            raise ValueError("HOST YOLO requires user_explicit=true")
        return self


class Principal(DomainModel):
    """Stable authenticated identity; bearer secrets never enter this model."""

    principal_id: PrincipalId
    label: Label = "default"


class WorkspaceGrant(DomainModel):
    """A finite session grant scoped to one server-owned Workspace identity."""

    principal_id: PrincipalId
    workspace_id: WorkspaceId
    access: tuple[ResourceAccess, ...] = (ResourceAccess.READ,)
    expires_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _access_is_unique_and_bounded(self) -> "WorkspaceGrant":
        if not self.access:
            raise ValueError("workspace grant access must not be empty")
        if len(set(self.access)) != len(self.access):
            raise ValueError("workspace grant access must not contain duplicates")
        return self


@unique
class SessionStatus(StrEnum):
    """Lifecycle of one authenticated Agent Session."""

    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"


class SessionCreateRequest(DomainModel):
    """Client-safe session request; it contains no host path or credential."""

    workspace_id: WorkspaceId
    access: tuple[ResourceAccess, ...] = (ResourceAccess.READ,)
    agent_options: AgentRequestOptions = Field(default_factory=AgentRequestOptions)
    expires_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _access_is_unique_and_bounded(self) -> "SessionCreateRequest":
        if not self.access:
            raise ValueError("session access must not be empty")
        if len(set(self.access)) != len(self.access):
            raise ValueError("session access must not contain duplicates")
        return self


class Session(DomainModel):
    """An authenticated, principal-owned Agent Session."""

    session_id: SessionId
    principal_id: PrincipalId
    workspace_id: WorkspaceId
    grant: WorkspaceGrant
    agent_options: AgentRequestOptions = Field(default_factory=AgentRequestOptions)
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: UtcTimestamp = Field(default_factory=utc_now)
    expires_at: UtcTimestamp | None = None
    revoked_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _scope_and_lifecycle_are_consistent(self) -> "Session":
        if (
            self.grant.principal_id != self.principal_id
            or self.grant.workspace_id != self.workspace_id
        ):
            raise ValueError("session grant must match the session principal and workspace")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("session expires_at must be after created_at")
        if self.status is SessionStatus.REVOKED and self.revoked_at is None:
            raise ValueError("a revoked session must have revoked_at")
        if self.status is SessionStatus.ACTIVE and self.revoked_at is not None:
            raise ValueError("an active session must not have revoked_at")
        return self


class ExecutionScope(DomainModel):
    """Owner-scoped persisted facts required to execute one Session run."""

    run_id: RunId
    session_id: SessionId
    principal_id: PrincipalId
    workspace_id: WorkspaceId
    grant: WorkspaceGrant
    agent_options: AgentRequestOptions = Field(default_factory=AgentRequestOptions)

    @model_validator(mode="after")
    def _scope_facts_match(self) -> "ExecutionScope":
        if self.grant.principal_id != self.principal_id:
            raise ValueError("execution grant must match the scope principal")
        if self.grant.workspace_id != self.workspace_id:
            raise ValueError("execution grant must match the scope workspace")
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
    routing: RoutingIntent | None = None
    agent_options: AgentRequestOptions = Field(default_factory=AgentRequestOptions)
    budget: Budget = Field(default_factory=Budget)
    output: OutputRequirement = Field(default_factory=OutputRequirement)

    @model_validator(mode="after")
    def _routing_matches_strategy(self) -> "NativeRunRequest":
        if self.routing_policy is RoutingPolicy.MANUAL and self.strategy is None:
            raise ValueError("manual routing requires an explicit strategy")
        if self.routing_policy is RoutingPolicy.AUTO and self.strategy is not None:
            raise ValueError("auto routing must not pin a strategy")
        if self.routing_policy is RoutingPolicy.MANUAL and self.routing is not None:
            raise ValueError("manual routing must not include automatic routing intent")
        return self


class Run(DomainModel):
    """A persisted run. ``strategy`` is empty until the controller decides."""

    run_id: RunId
    request: NativeRunRequest
    status: RunStatus = RunStatus.PENDING
    strategy: ExecutionStrategy | None = None
    graph_version: int = Field(default=1, ge=1)
    final_work_unit_id: WorkUnitId | None = None
    usage: Usage = Field(default_factory=Usage)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
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
    lineage_key: LineageKey | None = None
    dependency_fingerprint: Fingerprint | None = None
    content_fingerprint: Fingerprint | None = None
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
        fingerprint_fields = (
            self.lineage_key,
            self.dependency_fingerprint,
            self.content_fingerprint,
        )
        if any(value is not None for value in fingerprint_fields) and not all(
            value is not None for value in fingerprint_fields
        ):
            raise ValueError("lineage and fingerprints must be provided together")
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
    provider_request_id: ProviderRequestId | None = None
    usage: Usage | None = None
    cost: AttemptCost | None = None
    error: ErrorInfo | None = None
    created_at: UtcTimestamp = Field(default_factory=utc_now)
    started_at: UtcTimestamp | None = None
    completed_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _lifecycle_is_consistent(self) -> "Attempt":
        if self.usage is None and self.cost is not None:
            raise ValueError("an attempt with unknown usage cannot carry a cost")
        if self.usage is not None and self.cost is not None:
            if (
                self.cost.input_tokens != self.usage.input_tokens
                or self.cost.output_tokens != self.usage.output_tokens
            ):
                raise ValueError("attempt cost tokens must match usage")
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
        _assert_text_bytes(
            self.content,
            field_message="artifact content",
            max_bytes=MAX_ARTIFACT_CONTENT_BYTES,
        )
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


class GlobalCheck(DomainModel):
    """One deterministic fact contributing to a whole-round verdict."""

    kind: Label
    result: VerificationResult
    detail: NonBlankText
    evidence_ids: tuple[EvidenceId, ...] = ()
    fact_ids: tuple[FactId, ...] = ()

    @model_validator(mode="after")
    def _fact_ids_are_unique(self) -> "GlobalCheck":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("global check contains duplicate evidence ids")
        if len(set(self.fact_ids)) != len(self.fact_ids):
            raise ValueError("global check contains duplicate fact ids")
        return self


class GlobalVerificationReport(DomainModel):
    """Recoverable, model-independent verdict for one Progressive round."""

    run_id: RunId
    round_id: RoundId | None = None
    graph_version: int = Field(ge=1)
    result: VerificationResult = VerificationResult.INCONCLUSIVE
    checks: tuple[GlobalCheck, ...] = ()
    final_artifact_ids: tuple[ArtifactId, ...] = ()
    change_set_ids: tuple[MergeChangeSetId, ...] = ()
    evidence_ids: tuple[EvidenceId, ...] = ()
    syntax_report_count: int = Field(default=0, ge=0)
    command_result_count: int = Field(default=0, ge=0)
    usage: Usage | None = None
    metrics: RunMetrics | None = None

    @model_validator(mode="after")
    def _report_facts_are_closed(self) -> "GlobalVerificationReport":
        if len(set(self.final_artifact_ids)) != len(self.final_artifact_ids):
            raise ValueError("global report contains duplicate artifact ids")
        if len(set(self.change_set_ids)) != len(self.change_set_ids):
            raise ValueError("global report contains duplicate ChangeSet ids")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("global report contains duplicate evidence ids")
        kinds = [check.kind for check in self.checks]
        if len(kinds) != len(set(kinds)):
            raise ValueError("global report contains duplicate check kinds")
        results = tuple(check.result for check in self.checks)
        expected = (
            VerificationResult.INCONCLUSIVE
            if not results
            else VerificationResult.FAIL
            if VerificationResult.FAIL in results
            else VerificationResult.INCONCLUSIVE
            if VerificationResult.INCONCLUSIVE in results
            else VerificationResult.PASS
        )
        if self.result is not expected:
            raise ValueError("global report result does not match its checks")
        return self

    @property
    def passed(self) -> bool:
        """Whether every recorded global check is a proven pass."""
        return self.result is VerificationResult.PASS

    def summary(self) -> str:
        """Return a stable one-line summary suitable for a controller event."""
        counts = {
            state: sum(1 for check in self.checks if check.result is state)
            for state in VerificationResult
        }
        return (
            f"{self.result.value}: {counts[VerificationResult.PASS]} passed, "
            f"{counts[VerificationResult.FAIL]} failed, "
            f"{counts[VerificationResult.INCONCLUSIVE]} undecided"
        )


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
