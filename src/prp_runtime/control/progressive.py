"""Pure, auditable Progressive revision policy.

The policy consumes persisted verification/failure facts and a budget snapshot. It
never reads model text, calls a provider, or changes a graph. A revision is only
allowed when an explicit finite revision ceiling exists and the current fact is a
deterministic failure, an inconclusive verification, or a retryable provider
failure.
"""

from datetime import datetime
from enum import StrEnum, unique

from pydantic import Field, model_validator

from prp_runtime.domain.enums import RunStatus, WorkUnitStatus
from prp_runtime.domain.models import (
    Budget,
    DomainModel,
    ErrorCategory,
    ErrorInfo,
    Usage,
    VerificationResult,
)
from prp_runtime.planning.models import PlanRevisionReason
from prp_runtime.verification.verifier import VerificationReport

__all__ = [
    "ProgressiveDecision",
    "ProgressiveDisposition",
    "RevisionDecision",
    "RevisionDisposition",
    "RevisionStopReason",
    "decide_progressive_revision",
    "decide_revision",
]


@unique
class RevisionDisposition(StrEnum):
    """Whether the controller may submit a new graph version."""

    REVISE = "REVISE"
    STOP = "STOP"


ProgressiveDisposition = RevisionDisposition


@unique
class RevisionStopReason(StrEnum):
    """Stable reasons a revision request is not permitted."""

    PASS = "PASS"
    TERMINAL = "TERMINAL"
    CANCELLED = "CANCELLED"
    BUDGET = "BUDGET"
    REVISION_LIMIT = "REVISION_LIMIT"
    NO_REVISION_BUDGET = "NO_REVISION_BUDGET"
    NO_TRIGGER = "NO_TRIGGER"


class RevisionDecision(DomainModel):
    """A serializable Progressive decision with no private reasoning fields."""

    disposition: RevisionDisposition
    reason: PlanRevisionReason | None = None
    stop_reason: RevisionStopReason | None = None
    rationale: str = Field(min_length=1)
    graph_version: int = Field(ge=1)
    revision_count: int = Field(ge=0)
    next_graph_version: int | None = Field(default=None, ge=2)

    @model_validator(mode="after")
    def _shape_matches_disposition(self) -> "RevisionDecision":
        if self.disposition is RevisionDisposition.REVISE:
            if self.reason is None:
                raise ValueError("a revision decision must carry a revision reason")
            if self.stop_reason is not None:
                raise ValueError("a revision decision must not carry a stop reason")
            if self.next_graph_version != self.graph_version + 1:
                raise ValueError(
                    "a revision decision must advance exactly one graph version"
                )
        else:
            if self.reason is not None:
                raise ValueError("a stop decision must not carry a revision reason")
            if self.stop_reason is None:
                raise ValueError("a stop decision must carry a stop reason")
            if self.next_graph_version is not None:
                raise ValueError("a stop decision must not carry a next graph version")
        return self

    @property
    def should_revise(self) -> bool:
        """Whether a new graph version may be proposed."""
        return self.disposition is RevisionDisposition.REVISE

    @property
    def trigger_reason(self) -> PlanRevisionReason | None:
        """Compatibility-friendly name for the public revision reason."""
        return self.reason


ProgressiveDecision = RevisionDecision

_RETRYABLE_PROVIDER_CATEGORIES = frozenset(
    {ErrorCategory.TIMEOUT, ErrorCategory.RATE_LIMIT, ErrorCategory.NETWORK}
)


def _budget_is_exhausted(
    budget: Budget,
    *,
    usage: Usage | None,
    now: datetime | None,
) -> bool:
    if budget.deadline is not None and now is not None and now >= budget.deadline:
        return True
    if usage is None:
        return False
    if (
        budget.max_total_tokens is not None
        and usage.total_tokens >= budget.max_total_tokens
    ):
        return True
    return (
        budget.max_strong_model_tokens is not None
        and usage.strong_model_tokens >= budget.max_strong_model_tokens
    )


def _stop(
    *,
    reason: RevisionStopReason,
    rationale: str,
    graph_version: int,
    revision_count: int,
) -> RevisionDecision:
    return RevisionDecision(
        disposition=RevisionDisposition.STOP,
        stop_reason=reason,
        rationale=rationale,
        graph_version=graph_version,
        revision_count=revision_count,
    )


def decide_revision(
    *,
    budget: Budget,
    revision_count: int,
    graph_version: int = 1,
    verification: VerificationReport | None = None,
    verification_result: VerificationResult | None = None,
    error: ErrorInfo | None = None,
    run_status: RunStatus = RunStatus.RUNNING,
    work_unit_status: WorkUnitStatus | None = None,
    cancel_requested: bool = False,
    budget_exhausted: bool = False,
    usage: Usage | None = None,
    now: datetime | None = None,
) -> RevisionDecision:
    """Decide whether one current-graph fact permits a finite revision.

    ``verification`` is preferred when a complete report is available; the
    result-only argument keeps the policy useful at a Controller boundary where
    only the persisted aggregate verdict has been loaded. Contradictory signals
    are rejected instead of being resolved by truthiness.
    """
    if revision_count < 0:
        raise ValueError("revision_count must not be negative")
    if graph_version < 1:
        raise ValueError("graph_version must be at least 1")
    if verification is not None and verification_result is not None:
        if verification.result is not verification_result:
            raise ValueError("verification and verification_result disagree")
    result = verification.result if verification is not None else verification_result
    if result is not None and error is not None:
        raise ValueError("verification and error signals are mutually exclusive")

    if cancel_requested or run_status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
        return _stop(
            reason=RevisionStopReason.CANCELLED,
            rationale="cancellation takes precedence over Progressive revision",
            graph_version=graph_version,
            revision_count=revision_count,
        )
    if run_status.is_terminal:
        return _stop(
            reason=RevisionStopReason.TERMINAL,
            rationale="a terminal run cannot accept a Progressive revision",
            graph_version=graph_version,
            revision_count=revision_count,
        )
    if work_unit_status in (WorkUnitStatus.CANCELLED, WorkUnitStatus.INVALIDATED):
        return _stop(
            reason=RevisionStopReason.CANCELLED,
            rationale="the current work unit is no longer runnable",
            graph_version=graph_version,
            revision_count=revision_count,
        )
    if result is VerificationResult.PASS:
        return _stop(
            reason=RevisionStopReason.PASS,
            rationale="deterministic verification passed",
            graph_version=graph_version,
            revision_count=revision_count,
        )

    revision_reason: PlanRevisionReason | None = None
    if result is VerificationResult.FAIL:
        revision_reason = PlanRevisionReason.VERIFICATION_FAILED
    elif result is VerificationResult.INCONCLUSIVE:
        revision_reason = PlanRevisionReason.VERIFICATION_INCONCLUSIVE
    elif error is not None:
        if error.category is ErrorCategory.VERIFICATION_FAILED:
            revision_reason = PlanRevisionReason.VERIFICATION_FAILED
        elif error.category in _RETRYABLE_PROVIDER_CATEGORIES:
            revision_reason = PlanRevisionReason.PROVIDER_FAILED
        elif error.category is ErrorCategory.BUDGET_EXCEEDED:
            budget_exhausted = True

    if budget_exhausted or _budget_is_exhausted(budget, usage=usage, now=now):
        return _stop(
            reason=RevisionStopReason.BUDGET,
            rationale="the declared budget does not permit another revision",
            graph_version=graph_version,
            revision_count=revision_count,
        )
    if budget.max_plan_revisions is None:
        return _stop(
            reason=RevisionStopReason.NO_REVISION_BUDGET,
            rationale="Progressive revision requires an explicit revision budget",
            graph_version=graph_version,
            revision_count=revision_count,
        )
    if revision_count >= budget.max_plan_revisions:
        return _stop(
            reason=RevisionStopReason.REVISION_LIMIT,
            rationale="the Progressive revision budget is exhausted",
            graph_version=graph_version,
            revision_count=revision_count,
        )
    if revision_reason is None:
        return _stop(
            reason=RevisionStopReason.NO_TRIGGER,
            rationale="no deterministic Progressive revision trigger was recorded",
            graph_version=graph_version,
            revision_count=revision_count,
        )

    return RevisionDecision(
        disposition=RevisionDisposition.REVISE,
        reason=revision_reason,
        rationale=f"{revision_reason.value} permits one bounded revision",
        graph_version=graph_version,
        revision_count=revision_count,
        next_graph_version=graph_version + 1,
    )


decide_progressive_revision = decide_revision
