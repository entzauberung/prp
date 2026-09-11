"""Pure revision and reuse laws. No I/O, no model, no storage."""

from dataclasses import dataclass

from prp.vocabulary import (
    ReuseDisposition,
    ReuseReason,
    RevisionDisposition,
    RevisionStopReason,
    RunStatus,
    VerificationResult,
    WorkUnitStatus,
)

__all__ = [
    "ComparisonOutcome",
    "ReuseDecision",
    "RevisionDecision",
    "decide_reuse",
    "decide_revision",
]


class ComparisonOutcome:
    NO_GAIN = "NO_GAIN"
    REGRESSION = "REGRESSION"
    INCONCLUSIVE = "INCONCLUSIVE"
    IMPROVED = "IMPROVED"
    EVIDENCE_ADDED = "EVIDENCE_ADDED"
    BASELINE = "BASELINE"


@dataclass(frozen=True, slots=True)
class RevisionDecision:
    disposition: RevisionDisposition
    stop_reason: RevisionStopReason | None = None
    trigger: str | None = None
    rationale: str = ""
    graph_version: int = 1
    revision_count: int = 0
    next_graph_version: int | None = None

    def __post_init__(self) -> None:
        if self.disposition is RevisionDisposition.REVISE:
            if self.trigger is None or self.stop_reason is not None:
                raise ValueError("REVISE requires a trigger and no stop reason")
            if self.next_graph_version != self.graph_version + 1:
                raise ValueError("REVISE must advance exactly one graph version")
        else:
            if self.stop_reason is None or self.trigger is not None:
                raise ValueError("STOP requires a stop reason and no trigger")
            if self.next_graph_version is not None:
                raise ValueError("STOP must not carry a next graph version")


@dataclass(frozen=True, slots=True)
class ReuseDecision:
    disposition: ReuseDisposition
    reason: ReuseReason
    rationale: str

    def __post_init__(self) -> None:
        if self.disposition is ReuseDisposition.REUSE:
            if self.reason is not ReuseReason.ALL_FACTS_MATCH:
                raise ValueError("REUSE requires ALL_FACTS_MATCH")
        elif self.reason is ReuseReason.ALL_FACTS_MATCH:
            raise ValueError("RECOMPUTE cannot use ALL_FACTS_MATCH")


_RETRYABLE = frozenset({"TIMEOUT", "RATE_LIMIT", "NETWORK"})


def _stop(
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
    run_status: RunStatus = RunStatus.RUNNING,
    work_unit_status: WorkUnitStatus | None = None,
    verification_result: VerificationResult | None = None,
    error_category: str | None = None,
    cancel_requested: bool = False,
    budget_exhausted: bool = False,
    max_plan_revisions: int | None = None,
    revision_count: int = 0,
    graph_version: int = 1,
    comparison_outcome: str | None = None,
) -> RevisionDecision:
    if revision_count < 0:
        raise ValueError("revision_count must not be negative")
    if graph_version < 1:
        raise ValueError("graph_version must be at least 1")
    if verification_result is not None and error_category is not None:
        raise ValueError("verification and error signals are mutually exclusive")

    if cancel_requested or run_status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
        return _stop(
            RevisionStopReason.CANCELLED,
            "cancellation takes precedence over Progressive revision",
            graph_version,
            revision_count,
        )
    if run_status.is_terminal:
        return _stop(
            RevisionStopReason.TERMINAL,
            "a terminal run cannot accept a Progressive revision",
            graph_version,
            revision_count,
        )
    if work_unit_status in (WorkUnitStatus.CANCELLED, WorkUnitStatus.INVALIDATED):
        return _stop(
            RevisionStopReason.CANCELLED,
            "the current work unit is no longer runnable",
            graph_version,
            revision_count,
        )
    if verification_result is VerificationResult.PASS:
        return _stop(
            RevisionStopReason.PASS,
            "deterministic verification passed",
            graph_version,
            revision_count,
        )
    if verification_result is VerificationResult.INCONCLUSIVE and comparison_outcome is None:
        return _stop(
            RevisionStopReason.INCONCLUSIVE,
            "deterministic verification did not produce sufficient facts",
            graph_version,
            revision_count,
        )
    if comparison_outcome == ComparisonOutcome.NO_GAIN:
        return _stop(
            RevisionStopReason.NO_GAIN,
            "the candidate round added no protocol gain",
            graph_version,
            revision_count,
        )
    if comparison_outcome == ComparisonOutcome.REGRESSION:
        return _stop(
            RevisionStopReason.REGRESSION,
            "the candidate round regressed on recorded facts",
            graph_version,
            revision_count,
        )
    if (
        comparison_outcome == ComparisonOutcome.INCONCLUSIVE
        and verification_result is VerificationResult.INCONCLUSIVE
    ):
        return _stop(
            RevisionStopReason.INCONCLUSIVE,
            "the candidate round remained inconclusive",
            graph_version,
            revision_count,
        )

    trigger: str | None = None
    if verification_result is VerificationResult.FAIL:
        trigger = "VERIFICATION_FAILED"
    elif verification_result is VerificationResult.INCONCLUSIVE:
        trigger = "VERIFICATION_INCONCLUSIVE"
    elif error_category == "VERIFICATION_FAILED":
        trigger = "VERIFICATION_FAILED"
    elif error_category in _RETRYABLE:
        trigger = "PROVIDER_FAILED"
    elif error_category == "BUDGET_EXCEEDED":
        budget_exhausted = True

    if budget_exhausted:
        return _stop(
            RevisionStopReason.BUDGET,
            "the declared budget does not permit another revision",
            graph_version,
            revision_count,
        )
    if max_plan_revisions is None:
        return _stop(
            RevisionStopReason.NO_REVISION_BUDGET,
            "Progressive revision requires an explicit revision budget",
            graph_version,
            revision_count,
        )
    if revision_count >= max_plan_revisions:
        return _stop(
            RevisionStopReason.REVISION_LIMIT,
            "the Progressive revision budget is exhausted",
            graph_version,
            revision_count,
        )
    if trigger is None:
        return _stop(
            RevisionStopReason.NO_TRIGGER,
            "no deterministic Progressive revision trigger was recorded",
            graph_version,
            revision_count,
        )
    return RevisionDecision(
        disposition=RevisionDisposition.REVISE,
        trigger=trigger,
        rationale=f"{trigger} permits one bounded revision",
        graph_version=graph_version,
        revision_count=revision_count,
        next_graph_version=graph_version + 1,
    )


def decide_reuse(
    *,
    historical_succeeded: bool,
    historical_lineage: str | None,
    candidate_lineage: str | None,
    historical_content_fingerprint: str | None,
    candidate_content_fingerprint: str | None,
    historical_dependency_fingerprint: str | None,
    candidate_dependency_fingerprint: str | None,
) -> ReuseDecision:
    if not historical_succeeded:
        return ReuseDecision(
            ReuseDisposition.RECOMPUTE,
            ReuseReason.HISTORICAL_UNIT_NOT_SUCCEEDED,
            "the historical work unit did not produce a proven successful result",
        )
    facts = (
        historical_lineage,
        candidate_lineage,
        historical_content_fingerprint,
        candidate_content_fingerprint,
        historical_dependency_fingerprint,
        candidate_dependency_fingerprint,
    )
    if any(value is None for value in facts):
        return ReuseDecision(
            ReuseDisposition.RECOMPUTE,
            ReuseReason.MISSING_LINEAGE_OR_FINGERPRINT,
            "lineage and both execution fingerprints are required for reuse",
        )
    if historical_lineage != candidate_lineage:
        return ReuseDecision(
            ReuseDisposition.RECOMPUTE,
            ReuseReason.LINEAGE_CHANGED,
            "the candidate lineage differs from the historical lineage",
        )
    if historical_content_fingerprint != candidate_content_fingerprint:
        return ReuseDecision(
            ReuseDisposition.RECOMPUTE,
            ReuseReason.CONTENT_FINGERPRINT_CHANGED,
            "the public execution contract changed",
        )
    if historical_dependency_fingerprint != candidate_dependency_fingerprint:
        return ReuseDecision(
            ReuseDisposition.RECOMPUTE,
            ReuseReason.DEPENDENCY_FINGERPRINT_CHANGED,
            "the declared dependency lineage changed",
        )
    return ReuseDecision(
        ReuseDisposition.REUSE,
        ReuseReason.ALL_FACTS_MATCH,
        "lineage, execution fingerprints, and dependency facts match",
    )
