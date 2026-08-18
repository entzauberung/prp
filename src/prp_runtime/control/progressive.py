"""Pure, auditable Progressive revision policy.

The policy consumes persisted verification/failure facts and a budget snapshot. It
never reads model text, calls a provider, or changes a graph. A revision is only
allowed when an explicit finite revision ceiling exists and the current fact is a
deterministic failure, an inconclusive verification, or a retryable provider
failure.
"""

import re
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from enum import StrEnum, unique
from typing import Annotated
from uuid import uuid4

from pydantic import Field, StringConstraints, model_validator

from prp_runtime.domain.enums import RunStatus, WorkUnitStatus
from prp_runtime.domain.models import (
    Budget,
    DomainModel,
    ErrorCategory,
    ErrorInfo,
    EvidenceId,
    GlobalVerificationReport,
    RunMetrics,
    Usage,
    VerificationResult,
    WorkUnit,
)
from prp_runtime.domain.values import RunId, SnapshotId, UtcTimestamp
from prp_runtime.planning.models import PlanRevisionReason
from prp_runtime.verification.verifier import VerificationReport
from prp_runtime.workspace.changes import ChangeSetId

__all__ = [
    "RoundId",
    "RoundStatus",
    "ProgressiveRound",
    "new_round_id",
    "ProgressiveDecision",
    "ProgressiveDisposition",
    "ReuseDecision",
    "ReuseDisposition",
    "ReuseReason",
    "RevisionDecision",
    "RevisionDisposition",
    "RevisionStopReason",
    "ComparisonOutcome",
    "RoundComparison",
    "compare_rounds",
    "decide_progressive_revision",
    "decide_revision",
]

RoundId = Annotated[
    str,
    StringConstraints(pattern=r"^round_[A-Za-z0-9][A-Za-z0-9_-]{3,127}$"),
]


def new_round_id() -> str:
    """Generate one opaque immutable Progressive round identity."""
    return f"round_{uuid4().hex}"


@unique
class RoundStatus(StrEnum):
    """Persisted terminal or prepared state of one immutable round fact."""

    PLANNED = "PLANNED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProgressiveRound(DomainModel):
    """A reproducible Progressive input/output snapshot boundary."""

    round_id: RoundId
    run_id: RunId
    round_index: int = Field(ge=0)
    graph_version: int = Field(ge=1)
    base_snapshot_id: SnapshotId
    merged_snapshot_id: SnapshotId | None = None
    change_set_ids: tuple[ChangeSetId, ...] = ()
    evidence_ids: tuple[EvidenceId, ...] = ()
    status: RoundStatus = RoundStatus.PLANNED
    revision_of_round_id: RoundId | None = None
    revision_reason: str | None = Field(default=None, max_length=512)
    failure_reason: str | None = Field(default=None, max_length=512)
    created_at: UtcTimestamp
    completed_at: UtcTimestamp | None = None

    @model_validator(mode="after")
    def _round_state_is_closed(self) -> "ProgressiveRound":
        if len(set(self.change_set_ids)) != len(self.change_set_ids):
            raise ValueError("round contains duplicate ChangeSet ids")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("round contains duplicate Evidence ids")
        if self.revision_of_round_id == self.round_id:
            raise ValueError("round cannot revise itself")
        if self.revision_of_round_id is None and self.revision_reason is not None:
            raise ValueError("revision reason requires a predecessor round")
        if self.status is RoundStatus.PLANNED:
            if any(
                value is not None
                for value in (self.merged_snapshot_id, self.failure_reason, self.completed_at)
            ) or self.evidence_ids:
                raise ValueError("planned round cannot carry terminal facts")
        elif self.status is RoundStatus.VERIFIED:
            if self.merged_snapshot_id is None or not self.evidence_ids:
                raise ValueError("verified round requires merged snapshot and evidence")
            if self.completed_at is None or self.failure_reason is not None:
                raise ValueError("verified round has an invalid terminal shape")
        else:
            if self.completed_at is None or not self.failure_reason:
                raise ValueError("failed or cancelled round requires completion and reason")
            if self.merged_snapshot_id is not None:
                raise ValueError("failed or cancelled round cannot carry merged snapshot")
            if self.evidence_ids:
                raise ValueError("failed or cancelled round cannot carry evidence")
        return self


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
    NO_GAIN = "NO_GAIN"
    REGRESSION = "REGRESSION"
    INCONCLUSIVE = "INCONCLUSIVE"


@unique
class ReuseDisposition(StrEnum):
    """Whether a persisted successful result may be reused."""

    REUSE = "REUSE"
    RECOMPUTE = "RECOMPUTE"


@unique
class ReuseReason(StrEnum):
    """Stable public reasons for a reuse decision."""

    ALL_FACTS_MATCH = "ALL_FACTS_MATCH"
    HISTORICAL_UNIT_NOT_SUCCEEDED = "HISTORICAL_UNIT_NOT_SUCCEEDED"
    MISSING_LINEAGE_OR_FINGERPRINT = "MISSING_LINEAGE_OR_FINGERPRINT"
    LINEAGE_CHANGED = "LINEAGE_CHANGED"
    CONTENT_FINGERPRINT_CHANGED = "CONTENT_FINGERPRINT_CHANGED"
    DEPENDENCY_FINGERPRINT_CHANGED = "DEPENDENCY_FINGERPRINT_CHANGED"
    DEPENDENCY_ARTIFACT_HASH_MISSING_OR_MALFORMED = (
        "DEPENDENCY_ARTIFACT_HASH_MISSING_OR_MALFORMED"
    )
    DEPENDENCY_ARTIFACT_HASH_CHANGED = "DEPENDENCY_ARTIFACT_HASH_CHANGED"
    SNAPSHOT_FACTS_MISSING_OR_CHANGED = "SNAPSHOT_FACTS_MISSING_OR_CHANGED"
    MERGE_FACTS_MISSING_OR_CHANGED = "MERGE_FACTS_MISSING_OR_CHANGED"
    CHANGE_SET_FACTS_MISSING_OR_CHANGED = "CHANGE_SET_FACTS_MISSING_OR_CHANGED"
    EVIDENCE_FACTS_MISSING_OR_CHANGED = "EVIDENCE_FACTS_MISSING_OR_CHANGED"
    ATTEMPT_HISTORY_NOT_PROVEN = "ATTEMPT_HISTORY_NOT_PROVEN"


class ReuseDecision(DomainModel):
    """A serializable, conservative decision for one candidate node."""

    disposition: ReuseDisposition
    reason: ReuseReason
    rationale: str = Field(min_length=1)
    lineage_key: str | None = None

    @model_validator(mode="after")
    def _reason_matches_disposition(self) -> "ReuseDecision":
        if self.disposition is ReuseDisposition.REUSE:
            if self.reason is not ReuseReason.ALL_FACTS_MATCH:
                raise ValueError("REUSE requires ALL_FACTS_MATCH")
        elif self.reason is ReuseReason.ALL_FACTS_MATCH:
            raise ValueError("RECOMPUTE cannot use ALL_FACTS_MATCH")
        return self


@unique
class ComparisonOutcome(StrEnum):
    """Deterministic value of a candidate round against its predecessor."""

    BASELINE = "BASELINE"
    IMPROVED = "IMPROVED"
    EVIDENCE_ADDED = "EVIDENCE_ADDED"
    NO_GAIN = "NO_GAIN"
    REGRESSION = "REGRESSION"
    INCONCLUSIVE = "INCONCLUSIVE"


class RoundComparison(DomainModel):
    """Public comparison facts; no model score or hidden reasoning is stored."""

    base_round_id: RoundId | None = None
    candidate_round_id: RoundId | None = None
    base_result: VerificationResult | None = None
    candidate_result: VerificationResult
    base_evidence_ids: tuple[EvidenceId, ...] = ()
    candidate_evidence_ids: tuple[EvidenceId, ...] = ()
    new_evidence_ids: tuple[EvidenceId, ...] = ()
    base_usage: Usage | None = None
    candidate_usage: Usage | None = None
    base_metrics: RunMetrics | None = None
    candidate_metrics: RunMetrics | None = None
    outcome: ComparisonOutcome
    rationale: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _comparison_facts_are_unique(self) -> "RoundComparison":
        for name in (
            "base_evidence_ids",
            "candidate_evidence_ids",
            "new_evidence_ids",
        ):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate Evidence ids")
        if self.base_result is None and self.base_round_id is not None:
            raise ValueError("a base round id requires a base result")
        return self

    @property
    def accepted(self) -> bool:
        """Whether the candidate is acceptable on deterministic facts."""
        return self.candidate_result is VerificationResult.PASS and self.outcome in {
            ComparisonOutcome.BASELINE,
            ComparisonOutcome.IMPROVED,
            ComparisonOutcome.EVIDENCE_ADDED,
        }

    @property
    def token_delta(self) -> int | None:
        if self.base_usage is None or self.candidate_usage is None:
            return None
        return self.candidate_usage.total_tokens - self.base_usage.total_tokens

    @property
    def wall_clock_delta_ms(self) -> int | None:
        if self.base_metrics is None or self.candidate_metrics is None:
            return None
        if (
            self.base_metrics.wall_clock_ms is None
            or self.candidate_metrics.wall_clock_ms is None
        ):
            return None
        return self.candidate_metrics.wall_clock_ms - self.base_metrics.wall_clock_ms

    @property
    def cost_delta(self) -> Decimal | None:
        if self.base_metrics is None or self.candidate_metrics is None:
            return None
        if self.base_metrics.cost is None or self.candidate_metrics.cost is None:
            return None
        return self.candidate_metrics.cost - self.base_metrics.cost


def compare_rounds(
    base: GlobalVerificationReport | None,
    candidate: GlobalVerificationReport,
    *,
    base_round_id: RoundId | None = None,
    candidate_round_id: RoundId | None = None,
) -> RoundComparison:
    """Compare two global reports without inventing a quality score."""
    candidate_evidence = tuple(candidate.evidence_ids)
    base_evidence = () if base is None else tuple(base.evidence_ids)
    base_set = set(base_evidence)
    new_evidence = tuple(
        evidence_id
        for evidence_id in candidate_evidence
        if evidence_id not in base_set
    )

    if base is None:
        outcome = (
            ComparisonOutcome.BASELINE
            if candidate.result is VerificationResult.PASS
            else ComparisonOutcome.INCONCLUSIVE
        )
        rationale = (
            "the candidate is the first globally verified round"
            if outcome is ComparisonOutcome.BASELINE
            else "the first round did not produce a proven PASS verdict"
        )
    elif candidate.result is VerificationResult.PASS:
        if base.result is VerificationResult.PASS:
            outcome = (
                ComparisonOutcome.EVIDENCE_ADDED
                if new_evidence
                else ComparisonOutcome.NO_GAIN
            )
            rationale = (
                "the verdict is unchanged but new target Evidence was added"
                if outcome is ComparisonOutcome.EVIDENCE_ADDED
                else "the candidate verdict is unchanged and adds no target Evidence"
            )
        else:
            outcome = ComparisonOutcome.IMPROVED
            rationale = "the candidate improved the predecessor verdict to PASS"
    elif candidate.result is VerificationResult.INCONCLUSIVE:
        outcome = ComparisonOutcome.INCONCLUSIVE
        rationale = "the candidate lacks sufficient deterministic facts"
    elif base.result is VerificationResult.PASS:
        outcome = ComparisonOutcome.REGRESSION
        rationale = "the candidate regressed from PASS to FAIL"
    elif base.result is VerificationResult.FAIL and not new_evidence:
        outcome = ComparisonOutcome.NO_GAIN
        rationale = "the candidate remains FAIL without new target Evidence"
    elif base.result is VerificationResult.FAIL:
        outcome = ComparisonOutcome.EVIDENCE_ADDED
        rationale = "the candidate remains FAIL but records new target Evidence"
    else:
        outcome = ComparisonOutcome.REGRESSION
        rationale = "the candidate is a deterministic FAIL after an inconclusive verdict"

    return RoundComparison(
        base_round_id=base_round_id if base is not None else None,
        candidate_round_id=candidate_round_id,
        base_result=None if base is None else base.result,
        candidate_result=candidate.result,
        base_evidence_ids=base_evidence,
        candidate_evidence_ids=candidate_evidence,
        new_evidence_ids=new_evidence,
        base_usage=None if base is None else base.usage,
        candidate_usage=candidate.usage,
        base_metrics=None if base is None else base.metrics,
        candidate_metrics=candidate.metrics,
        outcome=outcome,
        rationale=rationale,
    )


class RevisionDecision(DomainModel):
    """A serializable Progressive decision with no private reasoning fields."""

    disposition: RevisionDisposition
    reason: PlanRevisionReason | None = None
    stop_reason: RevisionStopReason | None = None
    comparison: RoundComparison | None = None
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _reuse_recompute(
    reason: ReuseReason,
    rationale: str,
    lineage_key: str | None,
) -> ReuseDecision:
    return ReuseDecision(
        disposition=ReuseDisposition.RECOMPUTE,
        reason=reason,
        rationale=rationale,
        lineage_key=lineage_key,
    )


def _hashes_are_known(values: Sequence[str | None]) -> bool:
    return all(value is not None and _SHA256_RE.fullmatch(value) for value in values)


def _fact_sequences_match(
    historical: Sequence[str] | None,
    candidate: Sequence[str] | None,
) -> bool:
    return (
        historical is not None
        and candidate is not None
        and tuple(historical) == tuple(candidate)
    )


def decide_reuse(
    historical: WorkUnit,
    candidate: WorkUnit,
    *,
    historical_dependency_artifact_hashes: Sequence[str | None],
    candidate_dependency_artifact_hashes: Sequence[str | None],
    historical_base_snapshot_id: str | None = None,
    candidate_base_snapshot_id: str | None = None,
    historical_merged_snapshot_id: str | None = None,
    candidate_merged_snapshot_id: str | None = None,
    historical_merge_input_digest: str | None = None,
    candidate_merge_input_digest: str | None = None,
    historical_change_set_ids: Sequence[str] | None = None,
    candidate_change_set_ids: Sequence[str] | None = None,
    historical_evidence_ids: Sequence[str] | None = None,
    candidate_evidence_ids: Sequence[str] | None = None,
) -> ReuseDecision:
    """Decide reuse from persisted public facts only.

    The two hash sequences must be ordered by the candidate's declared
    dependency order. Missing or malformed hashes conservatively force a
    recomputation.
    """
    lineage_key = candidate.lineage_key
    if historical.status is not WorkUnitStatus.SUCCEEDED:
        return _reuse_recompute(
            ReuseReason.HISTORICAL_UNIT_NOT_SUCCEEDED,
            "the historical work unit did not produce a proven successful result",
            lineage_key,
        )
    if any(
        value is None
        for value in (
            historical.lineage_key,
            historical.dependency_fingerprint,
            historical.content_fingerprint,
            candidate.lineage_key,
            candidate.dependency_fingerprint,
            candidate.content_fingerprint,
        )
    ):
        return _reuse_recompute(
            ReuseReason.MISSING_LINEAGE_OR_FINGERPRINT,
            "lineage and both execution fingerprints are required for reuse",
            lineage_key,
        )
    if historical.lineage_key != candidate.lineage_key:
        return _reuse_recompute(
            ReuseReason.LINEAGE_CHANGED,
            "the candidate lineage differs from the historical lineage",
            lineage_key,
        )
    if historical.content_fingerprint != candidate.content_fingerprint:
        return _reuse_recompute(
            ReuseReason.CONTENT_FINGERPRINT_CHANGED,
            "the public execution contract changed",
            lineage_key,
        )
    if historical.dependency_fingerprint != candidate.dependency_fingerprint:
        return _reuse_recompute(
            ReuseReason.DEPENDENCY_FINGERPRINT_CHANGED,
            "the declared dependency lineage changed",
            lineage_key,
        )
    if not _hashes_are_known(historical_dependency_artifact_hashes) or not _hashes_are_known(
        candidate_dependency_artifact_hashes
    ):
        return _reuse_recompute(
            ReuseReason.DEPENDENCY_ARTIFACT_HASH_MISSING_OR_MALFORMED,
            "every dependency artifact hash must be a known SHA-256 value",
            lineage_key,
        )
    if tuple(historical_dependency_artifact_hashes) != tuple(
        candidate_dependency_artifact_hashes
    ):
        return _reuse_recompute(
            ReuseReason.DEPENDENCY_ARTIFACT_HASH_CHANGED,
            "a dependency artifact hash changed",
            lineage_key,
        )

    round_facts_supplied = any(
        value is not None
        for value in (
            historical_base_snapshot_id,
            candidate_base_snapshot_id,
            historical_merged_snapshot_id,
            candidate_merged_snapshot_id,
            historical_merge_input_digest,
            candidate_merge_input_digest,
            historical_change_set_ids,
            candidate_change_set_ids,
            historical_evidence_ids,
            candidate_evidence_ids,
        )
    )
    if round_facts_supplied:
        if not historical_base_snapshot_id or not candidate_base_snapshot_id:
            return _reuse_recompute(
                ReuseReason.SNAPSHOT_FACTS_MISSING_OR_CHANGED,
                "both historical and candidate base Snapshots are required for reuse",
                lineage_key,
            )
        if historical_base_snapshot_id != candidate_base_snapshot_id:
            return _reuse_recompute(
                ReuseReason.SNAPSHOT_FACTS_MISSING_OR_CHANGED,
                "the Progressive base Snapshot changed",
                lineage_key,
            )
        if not historical_merged_snapshot_id or not candidate_merged_snapshot_id:
            return _reuse_recompute(
                ReuseReason.SNAPSHOT_FACTS_MISSING_OR_CHANGED,
                "both historical and candidate merged Snapshots are required for reuse",
                lineage_key,
            )
        if historical_merged_snapshot_id != candidate_merged_snapshot_id:
            return _reuse_recompute(
                ReuseReason.SNAPSHOT_FACTS_MISSING_OR_CHANGED,
                "the merged Snapshot changed",
                lineage_key,
            )
        if not historical_merge_input_digest or not candidate_merge_input_digest:
            return _reuse_recompute(
                ReuseReason.MERGE_FACTS_MISSING_OR_CHANGED,
                "both merge input digests are required for reuse",
                lineage_key,
            )
        if historical_merge_input_digest != candidate_merge_input_digest:
            return _reuse_recompute(
                ReuseReason.MERGE_FACTS_MISSING_OR_CHANGED,
                "the merge input digest changed",
                lineage_key,
            )
        if not _fact_sequences_match(
            historical_change_set_ids, candidate_change_set_ids
        ):
            return _reuse_recompute(
                ReuseReason.CHANGE_SET_FACTS_MISSING_OR_CHANGED,
                "the persisted ChangeSet facts are missing or changed",
                lineage_key,
            )
        if not _fact_sequences_match(historical_evidence_ids, candidate_evidence_ids):
            return _reuse_recompute(
                ReuseReason.EVIDENCE_FACTS_MISSING_OR_CHANGED,
                "the persisted Evidence facts are missing or changed",
                lineage_key,
            )
    return ReuseDecision(
        disposition=ReuseDisposition.REUSE,
        reason=ReuseReason.ALL_FACTS_MATCH,
        rationale=(
            "lineage, execution fingerprints, and dependency artifact hashes match"
            if not round_facts_supplied
            else (
                "lineage, execution, Snapshot, Merge, ChangeSet, "
                "Evidence, and dependency facts match"
            )
        ),
        lineage_key=lineage_key,
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
    comparison: RoundComparison | None = None,
) -> RevisionDecision:
    return RevisionDecision(
        disposition=RevisionDisposition.STOP,
        stop_reason=reason,
        comparison=comparison,
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
    comparison: RoundComparison | None = None,
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
    if comparison is not None:
        if result is not None and result is not comparison.candidate_result:
            raise ValueError("comparison and verification result disagree")
        result = comparison.candidate_result
    if result is not None and error is not None:
        raise ValueError("verification and error signals are mutually exclusive")

    if cancel_requested or run_status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
        return _stop(
            reason=RevisionStopReason.CANCELLED,
            rationale="cancellation takes precedence over Progressive revision",
            graph_version=graph_version,
            revision_count=revision_count,
            comparison=comparison,
        )
    if run_status.is_terminal:
        return _stop(
            reason=RevisionStopReason.TERMINAL,
            rationale="a terminal run cannot accept a Progressive revision",
            graph_version=graph_version,
            revision_count=revision_count,
            comparison=comparison,
        )
    if work_unit_status in (WorkUnitStatus.CANCELLED, WorkUnitStatus.INVALIDATED):
        return _stop(
            reason=RevisionStopReason.CANCELLED,
            rationale="the current work unit is no longer runnable",
            graph_version=graph_version,
            revision_count=revision_count,
            comparison=comparison,
        )
    if result is VerificationResult.PASS:
        return _stop(
            reason=RevisionStopReason.PASS,
            rationale="deterministic verification passed",
            graph_version=graph_version,
            revision_count=revision_count,
            comparison=comparison,
        )
    if result is VerificationResult.INCONCLUSIVE and comparison is None:
        return _stop(
            reason=RevisionStopReason.INCONCLUSIVE,
            rationale="deterministic verification did not produce sufficient facts",
            graph_version=graph_version,
            revision_count=revision_count,
        )

    if comparison is not None:
        if comparison.outcome is ComparisonOutcome.NO_GAIN:
            return _stop(
                reason=RevisionStopReason.NO_GAIN,
                rationale=comparison.rationale,
                graph_version=graph_version,
                revision_count=revision_count,
                comparison=comparison,
            )
        if comparison.outcome is ComparisonOutcome.REGRESSION:
            return _stop(
                reason=RevisionStopReason.REGRESSION,
                rationale=comparison.rationale,
                graph_version=graph_version,
                revision_count=revision_count,
                comparison=comparison,
            )
        if (
            comparison.outcome is ComparisonOutcome.INCONCLUSIVE
            and comparison.candidate_result is VerificationResult.INCONCLUSIVE
        ):
            return _stop(
                reason=RevisionStopReason.INCONCLUSIVE,
                rationale=comparison.rationale,
                graph_version=graph_version,
                revision_count=revision_count,
                comparison=comparison,
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
            comparison=comparison,
        )
    if budget.max_plan_revisions is None:
        return _stop(
            reason=RevisionStopReason.NO_REVISION_BUDGET,
            rationale="Progressive revision requires an explicit revision budget",
            graph_version=graph_version,
            revision_count=revision_count,
            comparison=comparison,
        )
    if revision_count >= budget.max_plan_revisions:
        return _stop(
            reason=RevisionStopReason.REVISION_LIMIT,
            rationale="the Progressive revision budget is exhausted",
            graph_version=graph_version,
            revision_count=revision_count,
            comparison=comparison,
        )
    if revision_reason is None:
        return _stop(
            reason=RevisionStopReason.NO_TRIGGER,
            rationale="no deterministic Progressive revision trigger was recorded",
            graph_version=graph_version,
            revision_count=revision_count,
            comparison=comparison,
        )

    return RevisionDecision(
        disposition=RevisionDisposition.REVISE,
        reason=revision_reason,
        comparison=comparison,
        rationale=f"{revision_reason.value} permits one bounded revision",
        graph_version=graph_version,
        revision_count=revision_count,
        next_graph_version=graph_version + 1,
    )


decide_progressive_revision = decide_revision
