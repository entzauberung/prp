import pytest

from prp.revision import ComparisonOutcome, decide_reuse, decide_revision
from prp.vocabulary import (
    ReuseDisposition,
    ReuseReason,
    RevisionDisposition,
    RevisionStopReason,
    RunStatus,
    VerificationResult,
    WorkUnitStatus,
)


def test_pass_does_not_revise() -> None:
    decision = decide_revision(
        verification_result=VerificationResult.PASS,
        max_plan_revisions=3,
    )
    assert decision.disposition is RevisionDisposition.STOP
    assert decision.stop_reason is RevisionStopReason.PASS
    assert decision.next_graph_version is None


def test_inconclusive_is_not_failure() -> None:
    decision = decide_revision(
        verification_result=VerificationResult.INCONCLUSIVE,
        max_plan_revisions=3,
    )
    assert decision.stop_reason is RevisionStopReason.INCONCLUSIVE
    assert decision.trigger is None


def test_fail_opens_next_graph_version() -> None:
    decision = decide_revision(
        verification_result=VerificationResult.FAIL,
        max_plan_revisions=3,
        graph_version=2,
        revision_count=0,
    )
    assert decision.disposition is RevisionDisposition.REVISE
    assert decision.trigger == "VERIFICATION_FAILED"
    assert decision.next_graph_version == 3


def test_revision_requires_explicit_budget() -> None:
    decision = decide_revision(verification_result=VerificationResult.FAIL)
    assert decision.stop_reason is RevisionStopReason.NO_REVISION_BUDGET


def test_revision_limit_stops() -> None:
    decision = decide_revision(
        verification_result=VerificationResult.FAIL,
        max_plan_revisions=1,
        revision_count=1,
    )
    assert decision.stop_reason is RevisionStopReason.REVISION_LIMIT


def test_cancel_outranks_failure() -> None:
    decision = decide_revision(
        verification_result=VerificationResult.FAIL,
        cancel_requested=True,
        max_plan_revisions=3,
    )
    assert decision.stop_reason is RevisionStopReason.CANCELLED


def test_terminal_run_cannot_revise() -> None:
    decision = decide_revision(
        run_status=RunStatus.SUCCEEDED,
        max_plan_revisions=3,
    )
    assert decision.stop_reason is RevisionStopReason.TERMINAL


def test_no_gain_and_regression_stop() -> None:
    no_gain = decide_revision(
        verification_result=VerificationResult.FAIL,
        comparison_outcome=ComparisonOutcome.NO_GAIN,
        max_plan_revisions=3,
    )
    regression = decide_revision(
        verification_result=VerificationResult.FAIL,
        comparison_outcome=ComparisonOutcome.REGRESSION,
        max_plan_revisions=3,
    )
    assert no_gain.stop_reason is RevisionStopReason.NO_GAIN
    assert regression.stop_reason is RevisionStopReason.REGRESSION


def test_contradictory_signals_are_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        decide_revision(
            verification_result=VerificationResult.FAIL,
            error_category="TIMEOUT",
            max_plan_revisions=3,
        )


def test_invalidated_unit_cannot_revise() -> None:
    decision = decide_revision(
        work_unit_status=WorkUnitStatus.INVALIDATED,
        verification_result=VerificationResult.FAIL,
        max_plan_revisions=3,
    )
    assert decision.stop_reason is RevisionStopReason.CANCELLED


def test_reuse_is_conservative() -> None:
    missing = decide_reuse(
        historical_succeeded=True,
        historical_lineage="a",
        candidate_lineage="a",
        historical_content_fingerprint="c1",
        candidate_content_fingerprint="c1",
        historical_dependency_fingerprint=None,
        candidate_dependency_fingerprint="d1",
    )
    assert missing.disposition is ReuseDisposition.RECOMPUTE
    assert missing.reason is ReuseReason.MISSING_LINEAGE_OR_FINGERPRINT


def test_reuse_when_all_facts_match() -> None:
    decision = decide_reuse(
        historical_succeeded=True,
        historical_lineage="a",
        candidate_lineage="a",
        historical_content_fingerprint="c1",
        candidate_content_fingerprint="c1",
        historical_dependency_fingerprint="d1",
        candidate_dependency_fingerprint="d1",
    )
    assert decision.disposition is ReuseDisposition.REUSE
    assert decision.reason is ReuseReason.ALL_FACTS_MATCH


def test_failed_history_is_never_reused() -> None:
    decision = decide_reuse(
        historical_succeeded=False,
        historical_lineage="a",
        candidate_lineage="a",
        historical_content_fingerprint="c1",
        candidate_content_fingerprint="c1",
        historical_dependency_fingerprint="d1",
        candidate_dependency_fingerprint="d1",
    )
    assert decision.reason is ReuseReason.HISTORICAL_UNIT_NOT_SUCCEEDED
