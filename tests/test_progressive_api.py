from prp import (
    ComparisonOutcome,
    ProgressiveDecision,
    RevisionDisposition,
    RevisionStopReason,
    VerificationResult,
    decide_progressive,
)


def test_progressive_api_opens_one_new_graph_version_on_failure() -> None:
    decision = decide_progressive(
        verification_result=VerificationResult.FAIL,
        max_plan_revisions=2,
        graph_version=4,
    )
    assert isinstance(decision, ProgressiveDecision)
    assert decision.disposition is RevisionDisposition.REVISE
    assert decision.next_graph_version == 5


def test_progressive_api_stops_inconclusive_without_comparison_trigger() -> None:
    decision = decide_progressive(
        verification_result=VerificationResult.INCONCLUSIVE,
        max_plan_revisions=2,
    )
    assert decision.stop_reason is RevisionStopReason.INCONCLUSIVE


def test_progressive_api_stops_when_comparison_has_no_gain() -> None:
    decision = decide_progressive(
        verification_result=VerificationResult.FAIL,
        comparison_outcome=ComparisonOutcome.NO_GAIN,
        max_plan_revisions=2,
    )
    assert decision.stop_reason is RevisionStopReason.NO_GAIN
