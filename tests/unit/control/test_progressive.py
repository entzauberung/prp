"""Deterministic Progressive revision decision matrix."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError

from prp_runtime.control.controller import RunController
from prp_runtime.control.progressive import (
    ComparisonOutcome,
    ProgressiveRound,
    ReuseDecision,
    ReuseDisposition,
    ReuseReason,
    RevisionDecision,
    RevisionDisposition,
    RevisionStopReason,
    RoundStatus,
    compare_rounds,
    decide_reuse,
    decide_revision,
)
from prp_runtime.domain.enums import RunStatus, WorkUnitStatus
from prp_runtime.domain.errors import ErrorCode, StateError
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import (
    Budget,
    ErrorCategory,
    ErrorInfo,
    GlobalCheck,
    GlobalVerificationReport,
    NativeRunRequest,
    RunMetrics,
    Usage,
    VerificationResult,
    WorkUnit,
)
from prp_runtime.planning.compiler import CompiledPlan, compile_plan
from prp_runtime.planning.models import (
    PlanNode,
    PlanProposal,
    PlanRejection,
    PlanRevision,
    PlanRevisionReason,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.verification.verifier import VerificationReport

T0 = datetime(2026, 8, 11, tzinfo=UTC)


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "progressive-revision.db") as opened:
        yield opened


def report(result: VerificationResult) -> VerificationReport:
    return VerificationReport(
        run_id="run_progressive",
        work_unit_id="wu_current",
        artifact_id="art_current",
        result=result,
    )


def global_report(
    result: VerificationResult,
    *,
    round_id: str | None = None,
    evidence_ids: tuple[str, ...] = (),
    usage: Usage | None = None,
    metrics: RunMetrics | None = None,
) -> GlobalVerificationReport:
    return GlobalVerificationReport(
        run_id="run_progressive",
        round_id=round_id,
        graph_version=2,
        result=result,
        checks=(
            GlobalCheck(
                kind="EVIDENCE",
                result=result,
                detail="deterministic round fact",
                evidence_ids=evidence_ids,
                fact_ids=evidence_ids,
            ),
        ),
        evidence_ids=evidence_ids,
        usage=usage,
        metrics=metrics,
    )


def test_progressive_round_state_requires_verified_snapshot_and_evidence() -> None:
    planned = ProgressiveRound(
        round_id="round_" + "a" * 32,
        run_id="run_progressive",
        round_index=0,
        graph_version=1,
        base_snapshot_id="snap_" + "b" * 32,
        status=RoundStatus.PLANNED,
        created_at=T0,
    )
    assert planned.merged_snapshot_id is None
    with pytest.raises(ValidationError, match="merged snapshot and evidence"):
        ProgressiveRound(
            round_id="round_" + "c" * 32,
            run_id="run_progressive",
            round_index=1,
            graph_version=2,
            base_snapshot_id="snap_" + "d" * 32,
            status=RoundStatus.VERIFIED,
            created_at=T0,
            completed_at=T0,
        )
    with pytest.raises(ValidationError, match="completion and reason"):
        ProgressiveRound(
            round_id="round_" + "e" * 32,
            run_id="run_progressive",
            round_index=2,
            graph_version=3,
            base_snapshot_id="snap_" + "f" * 32,
            status=RoundStatus.FAILED,
            created_at=T0,
            completed_at=T0,
        )


def test_pass_never_revises_even_when_revision_budget_is_zero() -> None:
    decision = decide_revision(
        budget=Budget(max_plan_revisions=0),
        revision_count=0,
        verification=report(VerificationResult.PASS),
    )

    assert decision.disposition is RevisionDisposition.STOP
    assert decision.stop_reason is RevisionStopReason.PASS
    assert decision.reason is None
    assert decision.should_revise is False


def test_verification_failure_triggers_one_next_graph_version() -> None:
    decision = decide_revision(
        budget=Budget(max_plan_revisions=2),
        revision_count=1,
        graph_version=4,
        verification_result=VerificationResult.FAIL,
    )

    assert decision.disposition is RevisionDisposition.REVISE
    assert decision.reason is PlanRevisionReason.VERIFICATION_FAILED
    assert decision.next_graph_version == 5
    assert decision.should_revise is True


def test_inconclusive_verification_stops_without_claiming_progress() -> None:
    decision = decide_revision(
        budget=Budget(max_plan_revisions=2),
        revision_count=0,
        verification_result=VerificationResult.INCONCLUSIVE,
    )

    assert decision.disposition is RevisionDisposition.STOP
    assert decision.stop_reason is RevisionStopReason.INCONCLUSIVE


def test_retryable_provider_failure_is_a_revision_trigger() -> None:
    decision = decide_revision(
        budget=Budget(max_plan_revisions=1),
        revision_count=0,
        error=ErrorInfo(category=ErrorCategory.NETWORK, message="provider unavailable"),
    )

    assert decision.reason is PlanRevisionReason.PROVIDER_FAILED
    assert decision.next_graph_version == 2


def test_nonretryable_provider_failure_has_no_revision_trigger() -> None:
    decision = decide_revision(
        budget=Budget(max_plan_revisions=1),
        revision_count=0,
        error=ErrorInfo(category=ErrorCategory.AUTH, message="credentials rejected"),
    )

    assert decision.disposition is RevisionDisposition.STOP
    assert decision.stop_reason is RevisionStopReason.NO_TRIGGER


@pytest.mark.parametrize(
    ("run_status", "cancel_requested", "stop_reason"),
    [
        (RunStatus.SUCCEEDED, False, RevisionStopReason.TERMINAL),
        (RunStatus.RUNNING, True, RevisionStopReason.CANCELLED),
        (RunStatus.CANCELLING, False, RevisionStopReason.CANCELLED),
    ],
)
def test_terminal_and_cancel_signals_stop_before_revision(
    run_status: RunStatus,
    cancel_requested: bool,
    stop_reason: RevisionStopReason,
) -> None:
    decision = decide_revision(
        budget=Budget(max_plan_revisions=3),
        revision_count=0,
        verification_result=VerificationResult.FAIL,
        run_status=run_status,
        cancel_requested=cancel_requested,
    )

    assert decision.stop_reason is stop_reason


@pytest.mark.parametrize(
    ("budget", "revision_count", "expected"),
    [
        (Budget(), 0, RevisionStopReason.NO_REVISION_BUDGET),
        (Budget(max_plan_revisions=1), 1, RevisionStopReason.REVISION_LIMIT),
        (Budget(max_total_tokens=10), 0, RevisionStopReason.BUDGET),
        (Budget(max_plan_revisions=2), 0, RevisionStopReason.BUDGET),
    ],
)
def test_revision_budget_and_usage_are_hard_stops(
    budget: Budget,
    revision_count: int,
    expected: RevisionStopReason,
) -> None:
    kwargs: dict[str, object] = {
        "budget": budget,
        "revision_count": revision_count,
        "verification_result": VerificationResult.FAIL,
    }
    if budget.max_total_tokens is not None:
        kwargs["usage"] = Usage(input_tokens=8, output_tokens=2)
    if expected is RevisionStopReason.BUDGET and budget.max_total_tokens is None:
        kwargs["budget_exhausted"] = True

    decision = decide_revision(**kwargs)  # type: ignore[arg-type]

    assert decision.stop_reason is expected


def test_deadline_is_checked_only_against_an_explicit_clock() -> None:
    decision = decide_revision(
        budget=Budget(deadline=T0),
        revision_count=0,
        verification_result=VerificationResult.FAIL,
        now=T0 + timedelta(seconds=1),
    )
    assert decision.stop_reason is RevisionStopReason.BUDGET


def test_contradictory_signals_and_invalid_revision_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="disagree"):
        decide_revision(
            budget=Budget(max_plan_revisions=1),
            revision_count=0,
            verification=report(VerificationResult.PASS),
            verification_result=VerificationResult.FAIL,
        )
    with pytest.raises(ValueError, match="negative"):
        decide_revision(budget=Budget(max_plan_revisions=1), revision_count=-1)


def test_round_comparison_exposes_quality_outcome_and_measured_deltas() -> None:
    base_usage = Usage(input_tokens=2, output_tokens=3, elapsed_ms=10)
    candidate_usage = Usage(input_tokens=4, output_tokens=5, elapsed_ms=25)
    base = global_report(
        VerificationResult.PASS,
        round_id="round_" + "a" * 32,
        evidence_ids=("ev_" + "1" * 32,),
        usage=base_usage,
        metrics=RunMetrics(
            usage=base_usage,
            provider_elapsed_ms=10,
            wall_clock_ms=100,
            cost="1.25",
        ),
    )
    candidate = global_report(
        VerificationResult.PASS,
        round_id="round_" + "b" * 32,
        evidence_ids=("ev_" + "1" * 32,),
        usage=candidate_usage,
        metrics=RunMetrics(
            usage=candidate_usage,
            provider_elapsed_ms=25,
            wall_clock_ms=140,
            cost="1.75",
        ),
    )

    comparison = compare_rounds(base, candidate)

    assert comparison.outcome is ComparisonOutcome.NO_GAIN
    assert comparison.token_delta == 4
    assert comparison.wall_clock_delta_ms == 40
    assert comparison.cost_delta == 0.50
    decision = decide_revision(
        budget=Budget(max_plan_revisions=2),
        revision_count=0,
        comparison=comparison,
    )
    assert decision.stop_reason is RevisionStopReason.PASS
    assert decision.comparison == comparison


def test_round_comparison_stops_regression_and_no_gain() -> None:
    base = global_report(
        VerificationResult.PASS,
        round_id="round_" + "a" * 32,
        evidence_ids=("ev_" + "1" * 32,),
    )
    failed = global_report(
        VerificationResult.FAIL,
        round_id="round_" + "b" * 32,
        evidence_ids=("ev_" + "2" * 32,),
    )
    regression = compare_rounds(base, failed)
    assert regression.outcome is ComparisonOutcome.REGRESSION
    assert decide_revision(
        budget=Budget(max_plan_revisions=2),
        revision_count=0,
        comparison=regression,
    ).stop_reason is RevisionStopReason.REGRESSION

    no_gain = compare_rounds(failed, failed)
    assert no_gain.outcome is ComparisonOutcome.NO_GAIN
    assert decide_revision(
        budget=Budget(max_plan_revisions=2),
        revision_count=0,
        comparison=no_gain,
    ).stop_reason is RevisionStopReason.NO_GAIN


def test_first_failed_round_remains_a_revision_trigger_without_baseline() -> None:
    comparison = compare_rounds(
        None,
        global_report(VerificationResult.FAIL, round_id="round_" + "a" * 32),
    )

    assert comparison.outcome is ComparisonOutcome.INCONCLUSIVE
    decision = decide_revision(
        budget=Budget(max_plan_revisions=1),
        revision_count=0,
        comparison=comparison,
    )
    assert decision.disposition is RevisionDisposition.REVISE


def test_revision_decision_rejects_private_or_malformed_shapes() -> None:
    with pytest.raises(ValidationError):
        RevisionDecision.model_validate(
            {
                "disposition": "REVISE",
                "rationale": "revise",
                "graph_version": 1,
                "revision_count": 0,
                "next_graph_version": 2,
            }
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RevisionDecision.model_validate(
            {
                "disposition": "STOP",
                "stop_reason": "PASS",
                "rationale": "pass",
                "graph_version": 1,
                "revision_count": 0,
                "reasoning": "must not persist",
            }
        )


def test_cancelled_work_unit_stops_without_revising() -> None:
    decision = decide_revision(
        budget=Budget(max_plan_revisions=2),
        revision_count=0,
        verification_result=VerificationResult.FAIL,
        work_unit_status=WorkUnitStatus.CANCELLED,
    )
    assert decision.stop_reason is RevisionStopReason.CANCELLED


def graph_proposal(name: str) -> PlanProposal:
    return PlanProposal(
        summary=f"proposal for {name}",
        final_node=name,
        nodes=(
            PlanNode(
                key=name,
                name=name.title(),
                instruction=f"produce {name}",
            ),
        ),
    )


def reuse_unit(
    *,
    lineage_key: str | None = "stable-node",
    dependency_fingerprint: str | None = "1" * 64,
    content_fingerprint: str | None = "a" * 64,
    status: WorkUnitStatus = WorkUnitStatus.SUCCEEDED,
) -> WorkUnit:
    return WorkUnit(
        work_unit_id="wu_reuse_candidate",
        run_id="run_reuse",
        graph_version=2,
        lineage_key=lineage_key,
        dependency_fingerprint=dependency_fingerprint,
        content_fingerprint=content_fingerprint,
        name="node",
        instruction="produce node",
        status=status,
    )


def test_reuse_requires_all_persisted_facts_to_match() -> None:
    decision = decide_reuse(
        reuse_unit(),
        reuse_unit(),
        historical_dependency_artifact_hashes=(),
        candidate_dependency_artifact_hashes=(),
        historical_base_snapshot_id="snap_base",
        candidate_base_snapshot_id="snap_base",
        historical_merged_snapshot_id="snap_merged",
        candidate_merged_snapshot_id="snap_merged",
        historical_merge_input_digest="digest_same",
        candidate_merge_input_digest="digest_same",
        historical_change_set_ids=("cs_same",),
        candidate_change_set_ids=("cs_same",),
        historical_evidence_ids=("ev_same",),
        candidate_evidence_ids=("ev_same",),
    )
    assert decision == ReuseDecision(
        disposition=ReuseDisposition.REUSE,
        reason=ReuseReason.ALL_FACTS_MATCH,
        rationale=(
            "lineage, execution, Snapshot, Merge, ChangeSet, "
            "Evidence, and dependency facts match"
        ),
        lineage_key="stable-node",
    )
    assert ReuseDecision.model_validate_json(decision.model_dump_json()) == decision


def test_reuse_requires_progressive_snapshot_merge_changeset_and_evidence_facts() -> None:
    facts = {
        "historical_base_snapshot_id": "snap_base",
        "candidate_base_snapshot_id": "snap_base",
        "historical_merged_snapshot_id": "snap_merged",
        "candidate_merged_snapshot_id": "snap_merged",
        "historical_merge_input_digest": "digest_same",
        "candidate_merge_input_digest": "digest_same",
        "historical_change_set_ids": ("cs_same",),
        "candidate_change_set_ids": ("cs_same",),
        "historical_evidence_ids": ("ev_same",),
        "candidate_evidence_ids": ("ev_same",),
    }
    decision = decide_reuse(
        reuse_unit(),
        reuse_unit(),
        historical_dependency_artifact_hashes=(),
        candidate_dependency_artifact_hashes=(),
        **facts,
    )
    assert decision.disposition is ReuseDisposition.REUSE
    assert "Snapshot" in decision.rationale

    changed = decide_reuse(
        reuse_unit(),
        reuse_unit(),
        historical_dependency_artifact_hashes=(),
        candidate_dependency_artifact_hashes=(),
        **(facts | {"candidate_change_set_ids": ("cs_changed",)}),
    )
    assert changed.disposition is ReuseDisposition.RECOMPUTE
    assert changed.reason is ReuseReason.CHANGE_SET_FACTS_MISSING_OR_CHANGED


@pytest.mark.parametrize(
    ("historical", "candidate", "reason"),
    [
        (
            reuse_unit(status=WorkUnitStatus.FAILED),
            reuse_unit(),
            ReuseReason.HISTORICAL_UNIT_NOT_SUCCEEDED,
        ),
        (reuse_unit(), reuse_unit(lineage_key="different"), ReuseReason.LINEAGE_CHANGED),
        (
            reuse_unit(content_fingerprint="b" * 64),
            reuse_unit(),
            ReuseReason.CONTENT_FINGERPRINT_CHANGED,
        ),
        (
            reuse_unit(dependency_fingerprint="2" * 64),
            reuse_unit(),
            ReuseReason.DEPENDENCY_FINGERPRINT_CHANGED,
        ),
    ],
)
def test_reuse_recomputes_when_node_facts_change(
    historical: WorkUnit, candidate: WorkUnit, reason: ReuseReason
) -> None:
    decision = decide_reuse(
        historical,
        candidate,
        historical_dependency_artifact_hashes=(),
        candidate_dependency_artifact_hashes=(),
    )
    assert decision.disposition is ReuseDisposition.RECOMPUTE
    assert decision.reason is reason


@pytest.mark.parametrize(
    ("historical_hashes", "candidate_hashes", "reason"),
    [
        ((None,), ("a" * 64,), ReuseReason.DEPENDENCY_ARTIFACT_HASH_MISSING_OR_MALFORMED),
        (("bad",), ("a" * 64,), ReuseReason.DEPENDENCY_ARTIFACT_HASH_MISSING_OR_MALFORMED),
        (("a" * 64,), ("b" * 64,), ReuseReason.DEPENDENCY_ARTIFACT_HASH_CHANGED),
    ],
)
def test_reuse_recomputes_when_dependency_artifact_facts_are_unknown_or_changed(
    historical_hashes: tuple[str | None, ...],
    candidate_hashes: tuple[str | None, ...],
    reason: ReuseReason,
) -> None:
    decision = decide_reuse(
        reuse_unit(),
        reuse_unit(),
        historical_dependency_artifact_hashes=historical_hashes,
        candidate_dependency_artifact_hashes=candidate_hashes,
    )
    assert decision.disposition is ReuseDisposition.RECOMPUTE
    assert decision.reason is reason


def test_reuse_decision_rejects_private_or_inconsistent_shapes() -> None:
    with pytest.raises(ValidationError, match="REUSE requires"):
        ReuseDecision(
            disposition=ReuseDisposition.REUSE,
            reason=ReuseReason.LINEAGE_CHANGED,
            rationale="invalid",
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ReuseDecision(
            disposition=ReuseDisposition.RECOMPUTE,
            reason=ReuseReason.MISSING_LINEAGE_OR_FINGERPRINT,
            rationale="invalid",
            reasoning="private",
        )


def revision(base_graph_version: int, name: str = "replacement") -> PlanRevision:
    return PlanRevision(
        base_graph_version=base_graph_version,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="replace the failed graph",
        proposal=graph_proposal(name),
    )


@pytest.mark.asyncio
async def test_revision_commit_advances_version_and_preserves_old_graph(
    store: SqliteStore,
) -> None:
    controller = RunController(store, Settings(), {})
    run = await controller.create_run(NativeRunRequest(input="progressive run"))
    initial = compile_plan(
        graph_proposal("initial"),
        run_id=run.run_id,
        graph_version=2,
    )
    assert isinstance(initial, CompiledPlan)
    committed = await controller.commit_plan(
        run.run_id,
        initial,
        target_graph_version=2,
    )
    assert isinstance(committed, tuple)
    old_units = await store.list_work_units(run.run_id, graph_version=2)

    result = await controller.commit_revision(run.run_id, revision(2))

    assert isinstance(result, tuple)
    updated = await store.get_run(run.run_id)
    assert updated.graph_version == 3
    assert await store.list_work_units(run.run_id, graph_version=2) == old_units
    new_units = await store.list_work_units(run.run_id, graph_version=3)
    assert len(new_units) == 1
    assert new_units[0].work_unit_id != old_units[0].work_unit_id
    events = await store.list_events(run.run_id)
    revised = [event for event in events if event.event_type is EventType.PLAN_REVISED]
    assert len(revised) == 1
    assert revised[0].payload["graph_version"] == 3
    assert revised[0].payload["base_graph_version"] == 2
    committed_events = [
        event for event in events if event.event_type is EventType.PLAN_COMMITTED
    ]
    assert [event.payload["graph_version"] for event in committed_events] == [2, 3]


@pytest.mark.asyncio
async def test_stale_revision_is_rejected_without_writing_new_graph(
    store: SqliteStore,
) -> None:
    controller = RunController(store, Settings(), {})
    run = await controller.create_run(NativeRunRequest(input="progressive run"))
    initial = compile_plan(
        graph_proposal("initial"), run_id=run.run_id, graph_version=2
    )
    assert isinstance(initial, CompiledPlan)
    await controller.commit_plan(run.run_id, initial, target_graph_version=2)
    before = await store.list_work_units(run.run_id)

    result = await controller.commit_revision(run.run_id, revision(1, "stale"))

    assert isinstance(result, PlanRejection)
    assert await store.list_work_units(run.run_id) == before
    assert (await store.get_run(run.run_id)).graph_version == 2
    assert EventType.PLAN_REVISED not in {
        event.event_type for event in await store.list_events(run.run_id)
    }


@pytest.mark.asyncio
async def test_revision_store_failure_rolls_back_revision_event_and_graph(
    store: SqliteStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = RunController(store, Settings(), {})
    run = await controller.create_run(NativeRunRequest(input="progressive run"))
    initial = compile_plan(
        graph_proposal("initial"), run_id=run.run_id, graph_version=2
    )
    assert isinstance(initial, CompiledPlan)
    await controller.commit_plan(run.run_id, initial, target_graph_version=2)
    before = await store.list_work_units(run.run_id)

    async def fail_graph(units: tuple[WorkUnit, ...]) -> None:
        raise StateError("forced revision collision", code=ErrorCode.ILLEGAL_STATE_TRANSITION)

    monkeypatch.setattr(store, "create_graph", fail_graph)
    result = await controller.commit_revision(run.run_id, revision(2, "failed"))

    assert isinstance(result, PlanRejection)
    assert await store.list_work_units(run.run_id) == before
    assert (await store.get_run(run.run_id)).graph_version == 2
    events = await store.list_events(run.run_id)
    assert EventType.PLAN_REVISED not in {event.event_type for event in events}
    assert all(
        event.payload.get("graph_version") != 3
        for event in events
        if event.event_type is EventType.PLAN_COMMITTED
    )
