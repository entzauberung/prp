"""Deterministic Progressive revision decision matrix."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError

from prp_runtime.control.controller import RunController
from prp_runtime.control.progressive import (
    RevisionDecision,
    RevisionDisposition,
    RevisionStopReason,
    decide_revision,
)
from prp_runtime.domain.enums import RunStatus, WorkUnitStatus
from prp_runtime.domain.errors import ErrorCode, StateError
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import (
    Budget,
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
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


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (VerificationResult.FAIL, PlanRevisionReason.VERIFICATION_FAILED),
        (
            VerificationResult.INCONCLUSIVE,
            PlanRevisionReason.VERIFICATION_INCONCLUSIVE,
        ),
    ],
)
def test_verification_failure_classes_trigger_one_next_graph_version(
    result: VerificationResult,
    reason: PlanRevisionReason,
) -> None:
    decision = decide_revision(
        budget=Budget(max_plan_revisions=2),
        revision_count=1,
        graph_version=4,
        verification_result=result,
    )

    assert decision.disposition is RevisionDisposition.REVISE
    assert decision.reason is reason
    assert decision.next_graph_version == 5
    assert decision.should_revise is True


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
        nodes=(
            PlanNode(
                key=name,
                name=name.title(),
                instruction=f"produce {name}",
            ),
        ),
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
