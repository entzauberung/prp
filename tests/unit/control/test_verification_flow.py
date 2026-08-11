"""Control-Verifier ownership contract.

Invariants all verification integration (WO-001-ST-002 onwards) must preserve:

- Verifier writes only Evidence; it never changes Run, WorkUnit, or Attempt state.
- PASS is the only verdict that permits acceptance; FAIL and INCONCLUSIVE both
  yield is_pass=False but remain distinguishable as three-valued results.
- An empty plan is INCONCLUSIVE: nothing proved, nothing accepted.
- TEXT and JSON output requirements both produce a non-empty deterministic plan.
- Controller is the sole author of Run and WorkUnit state transitions.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from prp_runtime.domain.enums import ModelRole, RunStatus, WorkUnitStatus
from prp_runtime.domain.events import EVENT_REQUIRED_KEYS, EventType, RunEvent
from prp_runtime.domain.models import (
    Artifact,
    ArtifactKind,
    Attempt,
    ControllerAction,
    ControllerDecision,
    EvidenceKind,
    NativeRunRequest,
    OutputRequirement,
    Run,
    VerificationResult,
    WorkUnit,
    new_artifact_id,
    new_evidence_id,
)
from prp_runtime.domain.values import ModelRef, new_attempt_id, new_run_id, new_work_unit_id
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.verification.rules import (
    VerificationCheck,
    VerificationRule,
    plan_for_output,
)
from prp_runtime.verification.verifier import RuleVerifier, aggregate

# ---------------------------------------------------------------------------
# Shared identifiers
# ---------------------------------------------------------------------------

RUN_ID = new_run_id()
WORK_UNIT_ID = new_work_unit_id()
ATTEMPT_ID = new_attempt_id()

PASS = VerificationResult.PASS
FAIL = VerificationResult.FAIL
INCONCLUSIVE = VerificationResult.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifact(content: str, kind: ArtifactKind = ArtifactKind.TEXT) -> Artifact:
    return Artifact(
        artifact_id=new_artifact_id(),
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        attempt_id=ATTEMPT_ID,
        name="answer",
        kind=kind,
        content=content,
    )


def _json_artifact(document: object) -> Artifact:
    return _artifact(json.dumps(document), ArtifactKind.JSON)


# ---------------------------------------------------------------------------
# Section 1: Verdict semantics
# ---------------------------------------------------------------------------


def test_pass_is_the_only_accepting_verdict() -> None:
    assert PASS.is_pass is True
    assert FAIL.is_pass is False
    assert INCONCLUSIVE.is_pass is False


def test_fail_and_inconclusive_are_distinguishable_by_result() -> None:
    # Both fail is_pass, but the three-valued result must not be collapsed to a
    # boolean: a reader needs to know whether a check proved failure or merely
    # could not decide.
    assert FAIL is not INCONCLUSIVE
    assert FAIL.value != INCONCLUSIVE.value


def test_empty_plan_aggregates_to_inconclusive_not_pass() -> None:
    # An empty plan proves nothing. Treating it as a pass would let any artifact
    # through without inspection.
    result = aggregate(())
    assert result is INCONCLUSIVE
    assert result.is_pass is False


def test_a_single_pass_check_aggregates_to_pass() -> None:
    assert aggregate((PASS,)) is PASS


def test_fail_dominates_inconclusive_in_aggregation() -> None:
    # One failed check overrides an undecided check: the artifact is rejected.
    assert aggregate((FAIL, INCONCLUSIVE)) is FAIL
    assert aggregate((INCONCLUSIVE, FAIL)) is FAIL


def test_inconclusive_downgrades_an_otherwise_passing_plan() -> None:
    # All passing checks plus one undecided means the whole plan is undecided.
    assert aggregate((PASS, PASS, INCONCLUSIVE)) is INCONCLUSIVE


def test_all_pass_checks_aggregate_to_pass() -> None:
    assert aggregate((PASS, PASS, PASS)) is PASS


# ---------------------------------------------------------------------------
# Section 2: is_pass projection is derived, not stored
# ---------------------------------------------------------------------------


def test_pass_is_derived_from_result_not_an_independent_field() -> None:
    # Evidence.passed is a read-only property; it is always consistent with
    # result. The check here confirms the VerificationResult projection matches.
    for result in VerificationResult:
        assert result.is_pass is (result is PASS)


def test_inconclusive_is_not_stored_as_a_proven_failure() -> None:
    # FAIL and INCONCLUSIVE both produce is_pass=False, but must remain
    # distinguishable by their three-valued result.
    from prp_runtime.domain.models import Evidence

    row_fail = Evidence(
        evidence_id=new_evidence_id(),
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        artifact_id=new_artifact_id(),
        kind=EvidenceKind.DETERMINISTIC_CHECK,
        rule=VerificationRule.NON_EMPTY_OUTPUT.value,
        result=FAIL,
        detail="the output is empty",
    )
    row_undecided = Evidence(
        evidence_id=new_evidence_id(),
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        artifact_id=new_artifact_id(),
        kind=EvidenceKind.DETERMINISTIC_CHECK,
        rule=VerificationRule.MATCHES_JSON_SCHEMA.value,
        result=INCONCLUSIVE,
        detail="schema uses keywords this checker cannot decide",
    )
    # Both report passed=False …
    assert row_fail.passed is False
    assert row_undecided.passed is False
    # … but remain distinguishable by result.
    assert row_fail.result is FAIL
    assert row_undecided.result is INCONCLUSIVE


# ---------------------------------------------------------------------------
# Section 3: Plan construction for TEXT and JSON output requirements
# ---------------------------------------------------------------------------


def test_text_output_plan_is_non_empty() -> None:
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.TEXT))
    assert len(plan) >= 1
    rules = [check.rule for check in plan]
    assert VerificationRule.NON_EMPTY_OUTPUT in rules
    assert VerificationRule.OUTPUT_KIND_MATCHES in rules


def test_json_output_plan_includes_valid_json_check() -> None:
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.JSON))
    rules = [check.rule for check in plan]
    assert VerificationRule.VALID_JSON in rules
    assert VerificationRule.NON_EMPTY_OUTPUT in rules


def test_json_output_with_schema_plan_includes_schema_check() -> None:
    schema = json.dumps({"type": "object"})
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.JSON, json_schema=schema))
    rules = [check.rule for check in plan]
    assert VerificationRule.MATCHES_JSON_SCHEMA in rules
    assert VerificationRule.VALID_JSON in rules


def test_text_artifact_passes_non_empty_check() -> None:
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.TEXT))
    report = RuleVerifier().verify(_artifact("hello"), plan)
    assert report.result is PASS
    assert report.passed is True


def test_empty_text_artifact_fails_the_plan() -> None:
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.TEXT))
    blank = Artifact.model_construct(
        artifact_id=new_artifact_id(),
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        attempt_id=ATTEMPT_ID,
        name="answer",
        kind=ArtifactKind.TEXT,
        content="   ",
    )
    report = RuleVerifier().verify(blank, plan)
    assert report.result is FAIL
    assert report.passed is False


def test_valid_json_artifact_passes_json_plan() -> None:
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.JSON))
    report = RuleVerifier().verify(_json_artifact({"ok": True}), plan)
    assert report.result is PASS


def test_malformed_json_artifact_fails_json_plan() -> None:
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.JSON))
    bad = Artifact.model_construct(
        artifact_id=new_artifact_id(),
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        attempt_id=ATTEMPT_ID,
        name="answer",
        kind=ArtifactKind.JSON,
        content="{not: json}",
    )
    report = RuleVerifier().verify(bad, plan)
    assert report.result is FAIL


# ---------------------------------------------------------------------------
# Section 4: Controller decision model
# ---------------------------------------------------------------------------


def test_accept_artifact_decision_can_reference_evidence() -> None:
    evidence_id = new_evidence_id()
    decision = ControllerDecision(
        run_id=RUN_ID,
        action=ControllerAction.ACCEPT_ARTIFACT,
        rationale="all deterministic checks passed",
        work_unit_id=WORK_UNIT_ID,
        evidence_ids=(evidence_id,),
    )
    assert decision.action is ControllerAction.ACCEPT_ARTIFACT
    assert evidence_id in decision.evidence_ids


def test_reject_artifact_decision_on_fail_verdict() -> None:
    decision = ControllerDecision(
        run_id=RUN_ID,
        action=ControllerAction.REJECT_ARTIFACT,
        rationale="FAIL: NON_EMPTY_OUTPUT check failed",
        work_unit_id=WORK_UNIT_ID,
    )
    assert decision.action is ControllerAction.REJECT_ARTIFACT


def test_reject_artifact_decision_on_inconclusive_verdict() -> None:
    # INCONCLUSIVE must not be silently accepted. The controller must reject.
    decision = ControllerDecision(
        run_id=RUN_ID,
        action=ControllerAction.REJECT_ARTIFACT,
        rationale="INCONCLUSIVE: schema uses undecidable keywords",
        work_unit_id=WORK_UNIT_ID,
    )
    assert decision.action is ControllerAction.REJECT_ARTIFACT


def test_controller_decision_requires_evidence_ids_to_be_unique() -> None:
    import pytest as _pytest
    ev = new_evidence_id()
    with _pytest.raises(Exception):
        ControllerDecision(
            run_id=RUN_ID,
            action=ControllerAction.ACCEPT_ARTIFACT,
            rationale="should fail",
            evidence_ids=(ev, ev),
        )


# ---------------------------------------------------------------------------
# Section 5: EVIDENCE_RECORDED event payload contract
# ---------------------------------------------------------------------------


def test_evidence_recorded_event_requires_three_valued_result() -> None:
    required = EVENT_REQUIRED_KEYS[EventType.EVIDENCE_RECORDED]
    assert "result" in required
    assert "passed" not in required


def test_evidence_recorded_event_accepts_all_three_verdicts() -> None:
    for verdict in ("PASS", "FAIL", "INCONCLUSIVE"):
        event = RunEvent(
            run_id=RUN_ID,
            sequence=1,
            event_type=EventType.EVIDENCE_RECORDED,
            payload={
                "work_unit_id": WORK_UNIT_ID,
                "evidence_id": new_evidence_id(),
                "result": verdict,
            },
        )
        assert event.payload["result"] == verdict


def test_evidence_recorded_event_distinguishes_fail_from_inconclusive() -> None:
    fail_event = RunEvent(
        run_id=RUN_ID,
        sequence=1,
        event_type=EventType.EVIDENCE_RECORDED,
        payload={
            "work_unit_id": WORK_UNIT_ID,
            "evidence_id": new_evidence_id(),
            "result": "FAIL",
        },
    )
    inconclusive_event = RunEvent(
        run_id=new_run_id(),
        sequence=1,
        event_type=EventType.EVIDENCE_RECORDED,
        payload={
            "work_unit_id": new_work_unit_id(),
            "evidence_id": new_evidence_id(),
            "result": "INCONCLUSIVE",
        },
    )
    assert fail_event.payload["result"] != inconclusive_event.payload["result"]


# ---------------------------------------------------------------------------
# Section 6: Verifier ownership boundary (async, requires store)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "flow.db") as opened:
        yield opened


async def _seed(store: SqliteStore) -> Artifact:
    run = Run(run_id=RUN_ID, request=NativeRunRequest(input="contract test"))
    await store.create_run(run)
    unit = WorkUnit(
        work_unit_id=WORK_UNIT_ID,
        run_id=RUN_ID,
        name="direct",
        instruction="produce output",
    )
    await store.create_work_unit(unit)
    attempt = Attempt(
        attempt_id=ATTEMPT_ID,
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        role=ModelRole.WORKER,
        model=ModelRef(provider="openai_compatible", model="weak-model"),
    )
    await store.create_attempt(attempt)
    produced = _artifact("non-empty answer")
    await store.add_artifact(produced)
    return produced


@pytest.mark.asyncio
async def test_verifier_does_not_change_run_state(store: SqliteStore) -> None:
    produced = await _seed(store)
    before = await store.get_run(RUN_ID)
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.TEXT))

    await RuleVerifier().verify_and_record(store, produced, plan)

    after = await store.get_run(RUN_ID)
    assert after == before
    assert after.status is RunStatus.PENDING


@pytest.mark.asyncio
async def test_verifier_does_not_change_work_unit_state(store: SqliteStore) -> None:
    produced = await _seed(store)
    before = await store.get_work_unit(WORK_UNIT_ID)
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.TEXT))

    await RuleVerifier().verify_and_record(store, produced, plan)

    after = await store.get_work_unit(WORK_UNIT_ID)
    assert after == before
    assert after.status is WorkUnitStatus.PENDING


@pytest.mark.asyncio
async def test_verifier_does_not_emit_run_events(store: SqliteStore) -> None:
    # Emitting EVIDENCE_RECORDED is the controller's decision, not the verifier's.
    produced = await _seed(store)
    before_events = await store.list_events(RUN_ID)
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.TEXT))

    await RuleVerifier().verify_and_record(store, produced, plan)

    after_events = await store.list_events(RUN_ID)
    assert after_events == before_events


@pytest.mark.asyncio
async def test_fail_verdict_persists_as_fail_not_as_inconclusive(
    store: SqliteStore,
) -> None:
    produced = await _seed(store)
    too_short = (
        VerificationCheck(rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=1),
    )

    report = await RuleVerifier().verify_and_record(store, produced, too_short)

    assert report.result is FAIL
    stored = await store.list_evidence(WORK_UNIT_ID)
    assert stored[0].result is FAIL
    assert stored[0].result is not INCONCLUSIVE


@pytest.mark.asyncio
async def test_inconclusive_verdict_persists_as_inconclusive_not_as_fail(
    store: SqliteStore,
) -> None:
    await _seed(store)
    undecidable_schema = json.dumps({"type": "object", "oneOf": [{"type": "object"}]})
    plan = (
        VerificationCheck(
            rule=VerificationRule.MATCHES_JSON_SCHEMA, json_schema=undecidable_schema
        ),
    )
    # TEXT artifact: JSON parse will fail, not go undecided — use JSON artifact.
    json_produced = _json_artifact({"k": "v"})
    await store.add_artifact(json_produced)

    report = await RuleVerifier().verify_and_record(store, json_produced, plan)

    assert report.result is INCONCLUSIVE
    stored = await store.list_evidence(json_produced.work_unit_id)
    row = next(r for r in stored if r.artifact_id == json_produced.artifact_id)
    assert row.result is INCONCLUSIVE
    assert row.result is not FAIL


@pytest.mark.asyncio
async def test_empty_plan_writes_no_evidence(store: SqliteStore) -> None:
    produced = await _seed(store)

    report = await RuleVerifier().verify_and_record(store, produced, ())

    assert report.result is INCONCLUSIVE
    assert await store.list_evidence(WORK_UNIT_ID) == ()
