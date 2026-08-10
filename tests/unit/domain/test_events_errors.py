"""Targeted tests for the event ledger and the structured error layer."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import (
    ERROR_FAMILIES,
    RETRYABLE_CODES,
    BudgetError,
    DomainValidationError,
    ErrorCode,
    ErrorDetail,
    ErrorFamily,
    InternalError,
    ProtocolError,
    ProviderError,
    PrpError,
    StateError,
    state_error_from_transition,
)
from prp_runtime.domain.events import (
    EVENT_REQUIRED_KEYS,
    EventType,
    RunEvent,
    assert_sequence_chain,
    next_sequence,
    payload_from_model,
)
from prp_runtime.domain.models import (
    ControllerAction,
    ControllerDecision,
    NativeRunRequest,
    Usage,
)
from prp_runtime.domain.transitions import (
    IllegalStatusTransitionError,
    transition_run,
)
from prp_runtime.domain.values import new_attempt_id, new_run_id, new_work_unit_id

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
RUN_ID = new_run_id()
WORK_UNIT_ID = new_work_unit_id()
ATTEMPT_ID = new_attempt_id()

MINIMAL_PAYLOADS: dict[str, object] = {
    "request": {"input": "x"},
    "error": {"code": "internal_error", "message": "boom"},
    "strategy": "DIRECT",
    "from_strategy": "DIRECT",
    "to_strategy": "CASCADE",
    "decision": {"action": "CANCEL"},
    "graph_version": 1,
    "reason": "because",
    "work_unit_id": WORK_UNIT_ID,
    "attempt_id": ATTEMPT_ID,
    "artifact_id": "art_1",
    "evidence_id": "ev_1",
    "result": "INCONCLUSIVE",
    "usage": {"input_tokens": 1},
}


def make_event(event_type: EventType, sequence: int = 1, **overrides: object) -> RunEvent:
    payload: dict[str, object] = {
        key: MINIMAL_PAYLOADS[key] for key in EVENT_REQUIRED_KEYS[event_type]
    }
    payload.update(overrides)
    return RunEvent(
        run_id=RUN_ID,
        sequence=sequence,
        event_type=event_type,
        payload=payload,  # type: ignore[arg-type]
        timestamp=T0,
    )


# --- error codes and families ---------------------------------------------------


def test_every_error_code_has_exactly_one_family() -> None:
    assert set(ERROR_FAMILIES) == set(ErrorCode)
    assert set(ERROR_FAMILIES.values()) == set(ErrorFamily)


def test_only_transient_provider_codes_are_retryable() -> None:
    assert RETRYABLE_CODES == {
        ErrorCode.PROVIDER_RATE_LIMITED,
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.PROVIDER_UNAVAILABLE,
    }
    for code in RETRYABLE_CODES:
        assert ERROR_FAMILIES[code] is ErrorFamily.PROVIDER


def test_error_detail_is_built_from_the_code() -> None:
    detail = ErrorDetail.for_code(ErrorCode.PROVIDER_TIMEOUT, "upstream timed out")
    assert detail.family is ErrorFamily.PROVIDER
    assert detail.retryable is True
    assert detail.field is None
    assert ErrorDetail.for_code(ErrorCode.INVALID_REQUEST, "bad input").retryable is False


def test_error_detail_rejects_inconsistent_or_blank_content() -> None:
    with pytest.raises(ValidationError):
        ErrorDetail(
            code=ErrorCode.INVALID_REQUEST,
            family=ErrorFamily.PROVIDER,
            message="mismatched family",
            retryable=False,
        )
    with pytest.raises(ValidationError):
        ErrorDetail(
            code=ErrorCode.INVALID_REQUEST,
            family=ErrorFamily.VALIDATION,
            message="wrong retryability",
            retryable=True,
        )
    with pytest.raises(ValidationError):
        ErrorDetail.for_code(ErrorCode.INTERNAL_ERROR, "   ")


def test_error_detail_rejects_unknown_code_and_extra_field() -> None:
    with pytest.raises(ValidationError):
        ErrorDetail(
            code="not_a_code",
            family=ErrorFamily.VALIDATION,
            message="unknown",
            retryable=False,
        )
    with pytest.raises(ValidationError):
        ErrorDetail(
            code=ErrorCode.INVALID_REQUEST,
            family=ErrorFamily.VALIDATION,
            message="ok",
            retryable=False,
            traceback="Traceback (most recent call last)",
        )


def test_error_detail_carries_no_exception_or_stack_field() -> None:
    assert set(ErrorDetail.model_fields) == {
        "code",
        "family",
        "message",
        "retryable",
        "field",
    }


def test_error_detail_round_trips_through_json() -> None:
    detail = ErrorDetail.for_code(
        ErrorCode.UNSUPPORTED_TOOLS, "tools are not supported", field="tools"
    )
    assert ErrorDetail.model_validate_json(detail.model_dump_json()) == detail


# --- error classes --------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_class", "family"),
    [
        (DomainValidationError, ErrorFamily.VALIDATION),
        (StateError, ErrorFamily.STATE),
        (BudgetError, ErrorFamily.BUDGET),
        (ProviderError, ErrorFamily.PROVIDER),
        (ProtocolError, ErrorFamily.PROTOCOL),
        (InternalError, ErrorFamily.INTERNAL),
    ],
)
def test_error_classes_expose_their_family_and_default_code(
    error_class: type[PrpError], family: ErrorFamily
) -> None:
    error = error_class("something happened")
    assert isinstance(error, PrpError)
    assert error.detail.family is family
    assert ERROR_FAMILIES[error.code] is family
    assert str(error) == "something happened"


def test_error_class_rejects_a_code_from_another_family() -> None:
    with pytest.raises(ValueError, match="cannot be raised as"):
        BudgetError("wrong family", code=ErrorCode.PROVIDER_TIMEOUT)
    budget = BudgetError("deadline passed", code=ErrorCode.DEADLINE_EXCEEDED)
    assert budget.code is ErrorCode.DEADLINE_EXCEEDED
    assert budget.retryable is False


def test_provider_error_retryability_follows_the_code() -> None:
    rate_limited = ProviderError("rate limited", code=ErrorCode.PROVIDER_RATE_LIMITED)
    bad_response = ProviderError("bad response", code=ErrorCode.PROVIDER_INVALID_RESPONSE)
    assert rate_limited.retryable is True
    assert bad_response.retryable is False


def test_transition_violations_map_to_a_state_error() -> None:
    with pytest.raises(IllegalStatusTransitionError) as excinfo:
        transition_run(RunStatus.SUCCEEDED, RunStatus.RUNNING)
    mapped = state_error_from_transition(excinfo.value)
    assert isinstance(mapped, StateError)
    assert mapped.code is ErrorCode.ILLEGAL_STATE_TRANSITION
    assert mapped.retryable is False
    assert "SUCCEEDED -> RUNNING" in mapped.detail.message


# --- event envelope -------------------------------------------------------------


def test_every_event_type_declares_required_payload_keys() -> None:
    assert set(EVENT_REQUIRED_KEYS) == set(EventType)


def test_the_evidence_event_carries_the_verdict_not_its_boolean() -> None:
    required = EVENT_REQUIRED_KEYS[EventType.EVIDENCE_RECORDED]
    assert "result" in required
    # FAIL and INCONCLUSIVE share one boolean, so a ledger entry carrying only the
    # projection could not tell a proven failure from an undecided check.
    assert "passed" not in required


@pytest.mark.parametrize("verdict", ["PASS", "FAIL", "INCONCLUSIVE"])
def test_the_evidence_event_accepts_every_verdict(verdict: str) -> None:
    event = make_event(EventType.EVIDENCE_RECORDED, result=verdict)
    assert event.payload["result"] == verdict


def test_the_evidence_event_rejects_a_payload_without_a_verdict() -> None:
    with pytest.raises(ValidationError, match="result"):
        RunEvent(
            run_id=RUN_ID,
            sequence=1,
            event_type=EventType.EVIDENCE_RECORDED,
            payload={"work_unit_id": WORK_UNIT_ID, "evidence_id": "ev_1"},
            timestamp=T0,
        )


@pytest.mark.parametrize("event_type", list(EventType))
def test_every_event_type_can_be_built_and_round_tripped(event_type: EventType) -> None:
    event = make_event(event_type)
    assert event.sequence == 1
    assert event.timestamp == T0
    assert RunEvent.model_validate_json(event.model_dump_json()) == event


@pytest.mark.parametrize(
    "event_type",
    [
        EventType.WORK_UNIT_STARTED,
        EventType.ATTEMPT_FAILED,
        EventType.ARTIFACT_PRODUCED,
        EventType.USAGE_UPDATED,
    ],
)
def test_event_rejects_a_payload_missing_a_required_key(event_type: EventType) -> None:
    with pytest.raises(ValidationError, match="payload is missing"):
        RunEvent(
            run_id=RUN_ID,
            sequence=1,
            event_type=event_type,
            payload={},
            timestamp=T0,
        )


def test_event_rejects_unknown_type_and_extra_envelope_field() -> None:
    with pytest.raises(ValidationError):
        RunEvent(run_id=RUN_ID, sequence=1, event_type="RUN_EXPLODED", timestamp=T0)
    with pytest.raises(ValidationError):
        RunEvent(
            run_id=RUN_ID,
            sequence=1,
            event_type=EventType.RUN_STARTED,
            timestamp=T0,
            exception=ValueError("boom"),
        )


@pytest.mark.parametrize("bad_sequence", [0, -1])
def test_event_sequence_must_be_a_positive_integer(bad_sequence: int) -> None:
    with pytest.raises(ValidationError):
        RunEvent(
            run_id=RUN_ID,
            sequence=bad_sequence,
            event_type=EventType.RUN_STARTED,
            timestamp=T0,
        )


def test_event_rejects_a_foreign_run_id_and_naive_timestamp() -> None:
    with pytest.raises(ValidationError):
        RunEvent(
            run_id=WORK_UNIT_ID,
            sequence=1,
            event_type=EventType.RUN_STARTED,
            timestamp=T0,
        )
    with pytest.raises(ValidationError):
        RunEvent(
            run_id=RUN_ID,
            sequence=1,
            event_type=EventType.RUN_STARTED,
            timestamp=datetime(2026, 8, 10, 12, 0),
        )


def test_event_payload_rejects_a_non_json_value() -> None:
    with pytest.raises(ValidationError):
        RunEvent(
            run_id=RUN_ID,
            sequence=1,
            event_type=EventType.RUN_STARTED,
            payload={"model": object()},  # type: ignore[dict-item]
            timestamp=T0,
        )


def test_payload_from_domain_model_is_json_safe() -> None:
    decision = ControllerDecision(
        run_id=RUN_ID,
        action=ControllerAction.SELECT_STRATEGY,
        rationale="direct is sufficient",
        to_strategy=ExecutionStrategy.DIRECT,
        decided_at=T0,
    )
    event = RunEvent(
        run_id=RUN_ID,
        sequence=2,
        event_type=EventType.CONTROLLER_DECISION,
        payload=payload_from_model("decision", decision),
        timestamp=T0,
    )
    assert event.payload["decision"] == decision.model_dump(mode="json")
    assert RunEvent.model_validate_json(event.model_dump_json()) == event

    request_event = RunEvent(
        run_id=RUN_ID,
        sequence=1,
        event_type=EventType.RUN_CREATED,
        payload=payload_from_model("request", NativeRunRequest(input="hello")),
        timestamp=T0,
    )
    assert request_event.payload["request"]["routing_policy"] == "AUTO"  # type: ignore[index]

    usage_event = RunEvent(
        run_id=RUN_ID,
        sequence=3,
        event_type=EventType.USAGE_UPDATED,
        payload=payload_from_model("usage", Usage(input_tokens=3, output_tokens=4)),
        timestamp=T0,
    )
    assert usage_event.payload["usage"] == {
        "input_tokens": 3,
        "output_tokens": 4,
        "strong_model_tokens": 0,
        "elapsed_ms": 0,
    }


# --- sequence rules -------------------------------------------------------------


def test_next_sequence_starts_at_one_and_increases() -> None:
    assert next_sequence(None) == 1
    assert next_sequence(1) == 2
    assert next_sequence(41) == 42


@pytest.mark.parametrize("invalid", [0, -5])
def test_next_sequence_rejects_a_non_positive_previous_value(invalid: int) -> None:
    with pytest.raises(StateError) as excinfo:
        next_sequence(invalid)
    assert excinfo.value.code is ErrorCode.EVENT_SEQUENCE_INVALID


def test_sequence_chain_accepts_a_gapless_ledger() -> None:
    ledger = [
        make_event(EventType.RUN_CREATED, 1),
        make_event(EventType.RUN_STARTED, 2),
        make_event(EventType.RUN_SUCCEEDED, 3),
    ]
    assert assert_sequence_chain(ledger) is None
    assert assert_sequence_chain([]) is None


@pytest.mark.parametrize("sequences", [(1, 3), (1, 1), (2, 3), (1, 2, 2)])
def test_sequence_chain_rejects_gaps_repeats_and_wrong_start(sequences: tuple[int, ...]) -> None:
    ledger = [make_event(EventType.RUN_STARTED, sequence) for sequence in sequences]
    with pytest.raises(StateError) as excinfo:
        assert_sequence_chain(ledger)
    assert excinfo.value.code is ErrorCode.EVENT_SEQUENCE_INVALID


def test_sequence_chain_rejects_mixed_runs() -> None:
    other = RunEvent(
        run_id=new_run_id(),
        sequence=2,
        event_type=EventType.RUN_STARTED,
        timestamp=T0 + timedelta(seconds=1),
    )
    with pytest.raises(StateError, match="mixes run ids"):
        assert_sequence_chain([make_event(EventType.RUN_CREATED, 1), other])


# --- cross-layer consistency ----------------------------------------------------


RUN_STATUS_EVENTS: dict[RunStatus, EventType] = {
    RunStatus.PENDING: EventType.RUN_CREATED,
    RunStatus.RUNNING: EventType.RUN_STARTED,
    RunStatus.CANCELLING: EventType.RUN_CANCELLING,
    RunStatus.SUCCEEDED: EventType.RUN_SUCCEEDED,
    RunStatus.FAILED: EventType.RUN_FAILED,
    RunStatus.CANCELLED: EventType.RUN_CANCELLED,
}

WORK_UNIT_STATUS_EVENTS: dict[WorkUnitStatus, EventType] = {
    WorkUnitStatus.PENDING: EventType.WORK_UNIT_CREATED,
    WorkUnitStatus.READY: EventType.WORK_UNIT_READY,
    WorkUnitStatus.RUNNING: EventType.WORK_UNIT_STARTED,
    WorkUnitStatus.BLOCKED: EventType.WORK_UNIT_BLOCKED,
    WorkUnitStatus.SUCCEEDED: EventType.WORK_UNIT_SUCCEEDED,
    WorkUnitStatus.FAILED: EventType.WORK_UNIT_FAILED,
    WorkUnitStatus.CANCELLED: EventType.WORK_UNIT_CANCELLED,
    WorkUnitStatus.INVALIDATED: EventType.WORK_UNIT_INVALIDATED,
}

# A PENDING attempt is not observable: it becomes an event only when dispatched.
ATTEMPT_STATUS_EVENTS: dict[AttemptStatus, EventType] = {
    AttemptStatus.RUNNING: EventType.ATTEMPT_STARTED,
    AttemptStatus.SUCCEEDED: EventType.ATTEMPT_SUCCEEDED,
    AttemptStatus.FAILED: EventType.ATTEMPT_FAILED,
    AttemptStatus.CANCELLED: EventType.ATTEMPT_CANCELLED,
    AttemptStatus.INTERRUPTED: EventType.ATTEMPT_INTERRUPTED,
    AttemptStatus.UNKNOWN: EventType.ATTEMPT_UNKNOWN,
}


def test_every_lifecycle_status_has_a_distinct_event() -> None:
    assert set(RUN_STATUS_EVENTS) == set(RunStatus)
    assert set(WORK_UNIT_STATUS_EVENTS) == set(WorkUnitStatus)
    assert set(ATTEMPT_STATUS_EVENTS) == set(AttemptStatus) - {AttemptStatus.PENDING}
    mapped = (
        list(RUN_STATUS_EVENTS.values())
        + list(WORK_UNIT_STATUS_EVENTS.values())
        + list(ATTEMPT_STATUS_EVENTS.values())
    )
    assert len(set(mapped)) == len(mapped)
    assert set(mapped) <= set(EventType)
