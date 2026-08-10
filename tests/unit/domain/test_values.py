"""Targeted tests for domain enums, identifiers and base value objects."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.values import (
    ModelRef,
    ResourceClaim,
    RunId,
    UtcTimestamp,
    new_attempt_id,
    new_run_id,
    new_work_unit_id,
    utc_now,
    validate_attempt_id,
    validate_run_id,
    validate_work_unit_id,
)


def test_execution_strategy_has_exactly_four_members() -> None:
    assert [member.value for member in ExecutionStrategy] == [
        "DIRECT",
        "CASCADE",
        "PLANNED",
        "PROGRESSIVE",
    ]


def test_auto_is_a_routing_policy_not_an_execution_strategy() -> None:
    assert RoutingPolicy.AUTO.value == "AUTO"
    assert [member.value for member in RoutingPolicy] == ["AUTO", "MANUAL"]
    assert "AUTO" not in {member.value for member in ExecutionStrategy}
    with pytest.raises(ValueError):
        ExecutionStrategy("AUTO")


def test_execution_strategy_field_rejects_auto() -> None:
    adapter = TypeAdapter(ExecutionStrategy)
    assert adapter.validate_python("PLANNED") is ExecutionStrategy.PLANNED
    with pytest.raises(ValidationError):
        adapter.validate_python("AUTO")


def test_model_roles_are_explicit() -> None:
    assert [member.value for member in ModelRole] == ["PLANNER", "WORKER", "VERIFIER"]


def test_terminal_run_statuses() -> None:
    terminal = {status for status in RunStatus if status.is_terminal}
    assert terminal == {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
    assert not RunStatus.CANCELLING.is_terminal


def test_terminal_work_unit_statuses() -> None:
    terminal = {status for status in WorkUnitStatus if status.is_terminal}
    assert terminal == {
        WorkUnitStatus.SUCCEEDED,
        WorkUnitStatus.FAILED,
        WorkUnitStatus.CANCELLED,
        WorkUnitStatus.INVALIDATED,
    }
    assert not WorkUnitStatus.BLOCKED.is_terminal


def test_terminal_attempt_statuses_include_unconfirmed_outcomes() -> None:
    terminal = {status for status in AttemptStatus if status.is_terminal}
    assert terminal == {
        AttemptStatus.SUCCEEDED,
        AttemptStatus.FAILED,
        AttemptStatus.CANCELLED,
        AttemptStatus.INTERRUPTED,
        AttemptStatus.UNKNOWN,
    }
    assert not AttemptStatus.RUNNING.is_terminal


@pytest.mark.parametrize("unknown", ["", "auto", "DONE", "succeeded"])
def test_unknown_status_values_are_rejected(unknown: str) -> None:
    with pytest.raises(ValueError):
        RunStatus(unknown)


def test_generated_identifiers_are_prefixed_and_valid() -> None:
    run_id = new_run_id()
    work_unit_id = new_work_unit_id()
    attempt_id = new_attempt_id()
    assert run_id.startswith("run_")
    assert work_unit_id.startswith("wu_")
    assert attempt_id.startswith("att_")
    assert validate_run_id(run_id) == run_id
    assert validate_work_unit_id(work_unit_id) == work_unit_id
    assert validate_attempt_id(attempt_id) == attempt_id


def test_generated_identifiers_are_unique() -> None:
    assert len({new_run_id() for _ in range(50)}) == 50


@pytest.mark.parametrize("bad", ["", "   ", "run_", "run_-bad", "RUN_1", "run 1", "run_a b"])
def test_invalid_run_ids_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_run_id(bad)


def test_identifier_types_cannot_be_mixed() -> None:
    work_unit_id = new_work_unit_id()
    with pytest.raises(ValueError):
        validate_run_id(work_unit_id)
    with pytest.raises(ValueError):
        validate_attempt_id(work_unit_id)
    with pytest.raises(ValueError):
        validate_work_unit_id(new_attempt_id())


def test_run_id_annotation_rejects_empty_and_foreign_ids() -> None:
    adapter = TypeAdapter(RunId)
    run_id = new_run_id()
    assert adapter.validate_python(run_id) == run_id
    with pytest.raises(ValidationError):
        adapter.validate_python("")
    with pytest.raises(ValidationError):
        adapter.validate_python(new_work_unit_id())


def test_utc_now_is_timezone_aware_utc() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_timestamp_normalises_to_utc_and_rejects_naive_values() -> None:
    adapter = TypeAdapter(UtcTimestamp)
    offset = timezone(timedelta(hours=8))
    normalised = adapter.validate_python(datetime(2026, 8, 10, 20, 0, tzinfo=offset))
    assert normalised == datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    assert normalised.tzinfo is UTC
    with pytest.raises(ValidationError):
        adapter.validate_python(datetime(2026, 8, 10, 12, 0))


def test_resource_claim_requires_non_empty_resource() -> None:
    claim = ResourceClaim(resource="  report.md  ", access=ResourceAccess.READ)
    assert claim.resource == "report.md"
    with pytest.raises(ValidationError):
        ResourceClaim(resource="", access=ResourceAccess.READ)
    with pytest.raises(ValidationError):
        ResourceClaim(resource="   ", access=ResourceAccess.WRITE)


def test_resource_claim_rejects_unknown_access_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ResourceClaim(resource="a", access="APPEND")
    with pytest.raises(ValidationError):
        ResourceClaim(resource="a", access=ResourceAccess.READ, mode="shared")


def test_resource_claim_is_frozen() -> None:
    claim = ResourceClaim(resource="a", access=ResourceAccess.READ)
    with pytest.raises(ValidationError):
        claim.resource = "b"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((("a", ResourceAccess.READ)), ("a", ResourceAccess.READ), False),
        ((("a", ResourceAccess.READ)), ("a", ResourceAccess.WRITE), True),
        ((("a", ResourceAccess.WRITE)), ("a", ResourceAccess.WRITE), True),
        ((("a", ResourceAccess.WRITE)), ("b", ResourceAccess.WRITE), False),
    ],
)
def test_resource_claim_conflicts(
    left: tuple[str, ResourceAccess],
    right: tuple[str, ResourceAccess],
    expected: bool,
) -> None:
    first = ResourceClaim(resource=left[0], access=left[1])
    second = ResourceClaim(resource=right[0], access=right[1])
    assert first.conflicts_with(second) is expected
    assert second.conflicts_with(first) is expected


def test_model_ref_identity_and_validation() -> None:
    ref = ModelRef(provider="openai_compatible", model=" gpt-strong ")
    assert ref.model == "gpt-strong"
    assert ref.identifier == "openai_compatible/gpt-strong"
    with pytest.raises(ValidationError):
        ModelRef(provider="", model="gpt-strong")
    with pytest.raises(ValidationError):
        ModelRef(provider="openai_compatible", model="  ")
    with pytest.raises(ValidationError):
        ModelRef(provider="openai_compatible", model="gpt-strong", base_url="http://x")


def test_value_objects_round_trip_through_json() -> None:
    claim = ResourceClaim(resource="report.md", access=ResourceAccess.WRITE)
    assert ResourceClaim.model_validate_json(claim.model_dump_json()) == claim
    assert claim.model_dump() == {"resource": "report.md", "access": ResourceAccess.WRITE}

    ref = ModelRef(provider="anthropic", model="claude-weak")
    assert ModelRef.model_validate_json(ref.model_dump_json()) == ref
