"""Targeted tests for the native domain contracts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from prp_runtime.control.reservations import Reservation, ReservationRequest
from prp_runtime.domain.enums import (
    AgentMode,
    AttemptStatus,
    ExecutionLocation,
    ExecutionStrategy,
    IsolationMode,
    ModelRole,
    ReservationStatus,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.models import (
    MAX_AGENT_HISTORY_ITEMS,
    MAX_ARTIFACT_CONTENT_BYTES,
    MAX_PUBLIC_TEXT_CHARS,
    AgentHistoryRecord,
    AgentRequestOptions,
    AgentToolCall,
    AgentToolResult,
    AgentTurn,
    Artifact,
    ArtifactKind,
    Attempt,
    AttemptCost,
    Budget,
    ControllerAction,
    ControllerDecision,
    ErrorCategory,
    ErrorInfo,
    Evidence,
    EvidenceKind,
    ExecutionScope,
    NativeRunRequest,
    OutputRequirement,
    Principal,
    RoutingIntent,
    Run,
    RunMetrics,
    Session,
    SessionCreateRequest,
    SessionStatus,
    Usage,
    VerificationResult,
    WorkspaceGrant,
    WorkUnit,
    new_artifact_id,
    new_evidence_id,
)
from prp_runtime.domain.values import (
    ModelRef,
    ResourceClaim,
    new_attempt_id,
    new_principal_id,
    new_reservation_id,
    new_run_id,
    new_session_id,
    new_work_unit_id,
    new_workspace_id,
)

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
RUN_ID = new_run_id()
WORK_UNIT_ID = new_work_unit_id()
ATTEMPT_ID = new_attempt_id()
ARTIFACT_ID = new_artifact_id()
EVIDENCE_ID = new_evidence_id()
RESERVATION_ID = new_reservation_id()
WORKER_MODEL = ModelRef(provider="openai_compatible", model="weak-model")


@pytest.mark.parametrize(
    ("enum_type", "values"),
    [
        (AgentMode, ("NORMAL", "AUTO", "PLAN", "YOLO")),
        (IsolationMode, ("SANDBOXED", "HOST")),
        (ExecutionLocation, ("CLOUD", "BRIDGE")),
    ],
)
def test_agent_execution_vocabularies_are_closed_and_json_stable(
    enum_type: type[AgentMode | IsolationMode | ExecutionLocation],
    values: tuple[str, ...],
) -> None:
    assert tuple(member.value for member in enum_type) == values
    adapter = TypeAdapter(enum_type)
    for value in values:
        member = enum_type(value)
        assert adapter.validate_json(adapter.dump_json(member)) is member
    with pytest.raises(ValueError):
        enum_type("UNKNOWN")

ALL_CONTRACTS: tuple[type[BaseModel], ...] = (
    AgentRequestOptions,
    Principal,
    Artifact,
    Attempt,
    Budget,
    ControllerDecision,
    ErrorInfo,
    Evidence,
    NativeRunRequest,
    OutputRequirement,
    RoutingIntent,
    Session,
    SessionCreateRequest,
    WorkspaceGrant,
    Run,
    Usage,
    WorkUnit,
)


def make_request(**overrides: object) -> NativeRunRequest:
    data: dict[str, object] = {"input": "summarise the report"}
    data.update(overrides)
    return NativeRunRequest(**data)  # type: ignore[arg-type]


def make_history_record(**overrides: object) -> AgentHistoryRecord:
    data: dict[str, object] = {
        "run_id": RUN_ID,
        "work_unit_id": WORK_UNIT_ID,
        "attempt_id": ATTEMPT_ID,
        "sequence": 1,
        "idempotency_key": "turn-1",
        "item": AgentTurn(text="public answer"),
        "created_at": T0,
    }
    data.update(overrides)
    return AgentHistoryRecord(**data)  # type: ignore[arg-type]


def test_agent_history_record_is_bounded_public_and_discriminated() -> None:
    record = make_history_record()
    assert AgentHistoryRecord.model_validate_json(record.model_dump_json()) == record
    result_record = make_history_record(
        item=AgentToolResult(
            call_id="call-1",
            status=ToolCallStatus.SUCCEEDED,
            result={"content": "safe"},
        )
    )
    assert result_record.item.kind == "tool_result"
    with pytest.raises(ValidationError):
        make_history_record(sequence=MAX_AGENT_HISTORY_ITEMS + 1)
    with pytest.raises(ValidationError, match="history"):
        make_history_record(
            item=AgentTurn(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-secret",
                        tool_name="read_file",
                        arguments={"token": "must not persist"},
                    ),
                )
            )
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_public_tool_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValidationError):
        AgentToolCall(
            call_id="call-finite",
            tool_name="read_file",
            arguments={"value": value},
        )
    with pytest.raises(ValidationError):
        AgentToolResult(
            call_id="call-finite",
            status=ToolCallStatus.SUCCEEDED,
            result={"value": value},
        )


def test_public_text_and_artifact_payloads_are_bounded() -> None:
    with pytest.raises(ValidationError):
        make_request(input="x" * (MAX_PUBLIC_TEXT_CHARS + 1))
    # Keep the character count below the text limit while exceeding the byte
    # limit, so multibyte content cannot bypass the persisted payload bound.
    with pytest.raises(ValidationError):
        make_artifact(content="界" * (MAX_ARTIFACT_CONTENT_BYTES // 3 + 1))


def make_run(**overrides: object) -> Run:
    data: dict[str, object] = {
        "run_id": RUN_ID,
        "request": make_request(),
        "created_at": T0,
    }
    data.update(overrides)
    return Run(**data)  # type: ignore[arg-type]


def make_work_unit(**overrides: object) -> WorkUnit:
    data: dict[str, object] = {
        "work_unit_id": WORK_UNIT_ID,
        "run_id": RUN_ID,
        "name": "summarise",
        "instruction": "summarise section one",
    }
    data.update(overrides)
    return WorkUnit(**data)  # type: ignore[arg-type]


def test_work_unit_lineage_is_optional_for_legacy_single_unit_paths() -> None:
    assert make_work_unit().lineage_key is None
    assert make_work_unit(
        lineage_key="stable-node",
        dependency_fingerprint="a" * 64,
        content_fingerprint="b" * 64,
    ).lineage_key == "stable-node"
    with pytest.raises(ValidationError, match="proposal-local"):
        make_work_unit(lineage_key="wu_old")


def test_session_grant_is_principal_and_workspace_scoped() -> None:
    principal_id = new_principal_id()
    workspace_id = new_workspace_id()
    session = Session(
        session_id=new_session_id(),
        principal_id=principal_id,
        workspace_id=workspace_id,
        grant=WorkspaceGrant(
            principal_id=principal_id,
            workspace_id=workspace_id,
            access=(ResourceAccess.READ, ResourceAccess.WRITE),
        ),
        expires_at=T0 + timedelta(hours=1),
        created_at=T0,
    )
    assert session.status is SessionStatus.ACTIVE
    assert (
        SessionCreateRequest(workspace_id=workspace_id).agent_options.agent_mode
        is AgentMode.NORMAL
    )
    assert "host_path" not in Session.model_fields
    with pytest.raises(ValidationError, match="match the session principal"):
        Session(
            session_id=new_session_id(),
            principal_id=principal_id,
            workspace_id=workspace_id,
            grant=WorkspaceGrant(
                principal_id=new_principal_id(),
                workspace_id=workspace_id,
            ),
            created_at=T0,
        )


def test_execution_scope_is_owner_scoped_and_contains_no_host_root() -> None:
    principal_id = new_principal_id()
    workspace_id = new_workspace_id()
    scope = ExecutionScope(
        run_id=new_run_id(),
        session_id=new_session_id(),
        principal_id=principal_id,
        workspace_id=workspace_id,
        grant=WorkspaceGrant(
            principal_id=principal_id,
            workspace_id=workspace_id,
            access=(ResourceAccess.READ,),
        ),
    )

    assert ExecutionScope.model_validate_json(scope.model_dump_json()) == scope
    assert "root" not in scope.model_fields
    with pytest.raises(ValidationError):
        scope.workspace_id = new_workspace_id()  # type: ignore[misc]


def make_attempt(**overrides: object) -> Attempt:
    data: dict[str, object] = {
        "attempt_id": ATTEMPT_ID,
        "run_id": RUN_ID,
        "work_unit_id": WORK_UNIT_ID,
        "role": ModelRole.WORKER,
        "model": WORKER_MODEL,
    }
    data.update(overrides)
    return Attempt(**data)  # type: ignore[arg-type]


def make_artifact(**overrides: object) -> Artifact:
    data: dict[str, object] = {
        "artifact_id": ARTIFACT_ID,
        "run_id": RUN_ID,
        "work_unit_id": WORK_UNIT_ID,
        "attempt_id": ATTEMPT_ID,
        "name": "answer",
        "content": "the summary",
    }
    data.update(overrides)
    return Artifact(**data)  # type: ignore[arg-type]


def make_evidence(**overrides: object) -> Evidence:
    data: dict[str, object] = {
        "evidence_id": EVIDENCE_ID,
        "run_id": RUN_ID,
        "work_unit_id": WORK_UNIT_ID,
        "artifact_id": ARTIFACT_ID,
        "kind": EvidenceKind.DETERMINISTIC_CHECK,
        "rule": "REQUIRED_REFERENCES",
        "result": VerificationResult.PASS,
        "detail": "required sections present",
    }
    data.update(overrides)
    return Evidence(**data)  # type: ignore[arg-type]


# --- shared contract properties -------------------------------------------------


@pytest.mark.parametrize("contract", ALL_CONTRACTS)
def test_contracts_forbid_unknown_fields_and_are_frozen(contract: type[BaseModel]) -> None:
    assert contract.model_config.get("extra") == "forbid"
    assert contract.model_config.get("frozen") is True


@pytest.mark.parametrize("contract", ALL_CONTRACTS)
def test_contracts_never_expose_chain_of_thought(contract: type[BaseModel]) -> None:
    forbidden = {"chain_of_thought", "reasoning", "thoughts", "raw_response", "raw_request"}
    assert forbidden.isdisjoint(contract.model_fields)


# --- usage ----------------------------------------------------------------------


def test_usage_defaults_and_total() -> None:
    usage = Usage()
    assert (usage.input_tokens, usage.output_tokens, usage.elapsed_ms) == (0, 0, 0)
    assert usage.total_tokens == 0
    assert Usage(input_tokens=10, output_tokens=5).total_tokens == 15


def test_attempt_cost_distinguishes_unknown_usage_from_measured_zero() -> None:
    assert AttemptCost.from_usage(
        None,
        input_price_per_million_tokens=Decimal("3"),
        output_price_per_million_tokens=Decimal("15"),
    ) is None
    zero = AttemptCost.from_usage(
        Usage(input_tokens=0, output_tokens=0),
        input_price_per_million_tokens=Decimal("3"),
        output_price_per_million_tokens=Decimal("15"),
    )
    assert zero is not None
    assert zero.total_cost == Decimal("0")


def test_run_metrics_separate_provider_elapsed_and_run_wall_clock() -> None:
    metrics = RunMetrics.from_attempt(
        usage=Usage(input_tokens=2, output_tokens=1, elapsed_ms=40),
        wall_clock_ms=100,
    )
    assert metrics.provider_elapsed_ms == 40
    assert metrics.wall_clock_ms == 100
    assert metrics.provider_elapsed_ms != metrics.wall_clock_ms
    assert RunMetrics.model_validate_json(metrics.model_dump_json()) == metrics


def test_run_metrics_addition_propagates_unknown_facts_and_exact_cost() -> None:
    known_cost = AttemptCost.from_usage(
        Usage(input_tokens=1, output_tokens=1, elapsed_ms=10),
        input_price_per_million_tokens=Decimal("0.1"),
        output_price_per_million_tokens=Decimal("0.2"),
    )
    assert known_cost is not None
    known_cost_usage = Usage(input_tokens=1, output_tokens=1, elapsed_ms=10)
    first = RunMetrics.from_attempt(
        usage=known_cost_usage, cost=known_cost, wall_clock_ms=30
    )
    second = RunMetrics.from_attempt(
        usage=Usage(input_tokens=2, output_tokens=3, elapsed_ms=20),
        cost=AttemptCost.from_usage(
            Usage(input_tokens=2, output_tokens=3, elapsed_ms=20),
            input_price_per_million_tokens=Decimal("0.1"),
            output_price_per_million_tokens=Decimal("0.2"),
        ),
        wall_clock_ms=50,
    )
    total = first + second
    assert total.usage == known_cost_usage + Usage(input_tokens=2, output_tokens=3, elapsed_ms=20)
    assert total.provider_elapsed_ms == 30
    assert total.wall_clock_ms == 80
    assert total.cost == Decimal("0.0000011")
    assert (total + RunMetrics.from_attempt(usage=None)).cost is None


def test_run_metrics_reject_inconsistent_or_fabricated_cost_facts() -> None:
    with pytest.raises(ValidationError):
        RunMetrics(cost=Decimal("0.01"))
    with pytest.raises(ValidationError):
        RunMetrics(usage=Usage(elapsed_ms=3), provider_elapsed_ms=4)


def test_usage_addition_is_component_wise() -> None:
    first = Usage(input_tokens=10, output_tokens=5, strong_model_tokens=3, elapsed_ms=100)
    second = Usage(input_tokens=1, output_tokens=2, strong_model_tokens=1, elapsed_ms=50)
    assert first + second == Usage(
        input_tokens=11, output_tokens=7, strong_model_tokens=4, elapsed_ms=150
    )


def test_usage_rejects_negative_and_impossible_values() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1)
    with pytest.raises(ValidationError):
        Usage(elapsed_ms=-1)
    with pytest.raises(ValidationError):
        Usage(input_tokens=1, output_tokens=1, strong_model_tokens=3)


# --- budget ---------------------------------------------------------------------


def test_budget_is_unbounded_by_default() -> None:
    budget = Budget()
    assert budget.max_total_tokens is None
    assert budget.max_attempts is None
    assert budget.deadline is None


@pytest.mark.parametrize(
    "field",
    [
        "max_total_tokens",
        "max_strong_model_tokens",
        "max_attempts",
        "max_concurrency",
        "max_plan_revisions",
    ],
)
def test_budget_rejects_negative_ceilings(field: str) -> None:
    with pytest.raises(ValidationError):
        Budget(**{field: -1})


def test_budget_rejects_zero_attempts_and_zero_concurrency() -> None:
    with pytest.raises(ValidationError):
        Budget(max_attempts=0)
    with pytest.raises(ValidationError):
        Budget(max_concurrency=0)
    assert Budget(max_plan_revisions=0).max_plan_revisions == 0


def test_budget_strong_ceiling_cannot_exceed_total() -> None:
    with pytest.raises(ValidationError):
        Budget(max_total_tokens=100, max_strong_model_tokens=101)
    assert Budget(max_total_tokens=100, max_strong_model_tokens=100).max_total_tokens == 100


def test_budget_deadline_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        Budget(deadline=datetime(2026, 8, 10, 12, 0))
    assert Budget(deadline=T0).deadline == T0


# --- reservation ---------------------------------------------------------------


def make_reservation_request(**overrides: object) -> ReservationRequest:
    data: dict[str, object] = {
        "run_id": RUN_ID,
        "work_unit_id": WORK_UNIT_ID,
        "dispatch_key": "run-dispatch-1",
    }
    data.update(overrides)
    return ReservationRequest(**data)  # type: ignore[arg-type]


def make_reservation(**overrides: object) -> Reservation:
    data: dict[str, object] = {
        "reservation_id": RESERVATION_ID,
        "request": make_reservation_request(),
        "created_at": T0,
    }
    data.update(overrides)
    return Reservation(**data)  # type: ignore[arg-type]


def test_reservation_id_and_lifecycle_vocabularies_are_closed() -> None:
    assert RESERVATION_ID.startswith("res_")
    assert tuple(member.value for member in ReservationStatus) == (
        "PENDING",
        "HELD",
        "SETTLED",
        "RELEASED",
        "EXPIRED",
    )
    assert ReservationStatus.SETTLED.is_terminal is True
    assert ReservationStatus.HELD.is_terminal is False


def test_reservation_request_rejects_blank_or_unbounded_dispatch_facts() -> None:
    with pytest.raises(ValidationError):
        make_reservation_request(dispatch_key="   ")
    with pytest.raises(ValidationError):
        make_reservation_request(attempt_units=0)
    with pytest.raises(ValidationError):
        make_reservation_request(
            estimated_input_tokens=8,
            estimated_output_tokens=4,
            token_upper_bound=11,
        )
    with pytest.raises(ValidationError):
        make_reservation_request(extra_fact=True)


def test_reservation_lifecycle_requires_consistent_public_facts() -> None:
    assert make_reservation().status is ReservationStatus.PENDING
    with pytest.raises(ValidationError, match="held_at"):
        make_reservation(status=ReservationStatus.HELD)
    with pytest.raises(ValidationError, match="completed_at"):
        make_reservation(
            status=ReservationStatus.RELEASED,
            held_at=T0,
        )
    with pytest.raises(ValidationError, match="precede"):
        make_reservation(
            status=ReservationStatus.EXPIRED,
            held_at=T0,
            completed_at=T0 - timedelta(seconds=1),
        )


def test_held_and_terminal_reservations_round_trip_without_io() -> None:
    held = make_reservation(status=ReservationStatus.HELD, held_at=T0)
    settled = make_reservation(
        status=ReservationStatus.SETTLED,
        held_at=T0,
        completed_at=T0 + timedelta(seconds=1),
        measured_usage=Usage(input_tokens=3, output_tokens=2),
    )
    assert Reservation.model_validate_json(held.model_dump_json()) == held
    assert Reservation.model_validate_json(settled.model_dump_json()) == settled


# --- output requirement ---------------------------------------------------------


def test_output_requirement_defaults_to_text() -> None:
    assert OutputRequirement().kind is ArtifactKind.TEXT


def test_output_requirement_schema_rules() -> None:
    requirement = OutputRequirement(kind=ArtifactKind.JSON, json_schema='{"type": "object"}')
    assert requirement.json_schema is not None
    with pytest.raises(ValidationError):
        OutputRequirement(kind=ArtifactKind.TEXT, json_schema='{"type": "object"}')
    with pytest.raises(ValidationError):
        OutputRequirement(kind=ArtifactKind.JSON, json_schema="{not json")


#: The constants ``json.loads`` accepts but the JSON grammar does not define.
NON_STANDARD_JSON_CONSTANTS = ["NaN", "Infinity", "-Infinity"]


@pytest.mark.parametrize("constant", NON_STANDARD_JSON_CONSTANTS)
def test_output_requirement_rejects_non_standard_json_schema(constant: str) -> None:
    with pytest.raises(ValidationError):
        OutputRequirement(
            kind=ArtifactKind.JSON,
            json_schema='{"type": "number", "const": ' + constant + "}",
        )


def test_output_requirement_rejects_schema_number_that_overflows_to_infinity() -> None:
    with pytest.raises(ValidationError):
        OutputRequirement(kind=ArtifactKind.JSON, json_schema='{"maximum": 1e999}')


def test_output_requirement_keeps_ordinary_schema_values() -> None:
    requirement = OutputRequirement(
        kind=ArtifactKind.JSON,
        json_schema='{"type": "object", "properties": {"n": {"maximum": 1.5}}}',
    )
    assert requirement.json_schema is not None


# --- native run request ---------------------------------------------------------


def test_request_defaults_to_auto_routing() -> None:
    request = make_request()
    assert request.routing_policy is RoutingPolicy.AUTO
    assert request.strategy is None
    assert request.routing is None
    assert request.agent_options == AgentRequestOptions()
    assert request.budget == Budget()


def test_agent_request_options_keep_dimensions_independent() -> None:
    request = make_request(
        agent_options=AgentRequestOptions(
            agent_mode=AgentMode.YOLO,
            isolation_mode=IsolationMode.HOST,
            execution_location=ExecutionLocation.BRIDGE,
            user_explicit=True,
        )
    )
    assert request.agent_options.model_dump() == {
        "agent_mode": "YOLO",
        "isolation_mode": "HOST",
        "execution_location": "BRIDGE",
        "user_explicit": True,
    }
    assert NativeRunRequest.model_validate_json(request.model_dump_json()) == request


def test_agent_request_options_do_not_derive_permissions_from_routing_intent() -> None:
    request = make_request(routing=RoutingIntent(requires_plan=True))
    assert request.agent_options == AgentRequestOptions()


def test_host_yolo_requires_an_explicit_user_fact() -> None:
    with pytest.raises(ValidationError, match="user_explicit"):
        AgentRequestOptions(agent_mode=AgentMode.YOLO, isolation_mode=IsolationMode.HOST)
    options = AgentRequestOptions(
        agent_mode=AgentMode.YOLO,
        isolation_mode=IsolationMode.HOST,
        user_explicit=True,
    )
    assert options.user_explicit is True


def test_user_explicit_is_a_strict_boolean_fact() -> None:
    with pytest.raises(ValidationError):
        AgentRequestOptions(user_explicit="true")


def test_agent_request_options_reject_unknown_values_and_fields() -> None:
    with pytest.raises(ValidationError):
        AgentRequestOptions(agent_mode="ELEVATED")
    with pytest.raises(ValidationError):
        AgentRequestOptions(allow_all=True)


def test_routing_intent_is_closed_immutable_and_contains_only_public_facts() -> None:
    intent = RoutingIntent(requires_plan=True, desired_parallelism=2)
    assert intent.model_dump() == {
        "requires_cascade": False,
        "requires_plan": True,
        "requires_revision": False,
        "desired_parallelism": 2,
    }
    with pytest.raises(ValidationError):
        intent.requires_plan = False
    with pytest.raises(ValidationError):
        RoutingIntent(api_key="redacted")


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_request_rejects_blank_input(blank: str) -> None:
    with pytest.raises(ValidationError):
        make_request(input=blank)


def test_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        make_request(temperature=0.7)


def test_manual_routing_requires_a_strategy() -> None:
    with pytest.raises(ValidationError):
        make_request(routing_policy=RoutingPolicy.MANUAL)
    request = make_request(
        routing_policy=RoutingPolicy.MANUAL, strategy=ExecutionStrategy.CASCADE
    )
    assert request.strategy is ExecutionStrategy.CASCADE


def test_manual_routing_rejects_auto_routing_intent() -> None:
    with pytest.raises(ValidationError):
        make_request(
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
            routing=RoutingIntent(requires_cascade=True),
        )


def test_auto_routing_must_not_pin_a_strategy() -> None:
    with pytest.raises(ValidationError):
        make_request(strategy=ExecutionStrategy.PLANNED)


def test_request_round_trips_through_json() -> None:
    request = make_request(
        instructions="be concise",
        routing_policy=RoutingPolicy.MANUAL,
        strategy=ExecutionStrategy.PROGRESSIVE,
        budget=Budget(max_total_tokens=1000, max_attempts=3, deadline=T0),
        output=OutputRequirement(kind=ArtifactKind.JSON, json_schema='{"type": "object"}'),
    )
    assert NativeRunRequest.model_validate_json(request.model_dump_json()) == request


# --- run ------------------------------------------------------------------------


def test_pending_run_has_no_timestamps_beyond_creation() -> None:
    run = make_run()
    assert run.status is RunStatus.PENDING
    assert run.started_at is None
    assert run.completed_at is None
    assert run.usage == Usage()
    assert run.graph_version == 1


def test_pending_run_rejects_started_at() -> None:
    with pytest.raises(ValidationError):
        make_run(started_at=T0)


def test_running_run_requires_started_at() -> None:
    with pytest.raises(ValidationError):
        make_run(status=RunStatus.RUNNING)
    run = make_run(status=RunStatus.RUNNING, started_at=T0)
    assert run.started_at == T0


def test_terminal_run_requires_completed_at() -> None:
    with pytest.raises(ValidationError):
        make_run(status=RunStatus.SUCCEEDED, started_at=T0)
    run = make_run(
        status=RunStatus.SUCCEEDED,
        started_at=T0,
        completed_at=T0 + timedelta(seconds=2),
    )
    assert run.status.is_terminal


def test_non_terminal_run_rejects_completed_at() -> None:
    with pytest.raises(ValidationError):
        make_run(status=RunStatus.RUNNING, started_at=T0, completed_at=T0)


def test_failed_run_requires_error_and_others_forbid_it() -> None:
    error = ErrorInfo(category=ErrorCategory.PROVIDER_ERROR, message="upstream rejected request")
    with pytest.raises(ValidationError):
        make_run(status=RunStatus.FAILED, started_at=T0, completed_at=T0)
    failed = make_run(status=RunStatus.FAILED, started_at=T0, completed_at=T0, error=error)
    assert failed.error == error
    with pytest.raises(ValidationError):
        make_run(status=RunStatus.SUCCEEDED, started_at=T0, completed_at=T0, error=error)


def test_run_timestamps_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        make_run(status=RunStatus.RUNNING, started_at=T0 - timedelta(seconds=1))
    with pytest.raises(ValidationError):
        make_run(
            status=RunStatus.SUCCEEDED,
            started_at=T0 + timedelta(seconds=2),
            completed_at=T0 + timedelta(seconds=1),
        )


def test_manually_pinned_strategy_cannot_be_replaced() -> None:
    request = make_request(
        routing_policy=RoutingPolicy.MANUAL, strategy=ExecutionStrategy.DIRECT
    )
    with pytest.raises(ValidationError):
        make_run(request=request, strategy=ExecutionStrategy.PLANNED)
    run = make_run(request=request, strategy=ExecutionStrategy.DIRECT)
    assert run.strategy is ExecutionStrategy.DIRECT


def test_auto_run_may_record_the_decided_strategy() -> None:
    run = make_run(strategy=ExecutionStrategy.CASCADE)
    assert run.strategy is ExecutionStrategy.CASCADE


def test_run_rejects_foreign_identifier() -> None:
    with pytest.raises(ValidationError):
        make_run(run_id=WORK_UNIT_ID)


def test_run_round_trips_through_json() -> None:
    run = make_run(
        status=RunStatus.SUCCEEDED,
        strategy=ExecutionStrategy.DIRECT,
        started_at=T0,
        completed_at=T0 + timedelta(seconds=3),
        usage=Usage(input_tokens=12, output_tokens=30, strong_model_tokens=0, elapsed_ms=900),
    )
    assert Run.model_validate_json(run.model_dump_json()) == run


# --- work unit ------------------------------------------------------------------


def test_work_unit_defaults() -> None:
    unit = make_work_unit()
    assert unit.status is WorkUnitStatus.PENDING
    assert unit.depends_on == ()
    assert unit.resource_claims == ()
    assert unit.writes() == ()


def test_work_unit_rejects_self_dependency() -> None:
    with pytest.raises(ValidationError):
        make_work_unit(depends_on=(WORK_UNIT_ID,))


def test_work_unit_rejects_duplicate_dependency() -> None:
    other = new_work_unit_id()
    with pytest.raises(ValidationError):
        make_work_unit(depends_on=(other, other))
    unit = make_work_unit(depends_on=(other,))
    assert unit.depends_on == (other,)


def test_work_unit_rejects_dependency_with_wrong_identifier_type() -> None:
    with pytest.raises(ValidationError):
        make_work_unit(depends_on=(RUN_ID,))


def test_work_unit_resource_claims_are_deduplicated_and_typed() -> None:
    read = ResourceClaim(resource="report.md", access=ResourceAccess.READ)
    write = ResourceClaim(resource="report.md", access=ResourceAccess.WRITE)
    with pytest.raises(ValidationError):
        make_work_unit(resource_claims=(read, read))
    unit = make_work_unit(resource_claims=(read, write))
    assert unit.writes() == ("report.md",)


def test_work_unit_rejects_blank_instruction_and_unknown_field() -> None:
    with pytest.raises(ValidationError):
        make_work_unit(instruction="  ")
    with pytest.raises(ValidationError):
        make_work_unit(tools=["shell"])


def test_work_unit_round_trips_through_json() -> None:
    unit = make_work_unit(
        graph_version=2,
        acceptance_criteria="must mention every section",
        depends_on=(new_work_unit_id(),),
        resource_claims=(ResourceClaim(resource="a", access=ResourceAccess.WRITE),),
        created_at=T0,
    )
    assert WorkUnit.model_validate_json(unit.model_dump_json()) == unit


# --- attempt --------------------------------------------------------------------


def test_pending_attempt_defaults() -> None:
    attempt = make_attempt()
    assert attempt.status is AttemptStatus.PENDING
    assert attempt.attempt_index == 1
    assert attempt.usage is None
    assert attempt.started_at is None


def test_attempt_lifecycle_requires_consistent_timestamps() -> None:
    with pytest.raises(ValidationError):
        make_attempt(started_at=T0)
    with pytest.raises(ValidationError):
        make_attempt(status=AttemptStatus.RUNNING)
    running = make_attempt(status=AttemptStatus.RUNNING, started_at=T0)
    assert running.completed_at is None
    with pytest.raises(ValidationError):
        make_attempt(status=AttemptStatus.RUNNING, started_at=T0, completed_at=T0)
    with pytest.raises(ValidationError):
        make_attempt(status=AttemptStatus.SUCCEEDED, started_at=T0)
    with pytest.raises(ValidationError):
        make_attempt(
            status=AttemptStatus.SUCCEEDED,
            started_at=T0,
            completed_at=T0 - timedelta(seconds=1),
        )


def test_attempt_error_matches_status() -> None:
    error = ErrorInfo(category=ErrorCategory.TIMEOUT, message="upstream timed out")
    with pytest.raises(ValidationError):
        make_attempt(status=AttemptStatus.FAILED, started_at=T0, completed_at=T0)
    failed = make_attempt(
        status=AttemptStatus.FAILED, started_at=T0, completed_at=T0, error=error
    )
    assert failed.error == error
    with pytest.raises(ValidationError):
        make_attempt(
            status=AttemptStatus.SUCCEEDED, started_at=T0, completed_at=T0, error=error
        )


@pytest.mark.parametrize(
    "status", [AttemptStatus.INTERRUPTED, AttemptStatus.UNKNOWN, AttemptStatus.CANCELLED]
)
def test_unconfirmed_attempt_outcomes_are_representable(status: AttemptStatus) -> None:
    attempt = make_attempt(status=status, started_at=T0, completed_at=T0)
    assert attempt.status.is_terminal


def test_attempt_rejects_foreign_identifiers_and_zero_index() -> None:
    with pytest.raises(ValidationError):
        make_attempt(work_unit_id=RUN_ID)
    with pytest.raises(ValidationError):
        make_attempt(attempt_id=WORK_UNIT_ID)
    with pytest.raises(ValidationError):
        make_attempt(attempt_index=0)


def test_attempt_round_trips_through_json() -> None:
    attempt = make_attempt(
        attempt_index=2,
        role=ModelRole.PLANNER,
        model=ModelRef(provider="anthropic", model="strong-model"),
        status=AttemptStatus.SUCCEEDED,
        provider_request_id="req_123",
        usage=Usage(input_tokens=5, output_tokens=7, strong_model_tokens=12, elapsed_ms=42),
        created_at=T0,
        started_at=T0,
        completed_at=T0 + timedelta(milliseconds=42),
    )
    assert Attempt.model_validate_json(attempt.model_dump_json()) == attempt


# --- artifact and evidence ------------------------------------------------------


def test_artifact_requires_non_blank_content() -> None:
    with pytest.raises(ValidationError):
        make_artifact(content="")
    with pytest.raises(ValidationError):
        make_artifact(content="   \n")
    artifact = make_artifact(content="  padded but real  ")
    assert artifact.content == "  padded but real  "


def test_json_artifact_content_must_parse() -> None:
    with pytest.raises(ValidationError):
        make_artifact(kind=ArtifactKind.JSON, content="{not json")
    artifact = make_artifact(kind=ArtifactKind.JSON, content='{"ok": true}')
    assert artifact.kind is ArtifactKind.JSON


#: Every structural position a value can occupy, so no nesting depth is exempt.
NON_STANDARD_JSON_POSITIONS = [
    "{}",
    '{{"value": {}}}',
    "[1, {}]",
    '{{"a": [{{"b": {}}}]}}',
    '{{"a": {{"b": [[{}]]}}}}',
]


@pytest.mark.parametrize("template", NON_STANDARD_JSON_POSITIONS)
@pytest.mark.parametrize("constant", NON_STANDARD_JSON_CONSTANTS)
def test_json_artifact_rejects_non_standard_constants(constant: str, template: str) -> None:
    with pytest.raises(ValidationError):
        make_artifact(kind=ArtifactKind.JSON, content=template.format(constant))


@pytest.mark.parametrize("template", NON_STANDARD_JSON_POSITIONS)
@pytest.mark.parametrize("constant", NON_STANDARD_JSON_CONSTANTS)
def test_output_requirement_rejects_non_standard_constants_anywhere(
    constant: str, template: str
) -> None:
    with pytest.raises(ValidationError):
        OutputRequirement(
            kind=ArtifactKind.JSON, json_schema=template.format(constant)
        )


def test_json_artifact_rejects_number_that_overflows_to_infinity() -> None:
    with pytest.raises(ValidationError):
        make_artifact(kind=ArtifactKind.JSON, content='{"value": 1e999}')


def test_json_artifact_keeps_ordinary_json_values() -> None:
    artifact = make_artifact(
        kind=ArtifactKind.JSON,
        content='{"n": 1, "f": 1.5, "b": false, "z": null, "s": "NaN", "a": []}',
    )
    assert artifact.kind is ArtifactKind.JSON


def test_text_artifact_is_not_parsed_as_json() -> None:
    assert make_artifact(content="NaN").content == "NaN"
    assert make_artifact(content="{not json").content == "{not json"


def test_json_artifact_error_states_a_reason_without_a_traceback() -> None:
    with pytest.raises(ValidationError) as caught:
        make_artifact(kind=ArtifactKind.JSON, content="[Infinity]")
    message = str(caught.value)
    assert "JSON artifact content must be valid JSON" in message
    assert "Infinity" in message
    assert "Traceback" not in message


def test_artifact_references_are_type_checked() -> None:
    with pytest.raises(ValidationError):
        make_artifact(artifact_id=EVIDENCE_ID)
    with pytest.raises(ValidationError):
        make_artifact(attempt_id=WORK_UNIT_ID)


def test_evidence_must_reference_an_artifact_and_a_work_unit() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id=EVIDENCE_ID,
            run_id=RUN_ID,
            work_unit_id=WORK_UNIT_ID,
            kind=EvidenceKind.DETERMINISTIC_CHECK,
            rule="VALID_JSON",
            result=VerificationResult.PASS,
            detail="missing artifact reference",
        )
    with pytest.raises(ValidationError):
        make_evidence(artifact_id=ATTEMPT_ID)
    with pytest.raises(ValidationError):
        make_evidence(detail="   ")


@pytest.mark.parametrize("result", list(VerificationResult))
def test_every_verdict_is_representable(result: VerificationResult) -> None:
    evidence = make_evidence(result=result, detail=f"recorded {result.value}")
    assert evidence.result is result


def test_passed_is_derived_from_result_and_is_not_a_field() -> None:
    assert "passed" not in Evidence.model_fields
    assert make_evidence(result=VerificationResult.PASS).passed is True
    assert make_evidence(result=VerificationResult.FAIL).passed is False
    assert make_evidence(result=VerificationResult.INCONCLUSIVE).passed is False


def test_an_undecided_verdict_stays_distinct_from_a_proven_failure() -> None:
    undecided = make_evidence(result=VerificationResult.INCONCLUSIVE, detail="not decided")
    failed = make_evidence(result=VerificationResult.FAIL, detail="section two missing")
    # Both project to the same boolean, which is exactly why the boolean cannot be
    # the record: only ``result`` tells the two apart.
    assert undecided.passed == failed.passed
    assert undecided.result is not failed.result


@pytest.mark.parametrize("supplied", [True, False])
def test_passed_cannot_be_supplied_as_an_input(supplied: bool) -> None:
    with pytest.raises(ValidationError):
        make_evidence(passed=supplied)


def test_a_verdict_is_required() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id=EVIDENCE_ID,
            run_id=RUN_ID,
            work_unit_id=WORK_UNIT_ID,
            artifact_id=ARTIFACT_ID,
            kind=EvidenceKind.DETERMINISTIC_CHECK,
            rule="VALID_JSON",
            detail="no verdict was reached",
        )
    with pytest.raises(ValidationError):
        make_evidence(result=None)


def test_a_deterministic_check_must_name_its_rule() -> None:
    with pytest.raises(ValidationError, match="DETERMINISTIC_CHECK requires rule"):
        make_evidence(rule=None)
    with pytest.raises(ValidationError):
        make_evidence(rule="   ")


def test_a_model_review_may_omit_the_rule_but_not_the_verdict() -> None:
    reviewed = make_evidence(kind=EvidenceKind.MODEL_REVIEW, rule=None)
    assert reviewed.rule is None
    assert reviewed.result is VerificationResult.PASS
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id=EVIDENCE_ID,
            run_id=RUN_ID,
            work_unit_id=WORK_UNIT_ID,
            artifact_id=ARTIFACT_ID,
            kind=EvidenceKind.MODEL_REVIEW,
            detail="a reviewer looked at it",
        )


def test_evidence_serialises_the_verdict_and_not_the_boolean() -> None:
    payload = make_evidence(result=VerificationResult.INCONCLUSIVE).model_dump()
    assert payload["result"] is VerificationResult.INCONCLUSIVE
    assert "passed" not in payload


def test_artifact_and_evidence_round_trip_through_json() -> None:
    artifact = make_artifact(created_at=T0)
    assert Artifact.model_validate_json(artifact.model_dump_json()) == artifact
    evidence = make_evidence(created_at=T0)
    assert Evidence.model_validate_json(evidence.model_dump_json()) == evidence


# --- controller decision --------------------------------------------------------


def test_select_strategy_decision_requires_target() -> None:
    with pytest.raises(ValidationError):
        ControllerDecision(
            run_id=RUN_ID,
            action=ControllerAction.SELECT_STRATEGY,
            rationale="direct is sufficient",
        )
    decision = ControllerDecision(
        run_id=RUN_ID,
        action=ControllerAction.SELECT_STRATEGY,
        rationale="direct is sufficient",
        to_strategy=ExecutionStrategy.DIRECT,
    )
    assert decision.to_strategy is ExecutionStrategy.DIRECT


def test_escalation_decision_requires_two_distinct_strategies() -> None:
    with pytest.raises(ValidationError):
        ControllerDecision(
            run_id=RUN_ID,
            action=ControllerAction.ESCALATE_STRATEGY,
            rationale="verification failed twice",
            to_strategy=ExecutionStrategy.PLANNED,
        )
    with pytest.raises(ValidationError):
        ControllerDecision(
            run_id=RUN_ID,
            action=ControllerAction.ESCALATE_STRATEGY,
            rationale="verification failed twice",
            from_strategy=ExecutionStrategy.PLANNED,
            to_strategy=ExecutionStrategy.PLANNED,
        )
    decision = ControllerDecision(
        run_id=RUN_ID,
        action=ControllerAction.ESCALATE_STRATEGY,
        rationale="verification failed twice",
        from_strategy=ExecutionStrategy.CASCADE,
        to_strategy=ExecutionStrategy.PLANNED,
    )
    assert decision.from_strategy is ExecutionStrategy.CASCADE


def test_decision_requires_a_rationale_and_unique_evidence() -> None:
    with pytest.raises(ValidationError):
        ControllerDecision(
            run_id=RUN_ID, action=ControllerAction.CANCEL, rationale="  "
        )
    with pytest.raises(ValidationError):
        ControllerDecision(
            run_id=RUN_ID,
            action=ControllerAction.ACCEPT_ARTIFACT,
            rationale="evidence passed",
            evidence_ids=(EVIDENCE_ID, EVIDENCE_ID),
        )


def test_decision_round_trips_through_json() -> None:
    decision = ControllerDecision(
        run_id=RUN_ID,
        action=ControllerAction.ACCEPT_ARTIFACT,
        rationale="deterministic check passed",
        work_unit_id=WORK_UNIT_ID,
        evidence_ids=(EVIDENCE_ID,),
        decided_at=T0,
    )
    assert ControllerDecision.model_validate_json(decision.model_dump_json()) == decision
