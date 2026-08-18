"""Targeted tests for the CASCADE profile chain construction."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError

from prp_runtime.control.cascade import (
    CascadeDisposition,
    build_cascade_chain,
    decide_cascade,
    provider_failure_is_retryable,
)
from prp_runtime.control.controller import RunController
from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import (
    ArtifactKind,
    Budget,
    ControllerAction,
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
    Usage,
    VerificationResult,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_PROFILE = {
    "provider": "openai_compatible",
    "base_url": "https://models.internal/v1",
    "context_window_tokens": 32_000,
    "max_output_tokens": 4_000,
}


def _worker(alias: str, **overrides: object) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        model=alias,
        role=ModelRole.WORKER,
        **{**_BASE_PROFILE, **overrides},  # type: ignore[arg-type]
    )


def _planner(alias: str, **overrides: object) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        model=alias,
        role=ModelRole.PLANNER,
        **{**_BASE_PROFILE, **overrides},  # type: ignore[arg-type]
    )


W1 = _worker("w1")
W2 = _worker("w2")
W3 = _worker("w3")

# ---------------------------------------------------------------------------
# build_cascade_chain: happy paths
# ---------------------------------------------------------------------------


def test_single_worker_chain_is_valid() -> None:
    chain = build_cascade_chain(W1)
    assert chain == (W1,)


def test_chain_preserves_order() -> None:
    chain = build_cascade_chain(W1, (W2, W3))
    assert chain == (W1, W2, W3)


def test_chain_result_is_a_tuple() -> None:
    chain = build_cascade_chain(W1, (W2,))
    assert isinstance(chain, tuple)


def test_empty_escalation_tuple_is_allowed() -> None:
    chain = build_cascade_chain(W1, ())
    assert len(chain) == 1


# ---------------------------------------------------------------------------
# build_cascade_chain: role constraints
# ---------------------------------------------------------------------------


def test_base_worker_with_planner_role_is_rejected() -> None:
    planner = _planner("p1")
    with pytest.raises(ProviderError) as excinfo:
        build_cascade_chain(planner)
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED
    assert "WORKER" in str(excinfo.value)


def test_escalation_profile_with_planner_role_is_rejected() -> None:
    planner = _planner("p2")
    with pytest.raises(ProviderError) as excinfo:
        build_cascade_chain(W1, (planner,))
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED
    assert "WORKER" in str(excinfo.value)


def test_escalation_verifier_role_is_rejected() -> None:
    verifier = ModelProfile(
        alias="v1",
        model="v1",
        role=ModelRole.VERIFIER,
        **_BASE_PROFILE,  # type: ignore[arg-type]
    )
    with pytest.raises(ProviderError) as excinfo:
        build_cascade_chain(W1, (verifier,))
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


# ---------------------------------------------------------------------------
# build_cascade_chain: duplicate alias rejection
# ---------------------------------------------------------------------------


def test_duplicate_base_alias_in_escalation_is_rejected() -> None:
    with pytest.raises(ProviderError) as excinfo:
        build_cascade_chain(W1, (W1,))
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED
    assert "duplicate" in str(excinfo.value).lower()


def test_duplicate_alias_between_escalation_entries_is_rejected() -> None:
    w2_dup = _worker("w2")
    with pytest.raises(ProviderError) as excinfo:
        build_cascade_chain(W1, (W2, w2_dup))
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


# ---------------------------------------------------------------------------
# Settings: cascade_profiles field
# ---------------------------------------------------------------------------


def test_settings_defaults_cascade_profiles_to_empty_tuple() -> None:
    settings = Settings()
    assert settings.cascade_profiles == ()


def test_settings_accepts_cascade_profiles_as_python_list() -> None:
    settings = Settings(cascade_profiles=(W1, W2))
    assert len(settings.cascade_profiles) == 2
    assert settings.cascade_profiles[0].alias == "w1"


def test_settings_rejects_non_worker_cascade_profile() -> None:
    secret = "sk-must-not-leak"
    planner = _planner("planner", api_key=secret)

    with pytest.raises(ValidationError) as excinfo:
        Settings(cascade_profiles=(planner,))

    assert secret not in str(excinfo.value)


def test_settings_parses_cascade_profiles_from_json_array_string() -> None:
    profiles_json = json.dumps(
        [
            {
                "alias": "fast",
                "model": "fast-model",
                "provider": "openai_compatible",
                "role": "WORKER",
                "base_url": "https://models.internal/v1",
                "context_window_tokens": 16_000,
                "max_output_tokens": 2_000,
            }
        ]
    )
    settings = Settings.from_env({"PRP_CASCADE_PROFILES": profiles_json})
    assert len(settings.cascade_profiles) == 1
    assert settings.cascade_profiles[0].alias == "fast"
    assert settings.cascade_profiles[0].role is ModelRole.WORKER


def test_settings_rejects_cascade_profiles_non_array_json() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings.from_env({"PRP_CASCADE_PROFILES": '{"alias": "x"}'})


def test_settings_rejects_cascade_profiles_invalid_json() -> None:
    with pytest.raises((ValidationError, ValueError)):
        Settings.from_env({"PRP_CASCADE_PROFILES": "{not json"})


def test_settings_cascade_profiles_api_key_never_leaks() -> None:
    import json as _json

    secret = "sk-secret-0000"
    profiles = [
        {
            "alias": "secure",
            "model": "m",
            "provider": "openai_compatible",
            "role": "WORKER",
            "base_url": "https://models.internal/v1",
            "api_key": secret,
            "context_window_tokens": 8_000,
            "max_output_tokens": 1_000,
        }
    ]
    settings = Settings.from_env({"PRP_CASCADE_PROFILES": _json.dumps(profiles)})
    profile = settings.cascade_profiles[0]
    assert secret not in repr(profile)
    assert secret not in profile.model_dump_json()


# ---------------------------------------------------------------------------
# Deterministic escalation policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "has_next", "expected"),
    [
        (VerificationResult.PASS, False, CascadeDisposition.ACCEPT),
        (VerificationResult.PASS, True, CascadeDisposition.ACCEPT),
        (VerificationResult.FAIL, False, CascadeDisposition.STOP),
        (VerificationResult.FAIL, True, CascadeDisposition.ESCALATE),
        (VerificationResult.INCONCLUSIVE, False, CascadeDisposition.STOP),
        (VerificationResult.INCONCLUSIVE, True, CascadeDisposition.ESCALATE),
    ],
)
def test_verification_decision_matrix(
    result: VerificationResult,
    has_next: bool,
    expected: CascadeDisposition,
) -> None:
    decision = decide_cascade(
        verification_result=result,
        has_next_profile=has_next,
    )
    assert decision.disposition is expected
    assert result.value in decision.rationale or result is VerificationResult.PASS


@pytest.mark.parametrize(
    ("retryable", "has_next", "expected"),
    [
        (False, False, CascadeDisposition.STOP),
        (False, True, CascadeDisposition.STOP),
        (True, False, CascadeDisposition.STOP),
        (True, True, CascadeDisposition.ESCALATE),
    ],
)
def test_provider_failure_decision_matrix(
    retryable: bool,
    has_next: bool,
    expected: CascadeDisposition,
) -> None:
    assert (
        decide_cascade(
            provider_retryable=retryable,
            has_next_profile=has_next,
        ).disposition
        is expected
    )


def test_cascade_decision_requires_exactly_one_signal() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        decide_cascade(has_next_profile=True)
    with pytest.raises(ValueError, match="exactly one"):
        decide_cascade(
            has_next_profile=True,
            verification_result=VerificationResult.FAIL,
            provider_retryable=True,
        )


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (ErrorCategory.TIMEOUT, True),
        (ErrorCategory.RATE_LIMIT, True),
        (ErrorCategory.NETWORK, True),
        (ErrorCategory.AUTH, False),
        (ErrorCategory.PROVIDER_ERROR, False),
        (ErrorCategory.UNKNOWN, False),
    ],
)
def test_recorded_provider_failure_retryability(
    category: ErrorCategory, expected: bool
) -> None:
    error = ErrorInfo(category=category, message="provider failed")
    assert provider_failure_is_retryable(error) is expected


# ---------------------------------------------------------------------------
# Controller: bounded execution loop
# ---------------------------------------------------------------------------


class _CascadeAdapter:
    def __init__(self, *responses: object) -> None:
        self._responses = list(responses)
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "cascade-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        response = self._responses.pop(0)
        if callable(response):
            response = await response(request)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, ProviderResponse)
        return response


def _response(text: str, *, usage: Usage | None = None) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        usage=Usage(input_tokens=2, output_tokens=3, elapsed_ms=1)
        if usage is None
        else usage,
        finish_reason=FinishReason.STOP,
    )


@pytest_asyncio.fixture
async def cascade_store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "cascade.db") as opened:
        yield opened


@pytest.mark.asyncio
async def test_cascade_escalates_in_order_and_stops_on_verified_pass(
    cascade_store: SqliteStore,
) -> None:
    weak = _CascadeAdapter(
        ProviderError("weak model failed", code=ErrorCode.PROVIDER_UNAVAILABLE)
    )
    strong = _CascadeAdapter(_response('{"ok": true}'))
    settings = Settings(worker_profile=W1, cascade_profiles=(W2,))
    controller = RunController(
        cascade_store,
        settings,
        {"w1": weak, "w2": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="emit json",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
            output={"kind": ArtifactKind.JSON},
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.strategy is ExecutionStrategy.CASCADE
    assert finished.final_work_unit_id is None
    assert [request.alias for request in weak.requests + strong.requests] == [
        "w1",
        "w2",
    ]
    units = await cascade_store.list_work_units(run.run_id)
    assert len(units) == 1
    assert units[0].name == "cascade"
    assert units[0].status is WorkUnitStatus.SUCCEEDED
    attempts = await cascade_store.list_attempts(units[0].work_unit_id)
    assert [attempt.attempt_index for attempt in attempts] == [1, 2]
    assert [attempt.status for attempt in attempts] == [
        AttemptStatus.FAILED,
        AttemptStatus.SUCCEEDED,
    ]
    assert len(await cascade_store.list_artifacts(units[0].work_unit_id)) == 1
    ledger = await cascade_store.list_events(run.run_id)
    escalated = next(
        event for event in ledger if event.event_type is EventType.STRATEGY_ESCALATED
    )
    assert escalated.payload["from_profile"] == "w1"
    assert escalated.payload["to_profile"] == "w2"
    assert escalated.payload["reason"] == (
        "retryable provider failure permits the next profile"
    )
    decisions = [
        event.payload["decision"]
        for event in ledger
        if event.event_type is EventType.CONTROLLER_DECISION
    ]
    assert any(
        decision["action"] == ControllerAction.ESCALATE_MODEL.value
        for decision in decisions
    )


@pytest.mark.asyncio
async def test_non_retryable_provider_failure_stops_without_escalation(
    cascade_store: SqliteStore,
) -> None:
    weak = _CascadeAdapter(
        ProviderError("bad credentials", code=ErrorCode.PROVIDER_AUTH_FAILED)
    )
    strong = _CascadeAdapter(_response("must not run"))
    controller = RunController(
        cascade_store,
        Settings(worker_profile=W1, cascade_profiles=(W2,)),
        {"w1": weak, "w2": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="hello",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert strong.requests == []
    assert len(await cascade_store.list_run_attempts(run.run_id)) == 1
    types = [event.event_type for event in await cascade_store.list_events(run.run_id)]
    assert EventType.STRATEGY_ESCALATED not in types


@pytest.mark.asyncio
async def test_attempt_budget_blocks_pending_escalation(
    cascade_store: SqliteStore,
) -> None:
    weak = _CascadeAdapter(
        ProviderError("temporary outage", code=ErrorCode.PROVIDER_UNAVAILABLE)
    )
    strong = _CascadeAdapter(_response("must not run"))
    controller = RunController(
        cascade_store,
        Settings(worker_profile=W1, cascade_profiles=(W2,)),
        {"w1": weak, "w2": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="hello",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
            budget=Budget(max_attempts=1),
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert strong.requests == []
    types = [event.event_type for event in await cascade_store.list_events(run.run_id)]
    assert EventType.BUDGET_EXHAUSTED in types
    assert EventType.STRATEGY_ESCALATED not in types


@pytest.mark.asyncio
async def test_exact_token_ceilings_verify_current_artifact_but_block_escalation(
    cascade_store: SqliteStore,
) -> None:
    exact_usage = Usage(
        input_tokens=2,
        output_tokens=3,
        strong_model_tokens=5,
        elapsed_ms=1,
    )
    weak = _CascadeAdapter(_response('{"ok": false}', usage=exact_usage))
    strong = _CascadeAdapter(_response('{"ok": true}'))
    weak_profile = _worker("w1", supports_structured_output=True)
    strong_profile = _worker("w2", supports_structured_output=True)
    controller = RunController(
        cascade_store,
        Settings(worker_profile=weak_profile, cascade_profiles=(strong_profile,)),
        {"w1": weak, "w2": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="emit json",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
            output={
                "kind": ArtifactKind.JSON,
                "json_schema": json.dumps(
                    {
                        "type": "object",
                        "properties": {"ok": {"const": True}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    }
                ),
            },
            budget=Budget(max_total_tokens=5, max_strong_model_tokens=5),
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert len(weak.requests) == 1
    assert strong.requests == []
    assert len(await cascade_store.list_run_attempts(run.run_id)) == 1
    types = [event.event_type for event in await cascade_store.list_events(run.run_id)]
    assert EventType.EVIDENCE_RECORDED in types
    assert EventType.BUDGET_EXHAUSTED in types


@pytest.mark.asyncio
async def test_over_token_ceiling_rejects_cascade_result_before_verification(
    cascade_store: SqliteStore,
) -> None:
    weak = _CascadeAdapter(_response('{"ok": true}'))
    strong = _CascadeAdapter(_response("must not run"))
    controller = RunController(
        cascade_store,
        Settings(worker_profile=W1, cascade_profiles=(W2,)),
        {"w1": weak, "w2": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="emit json",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
            output={"kind": ArtifactKind.JSON},
            budget=Budget(max_total_tokens=4),
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert len(weak.requests) == 1
    assert strong.requests == []
    types = [event.event_type for event in await cascade_store.list_events(run.run_id)]
    assert EventType.EVIDENCE_RECORDED not in types
    assert EventType.BUDGET_EXHAUSTED in types


@pytest.mark.asyncio
async def test_cancel_blocks_pending_escalation(cascade_store: SqliteStore) -> None:
    controller: RunController | None = None
    run_id = ""

    async def cancel_then_fail(request: ProviderRequest) -> ProviderResponse:
        assert controller is not None
        await controller.cancel(run_id)
        raise ProviderError("temporary outage", code=ErrorCode.PROVIDER_UNAVAILABLE)

    weak = _CascadeAdapter(cancel_then_fail)
    strong = _CascadeAdapter(_response("must not run"))
    controller = RunController(
        cascade_store,
        Settings(worker_profile=W1, cascade_profiles=(W2,)),
        {"w1": weak, "w2": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="hello",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
        )
    )
    run_id = run.run_id

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.CANCELLED
    assert strong.requests == []
    types = [event.event_type for event in await cascade_store.list_events(run.run_id)]
    assert EventType.RUN_CANCELLING in types
    assert EventType.STRATEGY_ESCALATED not in types


@pytest.mark.asyncio
async def test_verification_failure_escalates_without_overwriting_evidence(
    cascade_store: SqliteStore,
) -> None:
    structured_w1 = W1.model_copy(update={"supports_structured_output": True})
    structured_w2 = W2.model_copy(update={"supports_structured_output": True})
    weak = _CascadeAdapter(_response('{"not_ok": true}'))
    strong = _CascadeAdapter(_response('{"ok": true}'))
    controller = RunController(
        cascade_store,
        Settings(
            worker_profile=structured_w1,
            cascade_profiles=(structured_w2,),
        ),
        {"w1": weak, "w2": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(
        NativeRunRequest(
            input="emit json",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
            output={
                "kind": ArtifactKind.JSON,
                "json_schema": '{"type":"object","required":["ok"]}',
            },
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    units = await cascade_store.list_work_units(run.run_id)
    artifacts = await cascade_store.list_artifacts(units[0].work_unit_id)
    evidence = await cascade_store.list_evidence(units[0].work_unit_id)
    assert len(artifacts) == 2
    assert len(evidence) == 8
    assert {row.artifact_id for row in evidence} == {
        artifact.artifact_id for artifact in artifacts
    }
    ledger = await cascade_store.list_events(run.run_id)
    event_evidence_ids = {
        event.payload["evidence_id"]
        for event in ledger
        if event.event_type is EventType.EVIDENCE_RECORDED
    }
    assert event_evidence_ids == {row.evidence_id for row in evidence}


@pytest.mark.asyncio
async def test_cascade_without_worker_profile_fails_before_start(
    cascade_store: SqliteStore,
) -> None:
    controller = RunController(cascade_store, Settings(), {})
    run = await controller.create_run(
        NativeRunRequest(
            input="hello",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.CASCADE,
        )
    )

    with pytest.raises(ProviderError) as excinfo:
        await controller.execute(run.run_id)

    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED
    assert (await cascade_store.get_run(run.run_id)).status is RunStatus.PENDING
    assert await cascade_store.list_work_units(run.run_id) == ()
