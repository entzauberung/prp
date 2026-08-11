"""v0.0.2 conformance invariants across Controller, Store, and event facts."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from prp_runtime.control.controller import RunController
from prp_runtime.domain.enums import (
    ExecutionStrategy,
    ModelRole,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import (
    ArtifactKind,
    ErrorCategory,
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


def _profile(alias: str, *, structured: bool = False) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="openai_compatible",
        model=f"{alias}-model",
        role=ModelRole.WORKER,
        base_url="https://models.internal/v1",
        context_window_tokens=32_000,
        max_output_tokens=4_000,
        supports_structured_output=structured,
    )


class FakeAdapter:
    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "v002-conformance-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if callable(outcome):
            outcome = await outcome(request)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, ProviderResponse)
        return outcome


def _response(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        usage=Usage(input_tokens=3, output_tokens=2, elapsed_ms=1),
        finish_reason=FinishReason.STOP,
    )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "v002-conformance.db") as opened:
        yield opened


def _request(*, output: dict[str, object] | None = None) -> NativeRunRequest:
    values: dict[str, object] = {
        "input": "produce the required result",
        "routing_policy": RoutingPolicy.MANUAL,
        "strategy": ExecutionStrategy.CASCADE,
    }
    if output is not None:
        values["output"] = output
    return NativeRunRequest.model_validate(values)


@pytest.mark.asyncio
async def test_inconclusive_chain_never_succeeds_or_overwrites_evidence(
    store: SqliteStore,
) -> None:
    weak_profile = _profile("weak", structured=True)
    strong_profile = _profile("strong", structured=True)
    weak = FakeAdapter(_response('{"value": 1}'))
    strong = FakeAdapter(_response('{"value": 2}'))
    controller = RunController(
        store,
        Settings(
            worker_profile=weak_profile,
            cascade_profiles=(strong_profile,),
        ),
        {"weak": weak, "strong": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(
        _request(
            output={
                "kind": ArtifactKind.JSON,
                "json_schema": '{"type":"object","oneOf":[{"type":"object"}]}',
            }
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.VERIFICATION_FAILED
    units = await store.list_work_units(run.run_id)
    assert units[0].status is WorkUnitStatus.FAILED
    artifacts = await store.list_artifacts(units[0].work_unit_id)
    evidence = await store.list_evidence(units[0].work_unit_id)
    assert len(artifacts) == 2
    assert len(evidence) == 8
    for artifact in artifacts:
        artifact_evidence = [row for row in evidence if row.artifact_id == artifact.artifact_id]
        assert len(artifact_evidence) == 4
        assert any(
            row.result is VerificationResult.INCONCLUSIVE for row in artifact_evidence
        )

    events = await store.list_events(run.run_id)
    assert assert_sequence_chain(events) is None
    evidence_event_ids = {
        event.payload["evidence_id"]
        for event in events
        if event.event_type is EventType.EVIDENCE_RECORDED
    }
    assert evidence_event_ids == {row.evidence_id for row in evidence}
    assert sum(
        event.event_type is EventType.STRATEGY_ESCALATED for event in events
    ) == 1
    assert events[-1].event_type is EventType.RUN_FAILED


@pytest.mark.asyncio
async def test_cancelled_retryable_failure_never_starts_the_next_attempt(
    store: SqliteStore,
) -> None:
    weak_profile = _profile("weak")
    strong_profile = _profile("strong")
    controller: RunController | None = None
    run_id = ""

    async def cancel_then_fail(request: ProviderRequest) -> ProviderResponse:
        assert controller is not None
        await controller.cancel(run_id)
        raise ProviderError("temporary outage", code=ErrorCode.PROVIDER_UNAVAILABLE)

    weak = FakeAdapter(cancel_then_fail)
    strong = FakeAdapter(_response("must not run"))
    controller = RunController(
        store,
        Settings(
            worker_profile=weak_profile,
            cascade_profiles=(strong_profile,),
        ),
        {"weak": weak, "strong": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(_request())
    run_id = run.run_id

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.CANCELLED
    assert len(await store.list_run_attempts(run.run_id)) == 1
    assert strong.requests == []
    events = await store.list_events(run.run_id)
    assert EventType.RUN_CANCELLING in {event.event_type for event in events}
    assert EventType.STRATEGY_ESCALATED not in {
        event.event_type for event in events
    }


@pytest.mark.asyncio
async def test_terminal_cascade_is_idempotent_and_usage_is_measured_once(
    store: SqliteStore,
) -> None:
    weak_profile = _profile("weak")
    strong_profile = _profile("strong")
    weak = FakeAdapter(
        ProviderError("temporary outage", code=ErrorCode.PROVIDER_TIMEOUT)
    )
    strong = FakeAdapter(_response("answer"))
    controller = RunController(
        store,
        Settings(
            worker_profile=weak_profile,
            cascade_profiles=(strong_profile,),
        ),
        {"weak": weak, "strong": strong},  # type: ignore[arg-type]
    )
    run = await controller.create_run(_request())

    first = await controller.execute(run.run_id)
    event_count = len(await store.list_events(run.run_id))
    second = await controller.execute(run.run_id)

    assert first.status is second.status is RunStatus.SUCCEEDED
    assert len(weak.requests) == len(strong.requests) == 1
    assert len(await store.list_events(run.run_id)) == event_count
    attempts = await store.list_run_attempts(run.run_id)
    measured = sum(
        attempt.usage.total_tokens
        for attempt in attempts
        if attempt.usage is not None
    )
    assert measured == 5
    assert (await store.get_run_usage(run.run_id)).total_tokens == measured
