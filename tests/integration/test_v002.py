"""v0.0.2 integration: Native API through control, verification, and storage."""

import asyncio
import json
from collections.abc import Awaitable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import prp_runtime.app as app_module
from prp_runtime.app import create_app
from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore


def run_async[T](coroutine: Awaitable[T]) -> T:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def _worker(alias: str, *, structured: bool = False) -> ModelProfile:
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
        return "v002-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, ProviderResponse)
        return outcome


def _response(text: str, *, tokens: int = 5) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        usage=Usage(input_tokens=tokens - 2, output_tokens=2, elapsed_ms=3),
        finish_reason=FinishReason.STOP,
    )


def _event_records(client: TestClient, run_id: str) -> list[dict[str, object]]:
    response = client.get(f"/v1/runs/{run_id}/events")
    assert response.status_code == 200
    records: list[dict[str, object]] = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            records.append(json.loads(line.removeprefix("data: ")))
    return records


def test_native_cascade_verification_failure_then_success_has_consistent_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "v002-cascade.db"
    weak_profile = _worker("weak", structured=True)
    strong_profile = _worker("strong", structured=True)
    weak = FakeAdapter(_response('{"wrong": true}'))
    strong = FakeAdapter(_response('{"ok": true}'))
    settings = Settings(
        database_path=database,
        worker_profile=weak_profile,
        cascade_profiles=(strong_profile,),
    )
    adapters = {"weak": weak, "strong": strong}
    monkeypatch.setattr(
        app_module,
        "OpenAICompatibleProvider",
        lambda profile: adapters[profile.alias],
    )
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            json={
                "input": "emit json",
                "routing_policy": "MANUAL",
                "strategy": "CASCADE",
                "output": {
                    "kind": "JSON",
                    "json_schema": '{"type":"object","required":["ok"]}',
                },
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == RunStatus.SUCCEEDED.value
        assert body["strategy"] == ExecutionStrategy.CASCADE.value
        assert json.loads(body["output_text"]) == {"ok": True}
        assert body["usage"]["input_tokens"] == 6
        assert body["usage"]["output_tokens"] == 4
        event_records = _event_records(client, body["run_id"])

    async def inspect() -> tuple[int, int, int, int, int]:
        async with SqliteStore(database) as store:
            units = await store.list_work_units(body["run_id"])
            attempts = await store.list_attempts(units[0].work_unit_id)
            artifacts = await store.list_artifacts(units[0].work_unit_id)
            evidence = await store.list_evidence(units[0].work_unit_id)
            events = await store.list_events(body["run_id"])
            assert units[0].status is WorkUnitStatus.SUCCEEDED
            assert [attempt.status for attempt in attempts] == [
                AttemptStatus.SUCCEEDED,
                AttemptStatus.SUCCEEDED,
            ]
            assert assert_sequence_chain(events) is None
            return (
                len(units),
                len(attempts),
                len(artifacts),
                len(evidence),
                (await store.get_run_usage(body["run_id"])).total_tokens,
            )

    assert run_async(inspect()) == (1, 2, 2, 8, 10)
    event_types = [record["event_type"] for record in event_records]
    assert EventType.STRATEGY_ESCALATED.value in event_types
    assert event_types[-1] == EventType.RUN_SUCCEEDED.value
    assert len(weak.requests) == len(strong.requests) == 1


def test_native_cascade_attempt_budget_blocks_the_second_adapter(tmp_path: Path) -> None:
    weak_profile = _worker("weak")
    strong_profile = _worker("strong")
    weak = FakeAdapter(
        ProviderError("temporary outage", code=ErrorCode.PROVIDER_UNAVAILABLE)
    )
    strong = FakeAdapter(_response("must not run"))
    settings = Settings(
        database_path=tmp_path / "v002-budget.db",
        worker_profile=weak_profile,
        cascade_profiles=(strong_profile,),
    )
    app = create_app(settings, adapters={"weak": weak, "strong": strong})  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            json={
                "input": "hello",
                "routing_policy": "MANUAL",
                "strategy": "CASCADE",
                "budget": {"max_attempts": 1},
            },
        )
        body = response.json()
        event_types = [
            record["event_type"] for record in _event_records(client, body["run_id"])
        ]

    assert response.status_code == 201
    assert body["status"] == RunStatus.FAILED.value
    assert body["error"]["category"] == "BUDGET_EXCEEDED"
    assert len(weak.requests) == 1
    assert strong.requests == []
    assert EventType.BUDGET_EXHAUSTED.value in event_types
    assert EventType.STRATEGY_ESCALATED.value not in event_types


def test_native_cascade_nonretryable_provider_error_stops_immediately(
    tmp_path: Path,
) -> None:
    weak_profile = _worker("weak")
    strong_profile = _worker("strong")
    weak = FakeAdapter(
        ProviderError("credentials rejected", code=ErrorCode.PROVIDER_AUTH_FAILED)
    )
    strong = FakeAdapter(_response("must not run"))
    settings = Settings(
        database_path=tmp_path / "v002-provider.db",
        worker_profile=weak_profile,
        cascade_profiles=(strong_profile,),
    )
    app = create_app(settings, adapters={"weak": weak, "strong": strong})  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            json={
                "input": "hello",
                "routing_policy": "MANUAL",
                "strategy": "CASCADE",
            },
        )
        body = response.json()

    assert response.status_code == 201
    assert body["status"] == RunStatus.FAILED.value
    assert body["error"]["category"] == "AUTH"
    assert len(weak.requests) == 1
    assert strong.requests == []


def test_native_direct_token_budget_still_uses_one_attempt(tmp_path: Path) -> None:
    worker_profile = _worker("worker")
    worker = FakeAdapter(_response("answer", tokens=20))
    settings = Settings(
        database_path=tmp_path / "v002-direct.db",
        worker_profile=worker_profile,
    )
    app = create_app(settings, adapters={"worker": worker})  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            json={"input": "hello", "budget": {"max_total_tokens": 10}},
        )
        body = response.json()

    assert response.status_code == 201
    assert body["strategy"] == ExecutionStrategy.DIRECT.value
    assert body["status"] == RunStatus.FAILED.value
    assert body["error"]["category"] == "BUDGET_EXCEEDED"
    assert len(worker.requests) == 1
