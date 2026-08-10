"""v0.0.1 integration: Native DIRECT API over ASGI with a fake provider.

No port is opened, no real network call is made, and every database is temporary.
"""

import asyncio
import json
from collections.abc import Awaitable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prp_runtime.app import create_app
from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    RoutingPolicy,
    RunStatus,
)
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import (
    Attempt,
    NativeRunRequest,
    Run,
    Usage,
    WorkUnit,
)
from prp_runtime.domain.values import (
    new_attempt_id,
    new_run_id,
    new_work_unit_id,
    utc_now,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="openai_compatible",
    model="weak-model",
    role=ModelRole.WORKER,
    base_url="https://models.internal/v1",
    context_window_tokens=32_000,
    max_output_tokens=4_000,
)

ANSWER = "the assembled answer"


def run_async[T](coroutine: Awaitable[T]) -> T:
    """Run one coroutine in its own loop, outside the TestClient loop."""
    return asyncio.run(coroutine)  # type: ignore[arg-type]


class FakeAdapter:
    """Returns queued outcomes and records the requests it received."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes: list[object] = list(outcomes)
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        outcome = self._outcomes.pop(0) if self._outcomes else _ok(ANSWER)
        if callable(outcome):
            outcome = await outcome(request)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, ProviderResponse)
        return outcome


def _ok(text: str) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        usage=Usage(input_tokens=11, output_tokens=7, elapsed_ms=9),
        finish_reason=FinishReason.STOP,
    )


def build_app(
    tmp_path: Path,
    adapter: FakeAdapter | None = None,
    *,
    settings: Settings | None = None,
) -> FastAPI:
    resolved = settings or Settings(
        database_path=tmp_path / "native.db", worker_profile=WORKER_PROFILE
    )
    adapters = {"worker": adapter} if adapter is not None else {}
    return create_app(resolved, adapters=adapters)  # type: ignore[arg-type]


@pytest.fixture
def adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def client(tmp_path: Path, adapter: FakeAdapter) -> Iterator[TestClient]:
    with TestClient(build_app(tmp_path, adapter)) as opened:
        yield opened


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse an SSE body into its event records."""
    records: list[dict[str, Any]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        record: dict[str, Any] = {}
        for line in block.splitlines():
            field, _, value = line.partition(": ")
            if field == "data":
                record["data"] = json.loads(value)
            else:
                record[field] = value
        records.append(record)
    return records


# --- create and read ------------------------------------------------------------


def test_create_run_executes_and_returns_the_answer(
    client: TestClient, adapter: FakeAdapter
) -> None:
    response = client.post(
        "/v1/runs", json={"input": "summarise the report", "instructions": "be terse"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == RunStatus.SUCCEEDED.value
    assert body["strategy"] == ExecutionStrategy.DIRECT.value
    assert body["output_text"] == ANSWER
    assert body["output_kind"] == "TEXT"
    assert body["usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "strong_model_tokens": 0,
        "elapsed_ms": 9,
    }
    assert body["error"] is None
    assert body["run_id"].startswith("run_")
    assert body["completed_at"] is not None
    assert len(adapter.requests) == 1
    assert adapter.requests[0].input == "summarise the report"


def test_get_run_matches_the_creation_response(client: TestClient) -> None:
    created = client.post("/v1/runs", json={"input": "hello"}).json()
    fetched = client.get(f"/v1/runs/{created['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created


def test_unknown_run_is_a_structured_404(client: TestClient) -> None:
    response = client.get(f"/v1/runs/{new_run_id()}")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == ErrorCode.RUN_NOT_FOUND.value
    assert error["family"] == "STATE"
    assert error["retryable"] is False
    assert "Traceback" not in response.text


def test_manual_direct_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/v1/runs",
        json={
            "input": "hello",
            "routing_policy": RoutingPolicy.MANUAL.value,
            "strategy": ExecutionStrategy.DIRECT.value,
        },
    )
    assert response.status_code == 201
    assert response.json()["strategy"] == ExecutionStrategy.DIRECT.value


def test_structured_output_round_trips(tmp_path: Path) -> None:
    adapter = FakeAdapter(_ok('{"ok": true}'))
    settings = Settings(
        database_path=tmp_path / "native.db",
        worker_profile=WORKER_PROFILE.model_copy(update={"supports_structured_output": True}),
    )
    with TestClient(build_app(tmp_path, adapter, settings=settings)) as client:
        response = client.post(
            "/v1/runs",
            json={
                "input": "emit json",
                "output": {"kind": "JSON", "json_schema": '{"type":"object"}'},
            },
        )
    assert response.status_code == 201
    body = response.json()
    assert body["output_kind"] == "JSON"
    assert json.loads(body["output_text"]) == {"ok": True}


# --- event replay ---------------------------------------------------------------


def test_events_replay_the_whole_ledger(client: TestClient) -> None:
    run_id = client.post("/v1/runs", json={"input": "hello"}).json()["run_id"]

    response = client.get(f"/v1/runs/{run_id}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    records = parse_sse(response.text)
    assert [record["event"] for record in records] == [
        EventType.RUN_CREATED.value,
        EventType.STRATEGY_SELECTED.value,
        EventType.RUN_STARTED.value,
        EventType.WORK_UNIT_CREATED.value,
        EventType.WORK_UNIT_READY.value,
        EventType.WORK_UNIT_STARTED.value,
        EventType.ATTEMPT_STARTED.value,
        EventType.ATTEMPT_SUCCEEDED.value,
        EventType.ARTIFACT_PRODUCED.value,
        EventType.USAGE_UPDATED.value,
        EventType.WORK_UNIT_SUCCEEDED.value,
        EventType.RUN_SUCCEEDED.value,
    ]
    assert [int(record["id"]) for record in records] == list(range(1, len(records) + 1))
    assert records[0]["data"]["run_id"] == run_id
    assert records[0]["data"]["sequence"] == 1


def test_last_event_id_resumes_the_stream(client: TestClient) -> None:
    run_id = client.post("/v1/runs", json={"input": "hello"}).json()["run_id"]
    everything = parse_sse(client.get(f"/v1/runs/{run_id}/events").text)
    cursor = int(everything[5]["id"])

    resumed = client.get(
        f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": str(cursor)}
    )

    records = parse_sse(resumed.text)
    assert [int(record["id"]) for record in records] == [
        int(record["id"]) for record in everything[6:]
    ]


def test_after_cursor_query_matches_the_header(client: TestClient) -> None:
    run_id = client.post("/v1/runs", json={"input": "hello"}).json()["run_id"]
    by_query = parse_sse(client.get(f"/v1/runs/{run_id}/events?after=3").text)
    by_header = parse_sse(
        client.get(f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": "3"}).text
    )
    assert [record["id"] for record in by_query] == [record["id"] for record in by_header]
    assert min(int(record["id"]) for record in by_query) == 4


def test_a_cursor_at_the_end_yields_an_empty_stream(client: TestClient) -> None:
    run_id = client.post("/v1/runs", json={"input": "hello"}).json()["run_id"]
    records = parse_sse(client.get(f"/v1/runs/{run_id}/events").text)
    last = max(int(record["id"]) for record in records)
    response = client.get(f"/v1/runs/{run_id}/events?after={last + 100}")
    assert response.status_code == 200
    assert response.text.strip() == ""


def test_a_malformed_last_event_id_is_rejected(client: TestClient) -> None:
    run_id = client.post("/v1/runs", json={"input": "hello"}).json()["run_id"]
    response = client.get(
        f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": "not-a-number"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value


def test_events_of_an_unknown_run_are_a_404(client: TestClient) -> None:
    response = client.get(f"/v1/runs/{new_run_id()}/events")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.RUN_NOT_FOUND.value


# --- cancellation ---------------------------------------------------------------


def test_cancelling_a_finished_run_changes_nothing(client: TestClient) -> None:
    created = client.post("/v1/runs", json={"input": "hello"}).json()
    response = client.post(f"/v1/runs/{created['run_id']}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == RunStatus.SUCCEEDED.value
    assert response.json()["completed_at"] == created["completed_at"]


def test_cancelling_an_unknown_run_is_a_404(client: TestClient) -> None:
    response = client.post(f"/v1/runs/{new_run_id()}/cancel")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.RUN_NOT_FOUND.value


def test_cancel_prevents_a_second_attempt(tmp_path: Path) -> None:
    adapter = FakeAdapter(
        ProviderError("upstream is down", code=ErrorCode.PROVIDER_UNAVAILABLE)
    )
    with TestClient(build_app(tmp_path, adapter)) as client:
        created = client.post("/v1/runs", json={"input": "hello"}).json()
        client.post(f"/v1/runs/{created['run_id']}/cancel")
        after = client.get(f"/v1/runs/{created['run_id']}").json()
    assert created["status"] == RunStatus.FAILED.value
    assert after["status"] == RunStatus.FAILED.value
    assert len(adapter.requests) == 1


# --- failures -------------------------------------------------------------------


def test_a_provider_failure_is_a_failed_run_not_an_http_error(tmp_path: Path) -> None:
    adapter = FakeAdapter(
        ProviderError("upstream timed out", code=ErrorCode.PROVIDER_TIMEOUT)
    )
    with TestClient(build_app(tmp_path, adapter)) as client:
        response = client.post("/v1/runs", json={"input": "hello"})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == RunStatus.FAILED.value
    assert body["error"]["category"] == "TIMEOUT"
    assert body["output_text"] is None


def test_a_missing_provider_configuration_is_a_503(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "native.db")
    with TestClient(build_app(tmp_path, settings=settings)) as client:
        response = client.post("/v1/runs", json={"input": "hello"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == ErrorCode.PROVIDER_NOT_CONFIGURED.value


@pytest.mark.parametrize(
    "strategy",
    [
        ExecutionStrategy.CASCADE.value,
        ExecutionStrategy.PLANNED.value,
        ExecutionStrategy.PROGRESSIVE.value,
    ],
)
def test_unimplemented_strategies_are_refused(client: TestClient, strategy: str) -> None:
    response = client.post(
        "/v1/runs",
        json={
            "input": "hello",
            "routing_policy": RoutingPolicy.MANUAL.value,
            "strategy": strategy,
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == ErrorCode.INVALID_REQUEST.value
    assert error["field"] == "strategy"


@pytest.mark.parametrize(
    "payload",
    [
        {"input": ""},
        {"input": "   "},
        {},
        {"input": "hello", "temperature": 0.7},
        {"input": "hello", "routing_policy": "AUTO", "strategy": "DIRECT"},
        {"input": "hello", "routing_policy": "MANUAL"},
        {"input": "hello", "budget": {"max_total_tokens": -1}},
    ],
)
def test_invalid_requests_are_rejected_with_a_stable_code(
    client: TestClient, payload: dict[str, Any]
) -> None:
    response = client.post("/v1/runs", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value
    assert "Traceback" not in response.text


def test_input_over_the_character_limit_is_a_413(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "native.db",
        worker_profile=WORKER_PROFILE,
        max_input_chars=10,
    )
    with TestClient(build_app(tmp_path, FakeAdapter(), settings=settings)) as client:
        response = client.post("/v1/runs", json={"input": "x" * 11})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == ErrorCode.INPUT_TOO_LARGE.value


def test_an_oversized_body_is_refused_before_parsing(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "native.db",
        worker_profile=WORKER_PROFILE,
        max_request_bytes=64,
    )
    with TestClient(build_app(tmp_path, FakeAdapter(), settings=settings)) as client:
        response = client.post("/v1/runs", json={"input": "x" * 500})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == ErrorCode.INPUT_TOO_LARGE.value


# --- wiring and persistence -----------------------------------------------------


def test_health_still_reports_liveness(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_a_run_survives_the_client_and_the_process(tmp_path: Path) -> None:
    database = tmp_path / "native.db"
    settings = Settings(database_path=database, worker_profile=WORKER_PROFILE)
    with TestClient(build_app(tmp_path, FakeAdapter(), settings=settings)) as client:
        run_id = client.post("/v1/runs", json={"input": "hello"}).json()["run_id"]

    async def inspect() -> tuple[Run, int]:
        async with SqliteStore(database) as store:
            return await store.get_run(run_id), len(await store.list_events(run_id))

    run, event_count = run_async(inspect())
    assert run.status is RunStatus.SUCCEEDED
    assert event_count == 12


def test_startup_recovers_an_interrupted_attempt(tmp_path: Path) -> None:
    database = tmp_path / "native.db"
    started = utc_now()
    run = Run(
        run_id=new_run_id(),
        request=NativeRunRequest(input="hello"),
        status=RunStatus.RUNNING,
        created_at=started,
        started_at=started,
    )
    unit = WorkUnit(
        work_unit_id=new_work_unit_id(),
        run_id=run.run_id,
        name="direct",
        instruction="hello",
    )
    attempt = Attempt(
        attempt_id=new_attempt_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        role=ModelRole.WORKER,
        model=WORKER_PROFILE.model_ref,
        status=AttemptStatus.RUNNING,
        started_at=run.created_at,
    )

    async def seed() -> None:
        async with SqliteStore(database) as store:
            await store.create_run(run)
            await store.create_work_unit(unit)
            await store.create_attempt(attempt)

    run_async(seed())

    settings = Settings(database_path=database, worker_profile=WORKER_PROFILE)
    app = build_app(tmp_path, FakeAdapter(), settings=settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        report = app.state.recovery
        assert report.changed is True
        assert report.interrupted_attempt_ids == (attempt.attempt_id,)

    async def check() -> Attempt:
        async with SqliteStore(database) as store:
            return await store.get_attempt(attempt.attempt_id)

    recovered = run_async(check())
    assert recovered.status is AttemptStatus.INTERRUPTED
    assert recovered.completed_at is not None


def test_error_bodies_never_carry_internals(client: TestClient) -> None:
    response = client.get(f"/v1/runs/{new_run_id()}")
    error = response.json()["error"]
    assert set(error) == {"code", "family", "message", "retryable", "field"}
    for forbidden in ("Traceback", "/home/", "site-packages", "sqlite3", "sk-"):
        assert forbidden not in response.text


def test_openapi_documents_only_the_native_surface(client: TestClient) -> None:
    paths: Sequence[str] = tuple(client.get("/openapi.json").json()["paths"])
    assert set(paths) == {
        "/health",
        "/v1/runs",
        "/v1/runs/{run_id}",
        "/v1/runs/{run_id}/cancel",
        "/v1/runs/{run_id}/events",
    }
