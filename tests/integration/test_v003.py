"""v0.0.3 integration: Native PLANNED requests through persisted results."""

import asyncio
import json
import time
from collections.abc import Awaitable
from pathlib import Path

from fastapi.testclient import TestClient

from prp_runtime.app import create_app
from prp_runtime.domain.enums import (
    ExecutionStrategy,
    ModelRole,
    RunStatus,
    WorkUnitStatus,
)
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


def _profile(alias: str, role: ModelRole) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="fake",
        model=f"{alias}-model",
        role=role,
        base_url="https://models.invalid/v1",
        supports_structured_output=True,
        context_window_tokens=16_000,
        max_output_tokens=2_000,
        max_concurrency=2,
    )


def _settings(database: Path) -> Settings:
    return Settings(
        database_path=database,
        leader_profile=_profile("planner", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
    )


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, object]:
    for _ in range(200):
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] in {status.value for status in RunStatus if status.is_terminal}:
            return body
        time.sleep(0.005)
    raise AssertionError(f"run {run_id} did not reach a terminal state")


def _node(key: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "key": key,
        "name": key.title(),
        "instruction": f"produce {key}",
    }
    values.update(overrides)
    return values


class PlanAdapter:
    def __init__(self, nodes: list[dict[str, object]]) -> None:
        self._nodes = nodes
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "v003-planner"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=json.dumps({
                "summary": "v0.0.3 fixed plan",
                "final_node": self._nodes[-1]["key"],
                "nodes": self._nodes,
            }),
            usage=Usage(input_tokens=2, output_tokens=2),
            finish_reason=FinishReason.STOP,
        )


class WorkerAdapter:
    def __init__(
        self,
        responses: dict[str, str],
        *,
        concurrent: frozenset[str] = frozenset(),
    ) -> None:
        self._responses = responses
        self._concurrent = concurrent
        self._both_started = asyncio.Event()
        self.requests: list[ProviderRequest] = []
        self.active = 0
        self.max_active = 0

    @property
    def name(self) -> str:
        return "v003-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        instruction = request.input.split("\n", maxsplit=1)[0]
        if instruction in self._concurrent:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            active_instructions = {
                candidate.input.split("\n", maxsplit=1)[0]
                for candidate in self.requests
                if candidate.input.split("\n", maxsplit=1)[0] in self._concurrent
            }
            if active_instructions == set(self._concurrent):
                self._both_started.set()
            await self._both_started.wait()
            self.active -= 1
        return ProviderResponse(
            text=self._responses[instruction],
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


def test_native_planned_parallel_graph_persists_verified_result(tmp_path: Path) -> None:
    database = tmp_path / "v003-planned.db"
    result_schema = json.dumps(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    planner = PlanAdapter(
        [
            _node("left"),
            _node("right"),
            _node(
                "join",
                instruction="combine inputs",
                depends_on=["left", "right"],
                output={"kind": "JSON", "json_schema": result_schema},
            ),
        ]
    )
    worker = WorkerAdapter(
        {
            "produce left": "left fact",
            "produce right": "right fact",
            "combine inputs": '{"answer":"combined"}',
        },
        concurrent=frozenset({"produce left", "produce right"}),
    )
    app = create_app(
        _settings(database),
        adapters={"planner": planner, "worker": worker},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            json={
                "input": "combine two facts",
                "routing_policy": "MANUAL",
                "strategy": "PLANNED",
                "budget": {"max_attempts": 4, "max_concurrency": 2},
            },
        )
        body = _wait_for_terminal(client, response.json()["run_id"])

    assert response.status_code == 201
    assert body["status"] == RunStatus.SUCCEEDED.value
    assert body["strategy"] == ExecutionStrategy.PLANNED.value
    assert body["graph_version"] == 2
    assert json.loads(body["output_text"]) == {"answer": "combined"}
    assert body["usage"]["input_tokens"] == 5
    assert body["usage"]["output_tokens"] == 5
    assert body["usage"]["strong_model_tokens"] == 4
    assert len(planner.requests) == 1
    assert len(worker.requests) == 3
    assert worker.max_active == 2
    join_request = next(
        request for request in worker.requests if request.input.startswith("combine inputs")
    )
    assert "left fact" in join_request.input
    assert "right fact" in join_request.input

    async def inspect() -> tuple[int, int, int, int]:
        async with SqliteStore(database) as store:
            units = await store.list_work_units(body["run_id"], graph_version=2)
            attempts = await store.list_run_attempts(body["run_id"])
            evidence_count = 0
            for unit in units:
                evidence_count += len(await store.list_evidence(unit.work_unit_id))
            events = await store.list_events(body["run_id"])
            assert all(unit.status is WorkUnitStatus.SUCCEEDED for unit in units)
            assert events[-1].event_type is EventType.RUN_SUCCEEDED
            assert assert_sequence_chain(events) is None
            return len(units), len(attempts), evidence_count, len(events)

    unit_count, attempt_count, evidence_count, event_count = run_async(inspect())
    assert (unit_count, attempt_count, evidence_count) == (3, 4, 8)
    assert event_count > evidence_count


def test_native_invalid_plan_fails_with_zero_user_graph_facts(tmp_path: Path) -> None:
    database = tmp_path / "v003-invalid.db"
    planner = PlanAdapter(
        [
            _node("a", depends_on=["b"]),
            _node("b", depends_on=["a"]),
        ]
    )
    worker = WorkerAdapter({})
    app = create_app(
        _settings(database),
        adapters={"planner": planner, "worker": worker},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            json={
                "input": "reject a cycle",
                "routing_policy": "MANUAL",
                "strategy": "PLANNED",
            },
        )
        body = _wait_for_terminal(client, response.json()["run_id"])

    assert response.status_code == 201
    assert body["status"] == RunStatus.FAILED.value
    assert body["graph_version"] == 1
    assert worker.requests == []

    async def inspect() -> tuple[int, int, list[EventType]]:
        async with SqliteStore(database) as store:
            units = await store.list_work_units(body["run_id"])
            attempts = await store.list_run_attempts(body["run_id"])
            events = await store.list_events(body["run_id"])
            return len(units), len(attempts), [event.event_type for event in events]

    unit_count, attempt_count, event_types = run_async(inspect())
    assert unit_count == attempt_count == 1
    assert EventType.PLAN_REJECTED in event_types
    assert EventType.PLAN_COMMITTED not in event_types
    assert event_types[-1] is EventType.RUN_FAILED


def test_native_planned_budget_stops_before_second_attempt(tmp_path: Path) -> None:
    database = tmp_path / "v003-budget.db"
    planner = PlanAdapter(
        [
            _node("first"),
            _node("second", depends_on=["first"]),
        ]
    )
    worker = WorkerAdapter(
        {"produce first": "first", "produce second": "must not run"}
    )
    app = create_app(
        _settings(database),
        adapters={"planner": planner, "worker": worker},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/runs",
            json={
                "input": "bounded plan",
                "routing_policy": "MANUAL",
                "strategy": "PLANNED",
                "budget": {"max_attempts": 2},
            },
        )
        body = _wait_for_terminal(client, response.json()["run_id"])

    assert response.status_code == 201
    assert body["status"] == RunStatus.FAILED.value
    assert body["error"]["category"] == "BUDGET_EXCEEDED"
    assert len(worker.requests) == 1

    async def inspect() -> tuple[int, list[EventType]]:
        async with SqliteStore(database) as store:
            attempts = await store.list_run_attempts(body["run_id"])
            events = await store.list_events(body["run_id"])
            return len(attempts), [event.event_type for event in events]

    attempt_count, event_types = run_async(inspect())
    assert attempt_count == 2
    assert event_types.count(EventType.BUDGET_EXHAUSTED) == 1
    assert event_types[-1] is EventType.RUN_FAILED
