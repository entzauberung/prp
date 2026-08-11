"""v0.0.4 conformance invariants for AUTO routing and strategy dispatch."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from prp_runtime.app import create_app
from prp_runtime.control.controller import RunController
from prp_runtime.domain.enums import ExecutionStrategy, ModelRole, RoutingPolicy, RunStatus
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import Budget, NativeRunRequest, Run, Usage
from prp_runtime.providers.base import FinishReason, ModelProfile, ProviderRequest, ProviderResponse
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore


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
    )


class FixedAdapter:
    """Return one strict plan for the planner and one text artifact for workers."""

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "v004-conformance-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if request.alias == "planner":
            text = json.dumps(
                {
                    "summary": "one bounded task",
                    "nodes": [
                        {
                            "key": "answer",
                            "name": "Answer",
                            "instruction": "produce the answer",
                        }
                    ],
                }
            )
        else:
            text = "answer"
        return ProviderResponse(
            text=text,
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "v004-conformance.db") as opened:
        yield opened


def _settings(database_path: Path = Path("prp_runtime.db")) -> Settings:
    return Settings(
        database_path=database_path,
        leader_profile=_profile("planner", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
    )


@pytest.mark.parametrize(
    ("routing", "expected", "budget", "graph_version"),
    [
        (None, ExecutionStrategy.DIRECT, None, 1),
        ({"requires_cascade": True}, ExecutionStrategy.CASCADE, None, 1),
        ({"requires_plan": True}, ExecutionStrategy.PLANNED, None, 2),
        (
            {"requires_revision": True},
            ExecutionStrategy.PROGRESSIVE,
            {"max_plan_revisions": 1},
            2,
        ),
    ],
)
def test_public_auto_dispatches_each_strategy_and_records_selection(
    tmp_path: Path,
    routing: dict[str, object] | None,
    expected: ExecutionStrategy,
    budget: dict[str, int] | None,
    graph_version: int,
) -> None:
    adapter = FixedAdapter()
    app = create_app(
        _settings(tmp_path / "public-auto.db"),
        adapters={"planner": adapter, "worker": adapter},  # type: ignore[arg-type]
    )
    payload: dict[str, object] = {"input": "route this request"}
    if routing is not None:
        payload["routing"] = routing
    if budget is not None:
        payload["budget"] = budget

    with TestClient(app) as client:
        response = client.post("/v1/runs", json=payload)
        assert response.status_code == 201
        body = response.json()
        events_response = client.get(f"/v1/runs/{body['run_id']}/events")

    assert body["status"] == RunStatus.SUCCEEDED.value
    assert body["strategy"] == expected.value
    assert body["graph_version"] == graph_version
    event_records = [
        json.loads(line.removeprefix("data: "))
        for line in events_response.text.splitlines()
        if line.startswith("data: ")
    ]
    selected = next(
        record
        for record in event_records
        if record["event_type"] == EventType.STRATEGY_SELECTED.value
    )
    decisions = [
        record["payload"]["decision"]
        for record in event_records
        if record["event_type"] == EventType.CONTROLLER_DECISION.value
        and record["payload"]["decision"]["action"] == "SELECT_STRATEGY"
    ]
    assert len(decisions) == 1
    assert selected["payload"]["strategy"] == expected.value
    assert decisions[0]["to_strategy"] == expected.value
    assert decisions[0]["rationale"] == selected["payload"]["rationale"]
    assert [request.alias for request in adapter.requests].count("worker") == 1
    if expected in (ExecutionStrategy.PLANNED, ExecutionStrategy.PROGRESSIVE):
        assert [request.alias for request in adapter.requests].count("planner") == 1


@pytest.mark.parametrize("strategy", list(ExecutionStrategy))
@pytest.mark.asyncio
async def test_manual_keeps_each_pinned_strategy(
    store: SqliteStore, strategy: ExecutionStrategy
) -> None:
    adapter = FixedAdapter()
    controller = RunController(store, _settings(), {"planner": adapter, "worker": adapter})
    budget = Budget(max_plan_revisions=1) if strategy is ExecutionStrategy.PROGRESSIVE else Budget()
    run = await controller.create_run(
        NativeRunRequest(
            input="manual strategy",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=strategy,
            budget=budget,
        )
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.strategy is strategy


def test_public_auto_rejects_progressive_budget_without_direct_fallback(
    tmp_path: Path,
) -> None:
    adapter = FixedAdapter()
    app = create_app(
        _settings(tmp_path / "progressive-budget.db"),
        adapters={"planner": adapter, "worker": adapter},  # type: ignore[arg-type]
    )
    created_ids: list[str] = []

    with TestClient(app) as client:
        controller = app.state.controller
        create_run = controller.create_run

        async def capture_created_run(request: NativeRunRequest) -> Run:
            created = await create_run(request)
            created_ids.append(created.run_id)
            return created

        controller.create_run = capture_created_run
        response = client.post(
            "/v1/runs",
            json={"input": "needs revision", "routing": {"requires_revision": True}},
        )
        pending = client.get(f"/v1/runs/{created_ids[0]}")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value
    assert pending.json()["status"] == RunStatus.PENDING.value
    assert adapter.requests == []


def test_public_auto_capability_failure_is_not_silently_downgraded(
    tmp_path: Path,
) -> None:
    worker = FixedAdapter()
    settings = Settings(
        database_path=tmp_path / "missing-planner.db",
        worker_profile=_profile("worker", ModelRole.WORKER),
    )
    app = create_app(settings, adapters={"worker": worker})  # type: ignore[arg-type]
    created_ids: list[str] = []

    with TestClient(app) as client:
        controller = app.state.controller
        create_run = controller.create_run

        async def capture_created_run(request: NativeRunRequest) -> Run:
            created = await create_run(request)
            created_ids.append(created.run_id)
            return created

        controller.create_run = capture_created_run
        response = client.post(
            "/v1/runs",
            json={"input": "needs a plan", "routing": {"requires_plan": True}},
        )
        pending = client.get(f"/v1/runs/{created_ids[0]}")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == ErrorCode.PROVIDER_NOT_CONFIGURED.value
    assert pending.json()["status"] == RunStatus.PENDING.value
    assert worker.requests == []
