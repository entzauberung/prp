"""Controller-owned atomic commit of compiled plans."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from prp_runtime.control.controller import RunController
from prp_runtime.domain.enums import (
    ExecutionStrategy,
    ModelRole,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import (
    Budget,
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
    Usage,
    WorkUnit,
)
from prp_runtime.domain.values import ResourceClaim, utc_now
from prp_runtime.planning.compiler import CompiledPlan, compile_plan
from prp_runtime.planning.frontier import compute_frontier
from prp_runtime.planning.models import PlanProposal, PlanRejection
from prp_runtime.planning.planner import new_planning_work_unit
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.runtime.assembler import assemble_run_result
from prp_runtime.runtime.scheduler import Scheduler, WaveOutcome, WaveStatus
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "planned-commit.db") as opened:
        yield opened


def proposal(*nodes: dict[str, object]) -> PlanProposal:
    return PlanProposal(
        summary="compile and commit",
        nodes=nodes,
    )


def node(key: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "key": key,
        "name": key.title(),
        "instruction": f"produce {key}",
    }
    values.update(overrides)
    return values


def profile(alias: str, role: ModelRole) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="fake",
        model=f"{alias}-model",
        role=role,
        base_url="https://models.invalid/v1",
        supports_structured_output=True,
        context_window_tokens=8_000,
        max_output_tokens=1_000,
        max_concurrency=2,
    )


class StaticPlannerAdapter:
    def __init__(self, plan: dict[str, object]) -> None:
        self._plan = plan
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "static-planner"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=json.dumps(self._plan),
            usage=Usage(input_tokens=5, output_tokens=5),
            finish_reason=FinishReason.STOP,
        )


class PlannedWorkerAdapter:
    def __init__(
        self,
        responses: dict[str, str],
        *,
        concurrent_instructions: frozenset[str] = frozenset(),
    ) -> None:
        self._responses = responses
        self._concurrent_instructions = concurrent_instructions
        self._all_concurrent_started = asyncio.Event()
        self.requests: list[ProviderRequest] = []
        self.active = 0
        self.max_active = 0

    @property
    def name(self) -> str:
        return "planned-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        instruction = request.input.split("\n", maxsplit=1)[0]
        if instruction in self._concurrent_instructions:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            started = {
                candidate.input.split("\n", maxsplit=1)[0]
                for candidate in self.requests
                if candidate.input.split("\n", maxsplit=1)[0]
                in self._concurrent_instructions
            }
            if started == set(self._concurrent_instructions):
                self._all_concurrent_started.set()
            await self._all_concurrent_started.wait()
            self.active -= 1
        return ProviderResponse(
            text=self._responses[instruction],
            usage=Usage(input_tokens=1, output_tokens=2),
            finish_reason=FinishReason.STOP,
        )


def planned_settings() -> Settings:
    return Settings(
        leader_profile=profile("planner", ModelRole.PLANNER),
        worker_profile=profile("worker", ModelRole.WORKER),
    )


async def create_controller_run(
    store: SqliteStore,
) -> tuple[RunController, str]:
    controller = RunController(store, Settings(), {})
    run = await controller.create_run(NativeRunRequest(input="plan this"))
    return controller, run.run_id


@pytest.mark.asyncio
async def test_controller_atomically_commits_graph_version_and_events(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    compiled = compile_plan(
        proposal(
            node("draft"),
            node(
                "review",
                depends_on=("draft",),
                acceptance_criteria="the output is accepted",
                resource_claims=(
                    ResourceClaim(resource="result", access=ResourceAccess.WRITE),
                ),
            ),
        ),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)

    committed = await controller.commit_plan(
        run_id,
        compiled,
        target_graph_version=2,
    )

    assert isinstance(committed, tuple)
    assert (await store.get_run(run_id)).graph_version == 2
    units = await store.list_work_units(run_id, graph_version=2)
    assert len(units) == 2
    by_name = {unit.name: unit for unit in units}
    assert by_name["Review"].depends_on == (by_name["Draft"].work_unit_id,)
    assert by_name["Review"].resource_claims == (
        ResourceClaim(resource="result", access=ResourceAccess.WRITE),
    )
    events = await store.list_events(run_id)
    assert assert_sequence_chain(events) is None
    assert [event.event_type for event in events] == [
        EventType.RUN_CREATED,
        EventType.PLAN_PROPOSED,
        EventType.WORK_UNIT_CREATED,
        EventType.WORK_UNIT_CREATED,
        EventType.CONTROLLER_DECISION,
        EventType.PLAN_COMMITTED,
    ]
    committed_event = events[-1]
    assert committed_event.payload["graph_version"] == 2
    assert set(committed_event.payload["work_unit_ids"]) == {
        unit.work_unit_id for unit in units
    }


@pytest.mark.asyncio
async def test_compile_rejection_records_events_and_writes_no_units(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    rejected = compile_plan(
        proposal(
            node("a", depends_on=("b",)),
            node("b", depends_on=("a",)),
        ),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(rejected, PlanRejection)

    result = await controller.commit_plan(
        run_id,
        rejected,
        target_graph_version=2,
    )

    assert result is rejected
    assert await store.list_work_units(run_id) == ()
    assert (await store.get_run(run_id)).graph_version == 1
    assert [event.event_type for event in await store.list_events(run_id)] == [
        EventType.RUN_CREATED,
        EventType.PLAN_PROPOSED,
        EventType.CONTROLLER_DECISION,
        EventType.PLAN_REJECTED,
    ]


@pytest.mark.asyncio
async def test_stale_or_mismatched_graph_version_is_rejected(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    compiled = compile_plan(
        proposal(node("only")),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)

    result = await controller.commit_plan(
        run_id,
        compiled,
        target_graph_version=1,
    )

    assert isinstance(result, PlanRejection)
    assert await store.list_work_units(run_id) == ()
    assert (await store.get_run(run_id)).graph_version == 1


@pytest.mark.asyncio
async def test_failed_store_write_rolls_back_graph_run_version_and_commit_events(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    compiled = compile_plan(
        proposal(
            node("first"),
            node("second", depends_on=("first",)),
        ),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)
    duplicate_id = compiled.nodes[-1].work_unit_id
    await store.create_work_unit(
        WorkUnit(
            work_unit_id=duplicate_id,
            run_id=run_id,
            graph_version=1,
            name="existing",
            instruction="already persisted",
            created_at=utc_now(),
        )
    )

    result = await controller.commit_plan(
        run_id,
        compiled,
        target_graph_version=2,
    )

    assert isinstance(result, PlanRejection)
    assert await store.list_work_units(run_id, graph_version=2) == ()
    assert len(await store.list_work_units(run_id, graph_version=1)) == 1
    assert (await store.get_run(run_id)).graph_version == 1
    event_types = [event.event_type for event in await store.list_events(run_id)]
    assert event_types.count(EventType.PLAN_PROPOSED) == 1
    assert EventType.PLAN_COMMITTED not in event_types
    assert event_types[-1] is EventType.PLAN_REJECTED


@pytest.mark.asyncio
async def test_compiler_commit_store_and_frontier_close_on_one_graph_version(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    compiled = compile_plan(
        proposal(
            node("join", depends_on=("left", "right")),
            node("right", depends_on=("root",)),
            node("root"),
            node("left", depends_on=("root",)),
        ),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)
    committed = await controller.commit_plan(
        run_id,
        compiled,
        target_graph_version=2,
    )
    assert isinstance(committed, tuple)

    persisted = await store.list_work_units(run_id, graph_version=2)
    frontier = compute_frontier(persisted, graph_version=2)
    by_name = {unit.name: unit for unit in persisted}
    assert frontier.ready == (by_name["Root"].work_unit_id,)
    assert set(frontier.waiting) == {
        by_name["Left"].work_unit_id,
        by_name["Right"].work_unit_id,
        by_name["Join"].work_unit_id,
    }
    assert frontier.blocked == ()
    assert frontier.complete == ()
    assert (await store.get_run(run_id)).graph_version == frontier.graph_version == 2


@pytest.mark.asyncio
async def test_internal_planning_unit_is_outside_user_frontier_and_result(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    planning_unit = new_planning_work_unit(run_id)
    await store.create_work_unit(planning_unit)
    compiled = compile_plan(
        proposal(node("answer")),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)

    await controller.commit_plan(run_id, compiled, target_graph_version=2)

    current_units = await store.list_work_units(run_id, graph_version=2)
    frontier = compute_frontier(current_units, graph_version=2)
    result = await assemble_run_result(store, run_id)
    assert planning_unit not in current_units
    assert planning_unit.work_unit_id not in frontier.ready
    assert planning_unit.work_unit_id not in frontier.waiting
    assert result.graph_version == 2
    assert result.artifact_id is None


@pytest.mark.asyncio
async def test_controller_dispatches_root_then_chain_successfully(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    compiled = compile_plan(
        proposal(node("root"), node("child", depends_on=("root",))),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)
    await controller.commit_plan(run_id, compiled, target_graph_version=2)

    async def execute(unit: WorkUnit) -> WaveOutcome:
        return WaveOutcome.success()

    first = await controller.dispatch_planned_wave(
        run_id,
        scheduler=Scheduler(),
        execute=execute,
    )
    assert first.status is WaveStatus.DISPATCHED
    assert len(first.succeeded) == 1
    persisted = await store.list_work_units(run_id, graph_version=2)
    assert sum(unit.status is WorkUnitStatus.SUCCEEDED for unit in persisted) == 1
    assert sum(unit.status is WorkUnitStatus.PENDING for unit in persisted) == 1

    second = await controller.dispatch_planned_wave(
        run_id,
        scheduler=Scheduler(),
        execute=execute,
    )
    assert second.status is WaveStatus.DISPATCHED
    assert len(second.succeeded) == 1
    assert all(
        unit.status is WorkUnitStatus.SUCCEEDED
        for unit in await store.list_work_units(run_id, graph_version=2)
    )


@pytest.mark.asyncio
async def test_controller_uses_budget_max_concurrency_for_the_wave(
    store: SqliteStore,
) -> None:
    controller = RunController(store, Settings(), {})
    run = await controller.create_run(
        NativeRunRequest(
            input="plan this",
            budget=Budget(max_concurrency=2),
        )
    )
    compiled = compile_plan(
        proposal(node("a"), node("b"), node("c")),
        run_id=run.run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)
    await controller.commit_plan(run.run_id, compiled, target_graph_version=2)
    both_started = asyncio.Event()
    active = 0
    max_active = 0

    async def execute(unit: WorkUnit) -> WaveOutcome:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            both_started.set()
        await both_started.wait()
        active -= 1
        return WaveOutcome.success()

    result = await controller.dispatch_planned_wave(
        run.run_id,
        scheduler=Scheduler(),
        execute=execute,
    )
    assert len(result.started) == 2
    assert len(result.deferred) == 1
    assert max_active == 2


@pytest.mark.asyncio
async def test_failed_dependency_blocks_downstream_but_independent_chain_finishes(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    compiled = compile_plan(
        proposal(
            node("root"),
            node("blocked_child", depends_on=("root",)),
            node("independent"),
            node("independent_tail", depends_on=("independent",)),
        ),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)
    await controller.commit_plan(run_id, compiled, target_graph_version=2)
    executed: list[str] = []
    root_error = ErrorInfo(
        category=ErrorCategory.PROVIDER_ERROR,
        message="root failed",
    )

    async def execute(unit: WorkUnit) -> WaveOutcome:
        executed.append(unit.name)
        if unit.name == "Root":
            return WaveOutcome.failure(root_error)
        return WaveOutcome.success()

    first = await controller.dispatch_planned_wave(
        run_id,
        scheduler=Scheduler(),
        execute=execute,
        max_concurrency=2,
    )

    assert first.status is WaveStatus.DISPATCHED
    assert set(executed) == {"Root", "Independent"}
    assert (await store.get_run(run_id)).status is RunStatus.RUNNING
    units = await store.list_work_units(run_id, graph_version=2)
    by_name = {unit.name: unit for unit in units}
    assert by_name["Root"].status is WorkUnitStatus.FAILED
    assert by_name["Blocked_Child"].status is WorkUnitStatus.BLOCKED
    assert by_name["Independent"].status is WorkUnitStatus.SUCCEEDED
    assert by_name["Independent_Tail"].status is WorkUnitStatus.PENDING
    blocked_event = next(
        event
        for event in await store.list_events(run_id)
        if event.event_type is EventType.WORK_UNIT_BLOCKED
    )
    assert blocked_event.payload["reason"] == (
        f"DEPENDENCY_FAILED: {by_name['Root'].work_unit_id}"
    )

    second = await controller.dispatch_planned_wave(
        run_id,
        scheduler=Scheduler(),
        execute=execute,
        max_concurrency=2,
    )

    assert second.succeeded == (by_name["Independent_Tail"].work_unit_id,)
    assert "Blocked_Child" not in executed
    assert (await store.get_run(run_id)).status is RunStatus.FAILED
    final_units = await store.list_work_units(run_id, graph_version=2)
    frontier = compute_frontier(final_units, graph_version=2)
    assert frontier.ready == frontier.waiting == ()
    assert frontier.blocked == (by_name["Blocked_Child"].work_unit_id,)
    assert assert_sequence_chain(await store.list_events(run_id)) is None


@pytest.mark.asyncio
async def test_cancel_before_first_wave_executes_nothing_and_cancels_graph(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    compiled = compile_plan(
        proposal(node("root"), node("child", depends_on=("root",))),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)
    await controller.commit_plan(run_id, compiled, target_graph_version=2)
    await controller.cancel(run_id)
    executed: list[str] = []

    async def execute(unit: WorkUnit) -> WaveOutcome:
        executed.append(unit.work_unit_id)
        return WaveOutcome.success()

    result = await controller.dispatch_planned_wave(
        run_id,
        scheduler=Scheduler(),
        execute=execute,
    )

    assert result.status is WaveStatus.CANCELLED
    assert result.started == ()
    assert executed == []
    assert (await store.get_run(run_id)).status is RunStatus.CANCELLED
    assert all(
        unit.status is WorkUnitStatus.CANCELLED
        for unit in await store.list_work_units(run_id, graph_version=2)
    )


@pytest.mark.asyncio
async def test_cancel_during_wave_settles_and_prevents_a_later_dispatch(
    store: SqliteStore,
) -> None:
    controller, run_id = await create_controller_run(store)
    compiled = compile_plan(
        proposal(node("root"), node("child", depends_on=("root",))),
        run_id=run_id,
        graph_version=2,
    )
    assert isinstance(compiled, CompiledPlan)
    await controller.commit_plan(run_id, compiled, target_graph_version=2)
    executed: list[str] = []

    async def execute(unit: WorkUnit) -> WaveOutcome:
        executed.append(unit.name)
        await controller.cancel(run_id)
        return WaveOutcome.success()

    first = await controller.dispatch_planned_wave(
        run_id,
        scheduler=Scheduler(),
        execute=execute,
    )

    assert first.status is WaveStatus.CANCELLED
    assert executed == ["Root"]
    assert (await store.get_run(run_id)).status is RunStatus.CANCELLED
    units = await store.list_work_units(run_id, graph_version=2)
    by_name = {unit.name: unit for unit in units}
    assert by_name["Root"].status is WorkUnitStatus.SUCCEEDED
    assert by_name["Child"].status is WorkUnitStatus.CANCELLED

    second = await controller.dispatch_planned_wave(
        run_id,
        scheduler=Scheduler(),
        execute=execute,
    )
    assert second.status is WaveStatus.CANCELLED
    assert second.started == ()
    assert executed == ["Root"]


@pytest.mark.asyncio
async def test_explicit_planned_executes_verified_graph_and_assembles_result(
    store: SqliteStore,
) -> None:
    result_schema = json.dumps(
        {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }
    )
    planner_adapter = StaticPlannerAdapter(
        {
            "summary": "produce two inputs and combine them",
            "nodes": [
                node("left", instruction="produce left"),
                node("right", instruction="produce right"),
                node(
                    "join",
                    instruction="combine results",
                    depends_on=("left", "right"),
                    output={"kind": "JSON", "json_schema": result_schema},
                ),
            ],
        }
    )
    worker_adapter = PlannedWorkerAdapter(
        {
            "produce left": "left result",
            "produce right": "right result",
            "combine results": '{"result":"combined"}',
        },
        concurrent_instructions=frozenset({"produce left", "produce right"}),
    )
    controller = RunController(
        store,
        planned_settings(),
        {"planner": planner_adapter, "worker": worker_adapter},
    )
    created = await controller.create_run(
        NativeRunRequest(
            input="combine two independently produced values",
            strategy=ExecutionStrategy.PLANNED,
            routing_policy=RoutingPolicy.MANUAL,
            budget=Budget(max_attempts=4, max_concurrency=2),
        )
    )

    completed = await controller.execute(created.run_id)

    assert completed.status is RunStatus.SUCCEEDED
    assert completed.strategy is ExecutionStrategy.PLANNED
    assert completed.graph_version == 2
    assert completed.usage.total_tokens == 19
    assert completed.usage.input_tokens == 8
    assert completed.usage.output_tokens == 11
    assert completed.usage.strong_model_tokens == 10
    assert len(planner_adapter.requests) == 1
    assert len(worker_adapter.requests) == 3
    assert worker_adapter.max_active == 2
    join_request = next(
        request
        for request in worker_adapter.requests
        if request.input.startswith("combine results")
    )
    assert "left result" in join_request.input
    assert "right result" in join_request.input
    assert join_request.json_schema == result_schema

    attempts = await store.list_run_attempts(created.run_id)
    assert len(attempts) == 4
    assert attempts[0].role is ModelRole.PLANNER
    assert attempts[0].usage is not None
    assert attempts[0].usage.strong_model_tokens == 10
    units = await store.list_work_units(created.run_id, graph_version=2)
    assert len(units) == 3
    assert all(unit.status is WorkUnitStatus.SUCCEEDED for unit in units)
    evidence = []
    for unit in units:
        evidence.extend(await store.list_evidence(unit.work_unit_id))
    assert len(evidence) == 8
    assert all(row.passed is True for row in evidence)
    assembled = await assemble_run_result(store, created.run_id)
    assert assembled.status is RunStatus.SUCCEEDED
    assert assembled.strategy is ExecutionStrategy.PLANNED
    assert assembled.output_text == '{"result":"combined"}'
    events = await store.list_events(created.run_id)
    assert EventType.PLAN_COMMITTED in {event.event_type for event in events}
    assert events[-1].event_type is EventType.RUN_SUCCEEDED
    assert assert_sequence_chain(events) is None


@pytest.mark.asyncio
async def test_planned_attempt_budget_stops_before_next_worker_call(
    store: SqliteStore,
) -> None:
    planner_adapter = StaticPlannerAdapter(
        {
            "summary": "a bounded chain",
            "nodes": [
                node("first", instruction="first task"),
                node("second", instruction="second task", depends_on=("first",)),
                node("third", instruction="third task", depends_on=("second",)),
            ],
        }
    )
    worker_adapter = PlannedWorkerAdapter(
        {
            "first task": "first result",
            "second task": "second result",
            "third task": "third result",
        }
    )
    controller = RunController(
        store,
        planned_settings(),
        {"planner": planner_adapter, "worker": worker_adapter},
    )
    created = await controller.create_run(
        NativeRunRequest(
            input="run a limited plan",
            strategy=ExecutionStrategy.PLANNED,
            routing_policy=RoutingPolicy.MANUAL,
            budget=Budget(max_attempts=2),
        )
    )

    completed = await controller.execute(created.run_id)

    assert completed.status is RunStatus.FAILED
    assert len(worker_adapter.requests) == 1
    assert len(await store.list_run_attempts(created.run_id)) == 2
    units = await store.list_work_units(created.run_id, graph_version=2)
    by_name = {unit.name: unit for unit in units}
    assert by_name["First"].status is WorkUnitStatus.SUCCEEDED
    assert by_name["Second"].status is WorkUnitStatus.FAILED
    assert by_name["Third"].status is WorkUnitStatus.BLOCKED
    event_types = [
        event.event_type for event in await store.list_events(created.run_id)
    ]
    assert event_types.count(EventType.BUDGET_EXHAUSTED) == 1
    assert event_types[-1] is EventType.RUN_FAILED


@pytest.mark.asyncio
async def test_planned_exact_total_ceiling_accepts_verified_worker_artifact(
    store: SqliteStore,
) -> None:
    planner_adapter = StaticPlannerAdapter(
        {"summary": "one step", "nodes": [node("only", instruction="only task")]}
    )
    worker_adapter = PlannedWorkerAdapter({"only task": "verified result"})
    controller = RunController(
        store,
        planned_settings(),
        {"planner": planner_adapter, "worker": worker_adapter},
    )
    created = await controller.create_run(
        NativeRunRequest(
            input="run exactly one step",
            strategy=ExecutionStrategy.PLANNED,
            routing_policy=RoutingPolicy.MANUAL,
            budget=Budget(max_total_tokens=13),
        )
    )

    completed = await controller.execute(created.run_id)

    assert completed.status is RunStatus.SUCCEEDED
    assert completed.usage.total_tokens == 13
    assert len(planner_adapter.requests) == 1
    assert len(worker_adapter.requests) == 1
    assert len(await store.list_run_attempts(created.run_id)) == 2
    event_types = [
        event.event_type for event in await store.list_events(created.run_id)
    ]
    assert EventType.EVIDENCE_RECORDED in event_types
    assert EventType.BUDGET_EXHAUSTED not in event_types


@pytest.mark.asyncio
async def test_planned_exact_strong_ceiling_commits_plan_but_blocks_worker(
    store: SqliteStore,
) -> None:
    planner_adapter = StaticPlannerAdapter(
        {"summary": "one step", "nodes": [node("blocked", instruction="blocked task")]}
    )
    worker_adapter = PlannedWorkerAdapter({"blocked task": "must not run"})
    controller = RunController(
        store,
        planned_settings(),
        {"planner": planner_adapter, "worker": worker_adapter},
    )
    created = await controller.create_run(
        NativeRunRequest(
            input="plan up to the ceiling",
            strategy=ExecutionStrategy.PLANNED,
            routing_policy=RoutingPolicy.MANUAL,
            budget=Budget(max_strong_model_tokens=10),
        )
    )

    completed = await controller.execute(created.run_id)

    assert completed.status is RunStatus.FAILED
    assert completed.error is not None
    assert completed.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert completed.graph_version == 2
    assert len(planner_adapter.requests) == 1
    assert worker_adapter.requests == []
    assert len(await store.list_run_attempts(created.run_id)) == 1
    assert len(await store.list_work_units(created.run_id, graph_version=2)) == 1


@pytest.mark.asyncio
async def test_planned_over_strong_ceiling_rejects_plan_before_commit(
    store: SqliteStore,
) -> None:
    planner_adapter = StaticPlannerAdapter(
        {"summary": "one step", "nodes": [node("unused")]}
    )
    worker_adapter = PlannedWorkerAdapter({"produce unused": "must not run"})
    controller = RunController(
        store,
        planned_settings(),
        {"planner": planner_adapter, "worker": worker_adapter},
    )
    created = await controller.create_run(
        NativeRunRequest(
            input="plan over the ceiling",
            strategy=ExecutionStrategy.PLANNED,
            routing_policy=RoutingPolicy.MANUAL,
            budget=Budget(max_strong_model_tokens=9),
        )
    )

    completed = await controller.execute(created.run_id)

    assert completed.status is RunStatus.FAILED
    assert completed.error is not None
    assert completed.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert completed.graph_version == 1
    assert len(planner_adapter.requests) == 1
    assert worker_adapter.requests == []
    assert await store.list_work_units(created.run_id, graph_version=2) == ()
