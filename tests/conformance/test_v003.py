"""v0.0.3 conformance invariants for planned scheduling and recovery."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

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
from prp_runtime.domain.events import EventType, assert_sequence_chain
from prp_runtime.domain.models import Attempt, Budget, NativeRunRequest, Usage, WorkUnit
from prp_runtime.domain.values import ModelRef, new_attempt_id, utc_now
from prp_runtime.planning.compiler import CompiledPlan, compile_plan
from prp_runtime.planning.models import PlanProposal
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.runtime.scheduler import Scheduler, WaveOutcome
from prp_runtime.settings import Settings
from prp_runtime.storage.recovery import recover_after_restart
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
        max_concurrency=3,
    )


def _settings() -> Settings:
    return Settings(
        leader_profile=_profile("planner", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
    )


def _request(*, max_concurrency: int = 2) -> NativeRunRequest:
    return NativeRunRequest(
        input="execute the fixed graph",
        strategy=ExecutionStrategy.PLANNED,
        routing_policy=RoutingPolicy.MANUAL,
        budget=Budget(max_concurrency=max_concurrency),
    )


def _node(key: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "key": key,
        "name": key.title(),
        "instruction": f"run {key}",
    }
    values.update(overrides)
    return values


class PlanAdapter:
    def __init__(self, nodes: list[dict[str, object]]) -> None:
        self._nodes = nodes
        self._final_node = str(nodes[-1]["key"])
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "v003-conformance-planner"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=json.dumps(
                {
                    "summary": "fixed conformance plan",
                    "final_node": self._final_node,
                    "nodes": self._nodes,
                }
            ),
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


class UnknownUsagePlanAdapter(PlanAdapter):
    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=json.dumps(
                {
                    "summary": "fixed conformance plan",
                    "final_node": self._final_node,
                    "nodes": self._nodes,
                }
            ),
            usage=None,
            finish_reason=FinishReason.STOP,
        )


class CancelledPlanAdapter:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "v003-cancelled-planner"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        raise asyncio.CancelledError


class ResourceWorkerAdapter:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []
        self.active: set[str] = set()
        self.first_batch: frozenset[str] | None = None
        self.batch_ready = asyncio.Event()
        self.release = asyncio.Event()
        self.conflict_observed = False

    @property
    def name(self) -> str:
        return "v003-resource-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        instruction = request.input.split("\n", maxsplit=1)[0]
        self.requests.append(request)
        if self.active and (
            instruction == "run writer" or "run writer" in self.active
        ):
            self.conflict_observed = True
        self.active.add(instruction)
        if instruction == "run writer" or self.active == {
            "run reader_a",
            "run reader_b",
        }:
            if self.first_batch is None:
                self.first_batch = frozenset(self.active)
            self.batch_ready.set()
        await self.release.wait()
        self.active.remove(instruction)
        return ProviderResponse(
            text=f"completed {instruction}",
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


class OutcomeWorkerAdapter:
    def __init__(
        self,
        outcomes: dict[str, str | BaseException],
    ) -> None:
        self._outcomes = outcomes
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "v003-outcome-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        instruction = request.input.split("\n", maxsplit=1)[0]
        outcome = self._outcomes[instruction]
        if isinstance(outcome, BaseException):
            raise outcome
        return ProviderResponse(
            text=outcome,
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "v003-conformance.db") as opened:
        yield opened


@pytest.mark.asyncio
async def test_resource_claims_never_overlap_a_writer_and_keep_auditable_facts(
    store: SqliteStore,
) -> None:
    planner = PlanAdapter(
        [
            _node(
                "reader_a",
                resource_claims=[{"resource": "document", "access": "READ"}],
            ),
            _node(
                "reader_b",
                resource_claims=[{"resource": "document", "access": "READ"}],
            ),
            _node(
                "writer",
                resource_claims=[{"resource": "document", "access": "WRITE"}],
            ),
        ]
    )
    worker = ResourceWorkerAdapter()
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_request(max_concurrency=3))

    execution = asyncio.create_task(controller.execute(run.run_id))
    await worker.batch_ready.wait()
    assert worker.first_batch in (
        frozenset({"run writer"}),
        frozenset({"run reader_a", "run reader_b"}),
    )
    assert worker.conflict_observed is False
    worker.release.set()
    finished = await execution

    assert finished.status is RunStatus.SUCCEEDED
    assert len(worker.requests) == 3
    assert worker.conflict_observed is False
    units = await store.list_work_units(run.run_id, graph_version=2)
    events = await store.list_events(run.run_id)
    assert all(unit.status is WorkUnitStatus.SUCCEEDED for unit in units)
    attempts = await store.list_run_attempts(run.run_id)
    assert len(attempts) == 4
    assert attempts[0].role is ModelRole.PLANNER
    assert attempts[0].usage == Usage(
        input_tokens=1,
        output_tokens=1,
        strong_model_tokens=2,
    )
    assert sum(
        event.event_type is EventType.WORK_UNIT_STARTED for event in events
    ) == 4
    assert sum(
        event.event_type is EventType.WORK_UNIT_SUCCEEDED for event in events
    ) == 4
    assert assert_sequence_chain(events) is None


@pytest.mark.asyncio
async def test_unknown_planner_usage_is_not_guessed_or_added_to_run(
    store: SqliteStore,
) -> None:
    planner = UnknownUsagePlanAdapter([_node("answer")])
    worker = OutcomeWorkerAdapter({"run answer": "answer"})
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.usage == Usage(input_tokens=1, output_tokens=1)
    attempts = await store.list_run_attempts(run.run_id)
    assert len(attempts) == 2
    assert attempts[0].role is ModelRole.PLANNER
    assert attempts[0].usage is None
    assert attempts[1].role is ModelRole.WORKER
    assert attempts[1].usage == Usage(input_tokens=1, output_tokens=1)
    events = await store.list_events(run.run_id)
    assert sum(event.event_type is EventType.USAGE_UPDATED for event in events) == 1


@pytest.mark.asyncio
async def test_cancelled_planner_call_records_unknown_without_user_graph(
    store: SqliteStore,
) -> None:
    planner = CancelledPlanAdapter()
    worker = OutcomeWorkerAdapter({})
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_request())

    with pytest.raises(asyncio.CancelledError):
        await controller.execute(run.run_id)

    attempts = await store.list_run_attempts(run.run_id)
    assert len(attempts) == 1
    assert attempts[0].role is ModelRole.PLANNER
    assert attempts[0].status is AttemptStatus.UNKNOWN
    assert attempts[0].usage is None
    planning_units = await store.list_work_units(run.run_id, graph_version=1)
    assert len(planning_units) == 1
    assert planning_units[0].status is WorkUnitStatus.CANCELLED
    assert await store.list_work_units(run.run_id, graph_version=2) == ()
    event_types = {
        event.event_type for event in await store.list_events(run.run_id)
    }
    assert EventType.ATTEMPT_UNKNOWN in event_types
    assert EventType.WORK_UNIT_CANCELLED in event_types


@pytest.mark.asyncio
async def test_failure_blocks_dependents_but_preserves_independent_success(
    store: SqliteStore,
) -> None:
    planner = PlanAdapter(
        [
            _node("root"),
            _node("child", depends_on=["root"]),
            _node("independent"),
        ]
    )
    worker = OutcomeWorkerAdapter(
        {
            "run root": ProviderError(
                "upstream unavailable",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
            ),
            "run child": "must not run",
            "run independent": "independent result",
        }
    )
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category.value == "NETWORK"
    called = {
        request.input.split("\n", maxsplit=1)[0] for request in worker.requests
    }
    assert called == {"run root", "run independent"}
    units = await store.list_work_units(run.run_id, graph_version=2)
    by_name = {unit.name: unit for unit in units}
    assert by_name["Root"].status is WorkUnitStatus.FAILED
    assert by_name["Child"].status is WorkUnitStatus.BLOCKED
    assert by_name["Independent"].status is WorkUnitStatus.SUCCEEDED
    blocked = next(
        event
        for event in await store.list_events(run.run_id)
        if event.event_type is EventType.WORK_UNIT_BLOCKED
    )
    assert blocked.payload["reason"] == (
        f"DEPENDENCY_FAILED: {by_name['Root'].work_unit_id}"
    )


@pytest.mark.asyncio
async def test_cancel_during_worker_prevents_all_later_graph_calls(
    store: SqliteStore,
) -> None:
    planner = PlanAdapter(
        [_node("root"), _node("child", depends_on=["root"])]
    )
    controller: RunController | None = None
    run_id = ""

    class CancellingWorker:
        def __init__(self) -> None:
            self.requests: list[ProviderRequest] = []

        @property
        def name(self) -> str:
            return "v003-cancelling-worker"

        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            assert controller is not None
            self.requests.append(request)
            await controller.cancel(run_id)
            return ProviderResponse(
                text="root result",
                usage=Usage(input_tokens=1, output_tokens=1),
                finish_reason=FinishReason.STOP,
            )

    worker = CancellingWorker()
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_request())
    run_id = run.run_id

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.CANCELLED
    assert len(worker.requests) == 1
    units = await store.list_work_units(run.run_id, graph_version=2)
    by_name = {unit.name: unit for unit in units}
    assert by_name["Root"].status is WorkUnitStatus.SUCCEEDED
    assert by_name["Child"].status is WorkUnitStatus.CANCELLED
    events = await store.list_events(run.run_id)
    assert EventType.RUN_CANCELLING in {event.event_type for event in events}
    assert events[-1].event_type is EventType.RUN_CANCELLED


@pytest.mark.asyncio
async def test_restart_interrupts_once_and_recomputed_frontier_never_reexecutes(
    store: SqliteStore,
) -> None:
    controller = RunController(store, Settings(), {})
    run = await controller.create_run(_request())
    proposal = PlanProposal(
        summary="root then child",
        final_node="child",
        nodes=(
            _node("root"),
            _node("child", depends_on=("root",)),
        ),
    )
    compiled = compile_plan(proposal, run_id=run.run_id, graph_version=2)
    assert isinstance(compiled, CompiledPlan)
    committed = await controller.commit_plan(
        run.run_id,
        compiled,
        target_graph_version=2,
    )
    assert isinstance(committed, tuple)
    root = next(unit for unit in committed if unit.name == "Root")
    interrupted_attempt_id = ""

    async def stop_process(unit: WorkUnit) -> WaveOutcome:
        nonlocal interrupted_attempt_id
        now = utc_now()
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=unit.work_unit_id,
            attempt_index=1,
            role=ModelRole.WORKER,
            model=ModelRef(provider="fake", model="worker"),
            status=AttemptStatus.RUNNING,
            created_at=now,
            started_at=now,
        )
        interrupted_attempt_id = attempt.attempt_id
        async with store.transaction():
            await store.create_attempt(attempt)
            await store.append_event(
                run.run_id,
                EventType.ATTEMPT_STARTED,
                {
                    "work_unit_id": unit.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                },
            )
        raise RuntimeError("process stopped")

    with pytest.raises(RuntimeError, match="process stopped"):
        await controller.dispatch_planned_wave(
            run.run_id,
            scheduler=Scheduler(),
            execute=stop_process,
        )
    assert (await store.get_work_unit(root.work_unit_id)).status is WorkUnitStatus.RUNNING

    first_recovery = await recover_after_restart(store)
    second_recovery = await recover_after_restart(store)
    reexecuted: list[str] = []

    async def must_not_execute(unit: WorkUnit) -> WaveOutcome:
        reexecuted.append(unit.work_unit_id)
        return WaveOutcome.success()

    await controller.dispatch_planned_wave(
        run.run_id,
        scheduler=Scheduler(),
        execute=must_not_execute,
    )

    assert first_recovery.interrupted_attempt_ids == (interrupted_attempt_id,)
    assert first_recovery.failed_work_unit_ids == (root.work_unit_id,)
    assert second_recovery.changed is False
    assert reexecuted == []
    assert (await store.get_attempt(interrupted_attempt_id)).status is AttemptStatus.INTERRUPTED
    units = await store.list_work_units(run.run_id, graph_version=2)
    by_name = {unit.name: unit for unit in units}
    assert by_name["Root"].status is WorkUnitStatus.FAILED
    assert by_name["Child"].status is WorkUnitStatus.BLOCKED
    assert (await store.get_run(run.run_id)).status is RunStatus.FAILED
    events = await store.list_events(run.run_id)
    assert sum(
        event.event_type is EventType.ATTEMPT_INTERRUPTED for event in events
    ) == 1
    assert assert_sequence_chain(events) is None
