"""Progressive integration: evidence-gated revision and bounded termination."""

import asyncio
import json
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
from prp_runtime.domain.models import Budget, ErrorCategory, NativeRunRequest, Usage
from prp_runtime.planning.models import (
    PlanProposal,
    PlanRevision,
    PlanRevisionReason,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

RESULT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"ok": {"const": True}},
        "required": ["ok"],
        "additionalProperties": False,
    }
)


def _profile(alias: str, role: ModelRole) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="fake",
        model=f"{alias}-model",
        role=role,
        base_url="https://models.invalid/v1",
        supports_structured_output=True,
        context_window_tokens=8_000,
        max_output_tokens=1_000,
    )


def _settings() -> Settings:
    return Settings(
        leader_profile=_profile("planner", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
    )


def _proposal(key: str) -> PlanProposal:
    return PlanProposal(
        summary=f"plan {key}",
        final_node=key,
        nodes=(
            {
                "key": key,
                "name": key.title(),
                "instruction": f"produce {key}",
                "output": {"kind": "JSON", "json_schema": RESULT_SCHEMA},
            },
        ),
    )


def _chain_proposal(*, changed_tail: bool = False) -> PlanProposal:
    return PlanProposal(
        summary="chain revision",
        final_node="tail",
        nodes=(
            {
                "key": "root",
                "lineage_key": "root-lineage",
                "name": "Root",
                "instruction": "produce root",
                "output": {"kind": "JSON", "json_schema": RESULT_SCHEMA},
            },
            {
                "key": "tail",
                "lineage_key": "tail-lineage",
                "name": "Tail",
                "instruction": (
                    "produce changed tail" if changed_tail else "produce tail"
                ),
                "depends_on": ["root"],
                "output": {"kind": "JSON", "json_schema": RESULT_SCHEMA},
            },
        ),
    )


class RevisionPlannerAdapter:
    def __init__(
        self,
        initial: PlanProposal,
        revision: PlanRevision | BaseException | None,
    ) -> None:
        self.initial = initial
        self.revision = revision
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "progressive-planner"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            text = self.initial.model_dump_json()
        else:
            assert self.revision is not None
            if isinstance(self.revision, BaseException):
                raise self.revision
            text = self.revision.model_dump_json()
        return ProviderResponse(
            text=text,
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


class RevisionWorkerAdapter:
    def __init__(self, responses: tuple[str, ...]) -> None:
        self.responses = responses
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "progressive-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        response = self.responses[len(self.requests) - 1]
        return ProviderResponse(
            text=response,
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    async with SqliteStore(tmp_path / "progressive-integration.db") as opened:
        yield opened


def _run_request(
    *,
    max_revisions: int | None = 1,
    max_attempts: int = 4,
    max_total_tokens: int | None = None,
    max_strong_model_tokens: int | None = None,
) -> NativeRunRequest:
    return NativeRunRequest(
        input="produce a verified JSON result",
        routing_policy=RoutingPolicy.MANUAL,
        strategy=ExecutionStrategy.PROGRESSIVE,
        budget=Budget(
            max_plan_revisions=max_revisions,
            max_attempts=max_attempts,
            max_total_tokens=max_total_tokens,
            max_strong_model_tokens=max_strong_model_tokens,
        ),
    )


@pytest.mark.asyncio
async def test_fail_evidence_revises_once_and_isolates_old_graph(
    store: SqliteStore,
) -> None:
    initial = _proposal("initial")
    revised = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="replace failed output graph",
        proposal=_proposal("revised"),
    )
    planner = RevisionPlannerAdapter(initial, revised)
    worker = RevisionWorkerAdapter(("{\"ok\":false}", "{\"ok\":true}"))
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_run_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED, finished.error
    assert finished.strategy is ExecutionStrategy.PROGRESSIVE
    assert finished.graph_version == 3
    assert finished.usage == Usage(
        input_tokens=4,
        output_tokens=4,
        strong_model_tokens=4,
    )
    assert len(planner.requests) == 2
    assert len(worker.requests) == 2
    old_units = await store.list_work_units(run.run_id, graph_version=2)
    new_units = await store.list_work_units(run.run_id, graph_version=3)
    persisted = await store.get_run(run.run_id)
    assert persisted.final_work_unit_id == new_units[0].work_unit_id
    assert persisted.final_work_unit_id != old_units[0].work_unit_id
    assert old_units[0].status is WorkUnitStatus.FAILED
    assert new_units[0].status is WorkUnitStatus.SUCCEEDED
    assert len(await store.list_evidence(old_units[0].work_unit_id)) == 4
    assert len(await store.list_evidence(new_units[0].work_unit_id)) == 4
    rounds = await store.list_rounds(run.run_id)
    assert [round_fact.round_index for round_fact in rounds] == [0, 1]
    assert [round_fact.graph_version for round_fact in rounds] == [2, 3]
    assert [round_fact.status.value for round_fact in rounds] == ["FAILED", "VERIFIED"]
    assert rounds[0].merged_snapshot_id is None
    assert rounds[1].merged_snapshot_id is not None
    assert rounds[0].base_snapshot_id == rounds[1].base_snapshot_id
    assert rounds[1].revision_of_round_id == rounds[0].round_id
    assert rounds[1].evidence_ids
    attempts = await store.list_run_attempts(run.run_id)
    planner_attempts = tuple(
        attempt for attempt in attempts if attempt.role is ModelRole.PLANNER
    )
    assert len(attempts) == 4
    assert [attempt.attempt_index for attempt in planner_attempts] == [1, 2]
    assert [attempt.status for attempt in planner_attempts] == [
        AttemptStatus.SUCCEEDED,
        AttemptStatus.SUCCEEDED,
    ]
    assert all(
        attempt.usage is not None
        and attempt.usage.strong_model_tokens == 2
        for attempt in planner_attempts
    )
    planning_units = await store.list_work_units(run.run_id, graph_version=1)
    assert len(planning_units) == 2
    assert all(unit.status is WorkUnitStatus.SUCCEEDED for unit in planning_units)
    events = await store.list_events(run.run_id)
    assert EventType.PLAN_REVISED in {event.event_type for event in events}
    global_reports = [
        event
        for event in events
        if event.event_type is EventType.CONTROLLER_DECISION
        and "global_report" in event.payload
    ]
    assert [event.payload["global_report"]["result"] for event in global_reports] == [
        "FAIL",
        "PASS",
    ]
    assert global_reports[1].payload["comparison"]["outcome"] == "IMPROVED"
    assert events[-1].event_type is EventType.RUN_SUCCEEDED
    assert assert_sequence_chain(events) is None


@pytest.mark.asyncio
async def test_progressive_pass_never_calls_revision_planner(store: SqliteStore) -> None:
    planner = RevisionPlannerAdapter(_proposal("success"), None)
    worker = RevisionWorkerAdapter(("{\"ok\":true}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.graph_version == 2
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    rounds = await store.list_rounds(run.run_id)
    assert len(rounds) == 1
    assert rounds[0].status.value == "VERIFIED"
    assert rounds[0].graph_version == 2
    assert rounds[0].merged_snapshot_id is not None
    assert rounds[0].evidence_ids
    assert EventType.PLAN_REVISED not in {
        event.event_type for event in await store.list_events(run.run_id)
    }
    reports = [
        event
        for event in await store.list_events(run.run_id)
        if event.event_type is EventType.CONTROLLER_DECISION
        and "global_report" in event.payload
    ]
    assert len(reports) == 1
    assert reports[0].payload["global_report"]["result"] == "PASS"
    assert reports[0].payload["comparison"]["outcome"] == "BASELINE"
    global_report = reports[0].payload["global_report"]
    assert global_report["round_id"] == rounds[0].round_id
    candidate_checks = [
        check for check in global_report["checks"] if check["kind"] == "CANDIDATE"
    ]
    assert len(candidate_checks) == 1
    assert rounds[0].merged_snapshot_id in candidate_checks[0]["fact_ids"]
    assert set(rounds[0].evidence_ids).issubset(candidate_checks[0]["fact_ids"])


@pytest.mark.asyncio
async def test_revision_reuses_unchanged_root_and_recomputes_changed_tail(
    store: SqliteStore,
) -> None:
    initial = _chain_proposal()
    revised = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="change only the tail",
        proposal=_chain_proposal(changed_tail=True),
    )
    planner = RevisionPlannerAdapter(initial, revised)
    worker = RevisionWorkerAdapter(
        ('{"ok":true}', '{"ok":false}', '{"ok":true}')
    )
    controller = RunController(
        store,
        _settings(),
        {"planner": planner, "worker": worker},
    )
    run = await controller.create_run(_run_request(max_attempts=8))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED, finished.error
    assert len(worker.requests) == 3
    old_units = await store.list_work_units(run.run_id, graph_version=2)
    new_units = await store.list_work_units(run.run_id, graph_version=3)
    old_by_lineage = {unit.lineage_key: unit for unit in old_units}
    new_by_lineage = {unit.lineage_key: unit for unit in new_units}
    old_root_artifacts = await store.list_artifacts(old_by_lineage["root-lineage"].work_unit_id)
    old_tail_artifacts = await store.list_artifacts(old_by_lineage["tail-lineage"].work_unit_id)
    new_root_artifacts = await store.list_artifacts(new_by_lineage["root-lineage"].work_unit_id)
    new_tail_artifacts = await store.list_artifacts(new_by_lineage["tail-lineage"].work_unit_id)

    assert old_by_lineage["root-lineage"].status is WorkUnitStatus.SUCCEEDED
    assert old_by_lineage["tail-lineage"].status is WorkUnitStatus.FAILED
    assert new_by_lineage["root-lineage"].status is WorkUnitStatus.SUCCEEDED
    assert new_by_lineage["tail-lineage"].status is WorkUnitStatus.SUCCEEDED
    assert [artifact.content for artifact in old_root_artifacts] == ["{\"ok\":true}"]
    assert [artifact.content for artifact in old_tail_artifacts] == ["{\"ok\":false}"]
    assert [artifact.content for artifact in new_root_artifacts] == ["{\"ok\":true}"]
    assert [artifact.content for artifact in new_tail_artifacts] == ["{\"ok\":true}"]
    assert old_root_artifacts[0].artifact_id != new_root_artifacts[0].artifact_id
    assert old_tail_artifacts[0].artifact_id != new_tail_artifacts[0].artifact_id

    events = await store.list_events(run.run_id)
    reused = [event for event in events if event.event_type is EventType.WORK_UNIT_REUSED]
    invalidated = [
        event for event in events if event.event_type is EventType.WORK_UNIT_INVALIDATED
    ]
    assert len(reused) == 1
    assert reused[0].payload["source_work_unit_id"] == old_by_lineage["root-lineage"].work_unit_id
    assert reused[0].payload["work_unit_id"] == new_by_lineage["root-lineage"].work_unit_id
    assert any(
        event.payload["work_unit_id"] == old_by_lineage["tail-lineage"].work_unit_id
        for event in invalidated
    )


@pytest.mark.asyncio
async def test_revision_budget_zero_stops_after_first_failed_graph(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("budget"), None)
    worker = RevisionWorkerAdapter(("{\"ok\":false}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_revisions=0))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category.value == "VERIFICATION_FAILED"
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert finished.graph_version == 2
    rounds = await store.list_rounds(run.run_id)
    assert len(rounds) == 1
    assert rounds[0].status.value == "FAILED"
    assert rounds[0].merged_snapshot_id is None
    assert rounds[0].evidence_ids == ()


@pytest.mark.asyncio
async def test_attempt_budget_stops_before_revision_planner_call(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("attempt-budget"), None)
    worker = RevisionWorkerAdapter(("{\"ok\":false}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_attempts=2))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert len(await store.list_run_attempts(run.run_id)) == 2
    assert await store.list_work_units(run.run_id, graph_version=3) == ()


@pytest.mark.asyncio
async def test_strong_token_budget_records_revision_but_prevents_graph_commit(
    store: SqliteStore,
) -> None:
    revision = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="replace failed output graph",
        proposal=_proposal("must-not-commit"),
    )
    planner = RevisionPlannerAdapter(_proposal("strong-budget"), revision)
    worker = RevisionWorkerAdapter(("{\"ok\":false}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(
        _run_request(max_strong_model_tokens=3)
    )

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert finished.graph_version == 2
    assert finished.usage.strong_model_tokens == 4
    assert len(planner.requests) == 2
    assert len(worker.requests) == 1
    assert await store.list_work_units(run.run_id, graph_version=3) == ()


@pytest.mark.asyncio
async def test_progressive_exact_total_ceiling_accepts_verified_current_artifact(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("success"), None)
    worker = RevisionWorkerAdapter(('{"ok":true}',))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_total_tokens=4))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.SUCCEEDED
    assert finished.usage.total_tokens == 4
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert len(await store.list_run_attempts(run.run_id)) == 2
    event_types = [event.event_type for event in await store.list_events(run.run_id)]
    assert EventType.EVIDENCE_RECORDED in event_types
    assert EventType.BUDGET_EXHAUSTED not in event_types


@pytest.mark.asyncio
async def test_progressive_exact_total_ceiling_blocks_revision_call(
    store: SqliteStore,
) -> None:
    revision = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="must not be requested",
        proposal=_proposal("unused"),
    )
    planner = RevisionPlannerAdapter(_proposal("exact-budget"), revision)
    worker = RevisionWorkerAdapter(('{"ok":false}',))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_total_tokens=4))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert len(await store.list_run_attempts(run.run_id)) == 2
    assert await store.list_work_units(run.run_id, graph_version=3) == ()
    assert any(
        event.event_type is EventType.EVIDENCE_RECORDED
        for event in await store.list_events(run.run_id)
    )


@pytest.mark.asyncio
async def test_progressive_over_total_ceiling_rejects_before_verification(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("over-budget"), None)
    worker = RevisionWorkerAdapter(('{"ok":true}',))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request(max_total_tokens=3))

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.BUDGET_EXCEEDED
    assert finished.usage.total_tokens == 4
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    assert len(await store.list_run_attempts(run.run_id)) == 2
    assert all(
        event.event_type is not EventType.EVIDENCE_RECORDED
        for event in await store.list_events(run.run_id)
    )


@pytest.mark.asyncio
async def test_revision_planner_failure_records_failed_attempt_without_new_graph(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(
        _proposal("provider-failure"),
        ProviderError("private upstream detail", code=ErrorCode.PROVIDER_TIMEOUT),
    )
    worker = RevisionWorkerAdapter(("{\"ok\":false}",))
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request())

    finished = await controller.execute(run.run_id)

    assert finished.status is RunStatus.FAILED
    assert finished.error is not None
    assert finished.error.category is ErrorCategory.TIMEOUT
    assert finished.graph_version == 2
    assert await store.list_work_units(run.run_id, graph_version=3) == ()
    planner_attempts = tuple(
        attempt
        for attempt in await store.list_run_attempts(run.run_id)
        if attempt.role is ModelRole.PLANNER
    )
    assert [attempt.status for attempt in planner_attempts] == [
        AttemptStatus.SUCCEEDED,
        AttemptStatus.FAILED,
    ]
    assert planner_attempts[-1].usage is None
    assert planner_attempts[-1].error is not None
    assert planner_attempts[-1].error.category is ErrorCategory.TIMEOUT


@pytest.mark.asyncio
async def test_cancel_during_progressive_worker_skips_revision_call(
    store: SqliteStore,
) -> None:
    planner = RevisionPlannerAdapter(_proposal("cancel"), None)
    cancel_started = asyncio.Event()
    controller: RunController | None = None
    run_id = ""

    class CancellingAdapter(RevisionWorkerAdapter):
        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            assert controller is not None
            self.requests.append(request)
            cancel_started.set()
            await controller.cancel(run_id)
            return ProviderResponse(
                text='{"ok":true}',
                usage=Usage(input_tokens=1, output_tokens=1),
                finish_reason=FinishReason.STOP,
            )

    worker = CancellingAdapter(())
    controller = RunController(store, _settings(), {"planner": planner, "worker": worker})
    run = await controller.create_run(_run_request())
    run_id = run.run_id

    finished = await controller.execute(run.run_id)

    assert cancel_started.is_set()
    assert finished.status is RunStatus.CANCELLED
    assert len(planner.requests) == 1
    assert len(worker.requests) == 1
    rounds = await store.list_rounds(run.run_id)
    assert len(rounds) == 1
    assert rounds[0].status.value == "CANCELLED"
    assert rounds[0].merged_snapshot_id is None
    assert rounds[0].failure_reason
    events = await store.list_events(run.run_id)
    assert events[-1].event_type is EventType.RUN_CANCELLED
    assert sum(event.event_type is EventType.ROUND_FAILED for event in events) == 1
