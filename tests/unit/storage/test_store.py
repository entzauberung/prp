"""Targeted tests for the SQLite operation set."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, InternalError
from prp_runtime.domain.events import EventType, assert_sequence_chain, payload_from_model
from prp_runtime.domain.models import (
    Artifact,
    ArtifactKind,
    Attempt,
    Budget,
    ErrorCategory,
    ErrorInfo,
    Evidence,
    EvidenceKind,
    NativeRunRequest,
    OutputRequirement,
    Run,
    Usage,
    VerificationResult,
    WorkUnit,
    new_artifact_id,
    new_evidence_id,
)
from prp_runtime.domain.values import (
    ModelRef,
    ResourceClaim,
    new_attempt_id,
    new_run_id,
    new_work_unit_id,
)
from prp_runtime.storage.sqlite import (
    DanglingReferenceError,
    DuplicateEntityError,
    MissingEntityError,
    SqliteStore,
)

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
WORKER = ModelRef(provider="openai_compatible", model="weak-model")
PLANNER = ModelRef(provider="anthropic", model="strong-model")


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "store-test.db"


@pytest_asyncio.fixture
async def store(database_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(database_path) as opened:
        yield opened


def make_run(**overrides: object) -> Run:
    data: dict[str, object] = {
        "run_id": new_run_id(),
        "request": NativeRunRequest(input="summarise the report"),
        "created_at": T0,
    }
    data.update(overrides)
    return Run(**data)  # type: ignore[arg-type]


def make_work_unit(run_id: str, **overrides: object) -> WorkUnit:
    data: dict[str, object] = {
        "work_unit_id": new_work_unit_id(),
        "run_id": run_id,
        "name": "unit",
        "instruction": "do the work",
        "created_at": T0,
    }
    data.update(overrides)
    return WorkUnit(**data)  # type: ignore[arg-type]


def make_attempt(run_id: str, work_unit_id: str, **overrides: object) -> Attempt:
    data: dict[str, object] = {
        "attempt_id": new_attempt_id(),
        "run_id": run_id,
        "work_unit_id": work_unit_id,
        "role": ModelRole.WORKER,
        "model": WORKER,
        "created_at": T0,
    }
    data.update(overrides)
    return Attempt(**data)  # type: ignore[arg-type]


async def seed(store: SqliteStore) -> tuple[Run, WorkUnit, Attempt]:
    run = make_run()
    await store.create_run(run)
    unit = make_work_unit(run.run_id)
    await store.create_work_unit(unit)
    attempt = make_attempt(run.run_id, unit.work_unit_id)
    await store.create_attempt(attempt)
    return run, unit, attempt


# --- runs -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_round_trips_with_every_field(store: SqliteStore) -> None:
    run = make_run(
        request=NativeRunRequest(
            input="  analyse   ",
            instructions="be concise",
            routing_policy=RoutingPolicy.MANUAL,
            strategy=ExecutionStrategy.PROGRESSIVE,
            budget=Budget(max_total_tokens=500, max_attempts=4, deadline=T0),
            output=OutputRequirement(kind=ArtifactKind.JSON, json_schema='{"type":"object"}'),
        ),
        status=RunStatus.RUNNING,
        strategy=ExecutionStrategy.PROGRESSIVE,
        graph_version=3,
        usage=Usage(input_tokens=7, output_tokens=9, strong_model_tokens=5, elapsed_ms=120),
        started_at=T0 + timedelta(seconds=1),
    )
    await store.create_run(run)
    assert await store.get_run(run.run_id) == run


@pytest.mark.asyncio
async def test_duplicate_run_id_is_rejected(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(DuplicateEntityError):
        await store.create_run(run)


@pytest.mark.asyncio
async def test_missing_run_reads_and_updates_fail_with_run_not_found(
    store: SqliteStore,
) -> None:
    with pytest.raises(MissingEntityError) as excinfo:
        await store.get_run("run_missing")
    assert excinfo.value.code is ErrorCode.RUN_NOT_FOUND
    with pytest.raises(MissingEntityError):
        await store.update_run(make_run())


@pytest.mark.asyncio
async def test_update_run_persists_terminal_state_and_error(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    failed = run.model_copy(
        update={
            "status": RunStatus.FAILED,
            "strategy": ExecutionStrategy.DIRECT,
            "started_at": T0,
            "completed_at": T0 + timedelta(seconds=2),
            "error": ErrorInfo(
                category=ErrorCategory.PROVIDER_ERROR, message="upstream rejected request"
            ),
        }
    )
    await store.update_run(failed)
    stored = await store.get_run(run.run_id)
    assert stored.status is RunStatus.FAILED
    assert stored.error is not None
    assert stored.error.category is ErrorCategory.PROVIDER_ERROR
    assert stored.completed_at == T0 + timedelta(seconds=2)


@pytest.mark.asyncio
async def test_run_usage_accumulates_atomically(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    assert await store.get_run_usage(run.run_id) == Usage()
    await store.add_run_usage(run.run_id, Usage(input_tokens=3, output_tokens=4, elapsed_ms=10))
    total = await store.add_run_usage(
        run.run_id, Usage(input_tokens=1, output_tokens=1, strong_model_tokens=2, elapsed_ms=5)
    )
    assert total == Usage(
        input_tokens=4, output_tokens=5, strong_model_tokens=2, elapsed_ms=15
    )
    assert (await store.get_run(run.run_id)).usage == total
    with pytest.raises(MissingEntityError):
        await store.add_run_usage("run_missing", Usage())


@pytest.mark.asyncio
async def test_update_run_does_not_clobber_accumulated_usage(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    await store.add_run_usage(run.run_id, Usage(input_tokens=10))
    await store.update_run(run.model_copy(update={"status": RunStatus.RUNNING, "started_at": T0}))
    assert (await store.get_run(run.run_id)).usage == Usage(input_tokens=10)


@pytest.mark.asyncio
async def test_concurrent_usage_updates_are_not_lost(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    await asyncio.gather(
        *(store.add_run_usage(run.run_id, Usage(input_tokens=1)) for _ in range(20))
    )
    assert (await store.get_run_usage(run.run_id)).input_tokens == 20


# --- work units -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_work_unit_round_trips_with_edges_and_claims(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    first = make_work_unit(run.run_id, name="collect")
    second = make_work_unit(
        run.run_id,
        name="summarise",
        graph_version=2,
        acceptance_criteria="mentions every section",
        depends_on=(first.work_unit_id,),
        resource_claims=(
            ResourceClaim(resource="report.md", access=ResourceAccess.READ),
            ResourceClaim(resource="summary.md", access=ResourceAccess.WRITE),
        ),
        output=OutputRequirement(kind=ArtifactKind.JSON, json_schema='{"type":"object"}'),
        status=WorkUnitStatus.READY,
    )
    await store.create_work_units([second, first])
    assert await store.get_work_unit(second.work_unit_id) == second
    assert await store.get_work_unit(first.work_unit_id) == first


@pytest.mark.asyncio
async def test_list_work_units_filters_by_graph_version(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    first = make_work_unit(run.run_id, name="v1")
    revised = make_work_unit(run.run_id, name="v2", graph_version=2)
    await store.create_work_units([first, revised])
    assert await store.list_work_units(run.run_id) == (first, revised)
    assert await store.list_work_units(run.run_id, graph_version=2) == (revised,)
    assert await store.list_work_units(run.run_id, graph_version=9) == ()


@pytest.mark.asyncio
async def test_work_unit_with_unknown_run_is_rejected(store: SqliteStore) -> None:
    with pytest.raises(DanglingReferenceError):
        await store.create_work_unit(make_work_unit(new_run_id()))


@pytest.mark.asyncio
async def test_work_unit_edge_to_unknown_unit_is_rejected(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(DanglingReferenceError):
        await store.create_work_unit(
            make_work_unit(run.run_id, depends_on=(new_work_unit_id(),))
        )


@pytest.mark.asyncio
async def test_update_work_unit_persists_status_only(store: SqliteStore) -> None:
    run, unit, _ = await seed(store)
    await store.update_work_unit(unit.model_copy(update={"status": WorkUnitStatus.SUCCEEDED}))
    stored = await store.get_work_unit(unit.work_unit_id)
    assert stored.status is WorkUnitStatus.SUCCEEDED
    assert stored.instruction == unit.instruction
    with pytest.raises(MissingEntityError) as excinfo:
        await store.update_work_unit(make_work_unit(run.run_id))
    assert excinfo.value.code is ErrorCode.WORK_UNIT_NOT_FOUND


# --- attempts -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attempt_round_trips_including_usage_and_error(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    assert await store.get_attempt(attempt.attempt_id) == attempt
    assert (await store.get_attempt(attempt.attempt_id)).usage is None

    completed = attempt.model_copy(
        update={
            "status": AttemptStatus.FAILED,
            "provider_request_id": "req_42",
            "usage": Usage(input_tokens=5, output_tokens=6, elapsed_ms=70),
            "error": ErrorInfo(category=ErrorCategory.TIMEOUT, message="upstream timed out"),
            "started_at": T0,
            "completed_at": T0 + timedelta(seconds=1),
        }
    )
    await store.update_attempt(completed)
    assert await store.get_attempt(attempt.attempt_id) == completed

    escalated = make_attempt(
        run.run_id,
        unit.work_unit_id,
        attempt_index=2,
        role=ModelRole.PLANNER,
        model=PLANNER,
        status=AttemptStatus.SUCCEEDED,
        usage=Usage(input_tokens=1, output_tokens=2, strong_model_tokens=3),
        started_at=T0,
        completed_at=T0 + timedelta(seconds=1),
    )
    await store.create_attempt(escalated)
    assert await store.list_attempts(unit.work_unit_id) == (completed, escalated)


@pytest.mark.asyncio
async def test_duplicate_attempt_index_is_rejected(store: SqliteStore) -> None:
    run, unit, _ = await seed(store)
    with pytest.raises(DuplicateEntityError):
        await store.create_attempt(make_attempt(run.run_id, unit.work_unit_id))


@pytest.mark.asyncio
async def test_attempt_with_unknown_work_unit_is_rejected(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(DanglingReferenceError):
        await store.create_attempt(make_attempt(run.run_id, new_work_unit_id()))


@pytest.mark.asyncio
async def test_list_run_attempts_filters_by_status(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    running = attempt.model_copy(update={"status": AttemptStatus.RUNNING, "started_at": T0})
    await store.update_attempt(running)
    assert await store.list_run_attempts(run.run_id) == (running,)
    assert await store.list_run_attempts(run.run_id, statuses=[AttemptStatus.RUNNING]) == (
        running,
    )
    assert await store.list_run_attempts(run.run_id, statuses=[AttemptStatus.SUCCEEDED]) == ()
    assert await store.list_run_attempts(run.run_id, statuses=[]) == ()


@pytest.mark.asyncio
async def test_missing_attempt_read_and_update_fail(store: SqliteStore) -> None:
    run, unit, _ = await seed(store)
    with pytest.raises(MissingEntityError):
        await store.get_attempt(new_attempt_id())
    with pytest.raises(MissingEntityError):
        await store.update_attempt(make_attempt(run.run_id, unit.work_unit_id, attempt_index=9))


# --- artifacts and evidence -----------------------------------------------------


@pytest.mark.asyncio
async def test_artifact_and_evidence_round_trip(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        name="answer",
        kind=ArtifactKind.JSON,
        content='{"ok": true}',
        created_at=T0,
    )
    await store.add_artifact(artifact)
    assert await store.get_artifact(artifact.artifact_id) == artifact
    assert await store.list_artifacts(unit.work_unit_id) == (artifact,)

    evidence = Evidence(
        evidence_id=new_evidence_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        artifact_id=artifact.artifact_id,
        kind=EvidenceKind.DETERMINISTIC_CHECK,
        rule="MATCHES_JSON_SCHEMA",
        result=VerificationResult.FAIL,
        detail="schema mismatch",
        created_at=T0,
    )
    await store.add_evidence(evidence)
    stored = await store.list_evidence(unit.work_unit_id)
    assert stored == (evidence,)
    assert stored[0].result is VerificationResult.FAIL
    assert stored[0].passed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("result", list(VerificationResult))
async def test_every_verdict_survives_a_round_trip(
    store: SqliteStore, result: VerificationResult
) -> None:
    """An undecided verdict must not come back as a proven failure."""
    run, unit, attempt = await seed(store)
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        name="answer",
        content="text",
        created_at=T0,
    )
    await store.add_artifact(artifact)
    evidence = Evidence(
        evidence_id=new_evidence_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        artifact_id=artifact.artifact_id,
        kind=EvidenceKind.DETERMINISTIC_CHECK,
        rule="VALID_JSON",
        result=result,
        detail=f"recorded {result.value}",
        created_at=T0,
    )
    await store.add_evidence(evidence)
    stored = await store.list_evidence(unit.work_unit_id)
    assert stored == (evidence,)
    assert stored[0].result is result
    assert stored[0].passed is result.is_pass


@pytest.mark.asyncio
@pytest.mark.parametrize("result", list(VerificationResult))
async def test_the_read_path_derives_passed_from_the_stored_verdict(
    store: SqliteStore, result: VerificationResult
) -> None:
    """Proved against a raw row, not one this store wrote.

    A round trip through ``add_evidence`` only shows the writer and the reader
    agree. Inserting the row directly proves the reader derives ``passed`` from
    the stored verdict rather than trusting whatever the writer believed.
    """
    run, unit, attempt = await seed(store)
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        name="answer",
        content="text",
        created_at=T0,
    )
    await store.add_artifact(artifact)
    async with store.transaction() as connection:
        await connection.execute(
            """
            INSERT INTO evidence (evidence_id, run_id, work_unit_id, artifact_id, kind,
                                  rule, result, detail, created_at)
            VALUES (?, ?, ?, ?, 'DETERMINISTIC_CHECK', 'VALID_JSON', ?, 'a detail', ?)
            """,
            (
                new_evidence_id(),
                run.run_id,
                unit.work_unit_id,
                artifact.artifact_id,
                result.value,
                T0.isoformat(),
            ),
        )

    stored = await store.list_evidence(unit.work_unit_id)
    assert len(stored) == 1
    assert stored[0].result is result
    assert stored[0].passed is result.is_pass


@pytest.mark.asyncio
async def test_a_model_review_round_trips_without_a_rule(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_id=attempt.attempt_id,
        name="answer",
        content="text",
        created_at=T0,
    )
    await store.add_artifact(artifact)
    evidence = Evidence(
        evidence_id=new_evidence_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        artifact_id=artifact.artifact_id,
        kind=EvidenceKind.MODEL_REVIEW,
        result=VerificationResult.INCONCLUSIVE,
        detail="a reviewer could not decide",
        created_at=T0,
    )
    await store.add_evidence(evidence)
    stored = await store.list_evidence(unit.work_unit_id)
    assert stored == (evidence,)
    assert stored[0].rule is None


@pytest.mark.asyncio
async def test_artifact_and_evidence_reject_dangling_references(store: SqliteStore) -> None:
    run, unit, attempt = await seed(store)
    with pytest.raises(DanglingReferenceError):
        await store.add_artifact(
            Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=unit.work_unit_id,
                attempt_id=new_attempt_id(),
                name="answer",
                content="text",
                created_at=T0,
            )
        )
    with pytest.raises(DanglingReferenceError):
        await store.add_evidence(
            Evidence(
                evidence_id=new_evidence_id(),
                run_id=run.run_id,
                work_unit_id=unit.work_unit_id,
                artifact_id=new_artifact_id(),
                kind=EvidenceKind.MODEL_REVIEW,
                result=VerificationResult.PASS,
                detail="looks right",
                created_at=T0,
            )
        )


@pytest.mark.asyncio
async def test_two_attempts_may_produce_the_same_artifact_name(store: SqliteStore) -> None:
    run, unit, first_attempt = await seed(store)
    second_attempt = make_attempt(run.run_id, unit.work_unit_id, attempt_index=2)
    await store.create_attempt(second_attempt)
    for attempt in (first_attempt, second_attempt):
        await store.add_artifact(
            Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                name="answer",
                content=f"from {attempt.attempt_index}",
                created_at=T0,
            )
        )
    assert len(await store.list_artifacts(unit.work_unit_id)) == 2


# --- events ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_are_appended_with_monotonic_sequences(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    assert await store.last_sequence(run.run_id) is None
    created = await store.append_event(
        run.run_id,
        EventType.RUN_CREATED,
        payload_from_model("request", run.request),
        timestamp=T0,
    )
    started = await store.append_event(run.run_id, EventType.RUN_STARTED, timestamp=T0)
    assert (created.sequence, started.sequence) == (1, 2)
    assert await store.last_sequence(run.run_id) == 2
    ledger = await store.list_events(run.run_id)
    assert ledger == (created, started)
    assert assert_sequence_chain(ledger) is None
    assert ledger[0].payload["request"] == run.request.model_dump(mode="json")


@pytest.mark.asyncio
async def test_event_payload_is_validated_before_it_is_written(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(ValueError, match="payload is missing"):
        await store.append_event(run.run_id, EventType.WORK_UNIT_STARTED)
    assert await store.list_events(run.run_id) == ()


@pytest.mark.parametrize(
    "bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
)
@pytest.mark.asyncio
async def test_non_finite_event_payload_is_rejected_and_not_persisted(
    store: SqliteStore, bad_value: float
) -> None:
    """A non-finite number in the payload is refused at the write, not stored.

    ``json.dumps(..., allow_nan=False)`` runs before any row is inserted, so the
    rejection must be provable from the ledger, not only from the raised error.
    """
    run = make_run()
    await store.create_run(run)
    with pytest.raises(ValueError):
        await store.append_event(run.run_id, EventType.RUN_RESUMED, {"value": bad_value})
    assert await store.last_sequence(run.run_id) is None
    assert await store.list_events(run.run_id) == ()
    # The store is still usable: a finite payload after the rejection is
    # accepted and round-trips normally.
    appended = await store.append_event(run.run_id, EventType.RUN_RESUMED, {"value": 1.5})
    assert appended.sequence == 1
    ledger = await store.list_events(run.run_id)
    assert ledger[0].payload["value"] == 1.5


@pytest.mark.asyncio
async def test_event_for_unknown_run_is_rejected(store: SqliteStore) -> None:
    with pytest.raises(DanglingReferenceError) as excinfo:
        await store.append_event(new_run_id(), EventType.RUN_STARTED)
    assert excinfo.value.code is ErrorCode.RUN_NOT_FOUND


@pytest.mark.asyncio
async def test_after_sequence_cursor_is_stable(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    for _ in range(5):
        await store.append_event(run.run_id, EventType.RUN_RESUMED)
    assert [event.sequence for event in await store.list_events(run.run_id)] == [1, 2, 3, 4, 5]
    assert [
        event.sequence for event in await store.list_events(run.run_id, after_sequence=3)
    ] == [4, 5]
    assert [
        event.sequence for event in await store.list_events(run.run_id, after_sequence=0, limit=2)
    ] == [1, 2]
    assert await store.list_events(run.run_id, after_sequence=5) == ()
    with pytest.raises(ValueError):
        await store.list_events(run.run_id, limit=0)


@pytest.mark.asyncio
async def test_ledgers_of_different_runs_are_independent(store: SqliteStore) -> None:
    first = make_run()
    second = make_run()
    await store.create_run(first)
    await store.create_run(second)
    await store.append_event(first.run_id, EventType.RUN_STARTED)
    await store.append_event(second.run_id, EventType.RUN_STARTED)
    assert (await store.last_sequence(first.run_id)) == 1
    assert (await store.last_sequence(second.run_id)) == 1
    assert len(await store.list_events(first.run_id)) == 1


@pytest.mark.asyncio
async def test_concurrent_appends_produce_unique_sequences(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    await asyncio.gather(
        *(store.append_event(run.run_id, EventType.RUN_RESUMED) for _ in range(25))
    )
    ledger = await store.list_events(run.run_id)
    assert [event.sequence for event in ledger] == list(range(1, 26))
    assert assert_sequence_chain(ledger) is None


@pytest.mark.asyncio
async def test_concurrent_appends_from_two_connections_do_not_duplicate(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as writer:
        run = make_run()
        await writer.create_run(run)
        async with SqliteStore(database_path) as other:
            await asyncio.gather(
                *(
                    connection.append_event(run.run_id, EventType.RUN_RESUMED)
                    for connection in (writer, other, writer, other, writer, other)
                )
            )
            ledger = await writer.list_events(run.run_id)
    assert [event.sequence for event in ledger] == [1, 2, 3, 4, 5, 6]


# --- transaction atomicity ------------------------------------------------------


@pytest.mark.asyncio
async def test_state_and_events_commit_together(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    async with store.transaction():
        await store.update_run(
            run.model_copy(update={"status": RunStatus.RUNNING, "started_at": T0})
        )
        await store.append_event(run.run_id, EventType.RUN_STARTED)
        assert store.in_transaction is True
    assert store.in_transaction is False
    assert (await store.get_run(run.run_id)).status is RunStatus.RUNNING
    assert len(await store.list_events(run.run_id)) == 1


@pytest.mark.asyncio
async def test_state_and_events_roll_back_together(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    with pytest.raises(RuntimeError):
        async with store.transaction():
            await store.update_run(
                run.model_copy(update={"status": RunStatus.RUNNING, "started_at": T0})
            )
            await store.append_event(run.run_id, EventType.RUN_STARTED)
            raise RuntimeError("controller failed")
    assert (await store.get_run(run.run_id)).status is RunStatus.PENDING
    assert await store.list_events(run.run_id) == ()
    assert store.in_transaction is False


@pytest.mark.asyncio
async def test_a_failed_graph_commit_leaves_no_partial_units(store: SqliteStore) -> None:
    run = make_run()
    await store.create_run(run)
    good = make_work_unit(run.run_id, name="good")
    broken = make_work_unit(run.run_id, name="broken", depends_on=(new_work_unit_id(),))
    with pytest.raises(DanglingReferenceError):
        await store.create_work_units([good, broken])
    assert await store.list_work_units(run.run_id) == ()


@pytest.mark.asyncio
async def test_writes_outside_a_transaction_commit_immediately(
    database_path: Path,
) -> None:
    run = make_run()
    async with SqliteStore(database_path) as writer:
        await writer.create_run(run)
    async with SqliteStore(database_path) as reader:
        assert (await reader.get_run(run.run_id)).run_id == run.run_id


@pytest.mark.asyncio
async def test_store_operations_require_an_open_store(database_path: Path) -> None:
    closed = SqliteStore(database_path)
    with pytest.raises(InternalError, match="not open"):
        await closed.get_run("run_1")
    with pytest.raises(InternalError, match="not open"):
        await closed.create_run(make_run())
