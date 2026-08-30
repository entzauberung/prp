"""Targeted tests for in-process LocalRuntime lifecycle."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionLocation,
    IsolationMode,
    ModelRole,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import BudgetError, DomainValidationError, ErrorCode, ProviderError
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import Attempt, NativeRunRequest, Run, Usage, WorkUnit
from prp_runtime.domain.values import ModelRef, new_attempt_id, new_run_id, new_work_unit_id
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.runtime.local import LocalRuntime, LocalRuntimeState
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
WORKER = ModelRef(provider="openai_compatible", model="weak-model")
WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="openai_compatible",
    model="weak-model",
    role=ModelRole.WORKER,
    base_url="https://models.invalid/v1",
    context_window_tokens=1_024,
    max_output_tokens=128,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.close_calls = 0

    @property
    def name(self) -> str:
        return "local-fake"

    async def aclose(self) -> None:
        self.close_calls += 1

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        del request
        return ProviderResponse(
            text="unused",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


@pytest.mark.asyncio
async def test_local_runtime_open_close_is_idempotent_and_has_no_listener(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "local.db"
    runtime = LocalRuntime(
        Settings(database_path=database_path),
        adapters={"worker": FakeAdapter()},
    )
    assert runtime.state is LocalRuntimeState.CLOSED
    assert runtime.defaults.execution_location is ExecutionLocation.LOCAL
    assert runtime.defaults.isolation_mode is IsolationMode.HOST
    assert str(database_path) not in repr(runtime)

    await runtime.close()
    async with runtime:
        assert runtime.state is LocalRuntimeState.OPEN
        assert runtime.recovery is not None
        assert runtime.controller is not None
        assert runtime.composition.supervisor is not None
        assert runtime.composition.supervisor.running
        facts = runtime.public_facts()
        assert facts["execution_location"] == "LOCAL"
        assert facts["isolation_mode"] == "HOST"
        assert runtime.composition.execution_location is ExecutionLocation.LOCAL
        assert runtime.composition.isolation_mode is IsolationMode.HOST
        composition_facts = runtime.composition.public_facts()
        assert composition_facts["execution_location"] == "LOCAL"
        assert composition_facts["isolation_mode"] == "HOST"
        assert facts["max_slots"] == 2
        assert facts["max_copied_bytes"] == 256 * 1024 * 1024
        assert facts["max_concurrency"] == 1
        assert facts["max_attempts"] == 8
        assert facts["max_total_tokens"] == 250_000
        assert runtime.composition.supervisor.max_concurrency == 1
        assert str(database_path) not in str(facts)
        assert str(database_path) not in repr(runtime)
    await runtime.close()
    assert runtime.state is LocalRuntimeState.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        _ = runtime.controller


@pytest.mark.asyncio
async def test_local_runtime_keeps_injected_store(tmp_path: Path) -> None:
    database_path = tmp_path / "injected-local.db"
    store = SqliteStore(database_path)
    await store.open()
    adapter = FakeAdapter()
    runtime = LocalRuntime(
        Settings(database_path=database_path),
        adapters={"worker": adapter},
        store=store,
    )
    await runtime.open()
    await runtime.close()
    assert store.is_open
    assert adapter.close_calls == 0
    await store.close()


async def _persist_pending_run(store: SqliteStore) -> str:
    run = Run(
        run_id=new_run_id(),
        request=NativeRunRequest(input="analyse the report"),
        status=RunStatus.PENDING,
        created_at=T0,
    )
    await store.create_run(run)
    await store.append_event(run.run_id, EventType.RUN_CREATED, {"request": {"input": "x"}})
    return run.run_id


async def _persist_running_attempt(store: SqliteStore) -> tuple[str, str]:
    run = Run(
        run_id=new_run_id(),
        request=NativeRunRequest(input="analyse the report"),
        status=RunStatus.RUNNING,
        created_at=T0,
        started_at=T0,
    )
    await store.create_run(run)
    unit = WorkUnit(
        work_unit_id=new_work_unit_id(),
        run_id=run.run_id,
        name="unit",
        instruction="do the work",
        status=WorkUnitStatus.RUNNING,
        created_at=T0,
    )
    await store.create_work_unit(unit)
    await store.append_event(run.run_id, EventType.RUN_CREATED, {"request": {"input": "x"}})
    await store.append_event(run.run_id, EventType.RUN_STARTED)
    attempt = Attempt(
        attempt_id=new_attempt_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_index=1,
        role=ModelRole.WORKER,
        model=WORKER,
        status=AttemptStatus.RUNNING,
        created_at=T0,
        started_at=T0,
    )
    await store.create_attempt(attempt)
    await store.append_event(
        run.run_id,
        EventType.ATTEMPT_STARTED,
        {"work_unit_id": unit.work_unit_id, "attempt_id": attempt.attempt_id},
    )
    return attempt.attempt_id, unit.work_unit_id


@pytest.mark.asyncio
async def test_local_runtime_recovers_interrupted_attempts_without_network(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "recover-local.db"
    async with SqliteStore(database_path) as store:
        attempt_id, work_unit_id = await _persist_running_attempt(store)

    adapter = FakeAdapter()
    runtime = LocalRuntime(
        Settings(database_path=database_path),
        adapters={"worker": adapter},
    )
    await runtime.open()
    try:
        report = runtime.recovery
        assert report is not None
        assert report.interrupted_attempt_ids == (attempt_id,)
        recovered = await runtime.store.get_attempt(attempt_id)
        assert recovered.status is AttemptStatus.INTERRUPTED
        assert recovered.status is not AttemptStatus.SUCCEEDED
        unit = await runtime.store.get_work_unit(work_unit_id)
        assert unit.status is WorkUnitStatus.FAILED
        supervisor = runtime.composition.supervisor
        assert supervisor is not None
        assert supervisor.running
    finally:
        await runtime.close()

    assert runtime.state is LocalRuntimeState.CLOSED
    assert adapter.close_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        _ = runtime.store
    async with SqliteStore(database_path) as store:
        recovered = await store.get_attempt(attempt_id)
        assert recovered.status is AttemptStatus.INTERRUPTED


@pytest.mark.asyncio
async def test_local_recovery_hold_survives_scan_until_bind_and_resume(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "hold-local.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with SqliteStore(database_path) as store:
        held_run_id = await _persist_pending_run(store)
        other_run_id = await _persist_pending_run(store)

    runtime = LocalRuntime(
        Settings(database_path=database_path, worker_profile=WORKER_PROFILE),
        adapters={"worker": FakeAdapter()},
    )
    await runtime.open()
    try:
        report = runtime.recovery
        assert report is not None
        assert held_run_id in report.recoverable_run_ids
        assert other_run_id in report.recoverable_run_ids
        supervisor = runtime.composition.supervisor
        assert supervisor is not None
        assert held_run_id in supervisor.held_run_ids
        assert other_run_id in supervisor.held_run_ids
        assert held_run_id not in supervisor._queued
        assert other_run_id not in supervisor._queued
        discovered = await supervisor.scan()
        assert held_run_id not in discovered
        assert other_run_id not in discovered
        assert held_run_id not in supervisor._queued
        assert held_run_id not in supervisor.active_run_ids
        persisted = await runtime.store.get_run(held_run_id)
        assert persisted.status is RunStatus.PENDING
        with pytest.raises(RuntimeError, match="not bound"):
            runtime._release_held_recovery_run(held_run_id)
        assert held_run_id in supervisor.held_run_ids
        runtime.bind_workspace(workspace)
        still_held = await supervisor.scan()
        assert held_run_id not in still_held
        assert other_run_id not in still_held
        runtime._release_held_recovery_run(held_run_id)
        assert held_run_id not in supervisor.held_run_ids
        assert other_run_id in supervisor.held_run_ids
        created = await runtime.run("summarise the report", workspace=workspace)
        assert created.status is RunStatus.SUCCEEDED
        assert created.run_id not in supervisor.held_run_ids
        leftover = await runtime.store.get_run(other_run_id)
        assert leftover.status is RunStatus.PENDING
        released = await runtime.store.get_run(held_run_id)
        assert released.run_id == held_run_id
        assert released.status is not RunStatus.PENDING
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_local_runtime_failed_open_stays_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeAdapter()

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("supervisor start failed")

    monkeypatch.setattr("prp_runtime.runtime.supervisor.RunSupervisor.start", boom)
    runtime = LocalRuntime(
        Settings(database_path=tmp_path / "fail-local.db"),
        adapters={"worker": adapter},
    )
    with pytest.raises(RuntimeError, match="supervisor start failed"):
        await runtime.open()
    assert runtime.state is LocalRuntimeState.CLOSED
    assert adapter.close_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        _ = runtime.controller


def test_local_runtime_admission_rejects_requests_that_raise_the_envelope(
    tmp_path: Path,
) -> None:
    runtime = LocalRuntime(
        Settings(
            database_path=tmp_path / "envelope.db",
            process_max_concurrency=1,
            process_max_attempts=2,
            process_max_total_tokens=100,
            isolation_max_slots=2,
            isolation_max_bytes=1024,
        ),
        adapters={"worker": FakeAdapter()},
    )
    with pytest.raises(DomainValidationError) as raised:
        runtime.admit(concurrency=2)
    assert raised.value.code is ErrorCode.INVALID_BUDGET
    assert raised.value.detail.field == "max_concurrency"
    with pytest.raises(DomainValidationError) as raised:
        runtime.admit(attempts=3)
    assert raised.value.detail.field == "max_attempts"
    with pytest.raises(DomainValidationError) as raised:
        runtime.admit(tokens=101)
    assert raised.value.detail.field == "max_total_tokens"
    with pytest.raises(DomainValidationError) as raised:
        runtime.admit(slots=3)
    assert raised.value.detail.field == "max_slots"
    with pytest.raises(DomainValidationError) as raised:
        runtime.admit(copied_bytes=1025)
    assert raised.value.detail.field == "max_copied_bytes"
    with pytest.raises(DomainValidationError):
        runtime.admit(concurrency=0)
    with pytest.raises(DomainValidationError):
        runtime.admit(attempts=-1)
    assert runtime.held_capacity()["max_concurrency"] == 0
    lease = runtime.admit(concurrency=1)
    with pytest.raises(BudgetError) as raised:
        runtime.admit(concurrency=1)
    assert raised.value.code is ErrorCode.RESOURCE_BUDGET_EXCEEDED
    assert raised.value.detail.field == "max_concurrency"
    runtime.release(lease)
    runtime.release(lease)
    assert runtime.held_capacity()["max_concurrency"] == 0

    wide = LocalRuntime(
        Settings(
            database_path=tmp_path / "envelope-wide.db",
            process_max_concurrency=8,
            process_max_attempts=2,
            process_max_total_tokens=100,
            isolation_max_slots=2,
            isolation_max_bytes=1024,
        ),
        adapters={"worker": FakeAdapter()},
    )
    attempts_lease = wide.admit(concurrency=1, attempts=2)
    with pytest.raises(BudgetError) as raised:
        wide.admit(concurrency=1, attempts=1)
    assert raised.value.code is ErrorCode.ATTEMPT_BUDGET_EXCEEDED
    wide.release(attempts_lease)
    token_lease = wide.admit(concurrency=1, attempts=1, tokens=100)
    with pytest.raises(BudgetError) as raised:
        wide.admit(concurrency=1, attempts=1, tokens=1)
    assert raised.value.code is ErrorCode.TOKEN_BUDGET_EXCEEDED
    wide.release(token_lease)
    slot_lease = wide.admit(concurrency=1, slots=2)
    with pytest.raises(BudgetError) as raised:
        wide.admit(concurrency=1, slots=1)
    assert raised.value.detail.field == "max_slots"
    wide.release(slot_lease)
    byte_lease = wide.admit(concurrency=1, copied_bytes=1024)
    with pytest.raises(BudgetError) as raised:
        wide.admit(concurrency=1, copied_bytes=1)
    assert raised.value.detail.field == "max_copied_bytes"
    wide.release(byte_lease)
    assert wide.held_capacity()["max_copied_bytes"] == 0
    assert runtime.resource_envelope.max_concurrency == 1
    assert runtime.resource_envelope.max_total_tokens == 100


@pytest.mark.asyncio
async def test_local_runtime_releases_envelope_on_success_failure_and_cancel(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = Settings(
        database_path=tmp_path / "envelope-run.db",
        worker_profile=WORKER_PROFILE,
        process_max_concurrency=1,
    )

    async with LocalRuntime(settings, adapters={"worker": FakeAdapter()}) as runtime:
        result = await runtime.run("summarise the report", workspace=workspace)
        assert result.status is RunStatus.SUCCEEDED
        assert runtime.held_capacity()["max_concurrency"] == 0
        assert runtime.held_capacity()["max_attempts"] == 0
        assert runtime.held_capacity()["max_total_tokens"] == 0
        assert str(workspace) not in str(runtime.public_facts())

    failing = FakeAdapter()

    async def fail_complete(request: ProviderRequest) -> ProviderResponse:
        del request
        raise ProviderError("upstream failed", code=ErrorCode.PROVIDER_UNAVAILABLE)

    failing.complete = fail_complete  # type: ignore[method-assign]
    async with LocalRuntime(settings, adapters={"worker": failing}) as runtime:
        failed = await runtime.run("hello", workspace=workspace)
        assert failed.status is RunStatus.FAILED
        assert runtime.held_capacity()["max_concurrency"] == 0

    async with LocalRuntime(
        Settings(
            database_path=tmp_path / "envelope-cancel.db",
            worker_profile=WORKER_PROFILE,
            process_max_concurrency=2,
        ),
        adapters={"worker": FakeAdapter()},
    ) as runtime:
        with pytest.raises(DomainValidationError) as raised:
            await runtime.run("hello", workspace=workspace, concurrency=2)
        assert raised.value.code is ErrorCode.INVALID_AGENT_OPTIONS
        assert raised.value.detail.field == "execution_copy_mode"
        assert runtime.held_capacity()["max_concurrency"] == 0
        lease = runtime.admit(concurrency=1)
        created = await runtime.controller.create_run(NativeRunRequest(input="cancel me"))
        runtime._run_leases[created.run_id] = lease
        assert runtime.held_capacity()["max_concurrency"] == 1
        cancelled = await runtime.cancel(created.run_id)
        assert cancelled.status is RunStatus.CANCELLED
        assert runtime.held_capacity()["max_concurrency"] == 0


@pytest.mark.asyncio
async def test_local_run_enforces_attempts_tokens_and_in_place_zero_copy(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = Settings(
        database_path=tmp_path / "envelope-actual.db",
        worker_profile=WORKER_PROFILE,
        process_max_concurrency=2,
        process_max_attempts=2,
        process_max_total_tokens=2,
        isolation_max_slots=2,
        isolation_max_bytes=1024,
    )
    async with LocalRuntime(settings, adapters={"worker": FakeAdapter()}) as runtime:
        assert runtime.resource_envelope.max_attempts == 2
        blocker = runtime.admit(concurrency=1, attempts=2)
        with pytest.raises(BudgetError) as raised:
            await runtime.run("hello", workspace=workspace)
        assert raised.value.code is ErrorCode.ATTEMPT_BUDGET_EXCEEDED
        assert raised.value.detail.field == "max_attempts"
        assert runtime.held_capacity()["max_attempts"] == 2
        runtime.release(blocker)
        slots = runtime.admit(concurrency=1, slots=2, copied_bytes=1024)
        result = await runtime.run("hello", workspace=workspace)
        assert result.status is RunStatus.SUCCEEDED
        assert result.usage.total_tokens == 2
        assert runtime.held_capacity()["max_attempts"] == 1
        assert runtime.held_capacity()["max_total_tokens"] == 0
        assert runtime.held_capacity()["max_slots"] == 2
        runtime.release(slots)
        token_blocker = runtime.admit(concurrency=1, tokens=2)
        with pytest.raises(BudgetError) as raised:
            await runtime.run("hello", workspace=workspace)
        assert raised.value.code is ErrorCode.TOKEN_BUDGET_EXCEEDED
        assert raised.value.detail.field == "max_total_tokens"
        runtime.release(token_blocker)
        assert runtime.held_capacity()["max_total_tokens"] == 0
        assert runtime.resource_envelope.max_total_tokens == 2

    tight = Settings(
        database_path=tmp_path / "envelope-tight-tokens.db",
        worker_profile=WORKER_PROFILE,
        process_max_concurrency=1,
        process_max_total_tokens=1,
    )
    async with LocalRuntime(tight, adapters={"worker": FakeAdapter()}) as runtime:
        with pytest.raises(BudgetError) as raised:
            await runtime.run("hello", workspace=workspace)
        assert raised.value.code is ErrorCode.TOKEN_BUDGET_EXCEEDED
        assert raised.value.detail.field == "max_total_tokens"
        assert runtime.held_capacity()["max_total_tokens"] == 0
        assert runtime.held_capacity()["max_concurrency"] == 0
        assert runtime.last_run_id is not None
        persisted = await runtime.store.get_run(runtime.last_run_id)
        assert persisted.status is not RunStatus.SUCCEEDED
        assert persisted.status is not RunStatus.PENDING


class SlowAdapter:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "slow-fake"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        await asyncio.sleep(self.delay)
        return ProviderResponse(
            text="slow answer",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


@pytest.mark.asyncio
async def test_local_wait_allows_slow_fake_inside_ceiling(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = SlowAdapter(0.2)
    async with LocalRuntime(
        Settings(
            database_path=tmp_path / "wait-ok.db",
            worker_profile=WORKER_PROFILE,
            local_wait_seconds=1,
        ),
        adapters={"worker": adapter},
    ) as runtime:
        result = await runtime.run("hello", workspace=workspace)
        assert result.status is RunStatus.SUCCEEDED
        assert result.output_text == "slow answer"
        assert runtime.held_capacity()["max_concurrency"] == 0


@pytest.mark.asyncio
async def test_local_wait_deadline_is_structured_and_bounded(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = SlowAdapter(0.3)
    async with LocalRuntime(
        Settings(
            database_path=tmp_path / "wait-fail.db",
            worker_profile=WORKER_PROFILE,
            local_wait_seconds=0.05,
        ),
        adapters={"worker": adapter},
    ) as runtime:
        with pytest.raises(BudgetError) as raised:
            await runtime.run("hello", workspace=workspace)
        assert raised.value.code is ErrorCode.DEADLINE_EXCEEDED
        assert raised.value.detail.field == "deadline"
        assert runtime.held_capacity()["max_concurrency"] == 0
        assert runtime.held_capacity()["max_attempts"] == 0
        assert runtime.held_capacity()["max_total_tokens"] == 0
        assert runtime.last_run_id is not None
        persisted = await runtime.store.get_run(runtime.last_run_id)
        assert persisted.status is not RunStatus.SUCCEEDED
        assert persisted.status in {RunStatus.CANCELLED, RunStatus.CANCELLING}

