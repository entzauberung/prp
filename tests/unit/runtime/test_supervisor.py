"""Targeted tests for the Store-backed Run supervisor."""

import asyncio
from collections.abc import Collection

import pytest

from prp_runtime.domain.enums import RunStatus
from prp_runtime.domain.models import NativeRunRequest, Run
from prp_runtime.domain.values import new_run_id
from prp_runtime.runtime.supervisor import RunSupervisor


def make_run(status: RunStatus = RunStatus.PENDING) -> Run:
    run = Run(run_id=new_run_id(), request=NativeRunRequest(input="run this"))
    if status is RunStatus.PENDING:
        return run
    return run.model_copy(
        update={
            "status": status,
            "started_at": run.created_at,
            "completed_at": run.created_at if status.is_terminal else None,
        }
    )


class FakeStore:
    def __init__(self, runs: Collection[Run] = ()) -> None:
        self.runs = {run.run_id: run for run in runs}

    async def list_pending_runs(self) -> Collection[Run]:
        return tuple(self.runs.values())

    def complete(self, run_id: str) -> None:
        run = self.runs[run_id]
        self.runs[run_id] = run.model_copy(
            update={
                "status": RunStatus.SUCCEEDED,
                "started_at": run.created_at,
                "completed_at": run.created_at,
            }
        )


async def wait_until(predicate: object, *, timeout: float = 1.0) -> None:
    async def check() -> None:
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0)

    await asyncio.wait_for(check(), timeout)


@pytest.mark.asyncio
async def test_scan_recovers_a_dropped_wake_and_executes_store_pending_run() -> None:
    run = make_run()
    store = FakeStore([run])
    executed: list[str] = []
    async def execute(run_id: str) -> Run:
        executed.append(run_id)
        store.complete(run_id)
        return store.runs[run_id]

    supervisor = RunSupervisor(store, execute)

    await supervisor.start()
    await supervisor.scan()
    await wait_until(lambda: executed == [run.run_id])
    await asyncio.wait_for(supervisor.stop(), timeout=1.0)

    assert supervisor.state.scans >= 1
    assert supervisor.state.executed == 1


@pytest.mark.asyncio
async def test_scan_dispatches_a_recoverable_running_approval_wait() -> None:
    run = make_run(RunStatus.RUNNING)
    store = FakeStore([run])
    executed: list[str] = []

    async def execute(run_id: str) -> Run:
        executed.append(run_id)
        store.complete(run_id)
        return store.runs[run_id]

    supervisor = RunSupervisor(store, execute)
    await supervisor.start(recoverable_run_ids=(run.run_id,))
    await wait_until(lambda: executed == [run.run_id])
    await asyncio.wait_for(supervisor.stop(), timeout=1.0)

    assert executed == [run.run_id]


@pytest.mark.asyncio
async def test_recovery_bootstrap_excludes_blocked_runs() -> None:
    recoverable = make_run()
    blocked = make_run()
    store = FakeStore([recoverable, blocked])
    executed: list[str] = []

    async def execute(run_id: str) -> Run:
        executed.append(run_id)
        store.complete(run_id)
        return store.runs[run_id]

    supervisor = RunSupervisor(store, execute)
    await supervisor.start(
        recoverable_run_ids=(recoverable.run_id,),
        blocked_run_ids=(blocked.run_id,),
    )
    await wait_until(lambda: executed == [recoverable.run_id])
    await asyncio.wait_for(supervisor.stop(), timeout=1.0)

    assert executed == [recoverable.run_id]
    assert supervisor.state.executed == 1


@pytest.mark.asyncio
async def test_same_run_id_is_never_executed_concurrently() -> None:
    run = make_run()
    store = FakeStore([run])
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum = 0

    async def execute(run_id: str) -> Run:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        started.set()
        await release.wait()
        active -= 1
        store.complete(run_id)
        return run

    supervisor = RunSupervisor(store, execute)
    await supervisor.start()
    await supervisor.enqueue(run.run_id)
    await supervisor.enqueue(run.run_id)
    await started.wait()
    await supervisor.enqueue(run.run_id)
    await asyncio.sleep(0)
    assert maximum == 1
    release.set()
    await asyncio.wait_for(supervisor.stop(), timeout=1.0)


@pytest.mark.asyncio
async def test_terminal_runs_are_not_executed_and_failures_are_isolated() -> None:
    terminal = make_run(RunStatus.SUCCEEDED)
    pending = make_run()
    failing = make_run()
    store = FakeStore([terminal, pending, failing])
    executed: list[str] = []

    async def execute(run_id: str) -> Run:
        executed.append(run_id)
        if run_id == failing.run_id:
            store.complete(run_id)
            raise RuntimeError("one run failed")
        store.complete(run_id)
        return pending

    supervisor = RunSupervisor(store, execute, max_concurrency=2)
    await supervisor.start()
    await supervisor.scan()
    await wait_until(lambda: len(executed) == 2)
    await asyncio.wait_for(supervisor.stop(), timeout=1.0)

    assert terminal.run_id not in executed
    assert set(executed) == {pending.run_id, failing.run_id}
    assert supervisor.state.failed == 1
    assert supervisor.state.executed == 1


@pytest.mark.asyncio
async def test_stop_drain_waits_for_active_run() -> None:
    run = make_run()
    store = FakeStore([run])
    release = asyncio.Event()

    async def execute(run_id: str) -> Run:
        await release.wait()
        store.complete(run_id)
        return run

    supervisor = RunSupervisor(store, execute)
    await supervisor.start()
    await supervisor.scan()
    await wait_until(lambda: run.run_id in supervisor.active_run_ids)
    stopping = asyncio.create_task(supervisor.stop(drain=True))
    await asyncio.sleep(0)
    assert not stopping.done()
    release.set()
    await asyncio.wait_for(stopping, timeout=1.0)
    assert supervisor.running is False


@pytest.mark.asyncio
async def test_shutdown_rejects_new_work_and_leaves_no_tasks() -> None:
    run = make_run()
    store = FakeStore([run])

    async def execute(run_id: str) -> Run:
        store.complete(run_id)
        return store.runs[run_id]

    supervisor = RunSupervisor(store, execute)
    await supervisor.start()
    await asyncio.wait_for(supervisor.stop(), timeout=1.0)

    with pytest.raises(RuntimeError, match="stopped"):
        await supervisor.enqueue(run.run_id)
    with pytest.raises(RuntimeError, match="stopped"):
        await supervisor.scan()
    assert supervisor.running is False
    assert supervisor.active_run_ids == frozenset()
