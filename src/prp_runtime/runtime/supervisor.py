"""Durable Run wake-up supervision for one runtime process.

The Store remains the source of truth. The in-memory queue only wakes a bounded
scan; a dropped wake-up is recovered by the next scan and by an explicit
``scan`` call after restart.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection
from typing import Protocol, runtime_checkable

from prp_runtime.domain.enums import RunStatus
from prp_runtime.domain.models import Run

__all__ = [
    "PendingRunStore",
    "RunExecutor",
    "RunSupervisor",
    "SupervisorState",
]


@runtime_checkable
class PendingRunStore(Protocol):
    """The minimal persisted read contract needed by the supervisor."""

    async def list_pending_runs(self) -> Collection[Run]:
        """Return every currently persisted run eligible for execution."""


RunExecutor = Callable[[str], Awaitable[Run]]


class SupervisorState:
    """Public counters useful to tests and lifecycle owners."""

    def __init__(self) -> None:
        self.scans = 0
        self.enqueued = 0
        self.executed = 0
        self.failed = 0


class RunSupervisor:
    """Execute persisted pending Runs with bounded, idempotent wake-ups."""

    def __init__(
        self,
        store: PendingRunStore,
        execute: RunExecutor,
        *,
        max_concurrency: int = 1,
        scan_interval: float = 1.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if scan_interval <= 0:
            raise ValueError("scan_interval must be positive")
        self._store = store
        self._execute = execute
        self._max_concurrency = max_concurrency
        self._scan_interval = scan_interval
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._drain = False
        self._loop_task: asyncio.Task[None] | None = None
        self._active: dict[str, asyncio.Task[None]] = {}
        self._queued: set[str] = set()
        self._blocked_run_ids: set[str] = set()
        self._held_run_ids: set[str] = set()
        self._stopped = False
        self._slots = asyncio.Semaphore(max_concurrency)
        self.state = SupervisorState()

    @property
    def running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def active_run_ids(self) -> frozenset[str]:
        return frozenset(self._active)

    @property
    def held_run_ids(self) -> frozenset[str]:
        return frozenset(self._held_run_ids)

    def hold_runs(self, run_ids: Collection[str]) -> None:
        """Keep recovered runs off later scans until an explicit release."""
        for run_id in run_ids:
            if not run_id.strip():
                continue
            self._held_run_ids.add(run_id)
            self._queued.discard(run_id)

    def release_held_run(self, run_id: str) -> None:
        """Allow one previously held run to be queued again."""
        self._held_run_ids.discard(run_id)

    def _is_gated(self, run_id: str) -> bool:
        return run_id in self._blocked_run_ids or run_id in self._held_run_ids

    async def start(
        self,
        *,
        recoverable_run_ids: Collection[str] = (),
        blocked_run_ids: Collection[str] = (),
    ) -> None:
        """Start one loop and bootstrap only the recovery-approved Run ids.

        Later scans still discover newly-created pending runs. The blocked set
        remains excluded until the process is restarted and recovery produces a
        new durable decision.
        """
        if self._stopped:
            raise RuntimeError("supervisor has been stopped")
        if self.running:
            return
        self._blocked_run_ids = set(blocked_run_ids)
        self._queued.update(recoverable_run_ids)
        self._stop.clear()
        self._drain = False
        self._loop_task = asyncio.create_task(self._run_loop())
        self._wake.set()

    async def enqueue(self, run_id: str) -> None:
        """Wake the loop for a run; the id is not an execution fact."""
        if not run_id.strip():
            raise ValueError("run_id must not be blank")
        if self._stopped:
            raise RuntimeError("supervisor has been stopped")
        if self._is_gated(run_id):
            return
        self._queued.add(run_id)
        self.state.enqueued += 1
        self._wake.set()

    async def scan(self) -> tuple[str, ...]:
        """Scan Store truth and schedule eligible pending Runs."""
        if self._stopped:
            raise RuntimeError("supervisor has been stopped")
        discovered = await self._scan_store()
        self._wake.set()
        return discovered

    async def _scan_store(self) -> tuple[str, ...]:
        """Read Store truth without recursively waking the scan loop."""
        self.state.scans += 1
        runs = await self._store.list_pending_runs()
        discovered: list[str] = []
        for run in runs:
            if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                continue
            if self._is_gated(run.run_id):
                continue
            discovered.append(run.run_id)
            self._queued.add(run.run_id)
        return tuple(discovered)

    async def stop(self, *, drain: bool = True) -> None:
        """Stop accepting work and optionally await active work to finish."""
        if self._stopped and self._loop_task is None:
            return
        self._stopped = True
        self._drain = drain
        self._stop.set()
        self._wake.set()
        loop_task, self._loop_task = self._loop_task, None
        if loop_task is not None:
            await loop_task
        if drain and self._active:
            await asyncio.gather(*tuple(self._active.values()))
        elif not drain:
            for task in tuple(self._active.values()):
                task.cancel()
            if self._active:
                await asyncio.gather(*tuple(self._active.values()), return_exceptions=True)

    async def _run_loop(self) -> None:
        while True:
            await self._dispatch_queued()
            if self._stop.is_set():
                if self._drain and self._active:
                    await asyncio.gather(*tuple(self._active.values()))
                return
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._scan_interval)
            except TimeoutError:
                pass
            self._wake.clear()
            await self._scan_store()

    async def _dispatch_queued(self) -> None:
        if not self._queued:
            return
        runs = {run.run_id: run for run in await self._store.list_pending_runs()}
        pending = tuple(self._queued)
        self._queued.clear()
        for run_id in pending:
            if self._is_gated(run_id):
                continue
            run = runs.get(run_id)
            if run is None or run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                continue
            if run_id in self._active:
                continue
            task = asyncio.create_task(self._execute_one(run_id))
            self._active[run_id] = task

    async def _execute_one(self, run_id: str) -> None:
        current = asyncio.current_task()
        try:
            async with self._slots:
                await self._execute(run_id)
            self.state.executed += 1
        except asyncio.CancelledError:
            raise
        except BaseException:
            self.state.failed += 1
        finally:
            if current is not None and self._active.get(run_id) is current:
                del self._active[run_id]
