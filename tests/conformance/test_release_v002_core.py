"""Production-path conformance matrix for the v0.0.2 core contracts.

These scenarios deliberately use the public application, Store, and recovery
interfaces. The provider is a deterministic in-process fake; no network or
private selector is involved.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from prp_runtime.app import create_app
from prp_runtime.domain.enums import (
    AttemptStatus,
    ModelRole,
    ReservationStatus,
    RunStatus,
)
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import Attempt, NativeRunRequest, Run, Usage, WorkUnit
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
from prp_runtime.storage.recovery import recover_after_restart
from prp_runtime.storage.sqlite import SqliteStore

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="openai_compatible",
    model="conformance-model",
    role=ModelRole.WORKER,
    base_url="https://models.internal/v1",
    context_window_tokens=32_000,
    max_output_tokens=4_000,
)


class ConformanceProvider:
    """A deterministic provider adapter for local production-path tests."""

    def __init__(self, text: str = "conformance answer") -> None:
        self.text = text
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "conformance"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=self.text,
            usage=Usage(input_tokens=3, output_tokens=2, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )

    async def aclose(self) -> None:
        return None


def _build_app(database_path: Path, provider: ConformanceProvider) -> FastAPI:
    return create_app(
        Settings(database_path=database_path, worker_profile=WORKER_PROFILE),
        adapters={"worker": provider},
    )


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, Any]:
    for _ in range(200):
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] in {status.value for status in RunStatus if status.is_terminal}:
            return body
        time.sleep(0.005)
    raise AssertionError(f"run {run_id} did not reach a terminal state")


def _event_names(sse_body: str) -> list[str]:
    return [
        line.removeprefix("event: ")
        for line in sse_body.splitlines()
        if line.startswith("event: ")
    ]


def test_final_node_artifact_and_live_event_contract(tmp_path: Path) -> None:
    provider = ConformanceProvider()
    with TestClient(_build_app(tmp_path / "final.db", provider)) as client:
        response = client.post("/v1/runs", json={"input": "produce an answer"})
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        final = _wait_for_terminal(client, run_id)

        assert final["status"] == RunStatus.SUCCEEDED.value
        assert final["output_text"] == "conformance answer"
        events = _event_names(client.get(f"/v1/runs/{run_id}/events").text)

    assert EventType.ARTIFACT_PRODUCED.value in events
    assert events[-1] == EventType.RUN_SUCCEEDED.value
    assert len(provider.requests) == 1


def test_async_creation_and_replay_cursor_are_observable(tmp_path: Path) -> None:
    provider = ConformanceProvider()
    with TestClient(_build_app(tmp_path / "async.db", provider)) as client:
        started = time.monotonic()
        response = client.post("/v1/runs", json={"input": "run asynchronously"})
        elapsed = time.monotonic() - started
        assert response.status_code == 201
        run_id = response.json()["run_id"]
        final = _wait_for_terminal(client, run_id)
        replay = _event_names(client.get(f"/v1/runs/{run_id}/events").text)
        resumed = _event_names(
            client.get(
                f"/v1/runs/{run_id}/events",
                headers={"Last-Event-ID": "1"},
            ).text
        )

    assert elapsed < 1.0
    assert final["status"] == RunStatus.SUCCEEDED.value
    assert resumed == replay[1:]


def test_recovery_reuses_durable_facts_without_duplicate_events(tmp_path: Path) -> None:
    database_path = tmp_path / "recovery.db"
    started = utc_now()
    run = Run(
        run_id=new_run_id(),
        request=NativeRunRequest(input="recover safely"),
        status=RunStatus.RUNNING,
        created_at=started,
        started_at=started,
    )
    unit = WorkUnit(
        work_unit_id=new_work_unit_id(),
        run_id=run.run_id,
        name="conformance",
        instruction="recover",
        created_at=started,
    )
    attempt = Attempt(
        attempt_id=new_attempt_id(),
        run_id=run.run_id,
        work_unit_id=unit.work_unit_id,
        attempt_index=1,
        role=ModelRole.WORKER,
        model=WORKER_PROFILE.model_ref,
        status=AttemptStatus.RUNNING,
        created_at=started,
        started_at=started,
    )

    async def scenario() -> tuple[AttemptStatus, int, int]:
        async with SqliteStore(database_path) as store:
            await store.create_run(run)
            await store.create_work_unit(unit)
            await store.create_attempt(attempt)
            first = await recover_after_restart(store)
            sequence = await store.last_sequence(run.run_id)
            second = await recover_after_restart(store)
            events = await store.list_events(run.run_id)
            recovered = await store.get_attempt(attempt.attempt_id)
            assert first.changed is True
            assert second.changed is False
            return recovered.status, sequence or 0, len(events)

    status, sequence, event_count = asyncio.run(scenario())
    assert status is AttemptStatus.INTERRUPTED
    assert sequence == event_count


def test_orphan_reservation_is_released_without_provider_reuse(tmp_path: Path) -> None:
    database_path = tmp_path / "reservation.db"
    started = utc_now()
    run = Run(
        run_id=new_run_id(),
        request=NativeRunRequest(input="release orphan"),
        status=RunStatus.RUNNING,
        created_at=started,
        started_at=started,
    )
    unit = WorkUnit(
        work_unit_id=new_work_unit_id(),
        run_id=run.run_id,
        name="reservation",
        instruction="reserve",
        created_at=started,
    )

    async def scenario() -> ReservationStatus:
        from prp_runtime.control.reservations import ReservationRequest

        async with SqliteStore(database_path) as store:
            await store.create_run(run)
            await store.create_work_unit(unit)
            reservation = await store.reserve_reservation(
                ReservationRequest(
                    run_id=run.run_id,
                    work_unit_id=unit.work_unit_id,
                    dispatch_key="conformance-reservation",
                ),
                created_at=started,
                held_at=started,
            )
            report = await recover_after_restart(store)
            assert report.released_reservation_ids == (reservation.reservation_id,)
            return (await store.get_reservation(reservation.reservation_id)).status

    assert asyncio.run(scenario()) is ReservationStatus.RELEASED
