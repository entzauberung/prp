"""Real SSE lifecycle plus local cancellation/restart checks."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from external_tests.result_ledger import LedgerEntry, LedgerStore
from external_tests.support import ExternalConfig, validate_external_url
from external_tests.test_live_deepseek import _profile_for_runtime
from external_tests.test_live_protocols import PROTOCOL_RESULT_FILE, _profile_by_alias
from prp_runtime.app import create_app
from prp_runtime.domain.enums import ModelRole, RunStatus
from prp_runtime.domain.models import Usage
from prp_runtime.providers.base import FinishReason, ModelProfile, ProviderRequest, ProviderResponse
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

LIFECYCLE_RESULT_ID_PREFIX = "wo-003-st-002-real-lifecycle"
LOCAL_PROFILE = ModelProfile(
    alias="local",
    provider="local",
    model="local-model",
    role=ModelRole.WORKER,
    base_url="https://local.invalid",
    context_window_tokens=8_192,
    max_output_tokens=128,
)


class LocalAdapter:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "local"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text="local-ready",
            usage=Usage(input_tokens=2, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )

    async def aclose(self) -> None:
        return None


def _local_app(database: Path, adapter: LocalAdapter) -> FastAPI:
    return create_app(
        Settings(database_path=database, worker_profile=LOCAL_PROFILE),
        adapters={"local": adapter},
    )


def _wait_native(client: TestClient, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body.get("status") in {status.value for status in RunStatus if status.is_terminal}:
            return body
        time.sleep(0.01)
    raise AssertionError("local lifecycle run did not terminate")


def _event_ids(body: str) -> list[int]:
    return [int(line.removeprefix("id: ")) for line in body.splitlines() if line.startswith("id: ")]


async def _attempt_count(database: Path, run_id: str) -> int:
    async with SqliteStore(database) as store:
        return len(await store.list_run_attempts(run_id))


@pytest.mark.live_protocols
def test_lifecycle_real_sse_replay_and_cancel(
    external_config: ExternalConfig,
    temporary_resources: Any,
) -> None:
    alias = "DEEPSEEK_FLASH_RESPONSES"
    if not any(
        entry.alias == alias and entry.status == "PASS"
        for entry in LedgerStore(PROTOCOL_RESULT_FILE).read()
    ):
        pytest.skip("real lifecycle requires a prior Responses provider PASS")
    profile = _profile_by_alias(external_config, alias)
    validate_external_url(profile.base_url, external_config.allowed_hosts)
    runtime_profile = _profile_for_runtime(profile)
    run_id: str | None = None
    app = create_app(
        Settings(
            database_path=temporary_resources.database_path,
            worker_profile=runtime_profile,
        )
    )
    with TestClient(app) as test_client:
        created = test_client.post(
            "/v1/responses", json={"input": "Reply with one short word: ready."}
        )
        if created.status_code != 202:
            raise AssertionError("real Responses lifecycle create failed")
        run_id = created.json()["id"]
        completed = test_client.get(f"/v1/responses/{run_id}").json()
        for _ in range(200):
            if completed.get("status") == "completed":
                break
            time.sleep(0.02)
            completed = test_client.get(f"/v1/responses/{run_id}").json()
        if completed.get("status") != "completed":
            raise AssertionError("real Responses lifecycle did not complete")
        stream = test_client.get(f"/v1/responses/{run_id}/events")
        ids = _event_ids(stream.text)
        if not ids or ids != list(range(1, len(ids) + 1)):
            raise AssertionError("SSE sequence is not gapless")
        replay = test_client.get(f"/v1/responses/{run_id}/events?after={ids[0]}")
        if _event_ids(replay.text) != ids[1:]:
            raise AssertionError("SSE cursor replay does not resume after the cursor")
        cancelled = test_client.post(f"/v1/responses/{run_id}/cancel").json()
        if cancelled.get("status") != "completed":
            raise AssertionError("cancelling a terminal real run changed its status")

    assert run_id is not None
    attempts = asyncio.run(_attempt_count(temporary_resources.database_path, run_id))
    if attempts != 1:
        raise AssertionError("SSE replay/cancel created a duplicate provider attempt")
    LedgerStore(PROTOCOL_RESULT_FILE).merge(
        [
            LedgerEntry(
                scenario_id=f"{LIFECYCLE_RESULT_ID_PREFIX}-{run_id}",
                alias=profile.alias,
                model_id=profile.model_id,
                protocol="OPENAI_RESPONSES->SSE_REPLAY_CANCEL",
                endpoint_host="api.deepseek.com",
                run_id=run_id,
                attempt_id="one-persisted-attempt",
                status="PASS",
                actual_or_simulated="ACTUAL",
                input_tokens=None,
                output_tokens=None,
                known_cost="unknown",
                latency_ms=None,
                error_code=None,
                output_sha256=None,
                recorded_at="completed",
            )
        ],
        secret_values=[profile.api_key.get_secret_value()],
    )


@pytest.mark.live_protocols
def test_lifecycle_local_cancel_is_idempotent_and_restart_is_quiet(tmp_path: Path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    first_adapter = LocalAdapter()
    with TestClient(_local_app(database, first_adapter)) as client:
        created = client.post("/v1/runs", json={"input": "local lifecycle"}).json()
        finished = _wait_native(client, created["run_id"])
        first_cancel = client.post(f"/v1/runs/{created['run_id']}/cancel").json()
        second_cancel = client.post(f"/v1/runs/{created['run_id']}/cancel").json()
    if finished["status"] != RunStatus.SUCCEEDED.value:
        raise AssertionError("local lifecycle setup did not finish")
    if first_cancel != second_cancel or len(first_adapter.requests) != 1:
        raise AssertionError("duplicate cancel was not an idempotent terminal replay")

    second_adapter = LocalAdapter()
    with TestClient(_local_app(database, second_adapter)) as client:
        if client.get("/health").status_code != 200:
            raise AssertionError("restart health check failed")
    if second_adapter.requests:
        raise AssertionError("restart re-dispatched a terminal run")
