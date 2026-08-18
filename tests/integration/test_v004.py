"""v0.0.4 protocol fixtures through the production app wiring."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prp_runtime.app import create_app
from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.models import Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "protocol"

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="fake",
    model="fixture-worker",
    role=ModelRole.WORKER,
    base_url="https://models.invalid/v1",
    context_window_tokens=16_000,
    max_output_tokens=2_000,
)


class FakeAdapter:
    def __init__(self, text: str = "fixture answer") -> None:
        self.text = text
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "v004-fixture-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=self.text,
            usage=Usage(input_tokens=2, output_tokens=2),
            finish_reason=FinishReason.STOP,
        )


def _app(tmp_path: Path, adapter: FakeAdapter) -> FastAPI:
    return create_app(
        Settings(database_path=tmp_path / "v004.db", worker_profile=WORKER_PROFILE),
        adapters={"worker": adapter},  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("fixture", "path"),
    [
        ("openai_responses_text.json", "/v1/responses"),
        ("openai_chat_text.json", "/v1/chat/completions"),
        ("anthropic_messages_text.json", "/v1/messages"),
    ],
)
def test_protocol_fixture_uses_one_persisted_run_fact(
    tmp_path: Path,
    fixture: str,
    path: str,
) -> None:
    payload = json.loads((FIXTURE_ROOT / fixture).read_text(encoding="utf-8"))
    adapter = FakeAdapter("fixture answer")
    with TestClient(_app(tmp_path, adapter)) as client:
        response = client.post(path, json=payload)
        assert response.status_code == (202 if path == "/v1/responses" else 200)
        created = response.json()
        run_id = created["id"]
        if path == "/v1/responses":
            events = client.get(f"{path}/{run_id}/events")
            assert events.status_code == 200
        queried = client.get(f"{path}/{run_id}")
        body = queried.json()

    assert body["status"] == "completed"
    assert queried.status_code == 200
    assert body["error"] is None
    assert "strategy" not in body
    assert len(adapter.requests) == 1


def test_fixture_run_ledger_is_serializable_and_has_stable_terminal_events(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (FIXTURE_ROOT / "openai_responses_text.json").read_text(encoding="utf-8")
    )
    with TestClient(_app(tmp_path, FakeAdapter())) as client:
        response = client.post("/v1/responses", json=payload)
        run_id = response.json()["id"]
        events = client.get(f"/v1/runs/{run_id}/events")

    assert events.status_code == 200
    assert "RUN_SUCCEEDED" in events.text
    assert "CONTROLLER_DECISION" in events.text
