"""ASGI integration for all inbound bindings with fake providers."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prp_runtime.app import create_app
from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.models import Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="fake",
    model="fake-worker",
    role=ModelRole.WORKER,
    base_url="https://models.invalid/v1",
    context_window_tokens=16_000,
    max_output_tokens=2_000,
)


class FakeAdapter:
    def __init__(self, text: str = "bound answer") -> None:
        self.text = text
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "binding-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=self.text,
            usage=Usage(input_tokens=3, output_tokens=2, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


def _app(tmp_path: Path, adapter: FakeAdapter) -> FastAPI:
    app = create_app(
        Settings(database_path=tmp_path / "bindings.db", worker_profile=WORKER_PROFILE),
        adapters={"worker": adapter},  # type: ignore[arg-type]
    )
    return app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(_app(tmp_path, FakeAdapter())) as opened:
        yield opened


def wait_for_response(client: TestClient, run_id: str) -> dict[str, object]:
    for _ in range(200):
        body = client.get(f"/v1/responses/{run_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
    raise AssertionError("Responses run did not reach a terminal state")


def test_responses_create_query_and_cancel_share_one_envelope(
    client: TestClient,
) -> None:
    created = client.post(
        "/v1/responses",
        json={"input": "hello", "instructions": "be terse"},
    )
    assert created.status_code == 202
    created_body = created.json()
    assert created_body["status"] in {"pending", "in_progress", "completed"}
    body = wait_for_response(client, created_body["id"])
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output_text"] == "bound answer"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert "strategy" not in body
    assert "graph_version" not in body

    run_id = body["id"]
    assert client.get(f"/v1/responses/{run_id}").json() == body
    events = client.get(f"/v1/responses/{run_id}/events")
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "response.run_created" in events.text
    cancelled = client.post(f"/v1/responses/{run_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "completed"


def test_chat_create_maps_system_and_user_text(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "bound answer",
    }
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_anthropic_create_query_and_cancel(client: TestClient) -> None:
    created = client.post(
        "/v1/messages",
        json={
            "system": "be concise",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["status"] == "completed"
    assert body["content"] == [{"type": "text", "text": "bound answer"}]
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert "strategy" not in body
    run_id = body["id"]
    assert client.get(f"/v1/messages/{run_id}").json() == body
    assert client.post(f"/v1/messages/{run_id}/cancel").json()["status"] == "completed"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/responses",
            {"input": "hello", "routing": {"requires_cascade": True}},
        ),
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "routing": {"requires_cascade": True},
            },
        ),
        (
            "/v1/messages",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "routing": {"requires_cascade": True},
            },
        ),
    ],
)
def test_external_routing_intent_reaches_controller(
    client: TestClient,
    path: str,
    payload: dict[str, object],
) -> None:
    created = client.post(path, json=payload)

    if path == "/v1/responses":
        assert created.status_code == 202
        body = wait_for_response(client, created.json()["id"])
    else:
        assert created.status_code == 200
        body = created.json()
    native = client.get(f"/v1/runs/{body['id']}")
    assert native.status_code == 200
    assert native.json()["strategy"] == "CASCADE"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/responses",
            {"input": "hello", "routing": {"desired_parallelism": 0}},
        ),
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "routing": {"desired_parallelism": 0},
            },
        ),
        (
            "/v1/messages",
            {
                "messages": [{"role": "user", "content": "hello"}],
                "routing": {"desired_parallelism": 0},
            },
        ),
    ],
)
def test_external_invalid_routing_value_is_structured(
    client: TestClient,
    path: str,
    payload: dict[str, object],
) -> None:
    response = client.post(path, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value
    assert response.json()["error"]["field"] == "routing"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            ErrorCode.UNSUPPORTED_STREAM_MODE,
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": "private"}],
                    }
                ]
            },
            ErrorCode.UNSUPPORTED_MODALITY,
        ),
    ],
)
def test_anthropic_errors_are_structured(
    client: TestClient,
    payload: dict[str, object],
    code: ErrorCode,
) -> None:
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code.value
    assert "private" not in response.text


@pytest.mark.parametrize(
    ("path", "payload", "code"),
    [
        (
            "/v1/responses",
            {"input": "hello", "stream": True},
            ErrorCode.UNSUPPORTED_STREAM_MODE,
        ),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}], "tools": []},
            ErrorCode.UNSUPPORTED_TOOLS,
        ),
        (
            "/v1/responses",
            {"input": "hello", "base_url": "https://private.invalid"},
            ErrorCode.UNSUPPORTED_FIELD,
        ),
    ],
)
def test_binding_errors_are_structured_and_redacted(
    client: TestClient,
    path: str,
    payload: dict[str, object],
    code: ErrorCode,
) -> None:
    response = client.post(path, json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code.value
    assert "private.invalid" not in response.text
